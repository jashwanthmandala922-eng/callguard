package dev.callguard.ml

import dev.callguard.dsp.Cue
import dev.callguard.dsp.Features
import dev.callguard.dsp.MelConfig
import dev.callguard.dsp.MelSpectrogram
import dev.callguard.dsp.RingBuffer
import kotlin.math.sqrt

/**
 * Turns a stream of PCM blocks into smoothed synthetic-speech scores.
 *
 * Streaming policy: keep a 2 s window, re-score every 0.5 s, then take the median of
 * the last few raw scores. Median smoothing is what stops the UI from flickering on
 * single odd windows.
 */
class DetectionEngine(
    private val detector: DeepfakeDetector,
    private val mel: MelSpectrogram = MelSpectrogram(),
) {

    class Update(
        val risk: Float,
        val rawRisk: Float,
        val cues: List<Cue>,
        val mel: Array<FloatArray>,
        val level: Float,
        val hasSpeech: Boolean,
    )

    private val ring = RingBuffer(MelConfig.WINDOW_SAMPLES + MelConfig.DECISION_HOP_SAMPLES)
    private var sinceDecision = 0
    private val recent = ArrayDeque<Float>()

    /** Feed captured PCM. Returns an update only when a new decision is due. */
    fun push(block: FloatArray): Update? {
        ring.write(block)
        sinceDecision += block.size
        if (sinceDecision < MelConfig.DECISION_HOP_SAMPLES) return null
        sinceDecision = 0
        val window = ring.readLast(MelConfig.WINDOW_SAMPLES) ?: return null
        return analyzeWindow(window)
    }

    /** Score an explicit 2 s window (also used for file analysis). */
    fun analyzeWindow(window: FloatArray): Update {
        val features = mel.logMel(window)
        val raw = detector.score(features)

        recent.addLast(raw)
        while (recent.size > MelConfig.SMOOTHING_WINDOW) recent.removeFirst()
        val risk = recent.sorted()[recent.size / 2]

        val level = rms(window)
        return Update(
            risk = risk,
            rawRisk = raw,
            cues = Features.cues(features),
            mel = features,
            level = level,
            hasSpeech = level > 0.004f,
        )
    }

    fun reset() {
        ring.clear()
        sinceDecision = 0
        recent.clear()
    }

    fun close() = detector.close()

    private fun rms(samples: FloatArray): Float {
        if (samples.isEmpty()) return 0f
        var acc = 0.0
        for (s in samples) acc += s.toDouble() * s
        return sqrt(acc / samples.size).toFloat()
    }
}
