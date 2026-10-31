package dev.callguard.dsp

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/**
 * One explainable observation about the audio.
 *
 * [synthetic] is 0..1 "how much this cue looks synthetic", using hand-tuned
 * demo heuristics. It is shown to the user next to the model score so the verdict
 * is not a black box -- but it is not a calibrated probability.
 */
data class Cue(
    val label: String,
    val synthetic: Float,
    val detail: String,
)

object Features {

    private val centers: DoubleArray by lazy { MelSpectrogram().binCenterHz }

    fun cues(mel: Array<FloatArray>): List<Cue> {
        if (mel.isEmpty() || mel[0].isEmpty()) return emptyList()
        val nBins = mel[0].size
        val nFrames = mel.size

        var hfEnergy = 0.0
        var totalEnergy = 0.0
        var flatnessSum = 0.0
        var fluxSum = 0.0
        var magnitudeSum = 0.0

        val hfStart = centers.indexOfFirst { it >= 5000.0 }.let { if (it < 0) nBins else it }

        var prev = FloatArray(nBins)
        for (f in 0 until nFrames) {
            val row = mel[f]
            var frameSum = 0.0
            var logSum = 0.0
            for (k in 0 until nBins) {
                val x = exp(row[k].toDouble())
                frameSum += x
                logSum += ln(x + 1e-9)
                if (k >= hfStart) hfEnergy += x
            }
            totalEnergy += frameSum

            val arithmetic = frameSum / nBins
            val geometric = exp(logSum / nBins)
            flatnessSum += (geometric / (arithmetic + 1e-12)).coerceIn(0.0, 1.0)

            if (f > 0) {
                var diff = 0.0
                for (k in 0 until nBins) {
                    diff += abs(row[k] - prev[k]).toDouble()
                    magnitudeSum += abs(row[k].toDouble())
                }
                fluxSum += diff
            }
            prev = row
        }

        val hfRatio = (hfEnergy / (totalEnergy + 1e-12)).toFloat()
        val flatness = (flatnessSum / nFrames).toFloat()
        // Both sums cover the same frames/bins, so the ratio is already normalised.
        val flux = if (nFrames > 1) {
            (fluxSum / (magnitudeSum + 1e-9)).toFloat()
        } else {
            0f
        }

        return listOf(
            Cue(
                label = "High-frequency energy",
                synthetic = clamp01((0.08f - hfRatio) / 0.08f),
                detail = "${(hfRatio * 100).toInt()}% of energy above 5 kHz",
            ),
            Cue(
                label = "Spectral flatness",
                synthetic = clamp01((0.20f - flatness) / 0.20f),
                detail = "${(flatness * 100).toInt()}% flat (smoother = more synthetic)",
            ),
            Cue(
                label = "Frame-to-frame change",
                synthetic = clamp01((0.35f - flux) / 0.35f),
                detail = "${(flux * 100).toInt()}% relative frame flux",
            ),
        )
    }

    private fun clamp01(v: Float): Float = v.coerceIn(0f, 1f)
}
