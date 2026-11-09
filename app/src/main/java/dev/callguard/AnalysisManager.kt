package dev.callguard

import android.content.Context
import android.net.Uri
import dev.callguard.audio.FileAudioDecoder
import dev.callguard.ml.DetectionEngine
import dev.callguard.ml.DetectorFactory
import dev.callguard.model.GuardBus
import dev.callguard.model.GuardUiState
import dev.callguard.NotificationHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.withContext

/** Manages AI analysis of audio recordings, particularly call recordings. */
object AnalysisManager {

    /** Analyze a recorded audio file using the AI model */
    fun analyzeRecording(recordingPath: String, context: Context) {
        GlobalScope.launch(Dispatchers.Default) {
            try {
                GuardBus.fileStarted()

                val decoder = FileAudioDecoder.decode(
                    context,
                    Uri.fromFile(java.io.File(recordingPath)),
                    maxSeconds = 300f,
                    onProgress = { progress ->
                        GuardBus.fileProgress(progress * 0.35f)
                    }
                )

                if (decoder.samples.size < MelConfig.WINDOW_SAMPLES) {
                    GuardBus.fail("Recording too short for analysis (need at least 2 seconds).")
                    return@launch
                }

                val detector = DetectorFactory.create(context)
                val engine = DetectionEngine(detector)

                // Analyze windows across the recording
                val window = MelConfig.WINDOW_SAMPLES
                val hop = MelConfig.SAMPLE_RATE // 1 second hop
                val totalWindows = max(1, (decoder.samples.size - window) / hop + 1)

                val scores = mutableListOf<Float>()
                val allCues = mutableListOf<dev.callguard.dsp.Cue>()
                val worstUpdate: MutableModel<DetectionEngine.Update?> = MutableModel(null)

                var position = 0
                var index = 0

                while (position + window <= decoder.samples.size) {
                    val update = engine.analyzeWindow(decoder.samples.copyOfRange(position, position + window))
                    scores.add(update.rawRisk)

                    // Track the worst (highest risk) window
                    if (worstUpdate.value == null || update.rawRisk > worstUpdate.value!!.rawRisk) {
                        worstUpdate.value = update
                    }

                    allCues.addAll(update.cues)

                    position += hop
                    index++

                    // Update progress every 3 windows or at the end
                    if (index % 3 == 0 || index == totalWindows) {
                        val p = 0.35f + 0.65f * (index.toFloat() / totalWindows)
                        GuardBus.fileProgress(p, (position + window).toFloat() / MelConfig.SAMPLE_RATE)
                    }
                }

                val best = worstUpdate.value
                if (best == null || scores.isEmpty()) {
                    GuardBus.fail("No audio could be analysed from the recording.")
                    return@launch
                }

                // Calculate median score
                val sortedScores = scores.sorted()
                val medianIndex = scores.size / 2
                val median = sortedScores[medianIndex]

                val verdict = GuardUiState.verdictFor(median)
                GuardBus.fileFinished(
                    risk = median,
                    cues = best.cues,
                    spectrogram = best.mel.toList(),
                    level = best.level,
                    seconds = decoder.durationSeconds,
                )

                // Show result notification
                NotificationHelper.showAnalysisResult(
                    context = context,
                    risk = median,
                    rawRisk = median,
                    verdict = verdict,
                    caveat = detector.caveat,
                    modelName = detector.name,
                    secondsAnalyzed = decoder.durationSeconds,
                )

            } catch (t: Throwable) {
                GuardBus.fail(t.message ?: "Could not analyse that recording.")
                // Still try to show a notification even on error
                try {
                    NotificationHelper.showAnalysisResult(
                        context = context,
                        risk = 0.5f,
                        rawRisk = 0.5f,
                        verdict = GuardUiState.Verdict.UNCERTAIN,
                        caveat = "Could not complete analysis",
                        modelName = "error",
                        secondsAnalyzed = 0f,
                    )
                } catch (e: Exception) {
                    Log.e("AnalysisManager", "Failed to show error notification", e)
                }
            }
        }
    }

    /** Simple mutable model wrapper for Kotlin */
    private class MutableModel<T>(var value: T?)