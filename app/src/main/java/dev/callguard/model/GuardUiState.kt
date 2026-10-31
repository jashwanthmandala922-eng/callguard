package dev.callguard.model

import dev.callguard.dsp.Cue

enum class Mode { IDLE, LIVE, FILE }

enum class Verdict { LIKELY_HUMAN, UNCERTAIN, LIKELY_SYNTHETIC }

data class GuardUiState(
    val mode: Mode = Mode.IDLE,
    val running: Boolean = false,
    val verdict: Verdict? = null,
    val risk: Float? = null,
    val rawRisk: Float? = null,
    val confidence: Float? = null,
    val cues: List<Cue> = emptyList(),
    val spectrogram: List<FloatArray> = emptyList(),
    val level: Float = 0f,
    val modelName: String = "loading…",
    val isDemoModel: Boolean = true,
    val progress: Float? = null,
    val secondsAnalyzed: Float = 0f,
    val error: String? = null,
) {
    companion object {
        fun verdictFor(risk: Float): Verdict = when {
            risk >= 0.65f -> Verdict.LIKELY_SYNTHETIC
            risk <= 0.35f -> Verdict.LIKELY_HUMAN
            else -> Verdict.UNCERTAIN
        }
    }
}
