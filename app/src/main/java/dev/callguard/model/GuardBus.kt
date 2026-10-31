package dev.callguard.model

import dev.callguard.dsp.Cue
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlin.math.abs

/**
 * Process-wide UI state. The capture service and the activity live in the same
 * process, so a simple StateFlow is enough -- no binding or broadcasts needed.
 */
object GuardBus {

    private val _state = MutableStateFlow(GuardUiState())
    val state: StateFlow<GuardUiState> = _state.asStateFlow()

    fun configureModel(name: String, isDemo: Boolean) = _state.update {
        it.copy(modelName = name, isDemoModel = isDemo)
    }

    fun liveStarted() = _state.update {
        it.copy(
            mode = Mode.LIVE,
            running = true,
            verdict = null,
            risk = null,
            rawRisk = null,
            confidence = null,
            cues = emptyList(),
            spectrogram = emptyList(),
            secondsAnalyzed = 0f,
            progress = null,
            error = null,
        )
    }

    fun liveStopped() = _state.update {
        it.copy(mode = Mode.IDLE, running = false, level = 0f, progress = null)
    }

    fun liveUpdate(
        risk: Float,
        rawRisk: Float,
        cues: List<Cue>,
        spectrogram: List<FloatArray>,
        level: Float,
        seconds: Float,
    ) = _state.update {
        it.copy(
            verdict = GuardUiState.verdictFor(risk),
            risk = risk,
            rawRisk = rawRisk,
            confidence = (abs(risk - 0.5f) * 2f).coerceIn(0f, 1f),
            cues = cues,
            spectrogram = spectrogram,
            level = level,
            secondsAnalyzed = seconds,
            error = null,
        )
    }

    fun fileStarted() = _state.update {
        it.copy(
            mode = Mode.FILE,
            running = false,
            verdict = null,
            risk = null,
            rawRisk = null,
            confidence = null,
            cues = emptyList(),
            spectrogram = emptyList(),
            progress = 0f,
            secondsAnalyzed = 0f,
            error = null,
        )
    }

    fun fileProgress(progress: Float, seconds: Float = 0f) = _state.update {
        it.copy(progress = progress.coerceIn(0f, 1f), secondsAnalyzed = seconds)
    }

    fun fileFinished(
        risk: Float,
        cues: List<Cue>,
        spectrogram: List<FloatArray>,
        level: Float,
        seconds: Float,
    ) = _state.update {
        it.copy(
            mode = Mode.FILE,
            running = false,
            verdict = GuardUiState.verdictFor(risk),
            risk = risk,
            rawRisk = risk,
            confidence = (abs(risk - 0.5f) * 2f).coerceIn(0f, 1f),
            cues = cues,
            spectrogram = spectrogram,
            level = level,
            progress = null,
            secondsAnalyzed = seconds,
            error = null,
        )
    }

    fun fail(message: String) = _state.update {
        it.copy(mode = Mode.IDLE, running = false, progress = null, error = message)
    }

    fun dismissError() = _state.update { it.copy(error = null) }

    fun reset() = _state.update { it.copy(mode = Mode.IDLE, running = false) }
}
