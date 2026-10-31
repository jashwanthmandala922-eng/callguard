package dev.callguard

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.callguard.audio.FileAudioDecoder
import dev.callguard.dsp.MelConfig
import dev.callguard.ml.DetectionEngine
import dev.callguard.ml.DetectorFactory
import dev.callguard.model.GuardBus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlin.math.max

class GuardViewModel(application: Application) : AndroidViewModel(application) {

    val state: StateFlow<dev.callguard.model.GuardUiState> = GuardBus.state

    init {
        val context = getApplication<Application>()
        val meta = DetectorFactory.meta(context)
        val name = meta?.optString("name") ?: "demo-cnn"
        val params = meta?.optInt("params") ?: 0
        val label = if (params > 0) "$name · ${params / 1000}k params" else name
        GuardBus.configureModel(label, DetectorFactory.isDemoModel(context))
    }

    /**
     * Decode the chosen file and slide a 2 s window over it. The reported risk is
     * the median window score; the spectrogram shown is the *worst* window, which is
     * more useful than a random one when you are reviewing a clip.
     */
    fun analyzeFile(uri: Uri) {
        if (GuardBus.state.value.running) return

        viewModelScope.launch(Dispatchers.Default) {
            GuardBus.fileStarted()
            val app = getApplication<Application>()
            var engine: DetectionEngine? = null

            try {
                val decoded = FileAudioDecoder.decode(app, uri) { progress ->
                    GuardBus.fileProgress(progress * 0.35f)
                }

                if (decoded.samples.size < MelConfig.WINDOW_SAMPLES) {
                    GuardBus.fail("That clip is shorter than 2 seconds.")
                    return@launch
                }

                engine = DetectionEngine(DetectorFactory.create(app))
                val samples = decoded.samples
                val window = MelConfig.WINDOW_SAMPLES
                val hop = MelConfig.SAMPLE_RATE // 1 s hop keeps long files quick
                val totalWindows = max(1, (samples.size - window) / hop + 1)

                val scores = ArrayList<Float>(totalWindows)
                var worst: DetectionEngine.Update? = null
                var position = 0
                var index = 0

                while (position + window <= samples.size) {
                    val update = engine.analyzeWindow(samples.copyOfRange(position, position + window))
                    scores.add(update.rawRisk)
                    if (worst == null || update.rawRisk > worst.rawRisk) worst = update

                    position += hop
                    index++
                    if (index % 3 == 0 || index == totalWindows) {
                        val p = 0.35f + 0.65f * (index.toFloat() / totalWindows)
                        GuardBus.fileProgress(p, (position + window).toFloat() / MelConfig.SAMPLE_RATE)
                    }
                }

                val best = worst
                if (best == null || scores.isEmpty()) {
                    GuardBus.fail("No audio could be analysed.")
                    return@launch
                }

                val median = scores.sorted()[scores.size / 2]
                GuardBus.fileFinished(
                    risk = median,
                    cues = best.cues,
                    spectrogram = best.mel.toList(),
                    level = best.level,
                    seconds = decoded.durationSeconds,
                )
            } catch (t: Throwable) {
                GuardBus.fail(t.message ?: "Could not analyse that file.")
            } finally {
                engine?.close()
            }
        }
    }

    fun clearError() = GuardBus.dismissError()

    fun reportError(message: String) = GuardBus.fail(message)
}
