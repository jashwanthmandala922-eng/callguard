package dev.callguard

import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.callguard.dsp.MelConfig
import dev.callguard.ml.DetectionEngine
import dev.callguard.ml.DetectorFactory
import dev.callguard.ml.TfliteDetector
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-end check on a real device/emulator: asset loading, TFLite inference, DSP.
 *
 * The fixtures are real clips from garystafford/deepfake-audio-detection (CC-BY-4.0),
 * a corpus the model is evaluated on but never trained on. This proves the shipped
 * artefact and the Kotlin feature path agree with the training pipeline; it is a
 * smoke test, not a measure of accuracy.
 */
@RunWith(AndroidJUnit4::class)
class DetectorInstrumentedTest {

    @Test
    fun fakeFixtureScoresHigherThanRealFixture() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val appContext = instrumentation.targetContext
        val fixtureContext = instrumentation.context

        val detector = DetectorFactory.create(appContext)
        assertTrue("expected the bundled TFLite model, got ${detector.name}", detector is TfliteDetector)

        val real = loadWav(fixtureContext, "real_speech.wav").copyOf(MelConfig.WINDOW_SAMPLES)
        val fake = loadWav(fixtureContext, "fake_speech.wav").copyOf(MelConfig.WINDOW_SAMPLES)

        val engine = DetectionEngine(detector)
        val realScore = engine.analyzeWindow(real).rawRisk
        engine.reset()
        val fakeScore = engine.analyzeWindow(fake).rawRisk

        Log.i(
            "CallGuard/Test",
            "real=$realScore fake=$fakeScore detector=${detector.name}",
        )
        println("real=$realScore fake=$fakeScore detector=${detector.name}")

        assertTrue(
            "fake fixture ($fakeScore) should score above the real fixture ($realScore)",
            fakeScore > realScore,
        )
    }

    private fun loadWav(context: Context, name: String): FloatArray {
        val bytes = context.assets.open(name).use { it.readBytes() }
        var offset = 12
        while (offset + 8 <= bytes.size) {
            val id = String(bytes, offset, 4, Charsets.US_ASCII)
            val size = leInt(bytes, offset + 4)
            if (id == "data") {
                val count = size / 2
                val out = FloatArray(count)
                for (i in 0 until count) {
                    val lo = bytes[offset + 8 + i * 2].toInt() and 0xFF
                    val hi = bytes[offset + 8 + i * 2 + 1].toInt()
                    out[i] = ((hi shl 8) or lo).toShort() / 32768f
                }
                return out
            }
            offset += 8 + size + (size and 1)
        }
        error("no data chunk in $name")
    }

    private fun leInt(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xFF) or
            ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
            ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
            ((bytes[offset + 3].toInt() and 0xFF) shl 24)
}
