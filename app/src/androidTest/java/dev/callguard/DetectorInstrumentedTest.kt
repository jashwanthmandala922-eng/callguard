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
 * The fixtures are synthetic (see tools/make_samples.py), so this only proves the
 * plumbing discriminates. It says nothing about real-world accuracy.
 */
@RunWith(AndroidJUnit4::class)
class DetectorInstrumentedTest {

    @Test
    fun syntheticFixtureScoresHigherThanHumanFixture() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val appContext = instrumentation.targetContext
        val fixtureContext = instrumentation.context

        val detector = DetectorFactory.create(appContext)
        assertTrue("expected the bundled TFLite model, got ${detector.name}", detector is TfliteDetector)

        val human = loadWav(fixtureContext, "human_like.wav").copyOf(MelConfig.WINDOW_SAMPLES)
        val synthetic = loadWav(fixtureContext, "synthetic_like.wav").copyOf(MelConfig.WINDOW_SAMPLES)

        val engine = DetectionEngine(detector)
        val humanScore = engine.analyzeWindow(human).rawRisk
        engine.reset()
        val syntheticScore = engine.analyzeWindow(synthetic).rawRisk

        Log.i(
            "CallGuard/Test",
            "human=$humanScore synthetic=$syntheticScore detector=${detector.name}",
        )
        println("human=$humanScore synthetic=$syntheticScore detector=${detector.name}")

        assertTrue(
            "synthetic fixture ($syntheticScore) should score above the human fixture ($humanScore)",
            syntheticScore > humanScore,
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
