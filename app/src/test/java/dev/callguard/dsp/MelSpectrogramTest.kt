package dev.callguard.dsp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Guards the Kotlin/Python contract.
 *
 * The fixture is produced by tools/train_demo_model.py (write_parity_fixture) from a
 * deterministic test tone. If this test fails, the app and the model disagree about
 * what the audio looks like and every score becomes meaningless.
 */
class MelSpectrogramTest {

    private val signal: FloatArray = FloatArray(8000) { i ->
        val t = i / 16000.0
        var v = 0.50 * sin(2 * PI * 440.0 * t) +
            0.30 * sin(2 * PI * 1234.5 * t) +
            0.15 * sin(2 * PI * 3001.0 * t) +
            0.05 * sin(2 * PI * 6400.0 * t)
        v *= 0.5 - 0.5 * cos(2 * PI * 3.0 * t)
        v.toFloat()
    }

    @Test
    fun matchesPythonReference() {
        val reference = loadReference()
        assertEquals("reference frame count", 48, reference.size)
        assertEquals("mel bins", MelConfig.N_MEL, reference[0].size)

        val mel = MelSpectrogram().logMel(signal)
        assertEquals("frame count", reference.size, mel.size)

        var maxDiff = 0.0
        var worst = ""
        for (f in mel.indices) {
            for (k in 0 until MelConfig.N_MEL) {
                val diff = abs(mel[f][k] - reference[f][k]).toDouble()
                if (diff > maxDiff) {
                    maxDiff = diff
                    worst = "frame=$f bin=$k kotlin=${mel[f][k]} python=${reference[f][k]}"
                }
            }
        }

        println("max |kotlin - python| = $maxDiff ($worst)")
        assertTrue(
            "log-mel diverges from the training pipeline: $worst (max diff $maxDiff)",
            maxDiff < 2e-3,
        )
    }

    @Test
    fun frameCountMatchesFormula() {
        val mel = MelSpectrogram()
        assertEquals(0, mel.frameCount(399))
        assertEquals(1, mel.frameCount(400))
        assertEquals(2, mel.frameCount(560))
        assertEquals(MelConfig.WINDOW_FRAMES, mel.frameCount(MelConfig.WINDOW_SAMPLES))
    }

    private fun loadReference(): Array<FloatArray> {
        val stream = javaClass.classLoader!!.getResourceAsStream("mel_reference.txt")
            ?: error("mel_reference.txt missing from test resources")
        val rows = stream.bufferedReader().use { reader ->
            reader.readLines()
                .filter { it.isNotBlank() && !it.startsWith("#") }
        }
        return rows.map { line ->
            line.trim().split(Regex("\\s+")).map { it.toFloat() }.toFloatArray()
        }.toTypedArray()
    }
}
