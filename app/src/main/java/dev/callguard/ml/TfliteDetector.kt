package dev.callguard.ml

import android.content.Context
import android.util.Log
import dev.callguard.dsp.MelConfig
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Runs the bundled TFLite model fully on device.
 *
 * Input contract (see tools/train_demo_model.py):
 *   float32 [1, N_MEL, WINDOW_FRAMES, 1], mel-major, z-scored over the window.
 * Output: float32 [1, 1] sigmoid -> P(synthetic).
 */
class TfliteDetector(
    context: Context,
    assetPath: String = "models/deepfake_detector.tflite",
    displayName: String = "callguard-cnn",
    override val isDemo: Boolean = false,
    override val caveat: String? = null,
    private val calibration: Calibration = Calibration.NONE,
) : DeepfakeDetector {

    override val name: String = displayName

    private val interpreter: Interpreter

    private val input = Array(1) {
        Array(MelConfig.N_MEL) { Array(MelConfig.WINDOW_FRAMES) { FloatArray(1) } }
    }
    private val output = Array(1) { FloatArray(1) }

    init {
        val model = loadModel(context, assetPath)
        interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(2) })

        val inShape = interpreter.getInputTensor(0).shape()
        val outShape = interpreter.getOutputTensor(0).shape()
        Log.i(TAG, "loaded $assetPath input=${inShape.toList()} output=${outShape.toList()}")

        require(inShape.size == 4 && inShape[1] == MelConfig.N_MEL && inShape[2] == MelConfig.WINDOW_FRAMES) {
            "Model input ${inShape.toList()} does not match " +
                "[1, ${MelConfig.N_MEL}, ${MelConfig.WINDOW_FRAMES}, 1]"
        }
    }

    override fun score(mel: Array<FloatArray>): Float {
        if (mel.isEmpty()) return 0.5f

        val frames = mel.size
        val start = max(0, frames - MelConfig.WINDOW_FRAMES)

        // z-score over exactly the values that go into the tensor, matching training
        var sum = 0.0
        var sumSq = 0.0
        for (b in 0 until MelConfig.N_MEL) {
            for (t in 0 until MelConfig.WINDOW_FRAMES) {
                val v = mel[min(start + t, frames - 1)][b].toDouble()
                sum += v
                sumSq += v * v
            }
        }
        val count = MelConfig.N_MEL * MelConfig.WINDOW_FRAMES
        val mean = sum / count
        val std = sqrt(max(sumSq / count - mean * mean, 0.0)) + 1e-5

        for (b in 0 until MelConfig.N_MEL) {
            for (t in 0 until MelConfig.WINDOW_FRAMES) {
                val v = mel[min(start + t, frames - 1)][b]
                input[0][b][t][0] = ((v - mean) / std).toFloat()
            }
        }

        return try {
            interpreter.run(input, output)
            calibration.apply(output[0][0].coerceIn(0f, 1f))
        } catch (t: Throwable) {
            Log.e(TAG, "inference failed", t)
            0.5f
        }
    }

    override fun close() {
        runCatching { interpreter.close() }
    }

    private fun loadModel(context: Context, path: String): ByteBuffer {
        val bytes = context.assets.open(path).use { it.readBytes() }
        val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()
        return buffer
    }

    private companion object {
        const val TAG = "CallGuard/TFLite"
    }
}
