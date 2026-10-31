package dev.callguard.audio

import dev.callguard.dsp.MelConfig
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Downmix + resample to the 16 kHz mono the model expects.
 *
 * Downsampling goes through a windowed-sinc low-pass first; plain linear
 * interpolation would alias everything above 8 kHz straight into the band the
 * detector looks at.
 */
object Resampler {

    private const val TAPS = 31
    private const val CUTOFF_HZ = 7200.0

    fun toMono16k(interleaved: FloatArray, sourceRate: Int, channels: Int): FloatArray {
        require(channels > 0) { "channels must be positive" }
        if (interleaved.isEmpty()) return FloatArray(0)

        val frames = interleaved.size / channels
        val mono = FloatArray(frames)
        if (channels == 1) {
            System.arraycopy(interleaved, 0, mono, 0, frames)
        } else {
            val inv = 1f / channels
            for (i in 0 until frames) {
                var acc = 0f
                val base = i * channels
                for (c in 0 until channels) acc += interleaved[base + c]
                mono[i] = acc * inv
            }
        }

        if (sourceRate == MelConfig.SAMPLE_RATE) return mono

        val filtered = if (sourceRate > MelConfig.SAMPLE_RATE) lowPass(mono, sourceRate) else mono
        return resampleLinear(filtered, sourceRate, MelConfig.SAMPLE_RATE)
    }

    private fun lowPass(input: FloatArray, sampleRate: Int): FloatArray {
        val fc = CUTOFF_HZ / sampleRate
        if (fc >= 0.49) return input

        val center = TAPS / 2
        val kernel = DoubleArray(TAPS)
        var sum = 0.0
        for (i in 0 until TAPS) {
            val n = i - center
            val sinc = if (n == 0) 2.0 * fc else sin(2.0 * PI * fc * n) / (PI * n)
            val hamming = 0.54 - 0.46 * cos(2.0 * PI * i / (TAPS - 1))
            kernel[i] = sinc * hamming
            sum += kernel[i]
        }
        if (sum != 0.0) for (i in 0 until TAPS) kernel[i] /= sum

        val out = FloatArray(input.size)
        for (i in input.indices) {
            var acc = 0.0
            for (k in 0 until TAPS) {
                val idx = (i + k - center).coerceIn(0, input.size - 1)
                acc += kernel[k] * input[idx]
            }
            out[i] = acc.toFloat()
        }
        return out
    }

    private fun resampleLinear(input: FloatArray, from: Int, to: Int): FloatArray {
        val outLength = (input.size.toDouble() * to / from).roundToInt()
        if (outLength <= 0) return FloatArray(0)
        val out = FloatArray(outLength)
        val step = from.toDouble() / to
        for (i in 0 until outLength) {
            val pos = i * step
            val i0 = pos.toInt().coerceIn(0, input.size - 1)
            val i1 = (i0 + 1).coerceAtMost(input.size - 1)
            val frac = (pos - i0).toFloat()
            out[i] = input[i0] * (1f - frac) + input[i1] * frac
        }
        return out
    }
}
