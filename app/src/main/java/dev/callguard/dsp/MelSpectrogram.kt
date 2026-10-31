package dev.callguard.dsp

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow

/**
 * Log-mel spectrogram, bit-for-bit compatible with the training pipeline.
 *
 * Output is [frames][melBin] of natural-log mel power, frames produced with
 * `1 + (n - WIN) / HOP` (no centring, no padding).
 */
class MelSpectrogram {

    private val nBins = MelConfig.N_FFT / 2 + 1

    private val window = DoubleArray(MelConfig.WIN) {
        0.5 - 0.5 * cos(2.0 * PI * it / MelConfig.WIN)
    }

    private val filters: Array<DoubleArray> = buildFilters()

    /** Centre frequency of each mel band, used for the explainability cues. */
    val binCenterHz: DoubleArray = buildCenters()

    private val bandLo = IntArray(MelConfig.N_MEL)
    private val bandHi = IntArray(MelConfig.N_MEL)

    init {
        for (k in 0 until MelConfig.N_MEL) {
            var lo = nBins
            var hi = 0
            for (b in 0 until nBins) {
                if (filters[k][b] != 0.0) {
                    if (b < lo) lo = b
                    if (b > hi) hi = b
                }
            }
            bandLo[k] = if (hi >= lo) lo else 0
            bandHi[k] = if (hi >= lo) hi else -1
        }
    }

    fun frameCount(sampleCount: Int): Int =
        if (sampleCount < MelConfig.WIN) 0 else 1 + (sampleCount - MelConfig.WIN) / MelConfig.HOP

    fun logMel(samples: FloatArray): Array<FloatArray> {
        val frames = frameCount(samples.size)
        if (frames <= 0) return emptyArray()

        val out = Array(frames) { FloatArray(MelConfig.N_MEL) }
        val re = DoubleArray(MelConfig.N_FFT)
        val im = DoubleArray(MelConfig.N_FFT)
        val power = DoubleArray(nBins)

        for (f in 0 until frames) {
            val offset = f * MelConfig.HOP
            java.util.Arrays.fill(re, 0.0)
            java.util.Arrays.fill(im, 0.0)
            for (i in 0 until MelConfig.WIN) {
                re[i] = samples[offset + i].toDouble() * window[i]
            }

            Fft.transform(re, im)

            for (b in 0 until nBins) {
                power[b] = re[b] * re[b] + im[b] * im[b]
            }

            val row = out[f]
            for (k in 0 until MelConfig.N_MEL) {
                var acc = 0.0
                val filter = filters[k]
                for (b in bandLo[k]..bandHi[k]) {
                    acc += filter[b] * power[b]
                }
                row[k] = ln(acc + MelConfig.FLOOR).toFloat()
            }
        }
        return out
    }

    private fun buildFilters(): Array<DoubleArray> {
        val melMin = hzToMel(MelConfig.FMIN)
        val melMax = hzToMel(MelConfig.FMAX)
        val step = (melMax - melMin) / (MelConfig.N_MEL + 1)
        val hz = DoubleArray(MelConfig.N_MEL + 2) { melToHz(melMin + it * step) }
        val bins = IntArray(MelConfig.N_MEL + 2) {
            floor((MelConfig.N_FFT + 1) * hz[it] / MelConfig.SAMPLE_RATE).toInt()
        }

        val fb = Array(MelConfig.N_MEL) { DoubleArray(nBins) }
        for (k in 0 until MelConfig.N_MEL) {
            val lo = bins[k]
            val mid = bins[k + 1]
            val hi = bins[k + 2]
            for (b in lo until hi) {
                if (b < 0 || b >= nBins) continue
                fb[k][b] = when {
                    b <= mid && mid > lo -> (b - lo).toDouble() / (mid - lo)
                    b > mid && hi > mid -> (hi - b).toDouble() / (hi - mid)
                    else -> 0.0
                }
            }
        }
        return fb
    }

    private fun buildCenters(): DoubleArray {
        val melMin = hzToMel(MelConfig.FMIN)
        val melMax = hzToMel(MelConfig.FMAX)
        val step = (melMax - melMin) / (MelConfig.N_MEL + 1)
        return DoubleArray(MelConfig.N_MEL) { melToHz(melMin + (it + 1) * step) }
    }

    private fun hzToMel(f: Double): Double = 2595.0 * log10(1.0 + f / 700.0)

    private fun melToHz(m: Double): Double = 700.0 * (10.0.pow(m / 2595.0) - 1.0)
}
