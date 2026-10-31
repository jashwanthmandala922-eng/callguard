package dev.callguard.dsp

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * In-place iterative radix-2 FFT. Sizes must be powers of two.
 *
 * Double precision on purpose: the Kotlin feature extractor is checked against the
 * numpy implementation in tools/ with a strict tolerance.
 */
object Fft {

    fun transform(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        require(n == im.size) { "re/im length mismatch" }
        require(n > 0 && n and (n - 1) == 0) { "FFT size must be a power of two" }

        // bit reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }

        var len = 2
        while (len <= n) {
            val angle = -2.0 * PI / len
            val wRe = cos(angle)
            val wIm = sin(angle)
            var start = 0
            while (start < n) {
                var curRe = 1.0
                var curIm = 0.0
                val half = len shr 1
                for (k in 0 until half) {
                    val a = start + k
                    val b = a + half
                    val vRe = re[b] * curRe - im[b] * curIm
                    val vIm = re[b] * curIm + im[b] * curRe
                    val uRe = re[a]
                    val uIm = im[a]
                    re[a] = uRe + vRe
                    im[a] = uIm + vIm
                    re[b] = uRe - vRe
                    im[b] = uIm - vIm

                    val nextRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nextRe
                }
                start += len
            }
            len = len shl 1
        }
    }
}
