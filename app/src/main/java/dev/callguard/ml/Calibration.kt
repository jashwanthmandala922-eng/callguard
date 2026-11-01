package dev.callguard.ml

import kotlin.math.exp
import kotlin.math.ln

/**
 * Monotone recalibration of the model's raw sigmoid output.
 *
 * The bundled CNN ranks well but its sigmoid is compressed: on unseen-generator
 * audio the equal-error operating point sits near 5e-4, so a UI that thresholds
 * at 0.5 would label everything human. Platt scaling,
 *
 *     calibrated = sigmoid(a * logit(raw) + b)
 *
 * maps that operating point to 0.5. Because the map is strictly increasing it
 * cannot change the measured EER -- it only makes the number and the decision
 * boundary meaningful. `a` and `b` are fitted offline (tools/calibrate_model.py)
 * on a disjoint half of the cross-dataset corpus and shipped in model_meta.json.
 */
data class Calibration(val a: Float, val b: Float) {

    fun apply(raw: Float): Float {
        val p = raw.coerceIn(EPS, 1f - EPS)
        val logit = ln(p / (1f - p))
        val z = a * logit + b
        return (1.0 / (1.0 + exp(-z.toDouble()))).toFloat().coerceIn(0f, 1f)
    }

    companion object {
        private const val EPS = 1e-7f

        /** Identity, used when metadata carries no calibration block. */
        val NONE = Calibration(1f, 0f)
    }
}
