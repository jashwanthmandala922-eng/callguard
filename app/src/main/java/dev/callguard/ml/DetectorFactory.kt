package dev.callguard.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject
import kotlin.math.roundToInt

object DetectorFactory {

    private const val TAG = "CallGuard/Detector"
    private const val MODEL_ASSET = "models/deepfake_detector.tflite"
    private const val META_ASSET = "models/model_meta.json"

    data class ModelInfo(
        val name: String,
        val label: String,
        val caveat: String?,
        val isDemo: Boolean,
    )

    fun info(context: Context): ModelInfo {
        val meta = meta(context)
        val name = meta?.optString("name")?.takeIf { it.isNotBlank() } ?: "callguard-cnn"
        val params = meta?.optInt("params") ?: 0
        val metrics = meta?.optJSONObject("metrics")
        val eer = metrics?.optDouble("headline_eer", Double.NaN)
            ?.takeIf { !it.isNaN() }
            ?: metrics?.optDouble("validation_eer", Double.NaN) ?: Double.NaN
        val headlineSet = metrics?.optString("headline_set").orEmpty()
        val setLabel = if (headlineSet.contains("cross", ignoreCase = true)) {
            "cross-dataset"
        } else {
            "lab"
        }

        val label = buildString {
            append(name)
            if (params > 0) append(" · ").append(params / 1000).append("k params")
            if (!eer.isNaN()) {
                append(" · ").append((eer * 100).roundToInt()).append("% EER (").append(setLabel).append(")")
            }
        }
        val caveat = meta?.optString("caveat")?.takeIf { it.isNotBlank() }
        return ModelInfo(
            name = name,
            label = label,
            caveat = caveat,
            isDemo = meta?.optBoolean("demo_model", false) ?: false,
        )
    }

    fun create(context: Context): DeepfakeDetector {
        val info = info(context)
        return try {
            TfliteDetector(
                context = context,
                assetPath = MODEL_ASSET,
                displayName = info.label,
                isDemo = info.isDemo,
                caveat = info.caveat,
                calibration = calibration(context),
            )
        } catch (t: Throwable) {
            Log.w(TAG, "TFLite model unavailable, falling back to heuristics", t)
            HeuristicDetector()
        }
    }

    /**
     * Reads the shipped Platt parameters. Without them the raw sigmoid is used,
     * which for this model means the 0.5 decision boundary is far off the
     * measured operating point.
     */
    fun calibration(context: Context): Calibration {
        val block = meta(context)?.optJSONObject("calibration") ?: return Calibration.NONE
        if (block.optString("method") != "platt") return Calibration.NONE
        val a = block.optDouble("a", Double.NaN)
        val b = block.optDouble("b", Double.NaN)
        if (a.isNaN() || b.isNaN()) return Calibration.NONE
        return Calibration(a.toFloat(), b.toFloat())
    }

    fun meta(context: Context): JSONObject? = runCatching {
        context.assets.open(META_ASSET).bufferedReader().use { JSONObject(it.readText()) }
    }.getOrNull()
}
