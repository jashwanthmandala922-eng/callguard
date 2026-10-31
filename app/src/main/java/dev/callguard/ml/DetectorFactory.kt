package dev.callguard.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject

object DetectorFactory {

    private const val TAG = "CallGuard/Detector"
    private const val MODEL_ASSET = "models/deepfake_detector.tflite"
    private const val META_ASSET = "models/model_meta.json"

    fun create(context: Context): DeepfakeDetector {
        val meta = meta(context)
        val displayName = buildString {
            append(meta?.optString("name") ?: "demo-cnn")
            meta?.optInt("params")?.takeIf { it > 0 }?.let { append(" · ").append(it / 1000).append("k params") }
        }
        return try {
            TfliteDetector(context, MODEL_ASSET, displayName, isDemo = true)
        } catch (t: Throwable) {
            Log.w(TAG, "TFLite model unavailable, falling back to heuristics", t)
            HeuristicDetector()
        }
    }

    fun meta(context: Context): JSONObject? = runCatching {
        context.assets.open(META_ASSET).bufferedReader().use { JSONObject(it.readText()) }
    }.getOrNull()

    fun isDemoModel(context: Context): Boolean =
        meta(context)?.optString("purpose")?.contains("DEMO", ignoreCase = true) ?: true
}
