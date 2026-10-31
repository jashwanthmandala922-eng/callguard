package dev.callguard.ml

import dev.callguard.dsp.Features

/**
 * Fallback used when the TFLite asset is missing or fails to load, and as the
 * explainability baseline. It is a hand-weighted combination of the same cues the
 * UI shows -- useful for keeping the app alive, useless as a real detector.
 */
class HeuristicDetector : DeepfakeDetector {

    override val name: String = "heuristic (no model)"
    override val isDemo: Boolean = true

    override fun score(mel: Array<FloatArray>): Float {
        val cues = Features.cues(mel)
        if (cues.isEmpty()) return 0.5f
        val s = cues.map { it.synthetic }
        val combined = 0.45f * s.getOrElse(0) { 0.5f } +
            0.35f * s.getOrElse(1) { 0.5f } +
            0.20f * s.getOrElse(2) { 0.5f }
        return (0.15f + 0.7f * combined).coerceIn(0f, 1f)
    }
}
