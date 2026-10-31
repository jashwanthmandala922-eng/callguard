package dev.callguard.ml

/**
 * Scores a window of log-mel features.
 *
 * Implementations are not thread-safe; one instance per pipeline.
 */
interface DeepfakeDetector {

    /** Human-readable identifier shown in the UI. */
    val name: String

    /** True when this is the demo model / heuristic, not a validated detector. */
    val isDemo: Boolean

    /**
     * @param mel [frames][melBin] natural-log mel power, frames oldest-first.
     * @return probability that the window is synthetic, 0..1.
     */
    fun score(mel: Array<FloatArray>): Float

    fun close() {}
}
