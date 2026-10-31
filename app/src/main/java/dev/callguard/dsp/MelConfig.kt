package dev.callguard.dsp

/**
 * Single source of truth for the feature extraction pipeline.
 *
 * These numbers MUST match tools/train_demo_model.py exactly. If you retrain with
 * a different window size or mel count, change both sides and re-run the parity
 * test (`./gradlew test`).
 */
object MelConfig {
    const val SAMPLE_RATE = 16000
    const val N_FFT = 512
    const val WIN = 400          // 25 ms analysis window
    const val HOP = 160          // 10 ms frame hop
    const val N_MEL = 64
    const val FMIN = 20.0
    const val FMAX = 8000.0
    const val FLOOR = 1e-6

    /** Model context: 2.0 s of audio -> 198 frames. */
    const val WINDOW_SAMPLES = 32000
    const val WINDOW_FRAMES = 198

    /** How often we emit a new decision while streaming. */
    const val DECISION_HOP_SAMPLES = 8000

    /** Number of recent raw scores combined by median smoothing. */
    const val SMOOTHING_WINDOW = 5
}
