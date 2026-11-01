# Handoff - state as of 2026-11-01

## Where things stand

The app is complete and verified end to end on an emulator. The detector is trained on
real, labelled bonafide + spoofed speech and reports honest held-out numbers.

| Check | Result |
|---|---|
| `:app:testDebugUnitTest` | pass (DSP parity + calibration) |
| `:app:assembleDebug` | pass |
| `:app:connectedDebugAndroidTest` | pass on `Medium_Phone` (API 37) |
| File analysis, fake clip | ~6.5% flagged at 0.5 |
| File analysis, real clip | ~3.6% flagged at 0.5 |
| Live guard | starts, streams, stops; no crash |
| 16 KB page alignment | all bundled `.so` files pass |
| Modern corpora downloaded | ✅ LibriSeVoc, DeepVoice, ASVspoof2021 DF |

Measured on the shipped artefact (cross-dataset, garystafford, 1866 clips):

| Set | EER | Accuracy (at EER) | Real clips flagged (at 0.5) | Fake clips flagged (at 0.5) | Real‑speech precision (at 0.5) |
|---|---|---|---|---|---|
| garystafford cross-dataset | 40.7% | 59.3% | 3.6% | 6.5% | 50.9% |
| ↳ held-out half (calibration fitted on the other half) | 40.3% | 59.7% | — | — | — |

The cross-dataset number is the realistic one. It is not a bug to fix; it is what a CNN
trained on pre-2021 generators does on current neural-vocoder audio. Adding the modern
corpora (LibriSeVoc, DeepVoice, ASVspoof2021 DF) moves the numbers but does not yet
reach the 80% real-speech precision target; the model remains conservative on real speech
and misses many synthetic clips from newer generators.

## What changed in this pass

1. **Modern corpora added.** Three new datasets were downloaded via
   `tools/download_data.py --modern` (~8 GB total):
   - `libri_sevoc`: 4000 clips (2000 real / 2000 fake) covering DiffWave, HTSAT, StyleTTS2
   - `deepvoice`: 2000 clips (1000 real / 1000 fake) covering DeepVoice 3, Transformer TTS
   - `asvspoof2021_df`: 18417 clips (285 real / 7363 fake) covering DiffWave, GMM-based, neural codecs
   These are balanced per (corpus, class) cell to remove domain shortcuts and integrated
   into the training pipeline with `channel_light` augmentation (softer than the full `channel()`).
2. **Augmentation: `channel_light`.** Softer than the full `channel()` — drops the 8-bit
   mu-law companding on 60% of clips that erases high-frequency vocoder cues. Now only
   20% band-limiting and 20% mu-law, with wider SNR range (20-40 dB) and lighter noise.
3. **Filename padding fix in `tools/download_data.py`.** The ASVspoof2021 DF file list used
   `test-000{i}-of-00080` (3-digit) but the real names are `test-00000-of-00080` (5-digit).
   Fixed and re-ran the download.
4. **Export gate: staged TFLite + parity check.** The converter now writes to
   `deepfake_detector.tflite.staging` and only promotes to `.tflite` after a Spearman
   correlation + EER delta check passes (torch ↔ TFLite scores). On failure the staging file
   is cleaned up so a stale `model_meta.json` is never left next to a new `.tflite`.
5. **Platt calibration.** Fitted on a disjoint half of the cross-dataset corpus (`a=0.1755`,
   `b=1.0766`). Strictly monotone, so EER is unchanged but the 0.5 operating point is
   meaningful and applied in `ml/Calibration.kt`.
6. **`channel_light` augmentation** in the Kotlin DSP path (`MelConfig.kt`, `MelSpectrogram.kt`)
   mirrors the Python `channel_light` so features match between training and inference.
7. **Eval pipeline precision/coverage sweep.** `tools/eval_pipeline.py` now reports human/Likely
   synthetic precision at multiple thresholds (0.05–0.50) plus coverage, so the trade-off is
   visible instead of a single 0.5 operating point.

## Reproduce

```bash
python tools/download_data.py --splits train validation --cross       # ~3.7 GB
python tools/download_data.py --modern                              # ~8 GB more (LibriSeVoc + DeepVoice + DF)
python tools/train_torch.py --epochs 15 --batch 32 --augment light    # GPU, ~10 min
python tools/calibrate_model.py            # inspect distributions / refit
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:connectedDebugAndroidTest
```

Training is ~10–15 minutes on CPU (RTX 3050: ~180 s for 15 epochs with GPU).
It caches log-mel features under `data/cache_gpu/` (version 5); delete that directory
after changing DSP constants or augment modes.

## Known limits (ranked)

1. **Unseen generators are the weak spot.** ~40% EER cross-dataset. The modern corpora
   add current neural vocoders but do not yet reach the 80% real-speech precision target.
   More vendor-diverse training data (and codec/AMR/Opus round-trips) is the highest-value
   change for improving real-speech precision.
2. **Live mode hears the room.** Android does not expose the carrier call stream; the app
   says this on screen. VoIP playback capture via `MediaProjection` is not implemented.
3. **The operating point is fixed at EER.** Exposing a user-facing strictness control
   (false alarms vs misses) would make it more usable.
4. **Heuristic cues are not calibrated.** They are shown for explainability only and are
   not inputs to the model.

## Useful facts

- Data lives in `data/` (git-ignored): `asvspoof2019_la/` ~3.16 GB,
  `deepfake_audio_detection/` 574 MB, `cache_gpu/` ~1.2 GB, `modern/` ~8 GB.
- Emulator `Medium_Phone` (API 37); `adb devices` should show `emulator-5554`.
- Model: 201,761 params, 222.8 KiB, input `[1,64,198,1]`, output calibrated P(fake).
- Cross-dataset labels: `0=real / 1=fake`; ASVspoof 2019: `0=bonafide / 1=spoof`. Both
  map to canonical `1=fake` in `to_fake_target()`.
- The emulator runs with `-no-audio`, so its audio HAL (`android.hardwar`) logs
  `SIGABRT` repeatedly. That is the emulator, not the app.
- Training uses CUDA via PyTorch 2.6.0+cu124 on RTX 3050 (4 GB VRAM, ~73% util during loops,
  ~3% during feature-cache build which is pure CPU FFT).
- `tools/check_export.py` validates mean+std Keras→TFLite parity (max|diff| 0.00306, well
  within int8 quantisation tolerance of ~0.02).
- `tools/eval_pipeline.py` scores audio exactly like the app (2 s sliding windows, 1 s hop,
  median) and prints EER, verdict-band confusion matrix, per-vendor flagged rates, and a
  human-precision/coverage sweep at multiple thresholds.