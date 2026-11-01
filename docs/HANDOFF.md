# Handoff — state as of 2026-11-01

## Where things stand

The app is complete and verified end to end on an emulator. The detector is trained on
real, labelled bonafide + spoofed speech and reports honest held-out numbers.

| Check | Result |
|---|---|
| `:app:testDebugUnitTest` | pass (DSP parity + calibration) |
| `:app:assembleDebug` | pass |
| `:app:connectedDebugAndroidTest` | pass on `Medium_Phone` (API 37) |
| File analysis, fake clip | 81% → "Likely synthetic" |
| File analysis, real clip | 53% → "Not sure" |
| Live guard | starts, streams, stops; no crash |
| 16 KB page alignment | all bundled `.so` files pass |

Measured on the shipped artefact:

| Set | EER | Accuracy |
|---|---|---|
| ASVspoof 2019 dev, speaker-disjoint (3,000 clips) | 2.53% | 97.5% |
| garystafford cross-dataset (1,866 clips) | 29.90% | 70.1% |
| ↳ held-out half (calibration fitted on the other half) | 29.15% | 70.9% |

The cross-dataset number is the realistic one. It is not a bug to fix; it is what an
ASVspoof-trained CNN does on unseen generators.

## What changed in this pass

1. **Training corpus.** The old ASVspoof 2021 mirror was corrupted (overlapping splits,
   acoustically identical validation classes). Replaced with
   `Bisher/ASVspoof_2019_LA`, the official 2019 LA protocol: 25,380 train / 24,844 dev,
   16 kHz, `speaker_id` present, `key` 0=bonafide / 1=spoof. The trainer now refuses to
   run if train and validation share a speaker (it prints `overlap=0`).
2. **Decision threshold.** The raw sigmoid is compressed: on cross-domain audio the EER
   point was `4.7e-4`, so the app's 0.35/0.65 thresholds never fired and everything read
   "Sounds human". Fixed with a Platt map fitted on a disjoint half of the cross corpus
   (`a=0.4930`, `b=3.7421`), stored in `model_meta.json` and applied in
   `ml/Calibration.kt`. The map is monotone, so the reported EER is unchanged; the
   held-out half is reported separately so there is no calibration leakage.
3. **16 KB pages.** `org.tensorflow:tensorflow-lite:2.17.0` ships 4 KB-aligned `.so`
   files, which triggered Android's "not 16 KB compatible" dialog. Switched to
   `com.google.ai.edge.litert:litert:1.4.2` (same `org.tensorflow.lite.Interpreter`
   API, 16 KB-aligned). Verified with `tools/check_elf_alignment.py`.
4. **UI honesty.** Removed the "confidence N%" line — with a ~30% EER, distance from 0.5
   overstates certainty. The model caveat and EER stay visible on the main screen.

## Reproduce

```bash
python tools/download_data.py --splits train validation --cross   # ~3.7 GB
python tools/train_real_model.py --train-limit 10000 --val-limit 3000 --epochs 10
python tools/calibrate_model.py            # inspect distributions / refit
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:connectedDebugAndroidTest
```

Training is ~8–10 minutes on CPU. It caches log-mel features under `data/cache/`
(version 4); delete that directory after changing DSP constants.

## Known limits (ranked)

1. **Unseen generators are the weak spot.** 30% EER cross-dataset. More vendor-diverse
   training data (and codec/AMR/Opus round-trips) is the highest-value change.
2. **Live mode hears the room.** Android does not expose the carrier call stream; the app
   says this on screen. VoIP playback capture via `MediaProjection` is not implemented.
3. **The operating point is fixed at EER.** Exposing a user-facing strictness control
   (false alarms vs misses) would make it more usable.
4. **Heuristic cues are not calibrated.** They are shown for explainability only and are
   not inputs to the model.

## Useful facts

- Data lives in `data/` (git-ignored): `asvspoof2019_la/` 3.16 GB,
  `deepfake_audio_detection/` 574 MB, `cache/` ~1.2 GB.
- The old `data/asvspoof2021_la/` (2.36 GB) is the corrupted mirror and is unused.
- Emulator `Medium_Phone` (API 37); `adb devices` should show `emulator-5554`.
- Model: 193,569 params, 214.8 KiB, input `[1,64,198,1]`, output calibrated P(fake).
- Cross-dataset labels: `0=real / 1=fake`; ASVspoof 2019: `0=bonafide / 1=spoof`. Both
  map to canonical `1=fake` in `to_fake_target()`.
- The emulator runs with `-no-audio`, so its audio HAL (`android.hardwar`) logs
  `SIGABRT` repeatedly. That is the emulator, not the app.
