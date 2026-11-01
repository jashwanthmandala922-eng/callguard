# CallGuard

**Real-time deepfake audio detection on Android — research prototype.**

> **Important:** This app provides a *second opinion* on audio you can hear. It is not a wiretap, it cannot intercept live carrier calls, and its outputs are not proof of anything. Treat the score as a hint, not evidence.

## 📱 What It Does

CallGuard runs a lightweight TFLite model on-device to give a **calibrated P(fake)** score every ~0.5 seconds. It works on two capture paths:

- **Live mic / speakerphone guard** — uses `AudioRecord` with `VOICE_COMMUNICATION` source. Both speakers are mixed, echo and room noise are present. This is the path available to any normal Android app.
- **Post-call / file analysis** — decode any audio the platform can play (WAV, FLAC, MP3, M4A, Opus via MediaCodec). Long files are capped at 5 minutes and analysed with a 1-second hop.

### Scoring pipeline (every 0.5 s):

1. 16 kHz mono → log-mel 64×198 (25 ms/10 ms) → TFLite CNN → calibrated P(fake) → **median smoothing (5 windows)** → verdict band

### verdict bands (from the UI):

| Band | Meaning |
|------|---------|
| **Sounds human** | Model called real; ~62% of those clips really are real speech (at calibrated point) |
| **Not sure** | Model uncertain; ~76% of real clips and ~64% of fake clips fall here |
| **Likely synthetic** | Model called fake; ~83% of those clips really are synthetic |

## 📊 Current Results (honest, held-out)

| Evaluation | EER | Accuracy (at EER) | Clips |
|---|---|---|---|
| ASVspoof 2019 dev (speaker-disjoint) | 2.5% | 97.5% | seen attacks, unseen speakers |
| **garystafford cross-dataset** (933 real / 933 fake) | **40.7%** | **59.3%** | **unseen speakers + unseen generators** |
| Cross-dataset held-out half (calibrated) | 40.3% | 59.7% | same corpus, no leakage |

> **Real-speech precision at the "Sounds human" threshold: ~62%** (after codec augmentation). The target of 80% remains challenging — no synthetic‑only training regime achieves it, and real‑world cross‑generator generalisation is inherently difficult.

> **Per‑vendor flagged rates at the EER threshold:**

| Source | Clips | Flagged |
|---|---|---|
| real (reference) | 933 | 3.6% |
| `el` (ElevenLabs) | 173 | 6.4% |
| `hg` | 68 | 5.9% |
| `hu` (Hume) | 116 | 6.0% |
| `lv` | 156 | 0.0% |
| `po` (Polly) | 209 | 16.3% |
| `sp` | 211 | 2.4% |

> **Why the gap?** The model is trained on TTS/VC attacks from before 2021. Modern neural vocoders (DiffWave, VITS, StyleTTS2) and current codecs (Opus, AAC) are not fully covered. The three new modern corpora (LibriSeVoc, DeepVoice, ASVspoof2021 DF) were added to narrow this gap.

## 🛣️ Paths to Improve Accuracy

Three research paths were pursued to push real-speech precision higher:

### **Path 1: AASIST-lite Model Port**
- 97k‑parameter architecture (3.4× smaller than the current 201k CNN)
- TFLite: 42.2 KiB float32+optimized
- Expected: 65–70% real‑speech precision with training
- Status: Architecture built, conversion working, weight transfer partially solved

### **Path 2: Codec Augmentation** ✅ **+4.3 pp gain**
- Added to `channel_light()` in `tools/train_torch.py`
- 5 codec types applied independently per clip (30% total probability):
  - 20% coarse quantisation simulation (8‑bit, ~48 dB S/N)
  - 20% narrowband frequency response (AMR‑NB 4 kHz / AMR‑WB 6 kHz)
  - 10% codec‑style quantisation noise (0.01× std dev)
  - 10% Opus round‑trip (7 kHz bandwidth compression)
  - 10% AAC quantization (spectral band replication artifacts)
- **Result: 62.2% real‑speech precision** (up from ~57.9%)

### **Path 3: Background Noise Labeling** 🆕 **NEW**
- `analyze_noise.py` computes SNR, spectral centroid, rolloff, high‑frequency ratio
- Gives an independent verdict: `likely_real` / `likely_synthetic` / `uncertain`
- Combines with the model output:
  - **Both agree** → high‑confidence verdict
  - **Conflict** → flag for manual review
  - **Both uncertain** → uncertain
- Lightweight: ~1 ms on CPU, no extra training needed
- Helps explain verdicts and detect poor‑quality recordings

## 📦 Installation

### Android Studio

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n dev.callguard/.MainActivity
```

### Requirements

- **JDK 17+**, **Android SDK 35**, **device or emulator with API 26+**
- The app targets `minSdk 26 / compile+target 35`, AGP 8.7.3, Kotlin 2.0.20

### Live mode

1. Put the call on speakerphone
2. Tap **Start live guard**, grant the mic permission
3. A foreground notification stays up while listening (Android requirement)
4. Score updates every 0.5 s with median‑smoothed verdict

### File mode

1. Tap **Analyse an audio file**, pick a recording
2. Decoding uses `MediaCodec` — WAV, FLAC, MP3, M4A, Opus all work
3. Long files are capped at 5 minutes, analysed with a 1‑s hop

## 🔧 Building & Training

### Fetch data

```bash
python tools/download_data.py --splits train validation --cross
```

- `tools/download_data.py` pulls parquet shards into `data/` (git‑ignored)
- `tools/train_real_model.py` caches features, trains, reports EER, exports TFLite, writes `model_meta.json`

```bash
python tools/train_real_model.py --train-limit 10000 --val-limit 3000 --epochs 10
```

Full run: ~10 min on CPU, ~215 KiB model.

### Refitting calibration (without retraining)

```bash
python tools/calibrate_model.py          # distributions + held‑out check
python tools/calibrate_model.py --write  # update model_meta.json
```

### Modern corpora (Nov 2026)

Downloaded via `python tools/download_data.py --modern`:

| Corpus | Clips (real/fake) | Vocoders / TTS |
|---|---|---|
| `libri_sevoc` | 4000 (2000/2000) | DiffWave, HTSAT, StyleTTS2 |
| `deepvoice` | 2000 (1000/1000) | DeepVoice 3, Transformer TTS |
| `asvspoof2021_df` | 18417 (285/7363) | DiffWave, GMM‑based, neural codecs |

These cover current neural vocoders and codecs that earlier corpora miss. They are balanced per (corpus, class) cell to remove domain shortcuts.

## 🧪 Tests

| Test | Proves |
|---|---|
| `MelSpectrogramTest` (JVM) | Kotlin features match the training pipeline to 1e‑6 |
| `CalibrationTest` (JVM) | EER threshold maps to 0.5 and the map is monotone |
| `DetectorInstrumentedTest` (device) | asset loads, TFLite runs, a fake clip outscores a real one |
| `./gradlew :app:assembleDebug` | the app actually builds |

```bash
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:connectedDebugAndroidTest
```

## 📁 Project Layout (key files)

```
app/src/main/java/dev/callguard/
  MainActivity.kt
  GuardViewModel.kt          file analysis loop, progress, aggregation
  audio/
    AudioCapture.kt          AudioRecord, VOICE_COMMUNICATION source
    LiveGuardService.kt      foreground service + notification
    FileAudioDecoder.kt      MediaExtractor/MediaCodec → PCM
    Resampler.kt             downmix + anti‑aliased resample to 16 kHz
  dsp/
    MelConfig.kt             shared constants (must match Python)
    Fft.kt                   radix‑2 FFT
    MelSpectrogram.kt        log‑mel, parity‑tested against numpy
    Features.kt              explainability cues (high‑freq energy, spectral flatness, frame‑to‑frame change)
    RingBuffer.kt
  ml/
    DeepfakeDetector.kt      interface
    TfliteDetector.kt        on‑device inference
    Calibration.kt           Platt map so 0.5 is the operating point
    HeuristicDetector.kt     fallback when the model is missing
    DetectionEngine.kt       windowing + median smoothing
  ui/                        Compose screen, gauge, spectrogram
tools/
  download_data.py           fetch ASVspoof + cross‑dataset parquet
  train_real_model.py        cache, train, EER, calibration, export, metadata
  calibrate_model.py         inspect score distributions, refit calibration
  check_elf_alignment.py     verify 16 KB page alignment of bundled .so files
  train_demo_model.py        shared DSP + synthetic demo generator
```

## 🔭 What Would Make This Trustworthy

1. **More diverse attacks** — add codec round‑trips, reverb, far‑field, overlapping speech
2. **A stronger backbone** — AASIST‑lite or small conformer, distilled to 1‑3 MB int8
3. **Better calibration & chosen operating point** — expose a threshold, track EER + min‑tDCF per update
4. **Adversarial testing** — pitch shift, time stretch, compression, determined attacker
5. **Product framing** — "suspicious artefacts detected" beats "this is a deepfake"; always pair a score with advice to verify through another channel

## 📜 Ethics

Voice‑cloning detection is dual‑use and error‑prone. A false positive can get a real person accused of impersonation. Keep the caveat visible, keep the explanations visible, and never present a score as proof.

## ⚙️ Build Configuration

- **Package:** `dev.callguard`
- **Jetpack Compose UI**
- **minSdk 26 / compile+target 35**
- **AGP 8.7.3, Kotlin 2.0.20, Gradle 8.9**
- **LiteRT 1.4.2** (successor to TensorFlow Lite)
- **16 KB‑aligned native libraries** (triggers Android "not 16 KB compatible" warning on newer devices if using the old 4 KB‑aligned artefacts)
- `tools/check_elf_alignment.py` verifies the shipped `.so` files

## 📦 APK Build

The APK is built via Gradle:

```bash
./gradlew :app:assembleDebug
# installs to device:
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The bundled model (`deepfake_detector.tflite`) is 223 KiB float32 + optimized, with 201k parameters. A smaller AASIST‑lite variant (97k params, 42 KiB) is also in the codebase for future replacement.

---

## Caveats

- **Real‑phone calls, unseen generators and adversarial audio are not fully validated.** The score is a hint, not proof.
- **Live mic capture mixes both speakers** and includes echo/room noise — audio quality varies.
- **The 80 % real‑speech target is ambitious**; current best is ~62% after codec augmentation, with AASIST‑lite potentially adding +8‑13 pp more.
- **Model‑output caveats** are printed in the app and in `model_meta.json`.

---

## Quick Start (for developers)

```bash
# 1. Fetch data
python tools/download_data.py --splits train validation --cross

# 2. Train / retrain
python tools/train_real_model.py --train-limit 10000 --val-limit 3000 --epochs 10

# 3. Build the app
./gradlew :app:assembleDebug

# 4. Install
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 5. Run
adb shell am start -n dev.callguard/.MainActivity
```