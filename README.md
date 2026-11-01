# CallGuard

**Real-time deepfake audio detection on Android — research prototype.**

> **Important:** This app provides a *second opinion* on audio you can hear. It is not a wiretap, it cannot intercept live carrier calls, and its outputs are not proof of anything. Treat the score as a hint, not evidence.

## 📱 What It Does

CallGuard runs a lightweight TFLite model on-device to give a **calibrated P(fake)** score every ~0.5 seconds. It works on two capture paths:

- **Live mic / speakerphone guard** — uses `AudioRecord` with `VOICE_COMMUNICATION` source. Both speakers are mixed, echo and room noise are present. This is the path available to any normal Android app.
- **Post-call / file analysis** — decode any audio the platform can play (WAV, FLAC, MP3, M4A, Opus via MediaCodec). Long files are capped at 5 minutes and analysed with a 1-second hop.

### Scoring pipeline (every 0.5 s):

1. 16 kHz mono → log-mel 64×198 (25 ms/10 ms) → TFLite CNN → calibrated P(fake) → **median smoothing (5 windows)** → verdict band

### Verdict bands (from the UI):

| Band | Meaning |
|------|---------|
| **Sounds human** | Model called real; ~62% of those clips really are real speech |
| **Not sure** | Model uncertain; ~76% of real clips and ~64% of fake clips fall here |
| **Likely synthetic** | Model called fake; ~83% of those clips really are synthetic |

## 📊 Current Results (honest, held-out)

| Evaluation | EER | Accuracy (at EER) |
|------------|-----|-------------------|
| ASVspoof 2019 dev (speaker-disjoint) | 2.5% | 97.5% |
| **garystafford cross-dataset** (933 real / 933 fake) | **40.7%** | **59.3%** |
| Cross-dataset held-out half (calibrated) | 40.3% | 59.7% |

> **Real-speech precision at the "Sounds human" threshold: ~62%** (after codec augmentation). The target of 80% remains challenging — no synthetic-only training regime achieves it, and real-world cross-generator generalisation is inherently difficult.

### Per-vendor flagged rates at the EER threshold:

| Source | Clips | Flagged |
|--------|-------|---------|
| real (reference) | 933 | 3.6% |
| `el` (ElevenLabs) | 173 | 6.4% |
| `hg` | 68 | 5.9% |
| `hu` (Hume) | 116 | 6.0% |
| `lv` | 156 | 0.0% |
| `po` (Polly) | 209 | 16.3% |
| `sp` | 211 | 2.4% |

## 🛣️ Paths to Improve Accuracy

Three research paths were pursued to push real-speech precision higher:

### Path 1: AASIST-lite Model Port
- 97k-parameter architecture (3.4× smaller than the current 201k CNN)
- TFLite: 42.2 KiB float32+optimized
- Expected: 65–70% real-speech precision with training

### Path 2: Codec Augmentation ✅ **+4.3 pp gain**
- Added to `channel_light()` in `scripts/`
- 5 codec types applied independently per clip:
  - Coarse quantisation simulation (8-bit)
  - Narrowband frequency response (AMR-NB/AMR-WB)
  - Codec-style quantisation noise
  - Opus round-trip simulation
  - AAC quantization simulation
- **Result: 62.2% real-speech precision** (up from ~57.9% baseline)

### Path 3: Background Noise Labeling 🆕
- `scripts/analyze_noise.py` computes SNR, spectral centroid, rolloff, high-frequency ratio
- Independent verdict: `likely_real` / `likely_synthetic` / `uncertain`
- Combines with model output: both agree → higher confidence; conflict → flag for review
- Lightweight: ~1ms on CPU, no extra training needed

## 📦 Installation

### Android Studio

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n dev.callguard/.MainActivity
```

### Requirements

- **JDK 17+**, **Android SDK 35**, **device or emulator with API 26+**
- App targets `minSdk 26 / compile+target 35`, AGP 8.7.3, Kotlin 2.0.20

### Live mode

1. Put the call on speakerphone
2. Tap **Start live guard**, grant the mic permission
3. A foreground notification stays up while listening (Android requirement)
4. Score updates every 0.5 s with median-smoothed verdict

### File mode

1. Tap **Analyse an audio file**, pick a recording
2. Decoding uses `MediaCodec` — WAV, FLAC, MP3, M4A, Opus all work
3. Long files are capped at 5 minutes, analysed with a 1-second hop

## 🔧 Building & Training

### Fetch data

```bash
python tools/download_data.py --splits train validation --cross
```

### Train

```bash
python tools/train_torch.py --epochs 15 --batch 32 --augment light
```

### Refitting calibration

```bash
python tools/calibrate_model.py --write
```

## 📁 Project Layout

```
app/                         Android app (Kotlin + Jetpack Compose)
  src/main/java/dev/callguard/
    audio/                   Audio capture, decoding, resampling
    dsp/                     FFT, log-mel, features, ring buffer
    ml/                      TFLite detection, calibration, heuristics
    model/                   Guard bus and UI state models
    ui/                      Compose screens, components, theme
  src/main/assets/models/    TFLite model files
tools/                       Training and data pipeline
  train_torch.py             PyTorch GPU trainer
  eval_pipeline.py           Evaluation pipeline
  download_data.py           Data downloader
  calibrate_model.py         Platt calibration
scripts/                     Analysis and experiment scripts
  analyze_noise.py           Background noise analysis
  convert_aasist.py          AASIST-lite TFLite conversion
  combine_noise_eval.py      Combined CNN + noise evaluation
docs/                        Handoff and design notes
```

## 🔬 What Would Make This Trustworthy

1. **More diverse attacks** — codec round-trips, reverb, far-field, overlapping speech
2. **A stronger backbone** — AASIST-lite or small conformer, distilled to int8
3. **Better calibration** — expose a user-facing threshold control
4. **Adversarial testing** — pitch shift, time stretch, compression
5. **Product framing** — "suspicious artefacts detected" beats "this is a deepfake"

## 📜 Ethics

Voice-cloning detection is dual-use and error-prone. A false positive can get a real person accused of impersonation. Keep the caveat visible, keep the explanations visible, and never present a score as proof.

## ⚙️ Build Configuration

- **Package:** `dev.callguard`
- **Jetpack Compose UI**
- **minSdk 26 / compile+target 35**
- **AGP 8.7.3, Kotlin 2.0.20, Gradle 8.9**
- **LiteRT 1.4.2** (successor to TensorFlow Lite)
- **16 KB-aligned native libraries**

## 📦 APK

Built via `./gradlew :app:assembleDebug`. Output: `app/build/outputs/apk/debug/app-debug.apk`

## Caveats

- Real-phone calls, unseen generators and adversarial audio are not fully validated
- Live mic capture mixes both speakers with echo/room noise
- The 80% real-speech target is ambitious; current best is ~62% after codec augmentation
