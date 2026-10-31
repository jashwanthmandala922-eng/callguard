#!/usr/bin/env python3
"""
CallGuard demo detector: training + TFLite export.

WHAT THIS IS
------------
A *demo* model. It is trained on synthetic speech (additive glottal synthesis)
and synthetic "vocoder-like" fakes (cepstral envelope smoothing + random phase
resynthesis). It exists to prove the whole on-device path works:

    PCM -> log-mel (Kotlin) -> TFLite -> score

It has NOT been trained on real deepfakes and has NO validated real-world
accuracy. Replace it with a model trained on ASVspoof 2019/2021 + WaveFake +
In-the-Wild before drawing any conclusion about a real call.

OUTPUTS
-------
  app/src/main/assets/models/deepfake_detector.tflite   (dynamic-range quantized)
  app/src/main/assets/models/model_meta.json
  app/src/test/resources/mel_reference.txt              (Kotlin parity fixture)

USAGE
-----
  python tools/train_demo_model.py --smoke
  python tools/train_demo_model.py --samples 2000 --epochs 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

# --------------------------------------------------------------------------
# DSP config -- MUST stay in sync with dsp/MelConfig.kt
# --------------------------------------------------------------------------
SR = 16000
N_FFT = 512
WIN = 400          # 25 ms
HOP = 160          # 10 ms
N_MEL = 64
FMIN = 20.0
FMAX = 8000.0
FLOOR = 1e-6

WINDOW_SECONDS = 2.0
N_FRAMES = 1 + (int(WINDOW_SECONDS * SR) - WIN) // HOP   # 198

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def hann_periodic(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


_WINDOW = hann_periodic(WIN).astype(np.float64)


def hz_to_mel(f: np.ndarray | float):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """(N_MEL, N_FFT//2+1) triangular HTK-scale filters."""
    n_bins = N_FFT // 2 + 1
    mel_pts = np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MEL + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((N_FFT + 1) * hz_pts / SR).astype(np.int64)
    fb = np.zeros((N_MEL, n_bins), dtype=np.float64)
    for k in range(N_MEL):
        lo, mid, hi = bins[k], bins[k + 1], bins[k + 2]
        for b in range(lo, hi):
            if b < 0 or b >= n_bins:
                continue
            if b <= mid and mid > lo:
                fb[k, b] = (b - lo) / (mid - lo)
            elif b > mid and hi > mid:
                fb[k, b] = (hi - b) / (hi - mid)
    return fb


_FB = mel_filterbank()


def frame_signal(x: np.ndarray) -> np.ndarray:
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def log_mel(x: np.ndarray) -> np.ndarray:
    """(T, N_MEL) natural-log mel power. x is float in [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < WIN:
        return np.zeros((0, N_MEL), dtype=np.float32)
    fr = frame_signal(x) * _WINDOW
    spec = np.fft.rfft(fr, n=N_FFT, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float64)
    mel = power @ _FB.T
    return np.log(mel + FLOOR).astype(np.float32)


def zscore(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    mu = float(m.mean())
    sd = float(m.std()) + 1e-5
    return (m - mu) / sd


def to_model_input(mel_tf: np.ndarray) -> np.ndarray:
    """(T, N_MEL) -> (1, N_MEL, T, 1), z-scored. Mel-major, as Keras expects."""
    return zscore(mel_tf.T)[None, :, :, None]


# --------------------------------------------------------------------------
# Synthetic speech
# --------------------------------------------------------------------------
FORMANT_TABLE = np.array([
    # male-ish, female-ish, and two in-betweens (F1, F2, F3, F4)
    [500, 1500, 2500, 3500],
    [300, 2300, 3000, 3700],
    [700, 1200, 2600, 3400],
    [400, 1900, 2800, 3600],
], dtype=np.float64)


def _envelope(freqs: np.ndarray, formants: np.ndarray) -> np.ndarray:
    """Gaussian formant envelope evaluated at arbitrary frequencies."""
    env = np.zeros_like(freqs)
    for i, (fc, bw) in enumerate(zip(formants, (90.0, 110.0, 140.0, 180.0))):
        env += (1.0 / (i + 1)) * np.exp(-0.5 * ((freqs - fc) / bw) ** 2)
    return env


def synth_speech(rng: np.random.Generator, dur: float = WINDOW_SECONDS) -> np.ndarray:
    """Additive glottal source shaped by time-varying formants + fricatives."""
    n = int(dur * SR)
    t = np.arange(n) / SR

    # --- prosody: f0 contour with vibrato and jitter -----------------------
    base = rng.uniform(95.0, 210.0)
    drift = rng.uniform(-0.25, 0.25)
    vib_hz = rng.uniform(3.0, 7.0)
    vib_amp = rng.uniform(0.0, 0.06)
    f0 = base * (1.0 + drift * t / dur + vib_amp * np.sin(2 * np.pi * vib_hz * t + rng.uniform(0, 6.28)))
    f0 = np.clip(f0, 70.0, 320.0)
    # slow random jitter (piecewise constant every 25 ms)
    jitter = rng.normal(0.0, 0.004, size=n // 400 + 2)
    f0 *= np.repeat(1.0 + jitter, 400)[:n]

    phase = 2 * np.pi * np.cumsum(f0) / SR

    # --- formant trajectory ------------------------------------------------
    formants = FORMANT_TABLE[rng.integers(0, len(FORMANT_TABLE))].copy()
    formants = formants * rng.uniform(0.9, 1.12)
    # glide: interpolate toward a random vowel over the utterance
    target = FORMANT_TABLE[rng.integers(0, len(FORMANT_TABLE))] * rng.uniform(0.9, 1.12)
    alpha = np.clip(t / dur, 0, 1)[:, None]
    fmt = formants[None, :] * (1 - alpha) + target[None, :] * alpha   # (n, 4)

    # --- harmonic sum ------------------------------------------------------
    k_max = int(min(60, FMAX / max(f0.min(), 1.0)))
    src = np.zeros(n, dtype=np.float64)
    phases = rng.uniform(0, 2 * np.pi, size=k_max)
    for k in range(1, k_max + 1):
        fk = k * f0
        # evaluate the formant envelope for this harmonic at every sample
        amp = np.zeros(n)
        for j in range(4):
            bw = (90.0, 110.0, 140.0, 180.0)[j]
            amp += (1.0 / (j + 1)) * np.exp(-0.5 * ((fk - fmt[:, j]) / bw) ** 2)
        amp = amp * (k ** -0.9)
        src += amp * np.sin(k * phase + phases[k - 1])

    # --- breathiness -------------------------------------------------------
    src += rng.normal(0.0, 0.02, size=n) * (1.0 + 0.5 * np.sin(2 * np.pi * 0.7 * t))

    # --- fricative bursts --------------------------------------------------
    for _ in range(rng.integers(0, 3)):
        start = rng.integers(0, max(1, n - 2000))
        length = rng.integers(800, 4000)
        end = min(n, start + length)
        seg = np.arange(end - start)
        noise = rng.normal(0.0, 1.0, size=end - start)
        # crude band-pass by differencing (keeps it broadband, ~2-7 kHz)
        noise = np.diff(np.concatenate([[0.0], noise]))
        fade = np.hanning(end - start)
        src[start:end] += 0.35 * noise * fade

    # --- utterance envelope, normalise -------------------------------------
    env = 0.55 + 0.45 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t + rng.uniform(0, 6.28))
    src *= np.clip(env, 0.0, None)
    peak = np.max(np.abs(src)) + 1e-9
    return (src / peak * 0.7).astype(np.float32)


# --------------------------------------------------------------------------
# "Vocoder" fake: cepstral envelope smoothing + random phase resynthesis
# --------------------------------------------------------------------------
def _stft(x: np.ndarray):
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    fr = xp[idx] * hann_periodic(N_FFT)[None, :]
    return np.fft.rfft(fr, axis=1), n, len(xp)


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    win = hann_periodic(N_FFT)
    fr = np.fft.irfft(spec, n=N_FFT, axis=1) * win[None, :]
    out = np.zeros(length, dtype=np.float64)
    wsum = np.zeros(length, dtype=np.float64)
    for i in range(spec.shape[0]):
        s = i * HOP
        out[s:s + N_FFT] += fr[i]
        wsum[s:s + N_FFT] += win ** 2
    wsum = np.maximum(wsum, 1e-8)
    out = out / wsum
    pad = N_FFT // 2
    return out[pad:pad + length]


def vocode(x: np.ndarray, rng: np.random.Generator, keep_cepstra: int = 26) -> np.ndarray:
    spec, _, length = _stft(x)
    mag = np.abs(spec) + 1e-8
    logmag = np.log(mag)

    # cepstral smoothing -> removes fine harmonic structure (classic vocoder tell)
    cep = np.fft.irfft(logmag, n=N_FFT, axis=1)
    cep[:, keep_cepstra:] = 0.0
    smooth = np.fft.rfft(cep, n=N_FFT, axis=1).real
    smooth = np.exp(smooth)

    # random phase -> incoherent, smeared transients
    ph = rng.uniform(-np.pi, np.pi, size=spec.shape)
    ph[:, 0] = 0.0

    # slight high-frequency tilt, typical of vocoders
    freqs = np.arange(spec.shape[1])
    tilt = (1.0 + freqs) ** -0.08
    y = _istft(smooth * tilt[None, :] * np.exp(1j * ph), length)
    y = y[:len(x)]
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak * 0.7).astype(np.float32)


# --------------------------------------------------------------------------
# Channel simulation, applied identically to both classes
# --------------------------------------------------------------------------
def channel(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.astype(np.float64).copy()
    n = len(y)

    # band limit: telephony (3.4k) .. wideband (8k)
    cutoff = rng.choice([3400.0, 4000.0, 6000.0, 8000.0], p=[0.25, 0.25, 0.25, 0.25])
    if cutoff < SR / 2:
        spec = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n, 1.0 / SR)
        # smooth rolloff above cutoff
        spec *= 1.0 / (1.0 + np.exp((freqs - cutoff) / 250.0))
        y = np.fft.irfft(spec, n=n)

    # mu-law round trip (8-bit telephony companding)
    if rng.random() < 0.6:
        mu = 255.0
        comp = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
        q = np.round(comp * 127.0) / 127.0
        y = np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu

    # additive noise
    snr_db = rng.uniform(12.0, 35.0)
    p_sig = np.mean(y ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    y += rng.normal(0.0, math.sqrt(p_noise), size=n)

    # packet loss: zero out a few 20 ms frames
    if rng.random() < 0.3:
        for _ in range(rng.integers(1, 4)):
            s = rng.integers(0, max(1, n - 320))
            y[s:s + 320] = 0.0

    # gain + occasional clipping
    y *= rng.uniform(0.3, 1.4)
    if rng.random() < 0.2:
        y = np.tanh(y * 1.8) / 1.8

    peak = np.max(np.abs(y)) + 1e-9
    if peak > 0.99:
        y = y / peak * 0.99
    return y.astype(np.float32)


def make_sample(rng: np.random.Generator, label: int) -> np.ndarray:
    speech = synth_speech(rng)
    if label == 1:
        speech = vocode(speech, rng)
    speech = channel(speech, rng)
    return log_mel(speech).astype(np.float32)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    inp = layers.Input(shape=(N_MEL, N_FRAMES, 1), name="logmel")
    x = inp
    for filters in (16, 32, 64):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        # GroupNorm, not BatchNorm: no train/eval moving-statistics mismatch, so the
        # exported model's sigmoid is on the same scale as the one we validated.
        x = layers.GroupNormalization(groups=8)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def batch_generator(rng, batch_size, samples_per_epoch):
    """Yields (x, y) forever, freshly synthesized."""
    steps = samples_per_epoch // batch_size
    while True:
        for _ in range(steps):
            xb = np.zeros((batch_size, N_MEL, N_FRAMES, 1), dtype=np.float32)
            yb = np.zeros((batch_size, 1), dtype=np.float32)
            for i in range(batch_size):
                label = int(rng.integers(0, 2))
                mel = make_sample(rng, label)
                xb[i] = to_model_input(mel)
                yb[i, 0] = float(label)
            yield xb, yb


def fixed_set(seed, n):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, N_MEL, N_FRAMES, 1), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        label = i % 2
        x[i] = to_model_input(make_sample(rng, label))
        y[i, 0] = float(label)
    return x, y


# --------------------------------------------------------------------------
# Parity fixture: deterministic signal, reproducible in Kotlin
# --------------------------------------------------------------------------
def parity_signal() -> np.ndarray:
    t = np.arange(8000, dtype=np.float64) / SR
    sig = (0.50 * np.sin(2 * np.pi * 440.0 * t)
           + 0.30 * np.sin(2 * np.pi * 1234.5 * t)
           + 0.15 * np.sin(2 * np.pi * 3001.0 * t)
           + 0.05 * np.sin(2 * np.pi * 6400.0 * t))
    sig *= 0.5 - 0.5 * np.cos(2 * np.pi * 3.0 * t)
    return sig.astype(np.float32)


def write_parity_fixture(path: str):
    mel = log_mel(parity_signal())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# frames={mel.shape[0]} bins={mel.shape[1]}\n")
        for row in mel:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    return mel.shape


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000, help="training samples")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    args = ap.parse_args()

    import tensorflow as tf
    tf.random.set_seed(1337)
    tf.get_logger().setLevel("ERROR")

    if args.smoke:
        args.samples, args.epochs, args.batch = 64, 1, 8

    print(f"[callguard] window={WINDOW_SECONDS}s frames={N_FRAMES} bins={N_MEL}")

    model = build_model()
    model.summary(line_length=96)

    rng = np.random.default_rng(7)
    steps = args.samples // args.batch
    print(f"[callguard] training {steps} steps x {args.epochs} epochs "
          f"({steps * args.batch * args.epochs} synthetic clips)")

    t0 = time.time()
    model.fit(batch_generator(rng, args.batch, args.samples),
              steps_per_epoch=steps, epochs=args.epochs, verbose=2)
    train_s = time.time() - t0
    print(f"[callguard] trained in {train_s:.1f}s")

    # ---- evaluation on a held-out synthetic set ----
    xv, yv = fixed_set(seed=99, n=240)
    metrics = model.evaluate(xv, yv, verbose=0, return_dict=True)
    probs = model.predict(xv, verbose=0).ravel()
    acc = float(np.mean((probs > 0.5) == (yv.ravel() > 0.5)))
    print(f"[callguard] holdout: acc={acc:.3f} auc={metrics.get('auc', float('nan')):.3f}")

    # ---- export TFLite (dynamic range quantization) ----
    out_dir = os.path.join(ROOT, "app", "src", "main", "assets", "models")
    os.makedirs(out_dir, exist_ok=True)
    tflite_path = os.path.join(out_dir, "deepfake_detector.tflite")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[callguard] wrote {tflite_path} ({len(tflite_model) / 1024:.1f} KiB)")

    # ---- sanity check the exported graph ----
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"[callguard] tflite input {inp_d['shape']} {inp_d['dtype'].__name__} "
          f"-> output {out_d['shape']} {out_d['dtype'].__name__}")

    sample = to_model_input(make_sample(np.random.default_rng(3), 1))
    interp.set_tensor(inp_d["index"], sample)
    interp.invoke()
    print(f"[callguard] smoke score on a fake clip: "
          f"{float(interp.get_tensor(out_d['index'])[0, 0]):.3f}")

    # ---- metadata ----
    meta = {
        "name": "callguard-demo-cnn",
        "version": "0.1.0",
        "purpose": "DEMO ONLY - trained on synthetic audio, no real-world validation",
        "input_shape": [1, N_MEL, N_FRAMES, 1],
        "output": "sigmoid fake probability",
        "dsp": {
            "sample_rate": SR, "n_fft": N_FFT, "win": WIN, "hop": HOP,
            "n_mel": N_MEL, "fmin": FMIN, "fmax": FMAX, "floor": FLOOR,
            "window_seconds": WINDOW_SECONDS, "frames": N_FRAMES,
            "normalization": "z-score over the whole window",
        },
        "holdout": {"accuracy": acc, "auc": float(metrics.get("auc", 0.0))},
        "train_seconds": round(train_s, 1),
        "params": int(model.count_params()),
        "size_kib": round(len(tflite_model) / 1024, 1),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[callguard] params={meta['params']}")

    fixture = os.path.join(ROOT, "app", "src", "test", "resources", "mel_reference.txt")
    shape = write_parity_fixture(fixture)
    print(f"[callguard] wrote parity fixture {fixture} {shape}")


if __name__ == "__main__":
    main()
