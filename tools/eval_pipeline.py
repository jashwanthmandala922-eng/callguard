#!/usr/bin/env python3
"""
Score audio the way the shipped app does, not the way training does.

Training evaluates one centred 2 s window per clip. The app slides a 2 s window
across the file and reports the median, which is a different (and more stable)
decision rule. This script mirrors the app so the reported accuracy matches what
a user actually sees.

  python tools/eval_pipeline.py                       # garystafford, full corpus
  python tools/eval_pipeline.py path/to/clip.wav      # single file

The TFLite model and its calibration are read from app/src/main/assets/models/.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
from math import gcd

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_demo_model import N_FRAMES, N_MEL, SR, log_mel, zscore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "app", "src", "main", "assets", "models")
CROSS_DIR = os.path.join(ROOT, "data", "deepfake_audio_detection")
WINDOW_SAMPLES = 32000   # 2 s
HOP_SAMPLES = 16000      # 1 s, matching GuardViewModel's file path


def load_meta() -> dict:
    with open(os.path.join(MODEL_DIR, "model_meta.json"), encoding="utf-8") as f:
        return json.load(f)


def platt(p: np.ndarray, cal: dict) -> np.ndarray:
    a = float(cal.get("a", 1.0))
    b = float(cal.get("b", 0.0))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(a * logit + b)))


def window_features(samples: np.ndarray) -> np.ndarray:
    mel = log_mel(samples.astype(np.float32))
    if mel.shape[0] < N_FRAMES:
        pad = N_FRAMES - mel.shape[0]
        mel = np.pad(mel, ((0, pad), (0, 0)), constant_values=float(mel.min()))
    else:
        mel = mel[:N_FRAMES]
    return zscore(mel.T)[None, :, :, None].astype(np.float32)


class Runner:
    def __init__(self, model_path: str, meta: dict, raw: bool = False):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.cal = {} if raw else meta.get("calibration", {})

    def score_window(self, samples: np.ndarray) -> float:
        self.interp.set_tensor(self.inp["index"], window_features(samples))
        self.interp.invoke()
        raw = float(self.interp.get_tensor(self.out["index"])[0, 0])
        return float(platt(np.array([raw]), self.cal)[0])

    def score_file(self, samples: np.ndarray) -> tuple[float, int]:
        """Median over sliding 2 s windows, exactly like the app's file path."""
        if len(samples) < WINDOW_SAMPLES:
            return float("nan"), 0
        scores = []
        position = 0
        while position + WINDOW_SAMPLES <= len(samples):
            scores.append(self.score_window(samples[position:position + WINDOW_SAMPLES]))
            position += HOP_SAMPLES
        scores.sort()
        return scores[len(scores) // 2], len(scores)


def to_16k(data: np.ndarray, rate: int) -> np.ndarray:
    x = data.mean(axis=1) if data.ndim > 1 else data
    if rate != SR:
        g = gcd(int(rate), SR)
        x = resample_poly(x, SR // g, int(rate) // g)
    return np.asarray(x, dtype=np.float32)


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    fake = np.sort(scores[labels == 1])
    real = np.sort(scores[labels == 0])
    thresholds = np.unique(np.concatenate([fake, real]))
    far = np.searchsorted(fake, thresholds, side="left") / max(len(fake), 1)
    frr = 1.0 - np.searchsorted(real, thresholds, side="left") / max(len(real), 1)
    idx = int(np.argmin(np.abs(far - frr)))
    value = float((far[idx] + frr[idx]) / 2)
    thr = float(thresholds[idx])
    acc = float(np.mean((scores >= thr) == (labels == 1)))
    return value, thr, acc


def eval_cross(runner: Runner, limit: int) -> None:
    rows: list[tuple[float, int, str]] = []
    files = sorted(glob.glob(os.path.join(CROSS_DIR, "*.parquet")))
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=["audio", "label"]):
            data = batch.to_pydict()
            for audio, label in zip(data["audio"], data["label"]):
                blob = audio["bytes"] if isinstance(audio, dict) else audio
                name = os.path.basename(str(audio.get("path", ""))) if isinstance(audio, dict) else ""
                try:
                    decoded, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
                except Exception:
                    continue
                samples = to_16k(decoded, rate)
                score, n = runner.score_file(samples)
                if n == 0 or np.isnan(score):
                    continue
                rows.append((score, int(label), name))
                if len(rows) % 200 == 0:
                    print(f"  {len(rows)} clips scored", flush=True)
                if limit and len(rows) >= limit:
                    break
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break

    scores = np.array([r[0] for r in rows])
    labels = np.array([r[1] for r in rows])
    eer_value, thr, acc = eer(scores, labels)
    print(f"\n[app-pipeline] clips={len(rows)} EER={eer_value * 100:.2f}% "
          f"accuracy@EER={acc * 100:.2f}%")
    print(f"[app-pipeline] real clips flagged: "
          f"{np.mean(scores[labels == 0] >= 0.5) * 100:.1f}%  "
          f"fake clips flagged: {np.mean(scores[labels == 1] >= 0.5) * 100:.1f}%")
    print(f"[app-pipeline] accuracy at the app's 0.5 point: "
          f"{np.mean((scores >= 0.5) == (labels == 1)) * 100:.2f}%")

    groups: dict[str, list[float]] = {}
    for score, label, name in rows:
        key = "real" if label == 0 else (name.split("_")[0].lower() or "fake")
        groups.setdefault(key, []).append(score)
    print("\n  vendor   clips   flagged@0.5")
    for key in sorted(groups):
        arr = np.array(groups[key])
        print(f"  {key:>6}   {len(arr):5d}   {np.mean(arr >= 0.5) * 100:5.1f}%")

    # What the user actually sees: three verdict bands, 0.35 / 0.65.
    human = scores <= 0.35
    synth = scores >= 0.65
    unsure = ~human & ~synth
    real = labels == 0
    print("\n  verdict            real clips   fake clips")
    print(f"  Sounds human       {np.mean(human[real]) * 100:8.1f}%   "
          f"{np.mean(human[~real]) * 100:8.1f}%")
    print(f"  Not sure           {np.mean(unsure[real]) * 100:8.1f}%   "
          f"{np.mean(unsure[~real]) * 100:8.1f}%")
    print(f"  Likely synthetic   {np.mean(synth[real]) * 100:8.1f}%   "
          f"{np.mean(synth[~real]) * 100:8.1f}%")
    if human.sum():
        print(f"\n  of clips called 'Sounds human', {np.mean(real[human]) * 100:.1f}% "
              f"really are real speech")
    if synth.sum():
        print(f"  of clips called 'Likely synthetic', {np.mean(~real[synth]) * 100:.1f}% "
              f"really are synthetic")

    # "When the app says human, be right at least 80% of the time" is a
    # precision target, not an accuracy target, so it depends on where the
    # human band is drawn. Sweep it and report the coverage cost.
    print("\n  human cut   precision   coverage   (real clips called human)")
    for cut in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        band = scores <= cut
        if not band.sum():
            print(f"  {cut:8.2f}         n/a       0.0%")
            continue
        precision = float(np.mean(real[band])) * 100
        coverage = float(np.mean(band[real])) * 100
        print(f"  {cut:8.2f}     {precision:6.1f}%    {coverage:6.1f}%")


def eval_single(runner: Runner, path: str) -> None:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    samples = to_16k(data, rate)
    score, n = runner.score_file(samples)
    print(f"{os.path.basename(path)}: P(fake)={score:.3f} over {n} window(s)")
    if score >= 0.65:
        print("  -> Likely synthetic")
    elif score <= 0.35:
        print("  -> Sounds human")
    else:
        print("  -> Not sure")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="single wav file instead of the corpus")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--raw", action="store_true",
                    help="ignore the shipped calibration (diagnostics only)")
    args = ap.parse_args()

    meta = load_meta()
    model = os.path.join(MODEL_DIR, "deepfake_detector.tflite")
    runner = Runner(model, meta, raw=args.raw)
    print(f"[app-pipeline] model {meta.get('name')} v{meta.get('version')} "
          f"({meta.get('params')} params, {meta.get('size_kib')} KiB)")

    if args.audio:
        eval_single(runner, args.audio)
    else:
        eval_cross(runner, args.limit)


if __name__ == "__main__":
    main()
