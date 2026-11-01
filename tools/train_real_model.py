#!/usr/bin/env python3
"""
Train CallGuard's detector on labelled real speech: ASVspoof 2019 Logical Access.

Data
----
  train      25,380 clips (bonafide + TTS/VC spoofs), speaker-disjoint from dev
  validation 24,844 clips, the official dev set
  cross      garystafford/deepfake-audio-detection, 933 real / 933 fake

This script downloads nothing itself -- run tools/download_data.py first. It reads
the parquet shards, caches log-mel features, trains a small CNN, reports EER on the
speaker-disjoint dev set and on the cross-dataset corpus, then exports TFLite.

Why EER: accuracy at 0.5 is meaningless for anti-spoofing. EER is the standard
operating-point-free summary and it is what the ASVspoof literature reports.

  python tools/train_real_model.py --train-limit 10000 --val-limit 3000 --epochs 6
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from math import gcd

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_demo_model import (  # noqa: E402
    HOP,
    N_FRAMES,
    N_MEL,
    SR,
    WIN,
    channel,
    log_mel,
    zscore,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "asvspoof2019_la")
CROSS_DIR = os.path.join(ROOT, "data", "deepfake_audio_detection")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
CACHE_VERSION = 4
MAX_SECONDS = 3.0
MAX_FRAMES = 1 + (int(MAX_SECONDS * SR) - WIN) // HOP  # 298


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def split_dir(split: str) -> str:
    return CROSS_DIR if split == "cross" else DATA_DIR


def label_column(split: str) -> str:
    """ASVspoof calls it `key`; garystafford calls it `label`."""
    return "label" if split == "cross" else "key"


def parquet_files(split: str) -> list[str]:
    directory = split_dir(split)
    if not os.path.isdir(directory):
        return []
    if split == "cross":
        names = [f for f in os.listdir(directory) if f.endswith(".parquet")]
    else:
        names = [
            f for f in os.listdir(directory)
            if f.startswith(split + "-") and f.endswith(".parquet")
        ]
    return [os.path.join(directory, n) for n in sorted(names)]


def to_fake_target(label: int, split: str) -> int:
    """Canonical target: 1 = fake, 0 = real.

    ASVspoof 2019 uses 0 = bonafide / 1 = spoof, and garystafford uses
    0 = real / 1 = fake, so both already match. A previous corpus labelled
    0 = fake; inverting it silently trained a detector that was perfectly
    backwards while still reporting high accuracy, so the mapping lives here.
    """
    return int(label)


def label_counts(split: str) -> dict[int, int]:
    """Count canonical classes cheaply (label column only, no audio)."""
    column = label_column(split)
    counts = {0: 0, 1: 0}
    for path in parquet_files(split):
        table = pq.read_table(path, columns=[column])
        for value in table.column(column).to_pylist():
            counts[to_fake_target(int(value), split)] += 1
    return counts


def speaker_ids(split: str) -> set[str]:
    ids: set[str] = set()
    for path in parquet_files(split):
        table = pq.read_table(path, columns=["speaker_id"])
        ids.update(table.column("speaker_id").to_pylist())
    return ids


def decode_to_features(
    blob: bytes, rng: np.random.Generator, augment: bool
) -> np.ndarray | None:
    """PCM -> 3 s log-mel clip. `augment` is train-only: eval stays clean."""
    try:
        data, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    except Exception:
        return None
    if data.size == 0:
        return None

    x = data.mean(axis=1)
    rate = int(rate)
    if rate != SR:
        g = gcd(rate, SR)
        x = resample_poly(x, SR // g, rate // g).astype(np.float32)

    if len(x) < SR // 2:  # shorter than 0.5 s -> not useful
        return None

    want = int(MAX_SECONDS * SR)
    if len(x) > want:
        start = int(rng.integers(0, len(x) - want))
        x = x[start:start + want]
    elif len(x) < want:
        x = np.pad(x, (0, want - len(x)))

    if augment:
        x = channel(x, rng)      # gain, noise, band-limit, mu-law, packet loss
    mel = log_mel(x)             # (frames, 64)
    if mel.shape[0] < MAX_FRAMES:
        pad = MAX_FRAMES - mel.shape[0]
        fill = float(mel.min()) if mel.size else float(np.log(1e-6))
        mel = np.pad(mel, ((0, pad), (0, 0)), constant_values=fill)
    else:
        mel = mel[:MAX_FRAMES]
    return mel.astype(np.float16)


def build_cache(split: str, limit: int, seed: int) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    x_path = os.path.join(CACHE_DIR, f"X_{split}.npy")
    y_path = os.path.join(CACHE_DIR, f"y_{split}.npy")
    meta_path = os.path.join(CACHE_DIR, f"meta_{split}.json")

    if os.path.exists(meta_path) and os.path.exists(x_path) and os.path.exists(y_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("version") == CACHE_VERSION and meta.get("limit", -1) >= limit:
            print(f"[{split}] cache hit: {meta['count']} clips")
            return meta

    files = parquet_files(split)
    if not files:
        raise SystemExit(f"no parquet files for split '{split}' in {split_dir(split)}. "
                         f"Run tools/download_data.py first.")

    # Both corpora already use 1=fake; to_fake_target() keeps that explicit.
    augment = split == "train"
    column = label_column(split)

    # Balance on the minority class so the cache never silently overshoots what
    # the corpus can supply (ASVspoof is ~10% bonafide, so 50/50 is the ceiling).
    available = label_counts(split)
    per_class_cap = min(limit // 2 + 1, min(available.values()) or 0)
    limit = min(limit, per_class_cap * 2)
    if per_class_cap <= 0:
        raise SystemExit(f"split '{split}' has no usable labels")
    print(f"[{split}] available {available}, using {per_class_cap} per class")

    rng = np.random.default_rng(seed)
    X = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float16, shape=(limit, MAX_FRAMES, N_MEL)
    )
    y = np.zeros(limit, dtype=np.int8)
    names: list[str] = []
    counts = {0: 0, 1: 0}
    n = 0
    t0 = time.time()

    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=["audio", column]):
            rows = batch.to_pydict()
            for blob, label in zip(rows["audio"], rows[column]):
                if n >= limit:
                    break
                label = to_fake_target(int(label), split)
                if counts[label] >= per_class_cap:
                    continue
                feats = decode_to_features(blob["bytes"], rng, augment=augment)
                if feats is None:
                    continue
                X[n] = feats
                y[n] = label
                names.append(os.path.basename(str(blob.get("path", ""))))
                n += 1
                counts[label] += 1
                if n % 250 == 0:
                    rate = n / max(time.time() - t0, 1e-6)
                    print(f"[{split}] {n}/{limit} clips ({rate:.1f}/s)", flush=True)
            if n >= limit:
                break
        if n >= limit:
            break

    X.flush()
    del X
    np.save(y_path, y[:n])
    with open(os.path.join(CACHE_DIR, f"names_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(names, f)
    meta = {
        "split": split,
        "version": CACHE_VERSION,
        "count": int(n),
        "limit": int(limit),
        "max_frames": MAX_FRAMES,
        "augmented": augment,
        "labels": {"0": int(counts[0]), "1": int(counts[1])},
        "seconds": round(time.time() - t0, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{split}] cached {n} clips ({counts[0]} real / {counts[1]} fake) "
          f"in {meta['seconds']}s")
    return meta


def load_cache(split: str):
    with open(os.path.join(CACHE_DIR, f"meta_{split}.json"), encoding="utf-8") as f:
        meta = json.load(f)
    X = np.load(os.path.join(CACHE_DIR, f"X_{split}.npy"), mmap_mode="r")[: meta["count"]]
    y = np.load(os.path.join(CACHE_DIR, f"y_{split}.npy"))
    return X, y


# --------------------------------------------------------------------------
# Batching / augmentation
# --------------------------------------------------------------------------
def spec_augment(window: np.ndarray, rng: np.random.Generator) -> None:
    """Zero out bands after z-scoring, i.e. replace with the mean."""
    for _ in range(2):
        width = int(rng.integers(0, 9))
        if width:
            start = int(rng.integers(0, N_MEL - width))
            window[start:start + width, :] = 0.0
    for _ in range(2):
        width = int(rng.integers(0, 21))
        if width:
            start = int(rng.integers(0, N_FRAMES - width))
            window[:, start:start + width] = 0.0


def make_batch(X, y, indices, train: bool, rng) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(indices), N_MEL, N_FRAMES, 1), dtype=np.float32)
    labels = np.zeros((len(indices), 1), dtype=np.float32)
    span = MAX_FRAMES - N_FRAMES
    for i, idx in enumerate(indices):
        clip = X[idx]
        if train:
            start = int(rng.integers(0, span + 1))
        else:
            start = span // 2
        window = zscore(np.asarray(clip[start:start + N_FRAMES], dtype=np.float32).T)
        if train:
            spec_augment(window, rng)
        out[i, :, :, 0] = window
        labels[i, 0] = float(y[idx])
    return out, labels


def batch_generator(X, y, batch_size: int, seed: int, train: bool):
    rng = np.random.default_rng(seed)
    n = len(y)
    while True:
        order = rng.permutation(n) if train else np.arange(n)
        for start in range(0, n - batch_size + 1, batch_size):
            yield make_batch(X, y, order[start:start + batch_size], train, rng)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """scores = P(fake), labels 1=fake / 0=real. Returns (eer, threshold, accuracy)."""
    fake = np.sort(scores[labels == 1])
    real = np.sort(scores[labels == 0])
    thresholds = np.unique(np.concatenate([fake, real]))
    far = np.searchsorted(fake, thresholds, side="left") / max(len(fake), 1)
    frr = 1.0 - np.searchsorted(real, thresholds, side="left") / max(len(real), 1)
    idx = int(np.argmin(np.abs(far - frr)))
    value = float((far[idx] + frr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    predictions = scores >= threshold
    accuracy = float(np.mean(predictions == (labels == 1)))
    return value, threshold, accuracy


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC for spoof (label 1) vs bonafide (label 0)."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    all_scores = np.concatenate([positive, negative])
    order = np.argsort(all_scores)
    ranks = np.empty(len(all_scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    return float((ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * logit(scores) + b)))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Regularised Platt scaling on the logit.

    The raw sigmoid ranks well but is compressed: on unseen-generator audio the
    EER operating point can sit near 5e-4, so a UI thresholding at 0.5 would
    never flag anything. Near-separable classes also make an unregularised fit
    diverge, which would collapse every score to 0 or 1. Platt's smoothed targets
    plus a mild slope penalty keep the map finite and strictly increasing.
    """
    from scipy.optimize import minimize

    x = logit(scores)
    y = labels.astype(np.float64)
    n_pos = max(y.sum(), 1.0)
    n_neg = max(len(y) - y.sum(), 1.0)
    t = np.where(y > 0.5, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def objective(params):
        a, b = params
        z = a * x + b
        bce = np.mean(np.logaddexp(0.0, z) - t * z)
        return bce + 1e-3 * (a - 1.0) ** 2 + 1e-4 * b * b

    result = minimize(
        objective,
        x0=np.array([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(0.01, 50.0), (-50.0, 50.0)],
    )
    return float(result.x[0]), float(result.x[1])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    inp = layers.Input(shape=(N_MEL, N_FRAMES, 1), name="logmel")
    x = inp
    for filters in (32, 64, 96, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.GroupNormalization(groups=8)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def predict(model, X, y, batch_size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = []
    for start in range(0, len(y), batch_size):
        idx = np.arange(start, min(start + batch_size, len(y)))
        xb, _ = make_batch(X, y, idx, train=False, rng=rng)
        out.append(model(xb, training=False).numpy().ravel())
    return np.concatenate(out)


def predict_tflite(path: str, X, y, batch_size: int = 64) -> np.ndarray:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    rng = np.random.default_rng(0)
    scores = []
    for start in range(0, len(y), batch_size):
        idx = np.arange(start, min(start + batch_size, len(y)))
        xb, _ = make_batch(X, y, idx, train=False, rng=rng)
        for row in xb:
            interp.set_tensor(inp["index"], row[None, ...])
            interp.invoke()
            scores.append(float(interp.get_tensor(out["index"])[0, 0]))
    return np.array(scores)


def source_breakdown(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Per-generator view of the cross-dataset test at the EER threshold.

    Fake clips are grouped by filename prefix (the TTS/VC vendor); real clips are
    pooled. A low `flagged_rate` on a fake vendor means the detector missed it.
    """
    names_path = os.path.join(CACHE_DIR, "names_cross.json")
    if not os.path.exists(names_path):
        return {}
    with open(names_path, encoding="utf-8") as f:
        names = json.load(f)[: len(labels)]

    groups: dict[str, list[float]] = {}
    for score, label, name in zip(scores, labels, names):
        key = "real" if int(label) == 0 else (name.split("_")[0].lower() or "fake")
        groups.setdefault(key, []).append(float(score))

    out = {}
    for key, values in sorted(groups.items()):
        arr = np.array(values)
        out[key] = {
            "clips": int(len(arr)),
            "mean_score": round(float(arr.mean()), 4),
            "flagged_rate": round(float(np.mean(arr >= threshold)), 4),
        }
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-limit", type=int, default=10000)
    ap.add_argument("--val-limit", type=int, default=3000)
    ap.add_argument("--cross-limit", type=int, default=1866)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.train_limit, args.val_limit, args.cross_limit, args.epochs = 200, 100, 100, 1

    import tensorflow as tf
    tf.random.set_seed(1337)
    tf.get_logger().setLevel("ERROR")

    print(f"[callguard] input {N_MEL} x {N_FRAMES} (2 s), cache {MAX_FRAMES} frames")

    # The dev set must not share speakers with train; ASVspoof's protocol
    # guarantees this, but a wrong mirror can silently break it.
    train_speakers = speaker_ids("train")
    val_speakers = speaker_ids("validation")
    overlap = train_speakers & val_speakers
    print(f"[callguard] speakers train={len(train_speakers)} "
          f"val={len(val_speakers)} overlap={len(overlap)}")
    if overlap:
        raise SystemExit(f"validation shares {len(overlap)} speakers with train; "
                         f"refusing to report a leaked EER")

    build_cache("train", args.train_limit, seed=11)
    build_cache("validation", args.val_limit, seed=12)

    Xtr, ytr = load_cache("train")
    Xva, yva = load_cache("validation")

    has_cross = bool(parquet_files("cross"))
    if has_cross:
        build_cache("cross", args.cross_limit, seed=13)
        Xcr, ycr = load_cache("cross")
        print(f"[callguard] train={len(ytr)} val={len(yva)} cross={len(ycr)}")
    else:
        Xcr = ycr = None
        print(f"[callguard] train={len(ytr)} val={len(yva)} "
              f"(no cross-dataset corpus in {CROSS_DIR})")

    model = build_model()
    model.summary(line_length=96)

    steps = max(1, len(ytr) // args.batch)
    print(f"[callguard] {steps} steps x {args.epochs} epochs")

    t0 = time.time()
    model.fit(
        batch_generator(Xtr, ytr, args.batch, seed=21, train=True),
        steps_per_epoch=steps,
        epochs=args.epochs,
        verbose=2,
    )
    train_seconds = time.time() - t0

    val_scores = predict(model, Xva, yva)
    val_eer, val_threshold, val_acc = eer(val_scores, yva)
    print(f"[callguard] val   EER={val_eer * 100:.2f}%  acc={val_acc:.3f}  "
          f"AUC={auc(val_scores, yva):.4f}")

    cross_eer = cross_threshold = cross_acc = None
    if has_cross:
        cross_scores = predict(model, Xcr, ycr)
        cross_eer, cross_threshold, cross_acc = eer(cross_scores, ycr)
        print(f"[callguard] cross EER={cross_eer * 100:.2f}%  acc={cross_acc:.3f}  "
              f"AUC={auc(cross_scores, ycr):.4f}")

    out_dir = os.path.join(ROOT, "app", "src", "main", "assets", "models")
    os.makedirs(out_dir, exist_ok=True)
    tflite_path = os.path.join(out_dir, "deepfake_detector.tflite")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite)
    print(f"[callguard] wrote {tflite_path} ({len(tflite) / 1024:.1f} KiB)")

    # verify the quantized artefact, not just the Keras model
    tflite_scores = predict_tflite(tflite_path, Xva, yva)
    tflite_eer, tflite_threshold, tflite_acc = eer(tflite_scores, yva)
    print(f"[callguard] tflite val EER={tflite_eer * 100:.2f}%  acc={tflite_acc:.3f}")

    per_source = {}
    calibration = None
    if has_cross:
        tflite_cross = predict_tflite(tflite_path, Xcr, ycr)
        tflite_cross_eer, tflite_cross_threshold, tflite_cross_acc = eer(tflite_cross, ycr)
        print(f"[callguard] tflite cross EER={tflite_cross_eer * 100:.2f}%  "
              f"acc={tflite_cross_acc:.3f}")
        per_source = source_breakdown(tflite_cross, ycr, tflite_cross_threshold)
        for name, stats in per_source.items():
            print(f"[callguard]   {name:10s} n={stats['clips']:4d} "
                  f"mean P(fake)={stats['mean_score']:.3f} "
                  f"flagged={stats['flagged_rate'] * 100:5.1f}%")

        # Pick the display operating point on a disjoint half of the cross corpus
        # (unseen speakers and generators -- the closest proxy for a real call)
        # and report on the other half.
        cal_rng = np.random.default_rng(7)
        order = cal_rng.permutation(len(ycr))
        calib_idx, held_idx = order[::2], order[1::2]
        cal_a, cal_b = fit_platt(tflite_cross[calib_idx], ycr[calib_idx])
        held_eer, held_threshold, held_acc = eer(
            apply_platt(tflite_cross[held_idx], cal_a, cal_b), ycr[held_idx]
        )
        calibration = {
            "method": "platt",
            "a": round(cal_a, 6),
            "b": round(cal_b, 6),
            "fit_on": "disjoint half of the cross-dataset corpus",
            "held_out_eer": round(float(held_eer), 5),
            "held_out_accuracy": round(float(held_acc), 5),
            "note": ("applies sigmoid(a*logit(p)+b); strictly monotone, so it "
                     "cannot change the measured EER. Maps the operating point "
                     "for unseen-generator audio to 0.5."),
        }
        print(f"[callguard] calibration a={cal_a:.4f} b={cal_b:.4f}  "
              f"held-out cross EER={held_eer * 100:.2f}% acc={held_acc:.3f}")
    else:
        tflite_cross_eer = tflite_cross_acc = tflite_cross_threshold = None

    headline_eer = tflite_cross_eer if has_cross else tflite_eer
    headline_set = ("garystafford cross-dataset" if has_cross else "ASVspoof validation")

    meta = {
        "name": "callguard-asvspoof-cnn",
        "version": "0.3.0",
        "training": {
            "dataset": "ASVspoof 2019 Logical Access (Bisher mirror of the official protocol)",
            "train_clips": int(len(ytr)),
            "validation_clips": int(len(yva)),
            "validation_speakers": len(val_speakers),
            "cross_clips": int(len(ycr)) if has_cross else 0,
            "cross_dataset": ("garystafford/deepfake-audio-detection (CC-BY-4.0)"
                              if has_cross else None),
            "epochs": args.epochs,
            "seconds": round(train_seconds, 1),
        },
        "metrics": {
            "headline_eer": round(float(headline_eer), 5),
            "headline_set": headline_set,
            "validation_eer": round(val_eer, 5),
            "validation_accuracy": round(val_acc, 5),
            "cross_dataset_eer": (round(cross_eer, 5) if has_cross else None),
            "cross_dataset_accuracy": (round(cross_acc, 5) if has_cross else None),
            "tflite_validation_eer": round(tflite_eer, 5),
            "tflite_cross_dataset_eer": (round(tflite_cross_eer, 5) if has_cross else None),
            "eer_threshold": round(float(tflite_cross_threshold if has_cross
                                         else tflite_threshold), 5),
            "calibrated_threshold": 0.5 if calibration else None,
        },
        "calibration": calibration,
        "per_source": per_source,
        "caveat": (
            "Trained on ASVspoof 2019 LA (TTS/VC attacks), validated on its "
            "speaker-disjoint dev set and on a separate CC-BY corpus. Real phone "
            "calls, unseen generators and adversarial audio are not validated. "
            "Treat the score as a hint, not proof."
        ),
        "purpose": "Research demo. Not a security control.",
        "input_shape": [1, N_MEL, N_FRAMES, 1],
        "output": "calibrated P(fake) in [0,1]; 0.5 is the fitted operating point",
        "dsp": {
            "sample_rate": SR,
            "n_fft": 512,
            "win": WIN,
            "hop": HOP,
            "n_mel": N_MEL,
            "fmin": 20.0,
            "fmax": 8000.0,
            "floor": 1e-6,
            "window_seconds": 2.0,
            "frames": N_FRAMES,
            "normalization": "z-score over the whole window",
        },
        "params": int(model.count_params()),
        "size_kib": round(len(tflite) / 1024, 1),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[callguard] params={meta['params']} size={meta['size_kib']} KiB")
    print("[callguard] done")


if __name__ == "__main__":
    main()
