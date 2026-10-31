#!/usr/bin/env python3
"""
Score the exported .tflite with fresh synthetic clips and report the distribution.

Use this after training to catch the classic demo failure: a model that ranks well
(AUC ~1) but whose sigmoid is shifted, so every clip reads "synthetic".

  python tools/check_model.py --n 24
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_demo_model import ROOT, make_sample, to_model_input  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="clips per class")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    import tensorflow as tf

    model_path = args.model or os.path.join(
        ROOT, "app", "src", "main", "assets", "models", "deepfake_detector.tflite"
    )
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    rng = np.random.default_rng(4242)
    scores = {0: [], 1: []}
    for label in (0, 1):
        for _ in range(args.n):
            mel = make_sample(rng, label)
            interp.set_tensor(inp["index"], to_model_input(mel))
            interp.invoke()
            scores[label].append(float(interp.get_tensor(out["index"])[0, 0]))

    human = np.array(scores[0])
    fake = np.array(scores[1])

    def describe(name, a):
        print(f"{name:8s} n={len(a):3d} min={a.min():.3f} p25={np.percentile(a, 25):.3f} "
              f"median={np.median(a):.3f} p75={np.percentile(a, 75):.3f} max={a.max():.3f}")

    describe("human", human)
    describe("synthetic", fake)

    acc = float(np.mean(np.concatenate([human < 0.5, fake >= 0.5])))
    # rank-based AUC, no sklearn dependency
    all_scores = np.concatenate([human, fake])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    auc = (ranks[len(human):].sum() - len(fake) * (len(fake) + 1) / 2) / (len(human) * len(fake))
    print(f"accuracy@0.5={acc:.3f}  auc={auc:.4f}")
    print(f"threshold midway = {(np.median(human) + np.median(fake)) / 2:.3f}")


if __name__ == "__main__":
    main()
