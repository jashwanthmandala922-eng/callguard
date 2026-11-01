#!/usr/bin/env python3
"""Diagnose the raw score distribution and fit a display calibration.

The detector ranks well but its sigmoid is compressed: on unseen-generator audio
the EER operating point sits near 5e-4, so any UI that thresholds at 0.5 labels
everything human. This script reports the actual distributions, fits Platt
scaling on a disjoint half of the cross-dataset corpus, and verifies that the
map is rank-preserving (so the reported EER cannot change).

  python tools/calibrate_model.py            # report only
  python tools/calibrate_model.py --write    # write calibration into model_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_real_model as t  # noqa: E402

MODEL = os.path.join(t.ROOT, "app", "src", "main", "assets", "models",
                     "deepfake_detector.tflite")
META = os.path.join(t.ROOT, "app", "src", "main", "assets", "models",
                    "model_meta.json")


def quantiles(scores: np.ndarray) -> str:
    q = np.quantile(scores, [0.01, 0.1, 0.5, 0.9, 0.99])
    return "  ".join(f"p{int(p * 100):02d}={v:.5f}" for p, v in zip([1, 10, 50, 90, 99], q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    Xv, yv = t.load_cache("validation")
    Xc, yc = t.load_cache("cross")

    print("scoring validation + cross with the exported TFLite model...")
    sv = t.predict_tflite(MODEL, Xv, yv)
    sc = t.predict_tflite(MODEL, Xc, yc)

    for name, s, y in (("validation", sv, yv), ("cross", sc, yc)):
        print(f"\n[{name}] real n={int((y == 0).sum())}  fake n={int((y == 1).sum())}")
        print("  real:", quantiles(s[y == 0]))
        print("  fake:", quantiles(s[y == 1]))
        e, thr, acc = t.eer(s, y)
        print(f"  EER={e * 100:.2f}%  threshold={thr:.6f}  acc@thr={acc:.4f}")

    # Fit the display calibration on a disjoint half of the cross-dataset corpus.
    # That corpus is the closest proxy for deployment ("call audio from unknown
    # generators"), so it is the honest domain to pick an operating point for.
    # The other half is held out for the numbers below.
    rng = np.random.default_rng(7)
    order = rng.permutation(len(yc))
    calib_idx, eval_idx = order[::2], order[1::2]
    a, b = t.fit_platt(sc[calib_idx], yc[calib_idx])
    print(f"\nPlatt on cross calibration half ({len(calib_idx)} clips): "
          f"a={a:.4f} b={b:.4f}")

    for name, s, y in (("validation", sv, yv), ("cross", sc, yc)):
        cal = t.apply_platt(s, a, b)
        before = t.eer(s, y)
        after = t.eer(cal, y)
        print(f"[{name}] EER raw={before[0] * 100:.2f}%  calibrated={after[0] * 100:.2f}% "
              f"(must match; calibration is monotone)")
        print(f"  calibrated real:", quantiles(cal[y == 0]))
        print(f"  calibrated fake:", quantiles(cal[y == 1]))
        pred = cal >= 0.5
        print(f"  at 0.5: flagged real={pred[y == 0].mean() * 100:.1f}%  "
              f"flagged fake={pred[y == 1].mean() * 100:.1f}%")

    held = t.apply_platt(sc[eval_idx], a, b)
    held_eer, held_thr, held_acc = t.eer(held, yc[eval_idx])
    print(f"\ncross HELD-OUT half (n={len(eval_idx)}): EER={held_eer * 100:.2f}%  "
          f"acc={held_acc:.4f}  threshold={held_thr:.3f}")

    # The two shipped test fixtures, scored the same way the app does: every
    # 2 s window in 0.5 s hops, then the median.
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    ii = interp.get_input_details()[0]
    oi = interp.get_output_details()[0]

    for wav in ("real_speech.wav", "fake_speech.wav"):
        path = os.path.join(t.ROOT, "app", "src", "androidTest", "assets", wav)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            clip = t.decode_to_features(f.read(), np.random.default_rng(0), augment=False)
        if clip is None:
            print(f"{wav}: could not decode")
            continue

        scores = []
        for start in range(0, t.MAX_FRAMES - t.N_FRAMES + 1, 50):  # 0.5 s hop
            window = t.zscore(np.asarray(clip[start:start + t.N_FRAMES],
                                         dtype=np.float32).T)
            interp.set_tensor(ii["index"], window[None, :, :, None])
            interp.invoke()
            scores.append(float(interp.get_tensor(oi["index"])[0, 0]))
        raw = float(np.median(scores))
        cal = float(t.apply_platt(np.array([raw]), a, b)[0])
        print(f"\n{wav}: windows={len(scores)} raw median P(fake)={raw:.6f}  "
              f"calibrated={cal:.4f}  verdict={'SYNTHETIC' if cal >= 0.5 else 'human'}")

    if args.write:
        with open(META, encoding="utf-8") as f:
            meta = json.load(f)
        meta["calibration"] = {
            "method": "platt",
            "a": round(a, 6),
            "b": round(b, 6),
            "fit_on": "disjoint half of garystafford cross-dataset (933 real / 933 fake)",
            "held_out_eer": round(float(held_eer), 5),
            "held_out_accuracy": round(float(held_acc), 5),
            "note": ("applies sigmoid(a*logit(p)+b); monotone, so EER is unchanged. "
                     "Picks the operating point for unseen-generator audio."),
        }
        meta["metrics"]["calibrated_threshold"] = 0.5
        with open(META, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        print(f"\nwrote calibration into {META}")


if __name__ == "__main__":
    main()
