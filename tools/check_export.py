#!/usr/bin/env python3
"""Verify that the mean+std pooling architecture converts to TFLite.

The GPU trainer copies PyTorch weights into a Keras model with the same
architecture and converts that to TFLite. This script builds the Keras half with
random weights and checks that the converted artefact produces the same outputs,
so a conversion failure surfaces before a long training run.

  python tools/check_export.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_demo_model import N_FRAMES, N_MEL  # noqa: E402

WIDTH = (32, 64, 96, 128)


def build_keras():
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = layers.Input(shape=(N_MEL, N_FRAMES, 1), name="logmel")
    x = inp
    for i, out_ch in enumerate(WIDTH):
        x = layers.Conv2D(out_ch, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.GroupNormalization(groups=8, epsilon=1e-5, name=f"gn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)

    mean = layers.GlobalAveragePooling2D(name="mean")(x)
    squared = layers.Multiply(name="square")([x, x])
    mean_sq = layers.GlobalAveragePooling2D(name="mean_sq")(squared)
    mean_squared = layers.Multiply(name="mean_squared")([mean, mean])
    var = layers.Subtract(name="variance")([mean_sq, mean_squared])
    std = layers.Lambda(lambda t: tf.sqrt(tf.maximum(t, 1e-6)), name="std")(var)
    x = layers.Concatenate(name="mean_std")([mean, std])

    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)
    return tf.keras.Model(inp, out)


def main() -> None:
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    model = build_keras()
    print(f"params={model.count_params()}")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, N_MEL, N_FRAMES, 1)).astype(np.float32)
    reference = model(x, training=False).numpy().ravel()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    print(f"tflite={len(tflite) / 1024:.1f} KiB")

    path = os.path.join(os.environ.get("TEMP", "."), "export_check.tflite")
    with open(path, "wb") as f:
        f.write(tflite)

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    got = []
    for row in x:
        interp.set_tensor(inp["index"], row[None, ...])
        interp.invoke()
        got.append(float(interp.get_tensor(out["index"])[0, 0]))
    got = np.array(got)

    diff = float(np.max(np.abs(got - reference)))
    print(f"keras={np.round(reference, 5)}")
    print(f"tflite={np.round(got, 5)}")
    print(f"max|diff|={diff:.5f} (quantisation tolerance ~0.02)")
    if diff > 0.05:
        raise SystemExit("TFLite output diverges from Keras")
    print("OK: mean+std architecture converts and matches")


if __name__ == "__main__":
    main()
