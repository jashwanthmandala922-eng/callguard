#!/usr/bin/env python3
"""
Generate the two audio fixtures used by the instrumented test and by manual QA.

  human_like.wav      additive glottal speech + channel simulation
  synthetic_like.wav  same speech pushed through the vocoder-style transform

These are synthetic stand-ins, not real recordings of people or real deepfakes.
They exist so the on-device test can assert that the pipeline *discriminates*,
not that it is accurate.

Usage:
  python tools/make_samples.py
"""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_demo_model import SR, channel, synth_speech, vocode  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DURATION = 3.2


def write_wav(path: str, samples: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {path} ({len(pcm) / SR:.2f}s, {os.path.getsize(path) / 1024:.0f} KiB)")


def main() -> None:
    human_rng = np.random.default_rng(2024)
    human = channel(synth_speech(human_rng, DURATION), human_rng)

    fake_rng = np.random.default_rng(2024)
    fake = channel(vocode(synth_speech(fake_rng, DURATION), fake_rng), fake_rng)

    targets = [
        os.path.join(ROOT, "samples"),
        os.path.join(ROOT, "app", "src", "androidTest", "assets"),
    ]
    for folder in targets:
        write_wav(os.path.join(folder, "human_like.wav"), human)
        write_wav(os.path.join(folder, "synthetic_like.wav"), fake)


if __name__ == "__main__":
    main()
