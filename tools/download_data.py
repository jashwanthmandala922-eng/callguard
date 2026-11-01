#!/usr/bin/env python3
"""
Download the labelled audio used for training CallGuard's detector.

Primary corpus: Bisher/ASVspoof_2019_LA on Hugging Face -- the official ASVspoof
2019 Logical Access protocol (25,380 train / 24,844 dev / 71,237 eval), 16 kHz
FLAC, with speaker IDs. Labels: 0 = bonafide, 1 = spoof.

Cross-dataset check: garystafford/deepfake-audio-detection (CC-BY-4.0), 933 real
YouTube clips + 933 synthetic clips from six vendors. Labels: 0 = real, 1 = fake.

Neither corpus is redistributed here; both are pulled into data/ (git-ignored).
Check the upstream licences before commercial use.

  python tools/download_data.py --splits train validation --cross
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

DATASET = "Bisher/ASVspoof_2019_LA"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "data", "asvspoof2019_la")

CROSS_DATASET = "garystafford/deepfake-audio-detection"
CROSS_BASE = f"https://huggingface.co/datasets/{CROSS_DATASET}/resolve/main"
CROSS_DIR = os.path.join(ROOT, "data", "deepfake_audio_detection")
CROSS_FILES = [
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
]

SPLIT_FILES = {
    "train": ["data/train-00000-of-00001.parquet"],
    "validation": ["data/validation-00000-of-00001.parquet"],
    "test": ["data/test-00000-of-00001.parquet"],
}


def download(url: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        print(f"  exists  {os.path.basename(destination)}")
        return

    part = destination + ".part"
    resume_at = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) + resume_at
        mode = "ab" if resume_at else "wb"
        done = resume_at
        last_pct = -5
        with open(part, mode) as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct >= last_pct + 5:
                        print(f"  {os.path.basename(destination)} {pct:3d}% "
                              f"({done / 1e6:.0f}/{total / 1e6:.0f} MB)", flush=True)
                        last_pct = pct
    os.replace(part, destination)
    print(f"  done    {os.path.basename(destination)} ({os.path.getsize(destination) / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--cross", action="store_true",
                    help="also fetch the garystafford cross-dataset corpus (~574 MB)")
    args = ap.parse_args()

    if args.cross:
        print(f"[cross] {CROSS_DATASET}")
        for relative in CROSS_FILES:
            download(f"{CROSS_BASE}/{relative}", os.path.join(CROSS_DIR, os.path.basename(relative)))

    for split in args.splits:
        if split not in SPLIT_FILES:
            sys.exit(f"unknown split {split}")
        files = SPLIT_FILES[split]
        print(f"[{split}] {len(files)} file(s)")
        for relative in files:
            download(f"{BASE}/{relative}", os.path.join(args.dir, os.path.basename(relative)))

    total = sum(
        os.path.getsize(os.path.join(args.dir, f))
        for f in os.listdir(args.dir)
        if f.endswith(".parquet")
    )
    print(f"total parquet on disk: {total / 1e9:.2f} GB in {args.dir}")


if __name__ == "__main__":
    main()
