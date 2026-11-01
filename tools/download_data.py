#!/usr/bin/env python3
"""
Download the labelled audio used for training CallGuard's detector.

Primary corpus: Bisher/ASVspoof_2019_LA on Hugging Face -- the official ASVspoof
2019 Logical Access protocol (25,380 train / 24,844 dev / 71,237 eval), 16 kHz
FLAC, with speaker IDs. Labels: 0 = bonafide, 1 = spoof.

Cross-dataset check: garystafford/deepfake-audio-detection (CC-BY-4.0), 933 real
YouTube clips + 933 synthetic clips from six vendors. Labels: 0 = real, 1 = fake.

Second training corpus (optional): SpeechAntiSpoofingBenchmarks/InTheWild, a
parquet repackaging of mueller91/In-The-Wild (CC-BY-SA-4.0) -- 37.9 h of real and
synthetic speech from online videos, 58 public figures. This is the real-world
domain that the lab corpus does not cover. Fetched as 8 shards (~2.3 GB) rather
than the 8.2 GB upstream zip.

None of these corpora is redistributed here; all are pulled into data/
(git-ignored). Check the upstream licences before commercial use.

  python tools/download_data.py --splits train validation --cross --in-the-wild
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

ITW_DATASET = "SpeechAntiSpoofingBenchmarks/InTheWild"
ITW_BASE = f"https://huggingface.co/datasets/{ITW_DATASET}/resolve/main"
ITW_DIR = os.path.join(ROOT, "data", "in_the_wild")
ITW_FILES = [f"data/test-0000{i}-of-00008.parquet" for i in range(8)]

# Modern generators. ASVspoof 2019 and In-the-Wild are both pre-2021; the
# vendors in the held-out corpus are current neural TTS. These three add
# current vocoders and codecs without needing a 34-84 GB full download.
MODERN = {
    "libri_sevoc": (
        "SpeechAntiSpoofingBenchmarks/LibriSeVoc",
        [f"data/test-0000{i}-of-00008.parquet" for i in range(8)],
    ),
    "deepvoice": (
        "SpeechAntiSpoofingBenchmarks/DeepVoice",
        [f"data/test-0000{i}-of-00002.parquet" for i in range(2)],
    ),
    "asvspoof2021_df": (
        "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF",
        [f"data/test-{i:05d}-of-00080.parquet" for i in range(10)],
    ),
}
MODERN_DIR = os.path.join(ROOT, "data", "modern")

SPLIT_FILES = {
    "train": ["data/train-00000-of-00001.parquet"],
    "validation": ["data/validation-00000-of-00001.parquet"],
    "test": ["data/test-00000-of-00001.parquet"],
}


def download(url: str, destination: str, connections: int = 1) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        print(f"  exists  {os.path.basename(destination)}")
        return

    if connections > 1:
        # Single streams get throttled hard by these CDNs; range requests are
        # several times faster and still resume.
        from fetch_parallel import download as parallel_download
        parallel_download(url, destination, connections)
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


def fetch_in_the_wild(connections: int = 8) -> None:
    """Download the In-the-Wild corpus as parquet shards (~2.3 GB total).

    The upstream mueller91 release is an 8.2 GB zip. This mirror repackages the
    identical audio as 16 kHz FLAC inside parquet, which is faster to fetch and
    needs no unpack step.
    """
    os.makedirs(ITW_DIR, exist_ok=True)
    for relative in ITW_FILES:
        download(f"{ITW_BASE}/{relative}",
                 os.path.join(ITW_DIR, os.path.basename(relative)),
                 connections=connections)


def fetch_modern(connections: int = 8) -> None:
    """Download the modern-generator corpora (LibriSeVoc, DeepVoice, DF subset)."""
    for name, (dataset, files) in MODERN.items():
        target = os.path.join(MODERN_DIR, name)
        os.makedirs(target, exist_ok=True)
        print(f"[modern] {dataset}")
        base = f"https://huggingface.co/datasets/{dataset}/resolve/main"
        for relative in files:
            download(f"{base}/{relative}",
                     os.path.join(target, os.path.basename(relative)),
                     connections=connections)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--cross", action="store_true",
                    help="also fetch the garystafford cross-dataset corpus (~574 MB)")
    ap.add_argument("--in-the-wild", action="store_true",
                    help="also fetch In-the-Wild (~2.3 GB of parquet shards)")
    ap.add_argument("--modern", action="store_true",
                    help="also fetch LibriSeVoc + DeepVoice + ASVspoof2021 DF subset (~8 GB)")
    ap.add_argument("--connections", type=int, default=8,
                    help="parallel range connections per file (1 = single stream)")
    args = ap.parse_args()

    if args.cross:
        print(f"[cross] {CROSS_DATASET}")
        for relative in CROSS_FILES:
            download(f"{CROSS_BASE}/{relative}",
                     os.path.join(CROSS_DIR, os.path.basename(relative)),
                     connections=args.connections)

    if args.in_the_wild:
        print(f"[in-the-wild] {ITW_DATASET}")
        fetch_in_the_wild(connections=args.connections)

    if args.modern:
        fetch_modern(connections=args.connections)

    for split in args.splits:
        if split not in SPLIT_FILES:
            sys.exit(f"unknown split {split}")
        files = SPLIT_FILES[split]
        print(f"[{split}] {len(files)} file(s)")
        for relative in files:
            download(f"{BASE}/{relative}",
                     os.path.join(args.dir, os.path.basename(relative)),
                     connections=args.connections)

    total = sum(
        os.path.getsize(os.path.join(args.dir, f))
        for f in os.listdir(args.dir)
        if f.endswith(".parquet")
    )
    print(f"total parquet on disk: {total / 1e9:.2f} GB in {args.dir}")


if __name__ == "__main__":
    main()
