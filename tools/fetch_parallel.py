#!/usr/bin/env python3
"""
Parallel HTTP range downloader for the big corpus files.

Why: the Hugging Face and PyTorch CDNs throttle a single connection to well
under 1 MB/s here, which turns a 2.5 GB wheel into an hour. Splitting the file
into range requests over several connections is much faster.

Each slice is written to its own `DEST.partN` file, so a rerun resumes every
slice independently and a half-written file is never mistaken for a complete one.

  python tools/fetch_parallel.py URL DEST [--connections 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import requests

CHUNK = 1 << 20  # 1 MiB per read


def head(url: str) -> tuple[int, bool]:
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    size = int(response.headers.get("content-length", 0))
    ranges = response.headers.get("accept-ranges", "").lower() == "bytes"
    return size, ranges


def fetch_slice(url: str, start: int, end: int, path: str, progress: list,
                lock: threading.Lock, stop: threading.Event, retries: int = 10) -> None:
    """Download [start, end] inclusive into `path`, resuming what is there."""
    total = end - start + 1
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have > total:
        os.remove(path)
        have = 0
    position = start + have
    attempt = 0
    while position <= end:
        if stop.is_set():
            return
        try:
            headers = {"Range": f"bytes={position}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(path, "ab") as handle:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        if stop.is_set():
                            return
                        handle.write(chunk)
                        position += len(chunk)
                        with lock:
                            progress[0] += len(chunk)
                        if position > end:
                            break
            attempt = 0
        except Exception as exc:  # pragma: no cover - network dependent
            attempt += 1
            if attempt > retries:
                print(f"\n[fetch] giving up at byte {position}: {exc}")
                stop.set()
                return
            time.sleep(min(2 * attempt, 10))
            have = os.path.getsize(path) if os.path.exists(path) else 0
            position = start + min(have, total)


def download(url: str, dest: str, connections: int) -> None:
    size, ranges = head(url)
    if not size:
        raise SystemExit("[fetch] server did not report a size")
    if not ranges:
        raise SystemExit("[fetch] server does not support range requests")

    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"[fetch] already complete: {dest}")
        return

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    bounds = []
    for i in range(connections):
        start = i * size // connections
        end = (i + 1) * size // connections - 1
        if start <= end:
            bounds.append((start, end))

    progress = [0]
    lock = threading.Lock()
    stop = threading.Event()
    t0 = time.time()

    threads = []
    for i, (start, end) in enumerate(bounds):
        path = f"{dest}.part{i}"
        have = min(os.path.getsize(path), end - start + 1) if os.path.exists(path) else 0
        progress[0] += have
        threads.append(threading.Thread(
            target=fetch_slice,
            args=(url, start, end, path, progress, lock, stop),
            daemon=True,
        ))

    for t in threads:
        t.start()

    last = 0.0
    while any(t.is_alive() for t in threads):
        time.sleep(1.0)
        if time.time() - last >= 10:
            last = time.time()
            got = progress[0]
            rate = got / max(time.time() - t0, 1e-6)
            print(f"  {os.path.basename(dest)} {got * 100 / size:5.1f}% "
                  f"({got / 1e6:.0f}/{size / 1e6:.0f} MB, {rate / 1e6:.2f} MB/s)",
                  flush=True)
    for t in threads:
        t.join()

    if stop.is_set() or progress[0] < size:
        raise SystemExit(f"[fetch] incomplete ({progress[0]}/{size} bytes), rerun to resume")

    with open(dest, "wb") as out:
        for i, (start, end) in enumerate(bounds):
            path = f"{dest}.part{i}"
            with open(path, "rb") as handle:
                while True:
                    chunk = handle.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
            os.remove(path)

    if os.path.getsize(dest) != size:
        raise SystemExit("[fetch] assembled size mismatch")
    print(f"[fetch] done {dest} ({size / 1e6:.0f} MB in {time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--connections", type=int, default=8)
    args = ap.parse_args()
    download(args.url, args.dest, max(1, args.connections))


if __name__ == "__main__":
    main()
