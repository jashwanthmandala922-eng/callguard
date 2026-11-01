#!/usr/bin/env python3
"""Minimal AASIST-lite training experiment."""
import os
import sys
import glob
import time
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

# Add tools to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_demo_model import HOP, N_MEL, SR, N_FRAMES, WIN, channel, log_mel, zscore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASV_DIR = os.path.join(ROOT, "data", "asvspoof2019_la")
ITW_DIR = os.path.join(ROOT, "data", "in_the_wild")
CROSS_DIR = os.path.join(ROOT, "data", "deepfake_audio_detection")
CACHE_DIR = os.path.join(ROOT, "data", "cache_gpu")
MODEL_DIR = os.path.join(ROOT, "app", "src", "main", "assets", "models")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SR = 16000
MAX_SECONDS = 3.0
MAX_FRAMES = 1 + (int(MAX_SECONDS * SR) - WIN) // HOP  # 298


def _finish(data, rate, mode):
    if data.size == 0:
        return None
    x = data.mean(axis=1)
    rate = int(rate)
    if rate != SR:
        g = gcd(rate, SR)
        x = resample_poly(x, SR // g, rate // g).astype(np.float32)
    if len(x) < SR // 2:
        return None
    want = int(MAX_SECONDS * SR)
    if len(x) > want:
        x = x[:want]
    elif len(x) < want:
        x = np.pad(x, (0, want - len(x)))
    mode = "light"  # simplified
    if mode == "full":
        # simplified channel simulation
        pass
    mel = log_mel(x)
    if mel.shape[0] < MAX_FRAMES:
        pad = MAX_FRAMES - mel.shape[0]
        fill = float(mel.min()) if mel.size else float(np.log(1e-6))
        mel = np.pad(mel, ((0, pad), (0, 0)), constant_values=fill)
    else:
        mel = mel[:MAX_FRAMES]
    return mel.astype(np.float16)


def _worker(job):
    kind, payload, mode, seed = job
    try:
        if kind == "bytes":
            data, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        else:
            data, rate = sf.read(payload, dtype="float32", always_2d=True)
    except Exception:
        return None
    return _finish(data, rate, "light")


def asv_jobs(split, limit=None, balanced=False, mode="light"):
    parquet_files = glob.glob(os.path.join(ASV_DIR, f"{split}-*.parquet"))
    parquet_files = [p for p in parquet_files if os.path.basename(p).startswith(split + "-")]
    jobs = []
    labels = []
    for path in parquet_files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["audio", "key"]):
            rows = batch.to_pydict()
            for audio, key in zip(rows["audio"], rows["key"]):
                blob = audio["bytes"] if isinstance(audio, dict) else audio
                jobs.append(("bytes", blob, mode if split == "train" else "none", len(jobs)))
                labels.append(int(key))
                if limit and len(jobs) >= limit:
                    return jobs, labels
    return jobs, labels


def itw_jobs(limit=None, mode="light"):
    parquet_files = sorted(glob.glob(os.path.join(ITW_DIR, "*.parquet")))
    jobs = []
    labels = []
    for path in parquet_files:
        pf = pq.ParquetFile(path)
        columns = ["audio", "label", "notes"]
        available = set(pf.schema_arrow.names)
        columns = [c for c in columns if c in available]
        for batch in pf.iter_batches(batch_size=256, columns=columns):
            rows = batch.to_pydict()
            for audio, label in zip(rows["audio"], rows["label"]):
                blob = audio["bytes"] if isinstance(audio, dict) else audio
                jobs.append(("bytes", blob, mode, len(jobs)))
                labels.append(int(label))
                if limit and len(jobs) >= limit:
                    return jobs, labels
    print(f"[itw] {len(jobs)} labelled clips ({sum(labels)} fake / {len(labels) - sum(labels)} real)")
    return jobs, labels


def cross_jobs(limit=None):
    jobs = []
    labels = []
    names = []
    for path in sorted(glob.glob(os.path.join(CROSS_DIR, "*.parquet"))):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["audio", "label"]):
            rows = batch.to_pydict()
            for audio, label in zip(rows["audio"], rows["label"]):
                blob = audio["bytes"] if isinstance(audio, dict) else audio
                name = os.path.basename(str(audio.get("path", ""))) if isinstance(audio, dict) else ""
                jobs.append(("bytes", blob, "none", len(jobs)))
                labels.append(int(label))
                names.append(name)
                if limit and len(jobs) >= limit:
                    return jobs, labels, names
    return jobs, labels, names


def build_cache(name, jobs, labels, workers=2):
    import multiprocessing as mp
    x_path = os.path.join(CACHE_DIR, f"X_{name}.npy")
    y_path = os.path.join(CACHE_DIR, f"y_{name}.npy")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    n = len(jobs)
    if n == 0:
        return None, None, None
    
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float16, shape=(n, MAX_FRAMES, N_MEL))
    y = np.zeros(n, dtype=np.int8)
    kept = []
    t0 = time.time()
    written = 0
    
    with mp.Pool(workers) as pool:
        for i, feat in enumerate(pool.imap(_worker, jobs, chunksize=16)):
            if feat is None:
                continue
            X[written] = feat
            y[written] = labels[i]
            kept.append(i)
            written += 1
            if written % 2000 == 0:
                rate = written / max(time.time() - t0, 1e-6)
                print(f"[cache] {name}: {written}/{n} ({rate:.0f}/s)")
    
    X.flush()
    np.save(y_path, y[:written])
    np.save(os.path.join(CACHE_DIR, f"kept_{name}.npy"), np.array(kept, dtype=np.int64))
    
    print(f"[cache] {name}: {written} clips ({int((y[:written] == 0).sum())} real / {int((y[:written] == 1).sum())} fake) in {time.time() - t0:.0f}s")
    return np.load(x_path, mmap_mode="r"), y[:written], np.array(kept, dtype=np.int64)


def eer(scores, labels):
    fake = np.sort(scores[labels == 1])
    real = np.sort(scores[labels == 0])
    thresholds = np.unique(np.concatenate([fake, real]))
    far = np.searchsorted(fake, thresholds, side="left") / max(len(fake), 1)
    frr = 1.0 - np.searchsorted(real, thresholds, side="left") / max(len(real), 1)
    idx = int(np.argmin(np.abs(far - frr)))
    value = float((far[idx] + frr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    acc = float(np.mean((scores >= threshold) == (labels == 1)))
    return value, threshold, acc


def predict(model, X, y, device="cpu"):
    model.eval()
    import torch
    with torch.no_grad():
        scores = []
        for start in range(0, len(y), 256):
            idx = np.arange(start, min(start + 256, len(y)))
            xb = torch.from_numpy(X[idx]).to(device).unsqueeze(1)
            with torch.cuda.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(xb)
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores)


def torch_auc(scores, labels):
    import numpy as np
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores)
    ranks = np.empty(len(all_scores))
    ranks[order] = np.arange(1, len(all_scores) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asv-limit", type=int, default=0, help="0 = all")
    ap.add_argument("--itw-limit", type=int, default=0, help="0 = all")
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--cross-limit", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--architecture", default="aasist", choices=["cnn", "aasist"])
    ap.add_argument("--width", default="16,32,64,128")
    ap.add_argument("--pooling", default="meanstd", choices=["meanstd", "avg"])
    ap.add_argument("--augment", default="light", choices=["light", "full", "none"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    
    if args.smoke:
        args.asv_limit, args.itw_limit = 400, 200
        args.val_limit, args.cross_limit = 200, 100
    
    width = tuple(int(w) for w in args.width.split(","))
    
    # Build model
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gpu] device={device}")
    
    # Simple AASIST-lite model
    class AASISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.LeakyReLU(0.3),
                nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.LeakyReLU(0.3),
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.3),
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.3),
            )
            self.attention = nn.Sequential(nn.Conv2d(128, 1, 1), nn.Sigmoid())
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, 1))
        
        def forward(self, x):
            h = self.features(x)
            att = self.attention(h)
            h = h * att
            h = self.global_pool(h).view(h.size(0), -1)
            h = self.classifier(h)
            return h.squeeze(1)
    
    model = AASISTNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params} params, pooling={args.pooling}, width={width}")
    
    # Data
    asv_train_jobs, asv_train_labels = asv_jobs("train", args.asv_limit or None, mode=args.augment)
    itw_train_jobs, itw_train_labels = itw_jobs(args.itw_limit or None, mode=args.augment)
    cross_jobs_list, cross_labels, cross_names = cross_jobs(args.cross_limit or None)
    
    jobs = asv_train_jobs + itw_train_jobs
    labels = asv_train_labels + itw_train_labels
    print(f"[data] ASVspoof {len(asv_train_jobs)} + In-the-Wild {len(itw_train_jobs)} = {len(jobs)} clips")
    
    # Cache
    from multiprocessing import Pool as mp_Pool
    mp = __import__('multiprocessing')
    X_train, y_train, kept_tr = build_cache("train", jobs, labels, workers=args.workers if hasattr(args, 'workers') else 2)
    if X_train is None:
        print("[data] no training clips")
        return
    
    val_jobs, val_labels = asv_jobs("validation", args.val_limit or None, balanced=True)
    X_val, y_val, _ = build_cache("validation", val_jobs, val_labels, workers=2)
    
    cross_X, cross_y, kept_cr = build_cache("cross", cross_jobs_list, cross_labels, workers=2)
    if len(cross_names) == len(cross_jobs_list):
        cross_names = [cross_names[i] for i in kept_cr]
    
    # Training
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    rng = np.random.default_rng(21)
    
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        loader = torch.utils.data.DataLoader(range(len(y_train)), batch_size=128, shuffle=True)
        for idx in loader:
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss) * len(yb)
            correct += int(((logits > 0).float() == yb).sum())
            total += len(yb)
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_scores = predict(model, X_val, y_val, device=device)
        val_eer, _, val_acc = eer(val_scores, y_val)
        
        cross_scores = predict(model, cross_X, cross_y, device=device)
        cross_eer, _, cross_acc = eer(cross_scores, cross_y)
        
        print(f"[epoch {epoch+1:02d}/{args.epochs}] loss={loss_sum/max(total,1):.4f} train_acc={correct/max(total,1):.3f} val EER={val_eer*100:.2f}% acc={val_acc:.3f} cross EER={cross_eer*100:.2f}% acc={cross_acc:.3f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_scores = predict(model, X_val, y_val, device=device)
        val_eer, _, val_acc = eer(val_scores, y_val)
        cross_scores = predict(model, cross_X, cross_y, device=device)
        cross_eer, _, cross_acc = eer(cross_scores, cross_y)
    
    print(f"[final] val   EER={val_eer*100:.2f}% acc={val_acc:.3f} AUC={torch_auc(val_scores, y_val):.4f}")
    print(f"[final] cross EER={cross_eer*100:.2f}% acc={cross_acc:.3f} AUC={torch_auc(cross_scores, cross_y):.4f}")
    
    # TFLite export (skipped for AASIST-lite due to weight transfer issues)
    print("[export] skipping TFLite export for AASIST-lite (weight transfer issues)")


if __name__ == "__main__":
    main()