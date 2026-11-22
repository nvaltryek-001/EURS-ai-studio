#!/usr/bin/env python3
"""Train EURS-Lite (CPU): residual-MLP restoration + MLP pixel classifier.

Train split -> fit;  val split -> model selection (hidden sizes, smoothing);  test split is
never touched here (use evaluate.py).

  python ml/train_lite.py --manifest data/splits/eurs/manifest.csv --out runs/eurs_lite
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

import eurs_ml  # noqa: F401
from eurs_core import NUM_CLASSES
from eurs_core.features import restoration_features, segmentation_features
from eurs_core.lite import LiteModel, NumpyMLP, _resize_max
from eurs_ml.discover import manifest_root, read_manifest
from eurs_ml.io import MaskDecoder, load_layout, read_rgb
from eurs_ml.metrics import confusion, psnr, seg_scores

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--work-side", type=int, default=384)
ap.add_argument("--px-per-image", type=int, default=1500)
ap.add_argument("--max-train-images", type=int, default=600)
ap.add_argument("--max-val-images", type=int, default=60)
ap.add_argument("--max-iter", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--install", action="store_true", help="copy weights to backend/models/")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
root = manifest_root(args.manifest)
report = json.loads(args.manifest.with_name("split_report.json").read_text())
layout = report.get("layout") or {}
decode = MaskDecoder(layout.get("mask_encoding", "auto"), layout.get("mask_palette"))
train = read_manifest(args.manifest, "train")
val = read_manifest(args.manifest, "val")
rng.shuffle(train)
train, val = train[: args.max_train_images], val[: args.max_val_images]
args.out.mkdir(parents=True, exist_ok=True)
log = {"dataset": report.get("dataset"), "n_train_images": len(train), "n_val_images": len(val),
       "args": {k: str(v) for k, v in vars(args).items()}}


def load(s, ws):
    x = _resize_max(read_rgb(root / s["input"]).astype(np.float32) / 255, ws)
    y = _resize_max(read_rgb(root / s["clean"]).astype(np.float32) / 255, ws) if s["clean"] else None
    m = None
    if s["mask"]:
        import cv2
        m = decode(root / s["mask"])
        m = cv2.resize(m, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_NEAREST)
    return x, y, m


t0 = time.time()
cache = {"train": [load(s, args.work_side) for s in train], "val": [load(s, args.work_side) for s in val]}
print(f"loaded {len(train)} train / {len(val)} val images in {time.time() - t0:.1f}s")

# ------------------------------------------------------------------ stage 1: restoration
restorer = None
pairs = [(x, y) for x, y, _ in cache["train"] if y is not None]
if pairs:
    X, Y = [], []
    for x, y in pairs:
        f = restoration_features(x)
        f = f.reshape(-1, f.shape[2])
        idx = rng.choice(len(f), min(args.px_per_image, len(f)), replace=False)
        X.append(f[idx])
        Y.append((y - x).reshape(-1, 3)[idx])
    X, Y = np.concatenate(X), np.concatenate(Y)
    scaler = StandardScaler().fit(X)
    best = None
    log["restoration_candidates"] = []
    for hidden in [(32,), (64, 32), (128, 64)]:
        t = time.time()
        mlp = MLPRegressor(hidden_layer_sizes=hidden, early_stopping=True, max_iter=args.max_iter,
                           learning_rate_init=1e-3, batch_size=1024, random_state=args.seed)
        mlp.fit(scaler.transform(X), Y)
        cand = LiteModel(NumpyMLP.from_sklearn(mlp, scaler, "identity"), None, args.work_side)
        vals = [psnr(cand.restore((x * 255).astype(np.uint8)), (y * 255).astype(np.uint8))
                for x, y, _ in cache["val"] if y is not None]
        score = float(np.mean(vals)) if vals else -mlp.best_loss_
        entry = {"hidden": hidden, "val_psnr": score, "epochs": mlp.n_iter_, "fit_s": round(time.time() - t, 1),
                 "loss_curve": [float(v) for v in mlp.loss_curve_],
                 "val_r2_curve": [float(v) for v in (mlp.validation_scores_ or [])]}
        log["restoration_candidates"].append(entry)
        print(f"[restore] hidden={hidden} val PSNR={score:.3f} ({entry['fit_s']}s)")
        if best is None or score > best[0]:
            best = (score, NumpyMLP.from_sklearn(mlp, scaler, "identity"), hidden)
    restorer = best[1]
    log["restoration_selected"] = {"hidden": best[2], "val_psnr": best[0]}

# ------------------------------------------------------------------ stage 2: segmentation
segmenter = None
base = LiteModel(restorer, None, args.work_side)


def restored(x):
    return base.restore((x * 255).astype(np.uint8)).astype(np.float32) / 255


seg_train = [(restored(x), m) for x, _, m in cache["train"] if m is not None]
seg_val = [(restored(x), m) for x, _, m in cache["val"] if m is not None]
if seg_train:
    X, Y = [], []
    per_class = max(50, args.px_per_image // NUM_CLASSES)
    for r, m in seg_train:
        f = segmentation_features(r)
        f, mm = f.reshape(-1, f.shape[2]), m.ravel()
        for c in range(NUM_CLASSES):  # class-balanced pixel sampling
            idx = np.flatnonzero(mm == c)
            if len(idx):
                idx = rng.choice(idx, min(per_class, len(idx)), replace=False)
                X.append(f[idx])
                Y.append(mm[idx])
    X, Y = np.concatenate(X), np.concatenate(Y)
    log["seg_train_pixels_per_class"] = np.bincount(Y, minlength=NUM_CLASSES).tolist()
    scaler = StandardScaler().fit(X)
    best = None
    log["segmentation_candidates"] = []
    for hidden in [(64,), (128, 64), (128, 128, 64)]:
        t = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=hidden, early_stopping=True, max_iter=args.max_iter,
                            batch_size=1024, random_state=args.seed)
        mlp.fit(scaler.transform(X), Y)
        net = NumpyMLP.from_sklearn(mlp, scaler, "softmax")
        if len(mlp.classes_) != NUM_CLASSES:  # pad missing classes so argmax ids stay aligned
            w = np.zeros((net.weights[-1].shape[0], NUM_CLASSES), np.float32)
            b = np.full(NUM_CLASSES, -1e4, np.float32)
            w[:, mlp.classes_] = net.weights[-1]
            b[mlp.classes_] = net.biases[-1]
            net.weights[-1], net.biases[-1] = w, b
        for sigma in (0.0, 1.0, 2.0, 4.0):
            cand = LiteModel(None, net, args.work_side, smooth_sigma=sigma)
            cm = sum(confusion(cand.segment((r * 255).astype(np.uint8)), m) for r, m in seg_val)
            miou = seg_scores(cm)["mIoU"] if seg_val else float(mlp.best_validation_score_)
            entry = {"hidden": hidden, "smooth_sigma": sigma, "val_mIoU": miou, "epochs": mlp.n_iter_,
                     "fit_s": round(time.time() - t, 1), "loss_curve": [float(v) for v in mlp.loss_curve_],
                     "val_acc_curve": [float(v) for v in (mlp.validation_scores_ or [])]}
            log["segmentation_candidates"].append(entry)
            print(f"[segment] hidden={hidden} sigma={sigma} val mIoU={miou:.4f}")
            if best is None or miou > best[0]:
                best = (miou, net, hidden, sigma)
    segmenter = best[1]
    log["segmentation_selected"] = {"hidden": best[2], "smooth_sigma": best[3], "val_mIoU": best[0]}
    smooth = best[3]
else:
    smooth = 2.0

meta = {"name": "eurs_lite", "display_name": "EURS-Lite (CPU MLP)", "dataset": report.get("dataset"),
        "restoration": "residual MLP on colour statistics", "segmentation": "MLP pixel classifier",
        "params": int(sum(w.size + b.size for n in (restorer, segmenter) if n for w, b in zip(n.weights, n.biases))),
        "selected": {k: log.get(k) for k in ("restoration_selected", "segmentation_selected")}}
model = LiteModel(restorer, segmenter, args.work_side, smooth, meta)
model.save(args.out / "eurs_lite.npz")
log["total_s"] = round(time.time() - t0, 1)
(args.out / "train_log.json").write_text(json.dumps(log, indent=2, default=str))
print(f"saved {args.out / 'eurs_lite.npz'}  ({meta['params']} parameters, {log['total_s']}s)")
if args.install:
    import shutil
    dst = eurs_ml.REPO_ROOT / "backend" / "models" / "eurs_lite.npz"
    shutil.copy(args.out / "eurs_lite.npz", dst)
    print(f"installed -> {dst}")
