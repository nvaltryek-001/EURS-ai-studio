#!/usr/bin/env python3
"""Train EURS-Net (GPU recommended). Requires: pip install -r ml/requirements-torch.txt

  python ml/train_net.py --manifest data/splits/eurs/manifest.csv --out runs/eurs_net \
      --encoder resnet18 --epochs 60 --batch 8 --size 384

Train split -> optimisation; val split -> checkpoint selection + early stopping.
The test split is not read here (use ml/evaluate.py on the exported ONNX).
Ablations: --no-fusion, --w-rest 0 (seg only), --w-seg 0 (restoration only), --encoder mobilenet_v3_large.
Embedded track (<=1.0M params, <=12 GFLOPs): --encoder tiny  (verify with ml/check_budget.py).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import eurs_ml  # noqa: F401
from eurs_core import NUM_CLASSES
from eurs_ml.discover import manifest_root
from eurs_ml.io import MaskDecoder
from eurs_ml.metrics import seg_scores
from eurs_torch.data import EURSDataset, class_weights_from_fraction
from eurs_torch.losses import MultiTaskLoss
from eurs_torch.model import EURSNet, count_params

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--encoder", default="resnet18", choices=["tiny", "resnet18", "resnet34", "mobilenet_v3_large"])
ap.add_argument("--no-pretrained", action="store_true")
ap.add_argument("--no-fusion", action="store_true")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--size", type=int, default=384)
ap.add_argument("--val-size", type=int, default=512)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--w-rest", type=float, default=1.0)
ap.add_argument("--w-seg", type=float, default=1.0)
ap.add_argument("--select", default="combined", choices=["miou", "psnr", "combined"])
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", type=Path, default=None)
args = ap.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
args.out.mkdir(parents=True, exist_ok=True)

report = json.loads(args.manifest.with_name("split_report.json").read_text())
layout = report.get("layout") or {}
decoder = MaskDecoder(layout.get("mask_encoding", "auto"), layout.get("mask_palette"))
root = manifest_root(args.manifest)
train_ds = EURSDataset(args.manifest, root, "train", args.size, True, decoder)
val_ds = EURSDataset(args.manifest, root, "val", args.val_size, False, decoder)
train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                      pin_memory=dev.type == "cuda", persistent_workers=args.workers > 0)
val_dl = DataLoader(val_ds, 1, shuffle=False, num_workers=args.workers)

model = EURSNet(NUM_CLASSES, args.encoder, not args.no_pretrained, not args.no_fusion).to(dev)
cw = class_weights_from_fraction(report["class_pixel_fraction"])
crit = MultiTaskLoss(NUM_CLASSES, cw, args.w_rest, args.w_seg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
steps = args.epochs * len(train_dl)
warm = min(500, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
start_epoch, best, bad, history = 0, -1e9, 0, []
if args.resume:
    ck = torch.load(args.resume, map_location=dev, weights_only=False)  # own checkpoint
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]

print(f"device={dev} params={count_params(model) / 1e6:.2f}M train={len(train_ds)} val={len(val_ds)} "
      f"class_weights={[round(w, 2) for w in cw]}")


@torch.no_grad()
def validate():
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    psnrs, losses = [], []
    for b in val_dl:
        x, y, m = b["image"].to(dev), b["clean"].to(dev), b["mask"].to(dev)
        hc, hm = b["has_clean"].to(dev), b["has_mask"].to(dev)
        restored, logits = model(x)
        loss, _ = crit(restored, logits, y, m, hc, hm)
        losses.append(loss.item())
        if hc.any():
            mse = ((restored[hc] - y[hc]) ** 2).mean().clamp_min(1e-10)
            psnrs.append((10 * torch.log10(1 / mse)).item())
        if hm.any():
            p = logits.argmax(1)[hm].flatten().cpu().numpy()
            g = m[hm].flatten().cpu().numpy()
            v = g != 255
            cm += np.bincount(g[v] * NUM_CLASSES + p[v], minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    model.train()
    miou = seg_scores(cm)["mIoU"] if cm.sum() else float("nan")
    return float(np.mean(losses)), float(np.mean(psnrs)) if psnrs else float("nan"), miou


for epoch in range(start_epoch, args.epochs):
    t0, run = time.time(), []
    for b in train_dl:
        x, y, m = b["image"].to(dev, non_blocking=True), b["clean"].to(dev), b["mask"].to(dev)
        hc, hm = b["has_clean"].to(dev), b["has_mask"].to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            restored, logits = model(x)
        loss, logs = crit(restored.float(), logits.float(), y, m, hc, hm)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run.append(logs["total"])
    vloss, vpsnr, vmiou = validate()
    score = {"miou": vmiou, "psnr": vpsnr,
             "combined": np.nan_to_num(vmiou) + np.nan_to_num(vpsnr) / 40.0}[args.select]
    history.append({"epoch": epoch, "train_loss": float(np.mean(run)), "val_loss": vloss, "val_psnr": vpsnr,
                    "val_mIoU": vmiou, "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)})
    print(json.dumps(history[-1]))
    ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
          "best": float(max(best, score)), "history": history, "args": vars(args) | {"manifest": str(args.manifest),
                                                                          "out": str(args.out), "resume": None}}
    torch.save(ck, args.out / "last.pt")
    if score > best:
        best, bad = score, 0
        torch.save(ck, args.out / "best.pt")
    else:
        bad += 1
        if bad >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

(args.out / "train_log.json").write_text(json.dumps({
    "model": "eurs_net", "dataset": report.get("dataset"), "encoder": args.encoder, "fusion": not args.no_fusion,
    "params": count_params(model), "best_val_score": best, "select": args.select,
    "class_weights": cw, "history": history, "args": {k: str(v) for k, v in vars(args).items()}}, indent=2))
print(f"best val {args.select}={best:.4f} -> {args.out / 'best.pt'}")
