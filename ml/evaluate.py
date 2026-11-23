#!/usr/bin/env python3
"""Evaluate a model on one split (default: test) and write metrics.json + per_image.csv.

  python ml/evaluate.py --manifest data/splits/eurs/manifest.csv --model classical
  python ml/evaluate.py --manifest ... --model runs/eurs_lite/eurs_lite.npz
  python ml/evaluate.py --manifest ... --model runs/eurs_net/eurs_net.onnx
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

import eurs_ml  # noqa: F401
from eurs_core import NUM_CLASSES, colorize, overlay
from eurs_ml.discover import manifest_root, read_manifest
from eurs_ml.io import MaskDecoder, read_rgb, write_rgb
from eurs_ml.models import load_model
from eurs_ml.metrics import bootstrap, confusion, psnr, seg_scores, ssim, uciqe


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, type=Path)
ap.add_argument("--model", required=True)
ap.add_argument("--split", default="test", choices=["train", "val", "test"])
ap.add_argument("--name", default=None)
ap.add_argument("--out", default=None, type=Path)
ap.add_argument("--save-preds", type=int, default=6, help="save N qualitative examples")
ap.add_argument("--bootstrap", type=int, default=1000)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

model = load_model(args.model)
meta = dict(getattr(model, "meta", {}) or {})
name = args.name or meta.get("name") or Path(args.model).stem
out = args.out or eurs_ml.REPO_ROOT / "results" / name
out.mkdir(parents=True, exist_ok=True)
root = manifest_root(args.manifest)
report = json.loads(args.manifest.with_name("split_report.json").read_text())
layout = report.get("layout") or {}
decode = MaskDecoder(layout.get("mask_encoding", "auto"), layout.get("mask_palette"))
rows = read_manifest(args.manifest, args.split)
if args.limit:
    rows = rows[: args.limit]

per_img, cms, times = [], [], []
vals = {"PSNR": [], "SSIM": [], "input_PSNR": [], "input_SSIM": [], "UCIQE": [], "input_UCIQE": []}
cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
for i, s in enumerate(rows):
    x = read_rgb(root / s["input"])
    t = time.perf_counter()
    restored, mask = model.predict(x)
    times.append((time.perf_counter() - t) * 1000)
    r = {"id": s["id"], "ms": round(times[-1], 2)}
    vals["UCIQE"].append(uciqe(restored))
    vals["input_UCIQE"].append(uciqe(x))
    if s["clean"]:
        y = read_rgb(root / s["clean"])
        r.update(PSNR=psnr(restored, y), SSIM=ssim(restored, y), input_PSNR=psnr(x, y), input_SSIM=ssim(x, y))
        for k in ("PSNR", "SSIM", "input_PSNR", "input_SSIM"):
            vals[k].append(r[k])
    if s["mask"]:
        gt = decode(root / s["mask"])
        cm = confusion(mask, gt)
        cms.append(cm)
        cm_total += cm
        r["mIoU"] = seg_scores(cm)["mIoU"]
    per_img.append(r)
    if i < args.save_preds:
        q = out / "qualitative"
        write_rgb(q / f"{s['id']}_input.png", x)
        write_rgb(q / f"{s['id']}_restored.png", restored)
        write_rgb(q / f"{s['id']}_mask.png", colorize(mask))
        write_rgb(q / f"{s['id']}_overlay.png", overlay(restored, mask))
        if s["mask"]:
            write_rgb(q / f"{s['id']}_gt.png", colorize(gt))
        if s["clean"]:
            write_rgb(q / f"{s['id']}_clean.png", y)

mean = {k: (float(np.mean(v)) if v else None) for k, v in vals.items()}
result = {
    "name": name,
    "display_name": meta.get("display_name", name),
    "model_spec": args.model,
    "dataset": report.get("dataset", "unknown"),
    "split": args.split,
    "n_images": len(rows),
    "restoration": mean,
    "segmentation": dict(seg_scores(cm_total), confusion_matrix=cm_total.tolist()) if cms else None,
    "ci95": bootstrap(cms, {k: vals[k] for k in ("PSNR", "SSIM")}, n_boot=args.bootstrap),
    "latency_ms": {"mean": float(np.mean(times)), "median": float(np.median(times)),
                   "hardware": f"{platform.processor() or platform.machine()} (CPU)"},
    "params": meta.get("params"),
}
(out / "metrics.json").write_text(json.dumps(result, indent=2))
with open(out / "per_image.csv", "w", newline="") as f:
    keys = sorted({k for r in per_img for k in r}, key=lambda k: (k != "id", k))
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(per_img)

seg = result["segmentation"] or {}
print(f"[{name}] {args.split} n={len(rows)}  PSNR={mean['PSNR']}  SSIM={mean['SSIM']}  "
      f"mIoU={seg.get('mIoU')}  latency={result['latency_ms']['median']:.1f}ms  -> {out}")
if result["dataset"] == "synthetic":
    print("NOTE: synthetic smoke-test data -- these are NOT benchmark results.")
