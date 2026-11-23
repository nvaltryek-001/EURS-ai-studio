#!/usr/bin/env python3
"""Run a model over a folder (e.g. the extracted eurs_dev_degraded archive).

Writes <out>/restored/<stem>.png (RGB) and <out>/masks/<stem>.png (single-channel class ids).
Check the official challenge page for the exact submission naming/format before uploading.

  python ml/predict.py --model runs/eurs_net/eurs_net.onnx --input data/eurs_dev_degraded --out submissions/dev
"""
import argparse
import time
from pathlib import Path

import eurs_ml  # noqa: F401
from eurs_ml.io import IMAGE_EXTS, read_rgb, write_mask, write_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--input", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
args = ap.parse_args()

from eurs_ml.models import load_model  # noqa: E402

model = load_model(args.model)
files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
if not files:
    raise SystemExit(f"No images under {args.input}")
t0 = time.time()
for i, p in enumerate(files, 1):
    restored, mask = model.predict(read_rgb(p))
    rel = p.relative_to(args.input).with_suffix(".png")
    write_rgb(args.out / "restored" / rel, restored)
    write_mask(args.out / "masks" / rel, mask)
    if i % 50 == 0 or i == len(files):
        print(f"{i}/{len(files)}  ({(time.time() - t0) / i * 1000:.0f} ms/img)")
print(f"done -> {args.out}")
