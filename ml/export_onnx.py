#!/usr/bin/env python3
"""Export a trained EURS-Net checkpoint to ONNX (+ metadata JSON read by the API).

Matches the EURS Track 2 contract: input "image" (1,3,H,W) RGB float32 in [0,1], static shape;
outputs "restored_image" (1,3,H,W) in [0,1] and "seg_logits" (1,5,H,W) raw logits; weights embedded.

  python ml/export_onnx.py --ckpt runs/eurs_net/best.pt --out runs/eurs_net/eurs_net.onnx [--install]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import eurs_ml  # noqa: F401
from eurs_core import NUM_CLASSES
from eurs_torch.model import EURSNet, count_params

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--opset", type=int, default=17)
ap.add_argument("--name", default="eurs_net")
ap.add_argument("--install", action="store_true", help="copy .onnx + .json to backend/models/")
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # own checkpoint
a = ck["args"]
model = EURSNet(NUM_CLASSES, a["encoder"], pretrained=False, fusion=not a["no_fusion"])
model.load_state_dict(ck["model"])
model.eval()
x = torch.rand(1, 3, args.size, args.size)
args.out.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(model, x, str(args.out), input_names=["image"], output_names=["restored_image", "seg_logits"],
                  opset_version=args.opset, do_constant_folding=True, dynamo=False)

# Numerical check against PyTorch.
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    r_onnx, l_onnx = sess.run(None, {"image": x.numpy()})
    with torch.no_grad():
        r_pt, l_pt = model(x)
    err = max(np.abs(r_onnx - r_pt.numpy()).max(), np.abs(l_onnx - l_pt.numpy()).max())
    print(f"max |onnx - torch| = {err:.2e}")
except ImportError:
    print("onnxruntime not installed; skipped numerical check")

report = json.loads(Path(a["manifest"]).with_name("split_report.json").read_text())
meta = {"name": args.name, "display_name": f"EURS-Net ({a['encoder']}{'' if not a['no_fusion'] else ', no fusion'})",
        "input_size": args.size, "encoder": a["encoder"], "fusion": not a["no_fusion"],
        "params": count_params(model), "dataset": report.get("dataset"), "epoch": ck["epoch"],
        "restoration": "U-Net decoder, residual output", "segmentation": "U-Net decoder with R2S fusion"}
args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(f"exported {args.out} ({meta['params'] / 1e6:.2f}M params)")
if args.install:
    dst = eurs_ml.REPO_ROOT / "backend" / "models"
    shutil.copy(args.out, dst / f"{args.name}.onnx")
    shutil.copy(args.out.with_suffix(".json"), dst / f"{args.name}.json")
    print(f"installed -> {dst}")
