#!/usr/bin/env python3
"""Count parameters and FLOPs of EURS-Net variants against the EURS Track 2 budget.

  python ml/check_budget.py --encoder tiny --size 512        # exits 1 if over budget
  python ml/check_budget.py --encoder resnet18 --size 512 --no-fail

FLOPs follow the common convention FLOPs = 2 x MACs for Conv2d/Linear layers (the official
check_submission_flops.py may count differently; run it before submitting).
"""
import argparse
import json

import torch
import torch.nn as nn

import eurs_ml  # noqa: F401
from eurs_torch.model import EURSNet

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="tiny")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--no-fusion", action="store_true")
ap.add_argument("--max-params", type=float, default=1.0e6)
ap.add_argument("--max-gflops", type=float, default=12.0)
ap.add_argument("--no-fail", action="store_true")
args = ap.parse_args()

model = EURSNet(5, args.encoder, pretrained=False, fusion=not args.no_fusion).eval()
params = sum(p.numel() for p in model.parameters())
macs = 0


def hook(m, inp, out):
    global macs
    if isinstance(m, nn.Conv2d):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs += out.numel() * k
    elif isinstance(m, nn.Linear):
        macs += m.in_features * m.out_features


for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.register_forward_hook(hook)
with torch.no_grad():
    model(torch.rand(1, 3, args.size, args.size))
gflops = 2 * macs / 1e9
ok = params <= args.max_params and gflops <= args.max_gflops
print(json.dumps({"encoder": args.encoder, "fusion": not args.no_fusion, "input": [1, 3, args.size, args.size],
                  "params": params, "params_M": round(params / 1e6, 3), "GMACs": round(macs / 1e9, 3),
                  "GFLOPs": round(gflops, 3), "within_track2_budget": ok}))
if not ok and not args.no_fail:
    raise SystemExit("over the Track 2 budget (<= 1.0M params, <= 12 GFLOPs)")
