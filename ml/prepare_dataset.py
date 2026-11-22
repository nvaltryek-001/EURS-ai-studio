#!/usr/bin/env python3
"""Index a dataset folder and write a reproducible train/val/test split.

  # Official EURS (after extracting eurs_train.zip):
  python ml/prepare_dataset.py --root data/eurs_train --out data/splits/eurs
  # With an explicit layout if auto-detection is not right:
  python ml/prepare_dataset.py --root data/eurs_train --layout ml/configs/layout.example.json ...
  # Synthetic smoke-test data:
  python ml/prepare_dataset.py --synthetic 300 --root data/synthetic --out data/splits/synthetic

The dev archive (eurs_dev_degraded.zip) has no ground truth: never put it in a split.
Use it only with ml/predict.py to produce challenge submissions.
"""
import argparse
import json
from pathlib import Path

import eurs_ml  # noqa: F401  (sets up imports)
from eurs_ml.discover import discover, split_samples, write_manifest
from eurs_ml.io import MaskDecoder, load_layout

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--layout", default=None)
ap.add_argument("--seed", type=int, default=2027)
ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic samples into --root first")
args = ap.parse_args()

if args.synthetic:
    from eurs_ml.synthetic import generate
    generate(args.root, n=args.synthetic)
    print(f"Generated {args.synthetic} SYNTHETIC samples in {args.root}")

layout = load_layout(args.layout)
decoder = MaskDecoder(layout.get("mask_encoding", "auto"), layout.get("mask_palette"))
samples = discover(args.root, layout)
samples, report = split_samples(samples, args.root, tuple(args.ratios), args.seed, decoder)
report["dataset_root"] = str(args.root.resolve())
report["dataset"] = "synthetic" if (args.root / "SYNTHETIC_DATA_README.txt").exists() else "eurs"
report["layout"] = layout

write_manifest(samples, args.out / "manifest.csv")
(args.out / "split_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("dataset", "n_total", "counts", "with_clean", "with_mask",
                                         "group_aware", "class_pixel_fraction")}, indent=2))
if report["with_mask"] == 0:
    print("WARNING: no masks detected -- check --layout (mask_dir / mask_encoding).")
if report["unknown_mask_pixels_subsampled"]:
    print(f"WARNING: {report['unknown_mask_pixels_subsampled']} mask pixels did not match the "
          "class encoding and were set to ignore (255). Check mask_encoding / mask_palette.")
