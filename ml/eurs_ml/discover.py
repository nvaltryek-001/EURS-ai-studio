"""Dataset discovery and leakage-aware train / val / test splitting.

The official EURS archive layout is not hard-coded. Files are grouped into samples by
filename stem, and each file's *role* (degraded input / clean reference / mask) is
inferred from directory and filename keywords -- or set explicitly in a layout JSON:

    {"input_dir": "train/degraded", "clean_dir": "train/clean", "mask_dir": "train/masks",
     "mask_encoding": "index", "mask_palette": [[0,0,0], ...]}
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .io import IMAGE_EXTS, MaskDecoder

ROLE_KEYWORDS = {
    "mask": ["mask", "label", "seg", "annot", "semantic", "class"],
    "clean": ["clean", "reference", "target", "enhanced", "clear", "hq", "gt_rgb", "restored"],
    "input": ["degraded", "input", "raw", "distort", "hazy", "lq", "low", "noisy"],
}
STEM_SUFFIXES = re.compile(
    r"([_\-.](mask|masks|label|labels|seg|gt|clean|ref|reference|target|degraded|input|raw|"
    r"lq|hq|enhanced|annotation|semantic))+$", re.I)


def norm_stem(p: Path) -> str:
    return STEM_SUFFIXES.sub("", p.stem).lower()


def infer_role(rel: Path) -> str | None:
    parts = [s.lower() for s in rel.parts]
    # Most specific first: the file name, then its directory chain (deepest first).
    for token in [parts[-1]] + parts[-2::-1]:
        for role, kws in ROLE_KEYWORDS.items():
            if any(k in token for k in kws):
                return role
    if any(p == "gt" for p in parts):
        return "gt?"  # ambiguous; resolved by channel count later
    return None


def discover(root: str | Path, layout: dict | None = None) -> list[dict]:
    root = Path(root)
    layout = layout or {}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise SystemExit(f"No images found under {root}")

    explicit = {r: layout.get(f"{r}_dir") for r in ("input", "clean", "mask")}
    samples: dict[str, dict] = defaultdict(dict)
    for p in files:
        rel = p.relative_to(root)
        role = None
        for r, d in explicit.items():
            if d and str(rel).replace("\\", "/").startswith(d.rstrip("/") + "/"):
                role = r
        if role is None and not any(explicit.values()):
            role = infer_role(rel)
        if role == "gt?":
            import cv2
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            role = "mask" if img is not None and img.ndim == 2 else "clean"
        if role is None:
            role = "input"  # images with no keyword are treated as inputs
        samples[norm_stem(p)].setdefault(role, str(rel))
    out = []
    for sid, roles in sorted(samples.items()):
        if "input" not in roles:
            continue
        out.append({"id": sid, "input": roles["input"], "clean": roles.get("clean", ""),
                    "mask": roles.get("mask", "")})
    return out


def group_key(sample_id: str) -> str:
    """Frames of one video/dive share a prefix; keep them in the same split."""
    return re.sub(r"[_\-]?\d+$", "", sample_id) or sample_id


def split_samples(samples: list[dict], root: Path, ratios=(0.8, 0.1, 0.1), seed: int = 2027,
                  decoder: MaskDecoder | None = None, min_groups: int = 20) -> tuple[list[dict], dict]:
    """Group-aware, rare-class-stratified split. Returns (samples with 'split', report)."""
    decoder = decoder or MaskDecoder()
    rng = random.Random(seed)
    # Per-sample class presence (subsampled read) for stratification and statistics.
    class_pixels = np.zeros(5, np.int64)
    for s in samples:
        s["classes"] = ""
        if s["mask"]:
            m = decoder(root / s["mask"])[::4, ::4]
            present = [c for c in range(5) if (m == c).any()]
            s["classes"] = "".join(map(str, present))
            class_pixels += np.bincount(m[m < 5].ravel(), minlength=5)[:5]

    groups = defaultdict(list)
    for s in samples:
        groups[group_key(s["id"])].append(s)
    grouped = len(groups) >= min_groups and len(groups) < len(samples)
    if not grouped:
        groups = {s["id"]: [s] for s in samples}

    # Stratify on the rarest class present in the group (elasmobranch is typically rarest).
    freq = class_pixels / max(1, class_pixels.sum())
    order = list(np.argsort(freq))

    def stratum(members):
        present = set("".join(m["classes"] for m in members))
        for c in order:
            if str(c) in present:
                return int(c)
        return -1

    strata = defaultdict(list)
    for g, members in groups.items():
        strata[stratum(members)].append(g)
    assign = {}
    for _, gs in sorted(strata.items()):
        rng.shuffle(gs)
        n = len(gs)
        n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        for i, g in enumerate(gs):
            assign[g] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    for g, members in groups.items():
        for s in members:
            s["split"] = assign[g]

    counts = Counter(s["split"] for s in samples)
    report = {
        "seed": seed, "ratios": list(ratios), "group_aware": grouped, "n_groups": len(groups),
        "counts": dict(counts), "n_total": len(samples),
        "with_clean": sum(bool(s["clean"]) for s in samples),
        "with_mask": sum(bool(s["mask"]) for s in samples),
        "class_pixel_fraction": {str(i): float(f) for i, f in enumerate(freq)},
        "unknown_mask_pixels_subsampled": decoder.unknown_pixels,
    }
    return samples, report


def write_manifest(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "input", "clean", "mask", "classes"])
        w.writeheader()
        for s in samples:
            w.writerow({k: s.get(k, "") for k in w.fieldnames})


def read_manifest(path: str | Path, split: str | None = None) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def manifest_root(manifest: str | Path) -> Path:
    meta = Path(manifest).with_name("split_report.json")
    return Path(json.loads(meta.read_text())["dataset_root"])
