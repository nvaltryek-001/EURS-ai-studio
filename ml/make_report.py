#!/usr/bin/env python3
"""Turn results/*/metrics.json into paper tables, paper figures and the website's results.json.

  # Real EURS results -> main paper tables (tag "main")
  python ml/make_report.py --tag main --results results/eurs --split-report data/splits/eurs/split_report.json \
      --train-log runs/eurs_lite/train_log.json --train-log runs/eurs_net/train_log.json
  # Synthetic smoke test -> appendix tables (tag "synthetic")
  python ml/make_report.py --tag synthetic --results results/synthetic ...

With --tag main and no results yet, placeholder tables ("--") are written so the paper compiles.
Synthetic results are refused for --tag main.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import eurs_ml  # noqa: E402,F401
from eurs_core import CLASS_NAMES  # noqa: E402

ROOT = eurs_ml.REPO_ROOT
ORDER = ["input", "classical", "eurs_lite", "eurs_net_tiny", "eurs_net"]
# Categorical slots 1-3 of the validated reference palette (fixed order, follows the model).
MODEL_COLORS = {"classical": "#2a78d6", "eurs_lite": "#eb6834", "eurs_net": "#1baf7a", "eurs_net_tiny": "#eda100"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, choices=["main", "synthetic"])
ap.add_argument("--results", type=Path, default=None)
ap.add_argument("--split-report", type=Path, default=None)
ap.add_argument("--train-log", type=Path, action="append", default=[])
ap.add_argument("--ablation", type=Path, default=None, help="folder of */metrics.json ablation runs")
ap.add_argument("--paper", type=Path, default=ROOT / "paper")
ap.add_argument("--web", type=Path, default=ROOT / "frontend" / "public" / "results.json")
args = ap.parse_args()

gen = args.paper / "generated"
figs = args.paper / "figures"
gen.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
T = args.tag

runs = []
if args.results and args.results.exists():
    for p in sorted(args.results.glob("*/metrics.json")):
        runs.append(json.loads(p.read_text()) | {"_dir": str(p.parent)})
if T == "main" and any(r["dataset"] == "synthetic" for r in runs):
    raise SystemExit("Refusing to put synthetic results into the main paper tables. Use --tag synthetic.")
runs.sort(key=lambda r: ORDER.index(r["name"]) if r["name"] in ORDER else 99)
split = json.loads(args.split_report.read_text()) if args.split_report and args.split_report.exists() else None


def f(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def ci(r, key, pct=False):
    c = (r.get("ci95") or {}).get(key)
    if not c:
        return ""
    lo, hi = (100 * c[0], 100 * c[1]) if pct else c
    return f"\\,{{\\scriptsize[{lo:.1f}, {hi:.1f}]}}"


def label(r):
    return {"classical": "Classical (GW+CLAHE / heuristic)", "eurs_lite": "EURS-Lite (ours, CPU)",
            "eurs_net": "EURS-Net (ours, ResNet-18)", "eurs_net_tiny": "EURS-Net-Tiny (ours, embedded)"}.get(r["name"], r.get("display_name", r["name"]).replace("_", r"\_"))


# ------------------------------------------------------------------ tables
def table_main():
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Method & PSNR$\uparrow$ & SSIM$\uparrow$ & UCIQE$\uparrow$ & mIoU (\%)$\uparrow$ & mF1 (\%)$\uparrow$ & EURS$^{\approx}\uparrow$ & ms/img$\downarrow$ \\",
             r"\midrule"]
    if runs:
        r0 = runs[0]["restoration"]
        lines.append(f"Degraded input (identity) & {f(r0.get('input_PSNR'))} & {f(r0.get('input_SSIM'), 3)} & "
                     f"{f(r0.get('input_UCIQE'), 3)} & -- & -- & -- & -- \\\\")
        for r in runs:
            rs, sg = r["restoration"], r.get("segmentation") or {}
            lines.append(f"{label(r)} & {f(rs.get('PSNR'))}{ci(r, 'PSNR')} & {f(rs.get('SSIM'), 3)} & "
                         f"{f(rs.get('UCIQE'), 3)} & {f(sg.get('mIoU'), 1, True)}{ci(r, 'mIoU', True)} & "
                         f"{f(sg.get('mean_f1'), 1, True)} & {f((r.get('eurs_score_approx') or {}).get('score'), 3)} & "
                         f"{f(r['latency_ms']['median'], 1)} \\\\")
    else:
        for name in ["Degraded input (identity)", "Classical (GW+CLAHE / heuristic)", "EURS-Lite (ours, CPU)",
                     "EURS-Net-Tiny (ours, embedded)", "EURS-Net (ours, ResNet-18)"]:
            lines.append(f"{name} & -- & -- & -- & -- & -- & -- & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_per_class():
    head = " & ".join(CLASS_NAMES)
    lines = [r"\begin{tabular}{l" + "c" * len(CLASS_NAMES) + "c}", r"\toprule",
             f"Method & {head} & mIoU \\\\", r"\midrule"]
    rows = runs if runs else [{"name": n} for n in ORDER[1:]]
    for r in rows:
        sg = r.get("segmentation") or {}
        pc = sg.get("per_class", {})
        cells = [f((pc.get(c) or {}).get("IoU"), 1, True) for c in CLASS_NAMES]
        lines.append(f"{label(r)} & " + " & ".join(cells) + f" & {f(sg.get('mIoU'), 1, True)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_split():
    if not split:
        return r"\begin{tabular}{lcccc}\toprule Split & Train & Val & Test & Total \\ \midrule Images & -- & -- & -- & -- \\ \bottomrule\end{tabular}"
    c = split["counts"]
    frac = split["class_pixel_fraction"]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Split & Train & Val & Test & Total \\", r"\midrule",
             f"Images & {c.get('train', 0)} & {c.get('val', 0)} & {c.get('test', 0)} & {split['n_total']} \\\\",
             r"\midrule",
             r"\multicolumn{5}{l}{Pixel share per class (\%): " +
             ", ".join(f"{CLASS_NAMES[int(k)]} {100 * v:.1f}" for k, v in frac.items()) + r"} \\",
             r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


(gen / f"{T}_table_main.tex").write_text(table_main() + "\n")
(gen / f"{T}_table_per_class.tex").write_text(table_per_class() + "\n")
(gen / f"{T}_table_split.tex").write_text(table_split() + "\n")


def table_ablation():
    abl = []
    if args.ablation and args.ablation.exists():
        abl = [json.loads(p.read_text()) for p in sorted(args.ablation.glob("*/metrics.json"))]
    full = next((r for r in runs if r["name"] == "eurs_net_tiny"), None)
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Variant & PSNR$\uparrow$ & SSIM$\uparrow$ & mIoU (\%)$\uparrow$ \\", r"\midrule"]
    rows = [("EURS-Net-Tiny, full (R2S fusion, joint loss)", full)]
    rows += [(r.get("display_name", r["name"]).replace("_", r"\_"), r) for r in abl] or [
        ("w/o R2S fusion", None), ("Segmentation only ($\\lambda_r{=}0$)", None),
        ("Restoration only ($\\lambda_s{=}0$)", None)]
    for name, r in rows:
        rs, sg = (r or {}).get("restoration") or {}, (r or {}).get("segmentation") or {}
        lines.append(f"{name} & {f(rs.get('PSNR'))} & {f(rs.get('SSIM'), 3)} & {f(sg.get('mIoU'), 1, True)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


(gen / f"{T}_table_ablation.tex").write_text(table_ablation() + "\n")


# ------------------------------------------------------------------ figures
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelcolor": TEXT, "pdf.fonttype": 42})
made = []
seg_runs = [r for r in runs if r.get("segmentation")]
if seg_runs:
    # Per-class IoU grouped bars (one colour per model, legend + axis labels carry identity).
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    k = len(seg_runs)
    w = 0.8 / k
    x = np.arange(len(CLASS_NAMES))
    for j, r in enumerate(seg_runs):
        v = [100 * ((r["segmentation"]["per_class"][c] or {}).get("IoU") or 0) for c in CLASS_NAMES]
        ax.bar(x + (j - (k - 1) / 2) * w, v, w * 0.92, color=MODEL_COLORS.get(r["name"], "#6250d6"),
               label=label(r).split(" (")[0], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, ["Bg.", "Coral", "Fish", "Elasmo.", "Other"])
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(0, 100)
    style(ax)
    ax.legend(frameon=False, fontsize=7, ncol=k, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(figs / f"{T}_per_class_iou.pdf")
    fig.savefig(figs / f"{T}_per_class_iou.png", dpi=200)
    plt.close(fig)
    made.append("per_class_iou")

    # Confusion matrix of the best model (row-normalised, single-hue sequential ramp).
    best = max(seg_runs, key=lambda r: r["segmentation"]["mIoU"])
    cm = np.array(best["segmentation"]["confusion_matrix"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{100 * cmn[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.55 else TEXT)
    short = ["Bg", "Coral", "Fish", "Elasmo.", "Other"]
    ax.set_xticks(range(5), short, fontsize=7)
    ax.set_yticks(range(5), short, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{label(best).split(' (')[0]}: row-normalised (%)", fontsize=8, color=TEXT)
    fig.tight_layout()
    fig.savefig(figs / f"{T}_confusion.pdf")
    fig.savefig(figs / f"{T}_confusion.png", dpi=200)
    plt.close(fig)
    made.append("confusion")

    # Qualitative grid from the best model's saved examples.
    qdir = Path(best["_dir"]) / "qualitative"
    ids = sorted({p.name.rsplit("_", 1)[0] for p in qdir.glob("*_input.png")})[:3]
    if ids:
        cols = [c for c in ["input", "restored", "clean", "overlay", "gt"] if (qdir / f"{ids[0]}_{c}.png").exists()]
        titles = {"input": "Degraded input", "restored": "Restored", "clean": "Reference",
                  "overlay": "Prediction", "gt": "Ground truth"}
        fig, axes = plt.subplots(len(ids), len(cols), figsize=(1.4 * len(cols), 1.05 * len(ids)))
        axes = np.atleast_2d(axes)
        for i, sid in enumerate(ids):
            for j, c in enumerate(cols):
                axes[i, j].imshow(plt.imread(qdir / f"{sid}_{c}.png"))
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(titles[c], fontsize=7)
        fig.subplots_adjust(wspace=0.03, hspace=0.05, left=0, right=1, top=0.92, bottom=0)
        fig.savefig(figs / f"{T}_qualitative.pdf", dpi=220)
        fig.savefig(figs / f"{T}_qualitative.png", dpi=160)
        plt.close(fig)
        made.append("qualitative")

# Training curves (sklearn: loss per epoch; torch: train/val curves) -- one panel per log.
logs = [json.loads(p.read_text()) | {"_path": str(p)} for p in args.train_log if p.exists()]
if logs:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1))
    for lg in logs:
        if "segmentation_candidates" in lg:  # EURS-Lite
            sel = lg["segmentation_selected"]
            c = next(c for c in lg["segmentation_candidates"] if list(c["hidden"]) == list(sel["hidden"]))
            axes[0].plot(c["loss_curve"], color=MODEL_COLORS["eurs_lite"], lw=2, label="EURS-Lite seg. (train loss)")
            if c.get("val_acc_curve"):
                axes[1].plot(c["val_acc_curve"], color=MODEL_COLORS["eurs_lite"], lw=2, label="EURS-Lite pixel acc. (held-out px)")
        if "history" in lg:  # EURS-Net
            h = lg["history"]
            axes[0].plot([e["train_loss"] for e in h], color=MODEL_COLORS["eurs_net"], lw=2, label="EURS-Net train loss")
            axes[0].plot([e["val_loss"] for e in h], color=MODEL_COLORS["eurs_net"], lw=2, ls="--", label="EURS-Net val loss")
            axes[1].plot([e["val_mIoU"] for e in h], color=MODEL_COLORS["eurs_net"], lw=2, label="EURS-Net val mIoU")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation score")
    for ax in axes:
        style(ax)
        ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(figs / f"{T}_training_curves.pdf")
    fig.savefig(figs / f"{T}_training_curves.png", dpi=200)
    plt.close(fig)
    made.append("training_curves")

# Placeholder figures so the main paper compiles before real results exist.
for name in ["per_class_iou", "confusion", "qualitative", "training_curves"]:
    if name not in made and not (figs / f"{T}_{name}.pdf").exists():
        fig, ax = plt.subplots(figsize=(3.2, 1.4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"[{name.replace('_', ' ')}]\nrun ml/run_pipeline.sh on EURS to generate",
                ha="center", va="center", fontsize=8, color=MUTED)
        fig.savefig(figs / f"{T}_{name}.pdf")
        plt.close(fig)

# ------------------------------------------------------------------ website JSON
web = json.loads(args.web.read_text()) if args.web.exists() else {}
web["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
web[T] = {
    "dataset": runs[0]["dataset"] if runs else None,
    "split": split and {"counts": split["counts"], "n_total": split["n_total"], "seed": split["seed"],
                        "group_aware": split["group_aware"], "class_pixel_fraction": split["class_pixel_fraction"]},
    "models": [{
        "name": r["name"], "display_name": label(r), "split": r["split"], "n_images": r["n_images"],
        "psnr": r["restoration"].get("PSNR"), "ssim": r["restoration"].get("SSIM"),
        "uciqe": r["restoration"].get("UCIQE"),
        "input_psnr": r["restoration"].get("input_PSNR"), "input_ssim": r["restoration"].get("input_SSIM"),
        "miou": (r.get("segmentation") or {}).get("mIoU"),
        "mf1": (r.get("segmentation") or {}).get("mean_f1"),
        "pixel_acc": (r.get("segmentation") or {}).get("pixel_acc"),
        "per_class_iou": {c: ((r.get("segmentation") or {}).get("per_class", {}).get(c) or {}).get("IoU")
                          for c in CLASS_NAMES},
        "eurs_score": (r.get("eurs_score_approx") or {}).get("score"),
        "ci95": r.get("ci95"), "latency_ms": r["latency_ms"]["median"], "params": r.get("params"),
    } for r in runs],
}
args.web.parent.mkdir(parents=True, exist_ok=True)
args.web.write_text(json.dumps(web, indent=2))
if "qualitative" in made:
    shutil.copy(figs / f"{T}_qualitative.png", args.web.parent / f"{T}_qualitative.png")
print(f"[{T}] tables -> {gen}, figures {made} -> {figs}, web -> {args.web}")
