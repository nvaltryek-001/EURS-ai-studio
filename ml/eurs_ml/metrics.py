"""Restoration and segmentation metrics, with image-level bootstrap confidence intervals."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from eurs_core import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


# ---------------------------------------------------------------- restoration
def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))


def uciqe(rgb: np.ndarray) -> float:
    """No-reference UCIQE (Yang & Sowmya, 2015) for images without a clean reference."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    l, a, b = lab[..., 0] / 255.0, (lab[..., 1] - 128) / 255.0, (lab[..., 2] - 128) / 255.0
    chroma = np.sqrt(a * a + b * b)
    sigma_c = chroma.std()
    lo, hi = np.percentile(l, [1, 99])
    con_l = hi - lo
    sat = chroma / np.maximum(l, 1e-6)
    mu_s = np.clip(sat, 0, 1).mean()
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


# ---------------------------------------------------------------- segmentation
def confusion(pred: np.ndarray, gt: np.ndarray, n: int = NUM_CLASSES) -> np.ndarray:
    valid = gt != IGNORE_INDEX
    g = gt[valid].astype(np.int64)
    p = np.clip(pred[valid].astype(np.int64), 0, n - 1)
    return np.bincount(g * n + p, minlength=n * n).reshape(n, n)


def seg_scores(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
    present = support > 0
    iou_p = np.where(present, iou, np.nan)
    freq = support / max(1.0, support.sum())

    def nm(x):
        x = x[present]
        x = x[~np.isnan(x)]
        return float(x.mean()) if len(x) else float("nan")

    return {
        "pixel_acc": float(tp.sum() / max(1.0, cm.sum())),
        "mIoU": nm(iou_p),
        "fwIoU": float(np.nansum(freq * np.nan_to_num(iou))),
        "mean_precision": nm(prec), "mean_recall": nm(rec), "mean_f1": nm(f1),
        "per_class": {
            CLASS_NAMES[i]: {
                "IoU": None if not present[i] or np.isnan(iou[i]) else float(iou[i]),
                "precision": None if np.isnan(prec[i]) else float(prec[i]),
                "recall": None if np.isnan(rec[i]) else float(rec[i]),
                "F1": None if np.isnan(f1[i]) else float(f1[i]),
                "support_px": int(support[i]),
            } for i in range(len(tp))
        },
    }


def bootstrap(per_image_cms: list[np.ndarray], per_image_vals: dict[str, list[float]],
              n_boot: int = 1000, seed: int = 0) -> dict:
    """95% CIs by resampling *images* (dataset-level mIoU recomputed per resample)."""
    rng = np.random.default_rng(seed)
    n = len(per_image_cms) or len(next(iter(per_image_vals.values()), []))
    if n == 0:
        return {}
    cms = np.stack(per_image_cms) if per_image_cms else None
    out: dict[str, list[float]] = {}
    samples = {k: [] for k in per_image_vals}
    miou = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if cms is not None:
            miou.append(seg_scores(cms[idx].sum(0))["mIoU"])
        for k, v in per_image_vals.items():
            if v:
                samples[k].append(float(np.mean(np.asarray(v)[idx])))
    if miou:
        out["mIoU"] = [float(np.nanpercentile(miou, 2.5)), float(np.nanpercentile(miou, 97.5))]
    for k, v in samples.items():
        if v:
            out[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


# ---------------------------------------------------------------- EURS composite (approximation)
def sharpness_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    """Laplacian-variance sharpness of the restoration relative to the reference, capped at 1."""
    def lap_var(x):
        return float(cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    return float(min(1.0, lap_var(pred) / max(lap_var(gt), 1e-6)))


def eurs_score_approx(seg: dict | None, psnr_mean: float | None, ssim_mean: float | None,
                      sharp_mean: float | None, uciqe_mean: float | None) -> dict:
    """Approximates the EURS composite published on the challenge page:

        F = 0.6 * Qseg + 0.3 * (0.45 * PSNRnorm + 0.45 * SSIM + 0.10 * sharpness_ratio) + 0.1 * NRquality

    Approximations (the organisers' evaluator is authoritative):
      * Qseg = mean F1 x mIoU x coverage, with coverage = 0 if any ground-truth class has recall < 1 %, else 1.
      * PSNRnorm = clip((PSNR - 15) / 20, 0, 1)  (the published 15-35 dB window).
      * sharpness_ratio = Laplacian-variance ratio restored / reference, capped at 1.
      * NRquality uses UCIQE / 0.7 (capped at 1) as a stand-in for mean(normalised UIQM, UCIQE, CCF).
    """
    parts = {}
    if seg:
        recalls = [v["recall"] for v in seg["per_class"].values() if v["support_px"] > 0]
        coverage = 0.0 if any(r is None or r < 0.01 for r in recalls) else 1.0
        parts["Qseg"] = float(np.nan_to_num(seg["mean_f1"]) * np.nan_to_num(seg["mIoU"]) * coverage)
    if psnr_mean is not None:
        parts["PSNRnorm"] = float(np.clip((psnr_mean - 15.0) / 20.0, 0, 1))
    if ssim_mean is not None:
        parts["SSIM"] = float(ssim_mean)
    if sharp_mean is not None:
        parts["sharpness_ratio"] = float(sharp_mean)
    if uciqe_mean is not None:
        parts["NRquality_proxy"] = float(min(1.0, uciqe_mean / 0.7))
    if all(k in parts for k in ("Qseg", "PSNRnorm", "SSIM", "sharpness_ratio", "NRquality_proxy")):
        rest = 0.45 * parts["PSNRnorm"] + 0.45 * parts["SSIM"] + 0.10 * parts["sharpness_ratio"]
        parts["score"] = 0.6 * parts["Qseg"] + 0.3 * rest + 0.1 * parts["NRquality_proxy"]
    return parts
