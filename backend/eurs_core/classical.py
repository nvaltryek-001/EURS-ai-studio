"""Training-free baselines. Used as the reference point in every experiment table."""
from __future__ import annotations

import cv2
import numpy as np


def restore_classical(rgb: np.ndarray) -> np.ndarray:
    """Gray-world white balance + CLAHE on L (Lab) + light bilateral denoise.

    Input/Output: HxWx3 uint8 RGB.
    """
    img = rgb.astype(np.float32)
    means = img.reshape(-1, 3).mean(0) + 1e-6
    img = np.clip(img * (means.mean() / means), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return cv2.bilateralFilter(out, 5, 30, 30)


def segment_heuristic(rgb: np.ndarray) -> np.ndarray:
    """Colour-threshold heuristic. NOT a semantic model; kept only as a lower bound."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = np.zeros(h.shape, dtype=np.uint8)
    coral = ((h < 20) | (h > 170)) & (s > 65) & (v > 45)
    cyan = (h >= 75) & (h <= 115) & (s > 70) & (v > 60)
    neutral = (s < 45) & (v > 100)
    mask[neutral] = 4
    mask[cyan] = 2
    mask[coral] = 1
    return mask
