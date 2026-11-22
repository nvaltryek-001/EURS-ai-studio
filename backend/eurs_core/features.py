"""Hand-crafted per-pixel features used by the CPU-trainable EURS-Lite model.

All functions take float32 RGB in [0, 1] (HxWx3) and return HxWxF float32.
Inference only needs NumPy + OpenCV, so the same code runs in training and in the API.
"""
from __future__ import annotations

import cv2
import numpy as np


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def restoration_features(rgb: np.ndarray) -> np.ndarray:
    """Features describing local and global colour cast / contrast loss."""
    h, w, _ = rgb.shape
    feats = [rgb]
    for sigma in (6.0, 24.0):
        mu = _blur(rgb, sigma)
        var = np.maximum(_blur(rgb * rgb, sigma) - mu * mu, 0.0)
        feats += [mu, np.sqrt(var + 1e-6)]
    g_mean = rgb.reshape(-1, 3).mean(0)
    g_std = rgb.reshape(-1, 3).std(0)
    feats.append(np.broadcast_to(g_mean, (h, w, 3)))
    feats.append(np.broadcast_to(g_std, (h, w, 3)))
    # Dark channel prior (He et al.) as a haze / transmission cue.
    dark = cv2.erode(rgb.min(axis=2), np.ones((15, 15), np.uint8))[..., None]
    feats.append(dark)
    # Normalised vertical position: depth/backscatter often varies with image row.
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None] * np.ones((1, w, 1), np.float32)
    feats.append(yy)
    return np.concatenate(feats, axis=2).astype(np.float32)


def segmentation_features(rgb: np.ndarray) -> np.ndarray:
    """Colour, multi-scale context and texture features for pixel classification."""
    h, w, _ = rgb.shape
    u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0] * (np.pi / 90.0)
    hsv_f = np.stack([np.cos(hue), np.sin(hue), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0], axis=2)
    feats = [rgb, lab, hsv_f]
    lum = np.ascontiguousarray(lab[..., 0])  # HxW
    for sigma in (2.0, 6.0, 16.0):
        feats.append(_blur(lab, sigma))
        mu = _blur(lum, sigma)
        std = np.sqrt(np.maximum(_blur(lum * lum, sigma) - mu * mu, 0) + 1e-6)
        feats.append(std[..., None])
    for sigma in (1.0, 3.0):
        s = _blur(lum, sigma)
        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
        feats.append(np.sqrt(gx * gx + gy * gy)[..., None])
        feats.append(cv2.Laplacian(s, cv2.CV_32F, ksize=3)[..., None])
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None] * np.ones((1, w, 1), np.float32)
    feats.append(yy)
    return np.concatenate(feats, axis=2).astype(np.float32)
