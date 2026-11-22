"""EURS-Lite: a CPU-only, two-stage learned model.

Stage 1 (restoration): an MLP regresses the per-pixel *residual* (clean - degraded)
from local/global colour statistics, predicted at a working resolution and
up-sampled (the residual is low-frequency, so detail is preserved from the input).

Stage 2 (segmentation): an MLP pixel classifier on colour/context/texture features
of the restored image, followed by spatial smoothing of class probabilities.

Weights are stored as a plain .npz (no pickle), and inference needs only NumPy+OpenCV.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .features import restoration_features, segmentation_features
from .palette import NUM_CLASSES


@dataclass
class NumpyMLP:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    mean: np.ndarray
    scale: np.ndarray
    output: str  # "identity" or "softmax"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.mean) / self.scale
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                np.maximum(h, 0, out=h)
        if self.output == "softmax":
            h = h - h.max(axis=1, keepdims=True)
            np.exp(h, out=h)
            h /= h.sum(axis=1, keepdims=True)
        return h

    @classmethod
    def from_sklearn(cls, mlp, scaler, output: str) -> "NumpyMLP":
        return cls(
            weights=[w.astype(np.float32) for w in mlp.coefs_],
            biases=[b.astype(np.float32) for b in mlp.intercepts_],
            mean=scaler.mean_.astype(np.float32),
            scale=scaler.scale_.astype(np.float32),
            output=output,
        )

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        d = {f"{prefix}_mean": self.mean, f"{prefix}_scale": self.scale,
             f"{prefix}_n": np.array(len(self.weights))}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            d[f"{prefix}_w{i}"] = w
            d[f"{prefix}_b{i}"] = b
        return d

    @classmethod
    def from_arrays(cls, z, prefix: str, output: str) -> "NumpyMLP":
        n = int(z[f"{prefix}_n"])
        return cls([z[f"{prefix}_w{i}"] for i in range(n)], [z[f"{prefix}_b{i}"] for i in range(n)],
                   z[f"{prefix}_mean"], z[f"{prefix}_scale"], output)


def _resize_max(img: np.ndarray, max_side: int, interp=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


class LiteModel:
    name = "eurs_lite"

    def __init__(self, restorer: NumpyMLP | None, segmenter: NumpyMLP | None,
                 work_side: int = 384, smooth_sigma: float = 2.0, meta: dict | None = None):
        self.restorer = restorer
        self.segmenter = segmenter
        self.work_side = work_side
        self.smooth_sigma = smooth_sigma
        self.meta = meta or {}

    # ---------------- persistence ----------------
    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        if self.restorer:
            arrays.update(self.restorer.to_arrays("res"))
        if self.segmenter:
            arrays.update(self.segmenter.to_arrays("seg"))
        meta = dict(self.meta, work_side=self.work_side, smooth_sigma=self.smooth_sigma,
                    has_restorer=self.restorer is not None, has_segmenter=self.segmenter is not None)
        arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "LiteModel":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(z["meta_json"]).decode())
        res = NumpyMLP.from_arrays(z, "res", "identity") if meta.get("has_restorer") else None
        seg = NumpyMLP.from_arrays(z, "seg", "softmax") if meta.get("has_segmenter") else None
        return cls(res, seg, meta.get("work_side", 384), meta.get("smooth_sigma", 2.0), meta)

    # ---------------- inference ----------------
    def restore(self, rgb_u8: np.ndarray) -> np.ndarray:
        if self.restorer is None:
            return rgb_u8
        full = rgb_u8.astype(np.float32) / 255.0
        small = _resize_max(full, self.work_side)
        f = restoration_features(small)
        resid = self.restorer(f.reshape(-1, f.shape[2])).reshape(small.shape).astype(np.float32)
        resid = cv2.GaussianBlur(resid, (0, 0), 1.0)
        if small.shape != full.shape:
            resid = cv2.resize(resid, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.clip((full + resid) * 255.0, 0, 255).astype(np.uint8)

    def segment_probs(self, rgb_u8: np.ndarray) -> np.ndarray:
        h, w = rgb_u8.shape[:2]
        if self.segmenter is None:
            p = np.zeros((h, w, NUM_CLASSES), np.float32)
            p[..., 0] = 1
            return p
        small = _resize_max(rgb_u8.astype(np.float32) / 255.0, self.work_side)
        f = segmentation_features(small)
        p = self.segmenter(f.reshape(-1, f.shape[2])).reshape(small.shape[0], small.shape[1], -1)
        p = p.astype(np.float32)
        if self.smooth_sigma > 0:
            p = cv2.GaussianBlur(p, (0, 0), self.smooth_sigma)
        if p.shape[:2] != (h, w):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        return p

    def segment(self, rgb_u8: np.ndarray) -> np.ndarray:
        return self.segment_probs(rgb_u8).argmax(axis=2).astype(np.uint8)

    def predict(self, rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        restored = self.restore(rgb_u8)
        return restored, self.segment(restored)
