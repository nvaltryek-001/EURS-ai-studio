"""Runtime for the deep EURS-Net exported from PyTorch (see ml/export_onnx.py).

Contract (written by export_onnx.py into <name>.json next to <name>.onnx):
  input  "image":    float32 [1, 3, H, W], RGB in [0, 1], H and W multiples of 32
  output "restored": float32 [1, 3, H, W] in [0, 1]
  output "logits":   float32 [1, C, H, W]
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class OnnxModel:
    def __init__(self, onnx_path: str | Path, session=None):
        onnx_path = Path(onnx_path)
        meta_path = onnx_path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.name = self.meta.get("name", onnx_path.stem)
        self.input_size = int(self.meta.get("input_size", 512))
        if session is None:
            import onnxruntime as ort  # imported lazily so the API runs without it
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.session = session

    def predict(self, rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single forward pass -> (restored RGB uint8, class-index mask uint8)."""
        h, w = rgb_u8.shape[:2]
        s = self.input_size
        x = cv2.resize(rgb_u8, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        restored, logits = self.session.run(["restored", "logits"], {"image": x})
        restored = np.clip(restored[0].transpose(1, 2, 0), 0, 1)
        # Re-inject full-resolution detail: upsample the predicted correction, not the image.
        full = rgb_u8.astype(np.float32) / 255.0
        small_in = x[0].transpose(1, 2, 0)
        resid = cv2.resize(restored - small_in, (w, h), interpolation=cv2.INTER_LINEAR)
        restored_full = np.clip((full + resid) * 255, 0, 255).astype(np.uint8)
        # Segmentation is predicted jointly from the degraded input; argmax at low res,
        # then nearest-neighbour upsampling keeps memory bounded for large images.
        mask = logits[0].argmax(axis=0).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return restored_full, mask
