"""Framework-agnostic request handling (unit-testable without FastAPI)."""
from __future__ import annotations

import base64
import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from eurs_core import CLASS_NAMES, ModelRegistry, class_counts, colorize, overlay

MAX_BYTES = 12 * 1024 * 1024
MAX_SIDE = 1600


class ServiceError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def encode_png(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb)
    if not ok:
        raise ServiceError(500, "Could not encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image(raw: bytes) -> np.ndarray:
    if len(raw) > MAX_BYTES:
        raise ServiceError(413, "Maximum upload size is 12 MB.")
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise ServiceError(400, "Invalid or unsupported image.")
    pil.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.array(pil)


def health(registry: ModelRegistry) -> dict:
    best = registry.best()
    return {
        "status": "ok",
        "default_model": best,
        "trained_model_loaded": best != "classical",
        "available_models": list(registry.models),
        "classes": CLASS_NAMES,
    }


def process(registry: ModelRegistry, raw: bytes, filename: str, model: str = "auto",
            alpha: float = 0.5) -> dict:
    rgb = decode_image(raw)
    try:
        name, m = registry.get(model)
    except KeyError:
        raise ServiceError(404, f"Unknown model '{model}'. Available: {', '.join(registry.models)}")

    start = time.perf_counter()
    restored, mask = m.predict(rgb)
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    counts = class_counts(mask)
    total = max(1, mask.size)
    meta = getattr(m, "meta", {}) or {}
    warning = None
    if name == "classical":
        warning = ("Segmentation uses a colour heuristic, not a trained model. "
                   "Do not use these masks as challenge predictions.")
    elif meta.get("dataset") == "synthetic":
        warning = ("This model was trained on the synthetic smoke-test set only. "
                   "Retrain on the official EURS data before interpreting outputs.")
    return {
        "filename": filename or "image",
        "model": name,
        "model_display_name": meta.get("display_name", name),
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "processing_ms": elapsed,
        "input_png": encode_png(rgb),
        "restored_png": encode_png(restored),
        "mask_png": encode_png(colorize(mask)),
        "mask_index_png": encode_png(mask),
        "overlay_png": encode_png(overlay(restored, mask, alpha)),
        "class_pixel_counts": counts,
        "class_percentages": {k: round(100 * v / total, 2) for k, v in counts.items()},
        "warning": warning,
    }
