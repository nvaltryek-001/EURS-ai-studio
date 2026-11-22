"""Image / mask IO with explicit, checked mask decoding."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from eurs_core import CLASS_COLORS, IGNORE_INDEX, NUM_CLASSES

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8))


class MaskDecoder:
    """Decodes label images into class-index masks.

    Supported encodings (set in the layout config, or auto-detected):
      * "index":   single-channel PNG whose values are class ids (255 = ignore)
      * "palette": RGB PNG; colours mapped via `palette` (list of [r,g,b] per class id)
    Unknown values/colours are mapped to IGNORE_INDEX and counted, never silently guessed.
    """

    def __init__(self, encoding: str = "auto", palette: list[list[int]] | None = None):
        self.encoding = encoding
        self.palette = np.array(palette if palette is not None else CLASS_COLORS, dtype=np.int32)
        self.unknown_pixels = 0

    def __call__(self, path: str | Path) -> np.ndarray:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"Cannot read mask: {path}")
        enc = self.encoding
        if enc == "auto":
            enc = "index" if raw.ndim == 2 or (raw.ndim == 3 and np.all(raw[..., 0] == raw[..., 1])
                                                  and np.all(raw[..., 1] == raw[..., 2])) else "palette"
        if enc == "index":
            m = raw if raw.ndim == 2 else raw[..., 0]
            m = m.astype(np.int32)
            bad = (m >= NUM_CLASSES) & (m != IGNORE_INDEX)
            self.unknown_pixels += int(bad.sum())
            m[bad] = IGNORE_INDEX
            return m.astype(np.uint8)
        rgb = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB).astype(np.int32)
        code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        out = np.full(code.shape, IGNORE_INDEX, np.uint8)
        for cid, (r, g, b) in enumerate(self.palette):
            out[code == ((r << 16) | (g << 8) | b)] = cid
        self.unknown_pixels += int((out == IGNORE_INDEX).sum())
        return out


def load_layout(path: str | Path | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text())
