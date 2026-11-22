"""Class taxonomy and colour palette for the EURS five-class segmentation task."""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ["Background", "Coral", "Fish", "Elasmobranch", "Other"]
NUM_CLASSES = len(CLASS_NAMES)

# RGB visualisation colours (one per class id).
CLASS_COLORS = np.array(
    [
        [24, 34, 48],     # 0 Background (deep water)
        [255, 138, 76],   # 1 Coral
        [56, 214, 255],   # 2 Fish
        [232, 72, 196],   # 3 Elasmobranch
        [120, 214, 120],  # 4 Other
    ],
    dtype=np.uint8,
)

IGNORE_INDEX = 255


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map an HxW class-index mask to an HxWx3 RGB image."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask < NUM_CLASSES
    out[valid] = CLASS_COLORS[mask[valid]]
    return out


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colourised mask over an RGB image. Background is left mostly untouched."""
    color = colorize(mask).astype(np.float32)
    base = rgb.astype(np.float32)
    a = np.where(mask[..., None] == 0, alpha * 0.25, alpha)
    return np.clip(base * (1 - a) + color * a, 0, 255).astype(np.uint8)


def class_counts(mask: np.ndarray) -> dict[str, int]:
    return {name: int(np.count_nonzero(mask == i)) for i, name in enumerate(CLASS_NAMES)}
