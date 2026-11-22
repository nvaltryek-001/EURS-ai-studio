"""EURS shared inference core (used by the API and by ml/ training & evaluation scripts)."""
from .palette import CLASS_COLORS, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, class_counts, colorize, overlay
from .registry import ClassicalModel, ModelRegistry

__all__ = [
    "CLASS_COLORS", "CLASS_NAMES", "IGNORE_INDEX", "NUM_CLASSES", "class_counts", "colorize",
    "overlay", "ClassicalModel", "ModelRegistry",
]
