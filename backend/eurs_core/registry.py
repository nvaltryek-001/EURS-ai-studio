"""Discovers available models and exposes a uniform predict() interface.

Priority for model="auto": EURS-Net (ONNX)  >  EURS-Lite (.npz)  >  classical baseline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .classical import restore_classical, segment_heuristic

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class ClassicalModel:
    name = "classical"
    meta = {
        "name": "classical",
        "display_name": "Classical baseline",
        "restoration": "gray-world + CLAHE + bilateral",
        "segmentation": "colour-threshold heuristic (not learned)",
        "trained": False,
    }

    def predict(self, rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        restored = restore_classical(rgb_u8)
        return restored, segment_heuristic(restored)


class ModelRegistry:
    def __init__(self, models_dir: str | Path | None = None):
        self.models_dir = Path(models_dir or os.getenv("EURS_MODELS_DIR", DEFAULT_MODELS_DIR))
        self.models: dict[str, object] = {"classical": ClassicalModel()}
        self.errors: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        lite = self.models_dir / "eurs_lite.npz"
        if lite.exists():
            try:
                from .lite import LiteModel
                m = LiteModel.load(lite)
                m.meta.setdefault("display_name", "EURS-Lite (CPU MLP)")
                m.meta["trained"] = True
                self.models["eurs_lite"] = m
            except Exception as e:  # pragma: no cover - surfaced via /api/models
                self.errors["eurs_lite"] = str(e)
        for onnx_path in sorted(self.models_dir.glob("*.onnx")):
            try:
                from .onnx_model import OnnxModel
                m = OnnxModel(onnx_path)
                m.meta.setdefault("display_name", "EURS-Net (ONNX)")
                m.meta["trained"] = True
                self.models[m.name] = m
            except Exception as e:  # pragma: no cover
                self.errors[onnx_path.stem] = str(e)

    def best(self) -> str:
        for name in self.models:
            if name not in ("classical", "eurs_lite"):
                return name  # an ONNX network
        return "eurs_lite" if "eurs_lite" in self.models else "classical"

    def get(self, name: str = "auto"):
        if name in (None, "", "auto"):
            name = self.best()
        if name not in self.models:
            raise KeyError(name)
        return name, self.models[name]

    def describe(self) -> list[dict]:
        out = []
        for name, m in self.models.items():
            meta = dict(getattr(m, "meta", {}) or {})
            meta["name"] = name
            meta.setdefault("display_name", name)
            out.append(json.loads(json.dumps(meta, default=str)))
        return out
