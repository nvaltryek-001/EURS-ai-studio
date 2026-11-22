"""Load any supported model spec: 'classical', a EURS-Lite .npz, or an EURS-Net .onnx."""
from pathlib import Path

from eurs_core import ClassicalModel


def load_model(spec: str):
    if spec == "classical":
        return ClassicalModel()
    p = Path(spec)
    if p.suffix == ".npz":
        from eurs_core.lite import LiteModel
        return LiteModel.load(p)
    if p.suffix == ".onnx":
        from eurs_core.onnx_model import OnnxModel
        return OnnxModel(p)
    raise SystemExit(f"Unknown model spec: {spec} (expected 'classical', *.npz or *.onnx)")
