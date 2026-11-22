# Model weights

The API auto-discovers weights in this folder (override with `EURS_MODELS_DIR`):

| File | Produced by | Loaded as |
|---|---|---|
| `eurs_lite.npz` | `python ml/train_lite.py ...` | `eurs_lite` (CPU MLP, NumPy inference) |
| `<name>.onnx` + `<name>.json` | `python ml/export_onnx.py ...` | `<name>` (EURS-Net via onnxruntime) |

If nothing is present, the API falls back to the training-free `classical` baseline.
`.npz` / `.onnx` files are git-ignored by default; commit them deliberately (or use Git LFS)
when you want the deployed API to serve a trained model.
