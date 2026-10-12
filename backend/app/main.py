import base64
import os
import time
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="EURS AI Studio API", version="1.0.0")

origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in origins if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_BYTES = 12 * 1024 * 1024
MAX_SIDE = 1600
CLASS_NAMES = ["Background", "Coral", "Fish", "Elasmobranch", "Other"]
CLASS_COLORS = [
    [30, 30, 30],
    [255, 145, 20],
    [0, 220, 255],
    [220, 40, 180],
    [80, 200, 100],
]

def encode_png(arr):
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("Could not encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def restore_image(bgr):
    # Classical baseline: gray-world color correction + CLAHE.
    # This is a starting point, not a learned restoration network.
    img = bgr.astype(np.float32)
    means = np.mean(img, axis=(0, 1)) + 1e-6
    target = float(np.mean(means))
    img = np.clip(img * (target / means), 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(result, 5, 30, 30)

def baseline_segmentation(bgr):
    # Explicitly a visualization heuristic, NOT a semantic AI model.
    # Train on the official pixel labels before treating masks as predictions.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    mask = np.zeros(h.shape, dtype=np.uint8)
    # Rough orange/red candidate regions for coral.
    coral = ((h < 20) | (h > 170)) & (s > 65) & (v > 45)
    # Cyan/blue candidates may include water, fish, or background.
    cyan = (h >= 75) & (h <= 115) & (s > 70) & (v > 60)
    # Neutral, non-dark regions are only weak candidates for other.
    neutral = (s < 45) & (v > 100)

    mask[neutral] = 4
    mask[cyan] = 2
    mask[coral] = 1

    # Colors are RGB for frontend visualization.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    for idx, color in enumerate(CLASS_COLORS):
        overlay[mask == idx] = (
            0.50 * overlay[mask == idx] + 0.50 * np.array(color)
        ).astype(np.uint8)

    return mask, overlay

@app.get("/")
def root():
    return {"name": "EURS AI Studio", "status": "ok"}

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "restoration": "classical_cv_baseline",
        "segmentation": "heuristic_demo_only",
        "trained_model_loaded": False,
        "classes": CLASS_NAMES,
    }

@app.post("/api/process")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Upload a supported image file.")

    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Maximum upload size is 12 MB.")

    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
        pil.thumbnail((MAX_SIDE, MAX_SIDE))
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        raise HTTPException(400, "Invalid or unsupported image.")

    start = time.perf_counter()
    restored = restore_image(bgr)
    mask, overlay = baseline_segmentation(restored)
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        mask_rgb[mask == idx] = color

    counts = {
        CLASS_NAMES[i]: int(np.count_nonzero(mask == i))
        for i in range(len(CLASS_NAMES))
    }

    return {
        "filename": file.filename or "image",
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "processing_ms": elapsed,
        "restored_png": encode_png(restored_rgb),
        "mask_png": encode_png(mask_rgb),
        "overlay_png": encode_png(overlay),
        "class_pixel_counts": counts,
        "warning": "Segmentation is a heuristic demo, not a trained model. "
                   "Do not use these masks as official challenge predictions."
    }
