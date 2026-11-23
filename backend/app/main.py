"""EURS AI Studio API (FastAPI)."""
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make eurs_core importable

from eurs_core import ModelRegistry  # noqa: E402

from . import service  # noqa: E402

app = FastAPI(title="EURS AI Studio API", version="2.0.0",
              description="Underwater image restoration + five-class semantic segmentation.")

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in origins if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

registry = ModelRegistry()


@app.get("/")
def root():
    return {"name": "EURS AI Studio", "status": "ok", "docs": "/docs"}


@app.get("/api/health")
def health():
    return service.health(registry)


@app.get("/api/models")
def models():
    return {"default": registry.best(), "models": registry.describe(), "errors": registry.errors}


@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    model: str = Query("auto", description="auto | classical | eurs_lite | <onnx model name>"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Upload a supported image file.")
    raw = await file.read(service.MAX_BYTES + 1)
    try:
        return service.process(registry, raw, file.filename, model, alpha)
    except service.ServiceError as e:
        raise HTTPException(e.status, e.detail)
