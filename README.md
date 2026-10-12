# EURS AI Studio
Underwater restoration and segmentation research application.

## Tracks
- Track 1: restore images and generate five-class segmentation masks.
- Track 2: train/export a compact model and validate ONNX constraints.

## Local development
Backend:
  cd backend
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  uvicorn app.main:app --reload --port 8000

Frontend (new terminal):
  cd frontend
  npm install
  npm run dev

Open http://localhost:5173

## Dataset
Download the official EURS dataset from:
https://drive.google.com/drive/folders/10nWoxDEdWjKoIzYQp3--KSnWca3Vn4C5

Extract the downloaded files under data/dataset.
Inspect the actual directory layout and official starter scripts before training.
Never use test labels for training.

The initial restoration and segmentation routines are classical CV baselines,
not a trained challenge-winning model. Validate every result against ground truth.
