# 🌊 EURS AI Studio

### Underwater Image Restoration & Semantic Segmentation

EURS AI Studio is a web-based computer vision project designed to improve the visibility and colour quality of underwater images and segment marine-scene pixels into five classes.

🌐 **Live Website:** https://eurs-ai-studio.vercel.app  
💻 **GitHub Repository:** https://github.com/nvaltryek-001/EURS-ai-studio  
⚙️ **Backend API:** https://eurs-ai-studio.onrender.com  
❤️ **Project Status:** Prototype deployed; dataset training and evaluation are in progress.

---

## ✨ Features

- 📤 Upload underwater images through a web interface.
- 🌊 Perform classical computer-vision image restoration.
- 🎨 Generate segmentation visualizations and overlays.
- 🐠 Display pixel counts for five semantic classes.
- 🖼️ Preview the input and processed outputs.
- 📥 Download available processing results.
- 🌐 Access the application through a deployed frontend and backend.

## 🎯 Project Aim

To develop a web-based system for underwater image restoration and five-class semantic segmentation using the official EURS dataset.

## 🐟 Segmentation Classes

| Class ID | Class Name |
|---|---|
| 0 | Background |
| 1 | Coral |
| 2 | Fish |
| 3 | Elasmobranch (Sharks/Rays) |
| 4 | Other |

## 🛠️ Technology Stack

**Frontend**
- React
- Vite
- CSS

**Backend**
- Python
- FastAPI
- OpenCV
- NumPy
- Pillow

**Deployment**
- Vercel — frontend hosting
- Render — backend hosting
- GitHub — source-code management

## 🏗️ System Architecture

```text
              USER
               |
               v
      React + Vite Frontend
        (Vercel Hosting)
               |
          Image Upload
               |
               v
       FastAPI Backend
       (Render Hosting)
               |
        Image Processing
          /          \
         v            v
   Restoration    Segmentation
     Baseline       Heuristic
          \          /
           v        v
        Output Generation
               |
               v
      Restored Image,
      Segmentation Overlay,
      Class-wise Pixel Counts
               |
               v
          User Interface
```

**Current implementation:** Restoration uses a classical computer-vision baseline, while segmentation uses heuristic rules. A trained deep-learning model is a future development milestone.

## 🔄 How It Works

1. Upload an underwater image.
2. The frontend sends the image to the backend API.
3. The backend applies the current restoration and segmentation pipeline.
4. The application generates an image, segmentation visualization and class-wise pixel counts.
5. View and download the available outputs.

## 📊 Dataset

The project is intended to use the official EURS dataset associated with the MaCVi @ WACV 2027 challenge.

**Official dataset folder:**  
https://drive.google.com/drive/folders/10nWoxDEdWjKoIzYQp3--KSnWca3Vn4C5?usp=sharing

**Official challenge page:**  
https://www.macvi.org/workshop/macvi27/challenges/eurs

The identified archives are:

- `eurs_train.zip` — training archive.
- `eurs_dev_degraded.zip` — development/evaluation archive.

Dataset extraction, structure verification, training and quantitative evaluation must be completed before reporting final dataset statistics or model performance. Evaluation data must be handled according to the official challenge rules.

## 🚀 Run Locally

### Requirements

- Node.js and npm
- Python
- Git

### 1. Clone the repository

```bash
git clone https://github.com/nvaltryek-001/EURS-ai-studio.git
cd EURS-ai-studio
```

### 2. Start the backend

Open a terminal:

```powershell
cd backend

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Backend health endpoint:

http://127.0.0.1:8000/api/health

API documentation:

http://127.0.0.1:8000/docs

### 3. Start the frontend

Open a second terminal from the repository root:

```powershell
cd frontend

npm install
npm run dev
```

Open the development URL printed by Vite, typically:

http://localhost:5173

Configure the frontend's `VITE_API_URL` environment variable to point to the backend base URL when required. For local development, use `http://localhost:8000`; for production, use the deployed Render URL.

## 🌐 Live Deployment

**Frontend:** https://eurs-ai-studio.vercel.app  
**Backend health:** https://eurs-ai-studio.onrender.com/api/health

The backend health endpoint should be checked independently if image processing fails. Free hosting instances may sleep after inactivity, causing an initial delay.

## 📈 Evaluation Plan

After dataset preparation and model training, evaluate using appropriate metrics:

**Image restoration**
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

**Semantic segmentation**
- Intersection over Union (IoU)
- Mean Intersection over Union (mIoU)
- F1 score
- Precision and recall

All reported results must come from actual experiments. No trained-model accuracy or final benchmark results are claimed at the current prototype stage.

## ⚠️ Current Limitations

- The current restoration method may introduce colour distortion.
- Heuristic segmentation can misclassify objects and water regions.
- The current model pipeline is not a trained semantic segmentation system.
- Training on the official dataset is still in progress.
- Quantitative performance metrics have not yet been established.

## 🔮 Future Scope

- Inspect and prepare the official dataset.
- Train restoration and semantic segmentation models.
- Evaluate models on appropriate held-out data.
- Compare trained models against the current baseline.
- Improve recognition of coral, fish and elasmobranchs.
- Export a compact ONNX model if pursuing the embedded challenge track.
- Generate reproducible training graphs and evaluation reports.

## 🤝 Contributions

Suggestions, feedback and contributions to improve the project are welcome.

## 📚 References

1. MaCVi @ WACV 2027 — EURS Challenge: https://www.macvi.org/workshop/macvi27/challenges/eurs
2. Official EURS Dataset: https://drive.google.com/drive/folders/10nWoxDEdWjKoIzYQp3--KSnWca3Vn4C5?usp=sharing

---

**Built as a computer vision research project for underwater image restoration and marine-scene analysis.** 🌊🐬
