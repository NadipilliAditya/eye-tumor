# 🚀 Quick Reference Card

## Project: Few-Shot Ocular Lesion Segmentation

### 📋 What This Does
Detects and highlights tumors/lesions in eye medical images using AI with:
- **MedSAM** (Medical Segment Anything Model)
- **Few-Shot Learning** (learns from just 5 examples)
- **Prompt Learning** (adaptive prompts)
- **All Required Metrics**: Accuracy, Dice Score, IoU, Precision, F1 Score, Recall

---

## ⚡ Quick Start (3 Commands)

### Windows:
```bash
# 1. Run setup
setup.bat

# 2. Start backend (Terminal 1)
cd backend && venv\Scripts\activate && python app.py

# 3. Start frontend (Terminal 2)
cd frontend && npm run dev
```

### Linux/Mac:
```bash
# 1. Run setup
chmod +x setup.sh && ./setup.sh

# 2. Start backend (Terminal 1)
cd backend && source venv/bin/activate && python app.py

# 3. Start frontend (Terminal 2)
cd frontend && npm run dev
```

Then open: **http://localhost:3000**

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `backend/models/medsam_model.py` | MedSAM + Few-Shot + Prompt Learning |
| `backend/utils/metrics.py` | All 6 evaluation metrics |
| `backend/training/train.py` | Training script |
| `backend/inference/predict.py` | Inference pipeline |
| `backend/app.py` | Flask API server |
| `frontend/src/App.jsx` | Main React app |
| `frontend/src/components/UploadSection.jsx` | Image upload UI |
| `frontend/src/components/ResultsSection.jsx` | Results viewer |
| `frontend/src/components/MetricsDisplay.jsx` | Metrics dashboard |

---

## 🎯 Training the Model

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Basic training
python training/train.py

# Custom parameters
python training/train.py --epochs 50 --batch_size 4 --k_shot 10 --lr 0.0001
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 2)
- `--k_shot`: Number of support examples (default: 5)
- `--lr`: Learning rate (default: 0.0001)
- `--checkpoint`: Path to pretrained checkpoint

---

## 🔍 Running Inference

```bash
cd backend
source venv/bin/activate

# Single image
python inference/predict.py \
  --image path/to/eye_image.jpg \
  --checkpoint checkpoints/best.pth \
  --output results/

# Results saved:
# - results/image_mask.png (binary mask)
# - results/image_confidence.png (heatmap)
# - results/image_visualization.png (overlay)
# - results/image_metadata.txt (metrics)
```

---

## 📊 Evaluation Metrics

All metrics are automatically computed:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Dice Score** | 2×TP/(2×TP+FP+FN) | Overlap measure |
| **IoU** | TP/(TP+FP+FN) | Jaccard index |
| **Precision** | TP/(TP+FP) | Positive predictive value |
| **Recall** | TP/(TP+FN) | Sensitivity |
| **F1 Score** | 2×(P×R)/(P+R) | Harmonic mean |

---

## 🌐 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Segment single image |
| `/batch_predict` | POST | Segment multiple images |
| `/metrics` | GET | Available metrics |

**Example API Call:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@eye_image.jpg" \
  -F "threshold=0.5"
```

---

## 📂 Dataset Structure

```
data/
├── train/
│   ├── images/          # Training eye images
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── masks/           # Corresponding masks
│       ├── img001.png
│       ├── img002.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Image Requirements:**
- Format: JPG, JPEG, PNG
- Masks: Binary PNG (0=background, 255=lesion)
- Naming: Mask filename must match image filename

---

## 🎨 UI Features

✅ Drag-and-drop image upload  
✅ Real-time preview  
✅ Adjustable threshold slider  
✅ 4 visualization modes:
  - Original image
  - Binary mask
  - Overlay with contours
  - Confidence heatmap  
✅ Interactive metrics dashboard  
✅ Zoom modal  
✅ Download results  
✅ Responsive design  

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version (need 3.8+)
python --version

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend won't start
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### CUDA out of memory
```bash
# Reduce batch size
python training/train.py --batch_size 1

# Or use CPU
python training/train.py --device cpu
```

### Model not loading
- Ensure checkpoint file exists in `checkpoints/`
- Check file path in config
- Verify model architecture matches checkpoint

---

## 📚 Documentation

- **README.md** - Project overview
- **SETUP.md** - Detailed setup instructions
- **PROJECT_SUMMARY.md** - Complete technical documentation
- **This file** - Quick reference

---

## 🎓 Key Technologies

**Backend:** Python, PyTorch, MedSAM, Flask, OpenCV  
**Frontend:** React, Vite, Framer Motion, Recharts  
**AI:** Few-Shot Learning, Prompt Learning, Vision Transformers  
**Metrics:** Accuracy, Dice, IoU, Precision, Recall, F1  

---

## ✨ Project Highlights

🔬 **Medical AI** - State-of-the-art MedSAM model  
🎯 **Few-Shot** - Learn from just 5 examples  
📊 **Complete Metrics** - All 6 required metrics  
🎨 **Beautiful UI** - Modern dark theme with animations  
⚡ **Real-time** - Fast inference with confidence scores  
📦 **Production Ready** - Full-stack application  

---

**Created by:** Aditya  
**Date:** February 2026  
**License:** MIT  

For support, see SETUP.md or PROJECT_SUMMARY.md
