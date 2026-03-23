# Few-Shot Ocular Lesion Segmentation using MedSAM and Prompt Learning

## Project Overview
This project implements a few-shot learning approach for ocular lesion segmentation in medical eye images using MedSAM (Medical Segment Anything Model) and advanced prompt learning techniques. The system can identify and highlight tumors/lesions in eye medical images with minimal training examples.

## Features
- 🔬 **MedSAM Integration**: Leverages the powerful Medical Segment Anything Model
- 🎯 **Few-Shot Learning**: Requires only a few labeled examples to learn new lesion types
- 📊 **Comprehensive Metrics**: Accuracy, Dice Score, IoU, Precision, F1 Score, Recall
- 🖼️ **Visual Interface**: Interactive web UI for image upload and segmentation visualization
- 💾 **Model Checkpoints**: Save and load trained models
- 📈 **Training Visualization**: Real-time training metrics and loss curves

## Project Structure
```
client/
├── backend/
│   ├── models/              # MedSAM and prompt learning models
│   ├── utils/               # Utility functions
│   ├── data/                # Dataset handling
│   ├── training/            # Training scripts
│   ├── inference/           # Inference pipeline
│   └── app.py              # Flask API server
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Application pages
│   │   ├── utils/          # Frontend utilities
│   │   └── App.jsx
│   └── package.json
├── data/
│   ├── train/              # Training images
│   ├── val/                # Validation images
│   ├── test/               # Test images
│   └── annotations/        # Segmentation masks
├── checkpoints/            # Saved model weights
├── results/                # Segmentation outputs
└── requirements.txt
```

## Installation

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## Usage

### Training
```bash
cd backend
python training/train.py --config configs/train_config.yaml
```

### Inference
```bash
cd backend
python inference/predict.py --image path/to/image.jpg --output results/
```

### Run Web Application
```bash
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

## Evaluation Metrics
- **Accuracy**: Overall pixel-wise classification accuracy
- **Dice Score**: Overlap between predicted and ground truth masks
- **IoU (Intersection over Union)**: Jaccard index for segmentation quality
- **Precision**: True positive rate
- **Recall**: Sensitivity of lesion detection
- **F1 Score**: Harmonic mean of precision and recall

## Technologies
- **Backend**: Python, PyTorch, MedSAM, Flask
- **Frontend**: React, Vite, TailwindCSS
- **Deep Learning**: Few-shot learning, Prompt learning, Vision Transformers
- **Medical Imaging**: OpenCV, PIL, SimpleITK

## License
MIT License

