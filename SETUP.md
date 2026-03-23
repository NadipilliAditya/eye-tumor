# Few-Shot Ocular Lesion Segmentation - Setup Guide

## Quick Start

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download MedSAM Checkpoint

Download the pretrained MedSAM checkpoint:
```bash
# Create checkpoints directory
mkdir checkpoints

# Download checkpoint (example - replace with actual URL)
# wget https://example.com/medsam_vit_b.pth -O checkpoints/medsam_pretrained.pth
```

### 3. Prepare Dataset

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── masks/
│       ├── image1.png
│       ├── image2.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 4. Train the Model

```bash
# Train with default config
python training/train.py

# Train with custom parameters
python training/train.py --epochs 50 --batch_size 4 --k_shot 10
```

### 5. Run Inference

```bash
# Single image prediction
python inference/predict.py --image path/to/image.jpg --checkpoint checkpoints/best.pth --output results/

# The results will be saved in the output directory with:
# - Binary segmentation mask
# - Confidence heatmap
# - Visualization overlay
# - Metadata file
```

### 6. Start Backend API

```bash
# Run Flask server
python app.py

# Server will start at http://localhost:5000
```

### 7. Frontend Setup

```bash
# Open new terminal
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Frontend will start at http://localhost:3000
```

## Usage

1. Open browser and navigate to `http://localhost:3000`
2. Upload an eye medical image
3. Adjust the detection threshold (0.0 - 1.0)
4. Click "Analyze Image"
5. View segmentation results and metrics

## Evaluation Metrics

The system provides comprehensive metrics:
- **Accuracy**: Pixel-wise classification accuracy
- **Dice Score**: Overlap between prediction and ground truth
- **IoU**: Intersection over Union
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean of precision and recall

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /batch_predict` - Batch prediction
- `GET /metrics` - Available metrics information

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use smaller model (vit_b instead of vit_l)
- Reduce image size

### Model Not Loading
- Ensure checkpoint path is correct
- Check if checkpoint file exists
- Verify model architecture matches checkpoint

### Poor Segmentation Results
- Increase number of training epochs
- Adjust k_shot (more support examples)
- Fine-tune threshold value
- Check data quality and annotations

## Advanced Configuration

Edit `configs/train_config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation
- Loss function weights
- Logging settings

## Citation

If you use this project, please cite:
```bibtex
@article{medsam2023,
  title={Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation},
  author={...},
  journal={...},
  year={2023}
}
```

## License

MIT License - See LICENSE file for details
