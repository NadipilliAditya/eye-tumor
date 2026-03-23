"""
Flask API server for ocular lesion segmentation
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import torch
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from inference.predict import OcularLesionPredictor
from utils.metrics import SegmentationMetrics

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
# Checkpoint Selection
# Get absolute path to parent directory (client/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

if os.path.exists(os.path.join(CHECKPOINT_DIR, 'best.pth')):
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best.pth')
elif os.path.exists(os.path.join(CHECKPOINT_DIR, 'medsam_vit_b.pth')):
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'medsam_vit_b.pth')
    print("Using base MedSAM checkpoint for inference.")
else:
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'medsam_vit_b.pth')  # Expect it here

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize predictor (lazy loading)
predictor = None


def get_predictor():
    """Get or initialize predictor"""
    global predictor
    if predictor is None:
        if os.path.exists(CHECKPOINT_PATH):
            predictor = OcularLesionPredictor(
                checkpoint_path=CHECKPOINT_PATH,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
        else:
            print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")
            print("Please train the model first or provide a valid checkpoint path")
    return predictor


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    pil_img = Image.fromarray(image.astype(np.uint8))
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode('utf-8')


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Ocular Lesion Segmentation API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Segment lesions in eye images',
            '/health': 'GET - Check API health',
            '/metrics': 'GET - Get model performance metrics',
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = predictor is not None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': device,
        'checkpoint_exists': os.path.exists(CHECKPOINT_PATH),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict lesion segmentation
    
    Expected input:
    - image: Image file
    - threshold: (optional) Segmentation threshold (default: 0.5)
    
    Returns:
    - original_image: Base64 encoded original image
    - segmentation_mask: Base64 encoded binary mask
    - visualization: Base64 encoded visualization
    - confidence_map: Base64 encoded confidence heatmap
    - metrics: Confidence scores and statistics
    """
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get threshold
        threshold = float(request.form.get('threshold', 0.5))
        
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get predictor
        pred = get_predictor()
        if pred is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Run prediction
        original_image, binary_mask, confidence_map, confidence = pred.predict(
            image_path=filepath,
            threshold=threshold,
        )
        
        # Create visualization
        visualization = pred.visualize_prediction(original_image, binary_mask)
        
        # Convert to base64
        original_b64 = numpy_to_base64(original_image)
        mask_b64 = numpy_to_base64(binary_mask * 255)
        vis_b64 = numpy_to_base64(visualization)
        
        # Create confidence heatmap
        import cv2
        conf_colored = cv2.applyColorMap(
            (confidence_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        conf_b64 = numpy_to_base64(cv2.cvtColor(conf_colored, cv2.COLOR_BGR2RGB))
        
        # Calculate statistics
        lesion_area = int(binary_mask.sum())
        total_area = binary_mask.size
        lesion_percentage = (lesion_area / total_area) * 100
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'original_image': original_b64,
            'segmentation_mask': mask_b64,
            'visualization': vis_b64,
            'confidence_map': conf_b64,
            'metrics': {
                'mean_confidence': confidence['mean_confidence'],
                'max_confidence': confidence['max_confidence'],
                'predicted_iou': confidence['predicted_iou'],
                'lesion_area_pixels': lesion_area,
                'total_area_pixels': total_area,
                'lesion_percentage': lesion_percentage,
                'tumor_location': confidence['location'],
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple images
    
    Expected input:
    - images: Multiple image files
    - threshold: (optional) Segmentation threshold
    
    Returns:
    - results: List of prediction results for each image
    """
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        threshold = float(request.form.get('threshold', 0.5))
        
        pred = get_predictor()
        if pred is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            # Save and process
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Predict
            original_image, binary_mask, confidence_map, confidence = pred.predict(
                image_path=filepath,
                threshold=threshold,
            )
            
            # Calculate statistics
            lesion_area = int(binary_mask.sum())
            total_area = binary_mask.size
            
            results.append({
                'filename': file.filename,
                'lesion_detected': lesion_area > 0,
                'lesion_area_pixels': lesion_area,
                'lesion_percentage': (lesion_area / total_area) * 100,
                'confidence': confidence['mean_confidence'],
            })
            
            # Clean up
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            'total_images': len(results),
            'results': results,
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/metrics')
def get_metrics():
    """Get available evaluation metrics"""
    return jsonify({
        'available_metrics': [
            'accuracy',
            'dice_score',
            'iou',
            'precision',
            'recall',
            'f1_score',
            'specificity',
        ],
        'description': {
            'accuracy': 'Pixel-wise classification accuracy',
            'dice_score': 'Dice coefficient (overlap measure)',
            'iou': 'Intersection over Union (Jaccard index)',
            'precision': 'True positive rate',
            'recall': 'Sensitivity',
            'f1_score': 'Harmonic mean of precision and recall',
            'specificity': 'True negative rate',
        }
    })


if __name__ == '__main__':
    print("=" * 60)
    print("OCULAR LESION SEGMENTATION API")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
