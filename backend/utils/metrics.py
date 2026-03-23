"""
Evaluation Metrics for Ocular Lesion Segmentation
Implements: Accuracy, Dice Score, IoU, Precision, F1 Score, Recall
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """
    Comprehensive metrics calculator for segmentation tasks
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        """
        Initialize metrics calculator
        
        Args:
            threshold: Threshold for binary segmentation
            smooth: Smoothing factor to avoid division by zero
        """
        self.threshold = threshold
        self.smooth = smooth
        
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics
        
        Args:
            predictions: Predicted masks [B, 1, H, W] or [B, H, W]
            targets: Ground truth masks [B, 1, H, W] or [B, H, W]
            
        Returns:
            Dictionary containing all metrics
        """
        # Ensure correct shape
        if predictions.dim() == 4:
            predictions = predictions.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)
            
        # Binarize predictions
        pred_binary = (predictions > self.threshold).float()
        target_binary = (targets > self.threshold).float()
        
        # Compute metrics
        metrics = {
            'accuracy': self.accuracy(pred_binary, target_binary),
            'dice_score': self.dice_score(pred_binary, target_binary),
            'iou': self.iou(pred_binary, target_binary),
            'precision': self.precision(pred_binary, target_binary),
            'recall': self.recall(pred_binary, target_binary),
            'f1_score': self.f1_score(pred_binary, target_binary),
            'specificity': self.specificity(pred_binary, target_binary),
        }
        
        return metrics
    
    def accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute pixel-wise accuracy
        
        Args:
            predictions: Binary predictions [B, H, W]
            targets: Binary targets [B, H, W]
            
        Returns:
            Accuracy score
        """
        correct = (predictions == targets).float()
        accuracy = correct.sum() / correct.numel()
        return accuracy.item()
    
    def dice_score(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute Dice coefficient (F1 score for segmentation)
        
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        
        Args:
            predictions: Binary predictions [B, H, W]
            targets: Binary targets [B, H, W]
            
        Returns:
            Dice score
        """
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice.item()
    
    def iou(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute Intersection over Union (Jaccard Index)
        
        IoU = |X ∩ Y| / |X ∪ Y|
        
        Args:
            predictions: Binary predictions [B, H, W]
            targets: Binary targets [B, H, W]
            
        Returns:
            IoU score
        """
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.item()
    
    def precision(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute precision (positive predictive value)
        
        Precision = TP / (TP + FP)
        
        Args:
            predictions: Binary predictions [B, H, W]
            targets: Binary targets [B, H, W]
            
        Returns:
            Precision score
        """
        true_positive = (predictions * targets).sum()
        predicted_positive = predictions.sum()
        
        precision = (true_positive + self.smooth) / (predicted_positive + self.smooth)
        return precision.item()
    
    def recall(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute recall (sensitivity, true positive rate)
        
        Recall = TP / (TP + FN)
        
        Args:
            predictions: Binary predictions [B, H, W]
            targets: Binary targets [B, H, W]
            
        Returns:
            Recall score
        """
        true_positive = (predictions * targets).sum()
        actual_positive = targets.sum()
        
        recall = (true_positive + self.smooth) / (actual_positive + self.smooth)
        return recall.item()
    
    def f1_score(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute F1 score (harmonic mean of precision and recall)
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            predictions: Binary predictions [B, H, W]
            targets: Binary targets [B, H, W]
            
        Returns:
            F1 score
        """
        prec = self.precision(predictions, targets)
        rec = self.recall(predictions, targets)
        
        f1 = (2 * prec * rec + self.smooth) / (prec + rec + self.smooth)
        return f1
    
    def specificity(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute specificity (true negative rate)
        
        Specificity = TN / (TN + FP)
        
        Args:
            predictions: Binary predictions [B, H, W]
            targets: Binary targets [B, H, W]
            
        Returns:
            Specificity score
        """
        true_negative = ((1 - predictions) * (1 - targets)).sum()
        actual_negative = (1 - targets).sum()
        
        specificity = (true_negative + self.smooth) / (actual_negative + self.smooth)
        return specificity.item()
    
    def confusion_matrix_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, int]:
        """
        Compute confusion matrix components
        
        Args:
            predictions: Binary predictions [B, H, W]
            targets: Binary targets [B, H, W]
            
        Returns:
            Dictionary with TP, TN, FP, FN counts
        """
        pred_flat = predictions.flatten().cpu().numpy()
        target_flat = targets.flatten().cpu().numpy()
        
        tn, fp, fn, tp = confusion_matrix(
            target_flat, pred_flat, labels=[0, 1]
        ).ravel()
        
        return {
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
        }


class MetricsTracker:
    """
    Track and aggregate metrics over multiple batches
    """
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.reset()
        
    def reset(self):
        """Reset all tracked metrics"""
        self.metrics_sum = {}
        self.count = 0
        
    def update(self, metrics: Dict[str, float]):
        """
        Update tracked metrics with new batch
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.metrics_sum:
                self.metrics_sum[key] = 0.0
            self.metrics_sum[key] += value
        self.count += 1
        
    def get_average(self) -> Dict[str, float]:
        """
        Get average metrics across all batches
        
        Returns:
            Dictionary of averaged metrics
        """
        if self.count == 0:
            return {}
        
        return {
            key: value / self.count
            for key, value in self.metrics_sum.items()
        }
    
    def get_summary(self) -> str:
        """
        Get formatted summary of metrics
        
        Returns:
            Formatted string with all metrics
        """
        avg_metrics = self.get_average()
        
        summary = "=" * 60 + "\n"
        summary += "SEGMENTATION METRICS SUMMARY\n"
        summary += "=" * 60 + "\n"
        
        for metric_name, value in avg_metrics.items():
            summary += f"{metric_name.replace('_', ' ').title():.<40} {value:.4f}\n"
        
        summary += "=" * 60 + "\n"
        
        return summary
