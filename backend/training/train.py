"""
Training script for few-shot ocular lesion segmentation
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter  # Disabled due to installation issues
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
import sys

# Dummy writer to replace TensorBoard
class SummaryWriter:
    def __init__(self, log_dir=None):
        pass
    def add_scalar(self, tag, scalar_value, global_step=None):
        pass
    def close(self):
        pass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.medsam_model import FewShotMedSAM
from data.dataset import get_dataloaders
from utils.metrics import SegmentationMetrics, MetricsTracker


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss
        
        Args:
            predictions: Predicted masks [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]
            
        Returns:
            Dice loss value
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined loss: BCE + Dice"""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss"""
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class Trainer:
    """Training manager for few-shot segmentation"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(log_dir)
        
        # Metrics
        self.metrics_calculator = SegmentationMetrics()
        self.best_val_dice = 0.0
        
    def train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            support_images = batch['support_images'].to(self.device)
            support_masks = batch['support_masks'].to(self.device)
            query_images = batch['query_images'].to(self.device)
            query_masks = batch['query_masks'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, iou_pred = self.model(
                query_images,
                support_images,
                support_masks,
            )
            
            # Compute loss
            loss = self.criterion(predictions, query_masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                metrics = self.metrics_calculator.compute_all_metrics(
                    predictions, query_masks
                )
                metrics_tracker.update(metrics)
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{metrics["dice_score"]:.4f}',
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        # Get average metrics
        avg_metrics = metrics_tracker.get_average()
        avg_metrics['loss'] = total_loss / len(self.train_loader)
        
        return avg_metrics
    
    def validate(self, epoch: int) -> dict:
        """
        Validate model
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for batch in pbar:
                # Move data to device
                support_images = batch['support_images'].to(self.device)
                support_masks = batch['support_masks'].to(self.device)
                query_images = batch['query_images'].to(self.device)
                query_masks = batch['query_masks'].to(self.device)
                
                # Forward pass
                predictions, iou_pred = self.model(
                    query_images,
                    support_images,
                    support_masks,
                )
                
                # Compute loss
                loss = self.criterion(predictions, query_masks)
                total_loss += loss.item()
                
                # Compute metrics
                metrics = self.metrics_calculator.compute_all_metrics(
                    predictions, query_masks
                )
                metrics_tracker.update(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{metrics["dice_score"]:.4f}',
                })
        
        # Get average metrics
        avg_metrics = metrics_tracker.get_average()
        avg_metrics['loss'] = total_loss / len(self.val_loader)
        
        # Log to tensorboard
        for metric_name, value in avg_metrics.items():
            self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'✓ Saved best model with Dice: {metrics["dice_score"]:.4f}')
    
    def train(self, num_epochs: int):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train
        """
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Dice: {train_metrics['dice_score']:.4f} | "
                  f"IoU: {train_metrics['iou']:.4f}")
            
            # Validate
            val_metrics = self.validate(epoch)
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Dice: {val_metrics['dice_score']:.4f} | "
                  f"IoU: {val_metrics['iou']:.4f}")
            
            # Print all metrics
            print("\nDetailed Validation Metrics:")
            for metric_name, value in val_metrics.items():
                if metric_name != 'loss':
                    print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['dice_score'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice_score']
            
            self.save_checkpoint(epoch, val_metrics, is_best)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print(f"Best Validation Dice Score: {self.best_val_dice:.4f}")
        print("=" * 60)
        
        self.writer.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Few-Shot MedSAM')
    parser.add_argument('--train_images', type=str, default='../data/train/images')
    parser.add_argument('--train_masks', type=str, default='../data/train/masks')
    parser.add_argument('--val_images', type=str, default='../data/val/images')
    parser.add_argument('--val_masks', type=str, default='../data/val/masks')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        train_image_dir=args.train_images,
        train_mask_dir=args.train_masks,
        val_image_dir=args.val_images,
        val_mask_dir=args.val_masks,
        batch_size=args.batch_size,
        few_shot=True,
        k_shot=args.k_shot,
    )
    
    # Create model
    print("Initializing model...")
    model = FewShotMedSAM(
        model_type='vit_b',
        checkpoint_path=args.checkpoint,
        num_prompts=10,
    )
    
    # Create optimizer and loss
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir='../checkpoints',
        log_dir='../logs',
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
