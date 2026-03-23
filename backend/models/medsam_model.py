"""
MedSAM Model Implementation for Few-Shot Ocular Lesion Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from typing import Tuple, Optional, List
import numpy as np


class MedSAM(nn.Module):
    """
    Medical Segment Anything Model (MedSAM) wrapper for ocular lesion segmentation
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = False,
    ):
        """
        Initialize MedSAM model
        
        Args:
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            checkpoint_path: Path to pretrained checkpoint
            freeze_image_encoder: Whether to freeze image encoder weights
            freeze_prompt_encoder: Whether to freeze prompt encoder weights
        """
        super().__init__()
        
        # Load SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        # Freeze components if specified
        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
                
        if freeze_prompt_encoder:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False
        
        self.model_type = model_type
        
    def forward(
        self,
        images: torch.Tensor,
        point_prompts: Optional[torch.Tensor] = None,
        box_prompts: Optional[torch.Tensor] = None,
        mask_prompts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MedSAM
        
        Args:
            images: Input images [B, 3, H, W]
            point_prompts: Point prompts [B, N, 2] with labels [B, N]
            box_prompts: Bounding box prompts [B, 4]
            mask_prompts: Mask prompts [B, 1, H, W]
            
        Returns:
            masks: Predicted segmentation masks [B, 1, H, W]
            iou_predictions: IoU predictions [B, 1]
        """
        # Encode image
        image_embeddings = self.sam.image_encoder(images)
        
        # Prepare prompts
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=point_prompts,
            boxes=box_prompts,
            masks=mask_prompts,
        )
        
        # Decode masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Upscale masks to original resolution
        masks = F.interpolate(
            low_res_masks,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        return masks, iou_predictions
    
    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get image embeddings for prompt learning
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            Image embeddings [B, C, H', W']
        """
        with torch.no_grad():
            embeddings = self.sam.image_encoder(images)
        return embeddings


class PromptLearner(nn.Module):
    """
    Learnable prompt generator for few-shot learning
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_prompts: int = 10,
        prompt_type: str = "point",
    ):
        """
        Initialize prompt learner
        
        Args:
            embedding_dim: Dimension of prompt embeddings
            num_prompts: Number of learnable prompts
            prompt_type: Type of prompts ('point', 'box', 'hybrid')
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_prompts = num_prompts
        self.prompt_type = prompt_type
        
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_prompts, embedding_dim)
        )
        
        # Prompt projection layers
        self.prompt_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        
        # Context encoder for few-shot adaptation
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        
    def forward(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adaptive prompts based on support and query features
        
        Args:
            support_features: Features from support images [B, K, C]
            query_features: Features from query image [B, C]
            
        Returns:
            Adaptive prompt embeddings [B, N, C]
        """
        batch_size = query_features.shape[0]
        
        # Expand prompt embeddings for batch
        prompts = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, N, C]
        
        # Encode support set context
        support_context = self.context_encoder(support_features)  # [B, K, C]
        
        # Compute attention between prompts and support context
        context_summary = support_context.mean(dim=1, keepdim=True)  # [B, 1, C]
        
        # Adapt prompts based on context
        adapted_prompts = prompts + context_summary
        
        # Project prompts
        adapted_prompts = self.prompt_proj(adapted_prompts)
        
        return adapted_prompts


class FewShotMedSAM(nn.Module):
    """
    Complete few-shot learning model combining MedSAM with prompt learning
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        num_prompts: int = 10,
    ):
        """
        Initialize few-shot MedSAM
        
        Args:
            model_type: SAM model type
            checkpoint_path: Path to pretrained checkpoint
            num_prompts: Number of learnable prompts
        """
        super().__init__()
        
        # Base MedSAM model
        self.medsam = MedSAM(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            freeze_image_encoder=True,
            freeze_prompt_encoder=False,
        )
        
        # Prompt learner
        self.prompt_learner = PromptLearner(
            embedding_dim=256,
            num_prompts=num_prompts,
        )
        
    def forward(
        self,
        query_images: torch.Tensor,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Few-shot segmentation forward pass
        
        Args:
            query_images: Query images to segment [B, 3, H, W]
            support_images: Support images with annotations [B, K, 3, H, W]
            support_masks: Support masks [B, K, 1, H, W]
            
        Returns:
            Predicted masks and IoU scores
        """
        batch_size, num_support = support_images.shape[:2]
        
        # Get embeddings for support images
        support_images_flat = support_images.view(
            batch_size * num_support, *support_images.shape[2:]
        )
        support_embeddings = self.medsam.get_image_embeddings(support_images_flat)
        support_embeddings = support_embeddings.view(
            batch_size, num_support, *support_embeddings.shape[1:]
        )
        
        # Get embeddings for query images
        query_embeddings = self.medsam.get_image_embeddings(query_images)
        
        # Generate adaptive prompts
        support_features = support_embeddings.mean(dim=[3, 4])  # Pool spatial dims
        query_features = query_embeddings.mean(dim=[2, 3])  # Pool spatial dims
        
        adaptive_prompts = self.prompt_learner(support_features, query_features)
        
        # Perform segmentation with learned prompts
        # The adaptive prompts are used as sparse embeddings for the prompt encoder
        # We need to reshape them for the MedSAM forward call if necessary, 
        # or use a different interface if medsam expects specific prompt types.
        # For now, we'll pass them to the underlying sam model's decoder via a custom forward or by setting the embeddings.
        
        # Encode query image
        image_embeddings = self.medsam.sam.image_encoder(query_images)
        
        # Use simple point prompt from query image center as a baseline + learned prompts
        h, w = query_images.shape[2:]
        coords = torch.tensor([[[w // 2, h // 2]]], device=query_images.device, dtype=torch.float)
        labels = torch.tensor([[1]], device=query_images.device, dtype=torch.long)
        
        sparse_embeddings, dense_embeddings = self.medsam.sam.prompt_encoder(
            points=(coords, labels),
            boxes=None,
            masks=None,
        )
        
        # Add adaptive prompts to sparse embeddings
        sparse_embeddings = torch.cat([sparse_embeddings, adaptive_prompts], dim=1)
        
        # Decode masks
        low_res_masks, iou_pred = self.medsam.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.medsam.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Upscale masks to original resolution
        masks = F.interpolate(
            low_res_masks,
            size=(query_images.shape[2], query_images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        return masks, iou_pred
