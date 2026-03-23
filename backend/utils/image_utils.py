"""
Utility functions for image processing and visualization
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True,
) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size (H, W)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )
    else:
        resized = cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_LINEAR)
    
    return resized


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Create overlay visualization
    
    Args:
        image: Original image
        mask: Binary mask
        alpha: Overlay transparency
        color: Overlay color (RGB)
        
    Returns:
        Overlay image
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return result


def create_heatmap(
    confidence_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Create confidence heatmap
    
    Args:
        confidence_map: Confidence values [0, 1]
        colormap: OpenCV colormap
        
    Returns:
        Colored heatmap
    """
    heatmap = (confidence_map * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, colormap)
    return colored


def draw_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw contours on image
    
    Args:
        image: Input image
        mask: Binary mask
        color: Contour color (RGB)
        thickness: Line thickness
        
    Returns:
        Image with contours
    """
    result = image.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1]
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def save_comparison_grid(
    images: list,
    titles: list,
    save_path: str,
    figsize: Tuple[int, int] = (15, 5),
):
    """
    Save comparison grid of images
    
    Args:
        images: List of images
        titles: List of titles
        save_path: Path to save figure
        figsize: Figure size
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
