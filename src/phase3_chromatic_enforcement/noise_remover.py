"""Connected component analysis for noise removal"""

import numpy as np
import cv2
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NoiseRemover:
    """Remove small noise blobs using connected component analysis"""
    
    def __init__(self, min_area_percent: float = 0.001):
        """Initialize noise remover"""
        self.min_area_percent = min_area_percent
    
    def remove_noise(
        self,
        image: np.ndarray,
        min_area_percent: Optional[float] = None
    ) -> np.ndarray:
        """
        Remove small noise blobs from quantized image
        
        Args:
            image: Input quantized image (RGB, uint8)
            min_area_percent: Override default min area percentage
        
        Returns:
            Denoised image
        """
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        min_area = min_area_percent or self.min_area_percent
        total_area = image.shape[0] * image.shape[1]
        min_pixels = int(total_area * min_area)
        
        # Process each color channel separately
        result = image.copy()
        
        # Get unique colors
        unique_colors = np.unique(image.reshape(-1, 3), axis=0)
        
        for color in unique_colors:
            # Create binary mask for this color
            mask = np.all(image == color, axis=2).astype(np.uint8) * 255
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            
            # Remove small components
            for label_id in range(1, num_labels):  # Skip background (label 0)
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < min_pixels:
                    # Find nearest color (majority vote of neighbors)
                    component_mask = (labels == label_id)
                    neighbor_colors = self._get_neighbor_colors(image, component_mask)
                    if len(neighbor_colors) > 0:
                        replacement_color = self._majority_vote(neighbor_colors)
                        result[component_mask] = replacement_color
                    else:
                        # Fallback: use most common color in image if no neighbors
                        unique_colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
                        if len(unique_colors) > 0:
                            most_common_idx = np.argmax(counts)
                            result[component_mask] = unique_colors[most_common_idx]
        
        removed_pixels = np.sum(result != image)
        logger.info(
            "noise_removed",
            min_area_pixels=min_pixels,
            removed_pixels=removed_pixels
        )
        
        return result.astype(np.uint8)
    
    def _get_neighbor_colors(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Get colors of neighboring pixels"""
        # Dilate mask to get neighbors
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        neighbor_mask = (dilated > 0) & (~mask)
        
        if neighbor_mask.sum() == 0:
            return np.array([])
        
        return image[neighbor_mask]
    
    def _majority_vote(self, colors: np.ndarray) -> np.ndarray:
        """Get most common color"""
        if len(colors) == 0:
            return np.array([0, 0, 0], dtype=np.uint8)
        
        # Find unique colors and counts
        unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
        most_common_idx = np.argmax(counts)
        return unique_colors[most_common_idx]

