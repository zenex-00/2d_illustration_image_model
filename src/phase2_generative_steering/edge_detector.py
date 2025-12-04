"""Canny edge detection with LAB de-shine filter"""

import numpy as np
import cv2
from typing import Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EdgeDetector:
    """Canny edge detection with LAB color space de-shine"""
    
    def __init__(
        self,
        low_threshold: int = 100,
        high_threshold: int = 200,
        apply_deshine: bool = True
    ):
        """Initialize edge detector"""
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.apply_deshine = apply_deshine
    
    def deshine(self, image: np.ndarray) -> np.ndarray:
        """
        Remove specular highlights using LAB color space filtering
        
        Args:
            image: Input image as numpy array (RGB, uint8)
        
        Returns:
            De-shined image
        """
        # Convert RGB to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply bilateral filter to L channel (lightness) to reduce specular highlights
        l_channel = lab[:, :, 0]
        l_filtered = cv2.bilateralFilter(l_channel, 9, 75, 75)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize lighting
        # This reduces the intensity of specular highlights (glare) as per plan section 4.2.2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_filtered = clahe.apply(l_filtered)
        
        # Reconstruct LAB image
        lab[:, :, 0] = l_filtered
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def detect_edges(
        self,
        image: np.ndarray,
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None
    ) -> np.ndarray:
        """
        Detect edges using Canny algorithm
        
        Args:
            image: Input image as numpy array (RGB, uint8)
            low_threshold: Override default low threshold
            high_threshold: Override default high threshold
        
        Returns:
            Binary edge map
        """
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Apply de-shine if enabled
        if self.apply_deshine:
            image = self.deshine(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Canny edge detection
        low = low_threshold or self.low_threshold
        high = high_threshold or self.high_threshold
        
        edges = cv2.Canny(gray, low, high)
        
        logger.info("edges_detected", edge_pixels=edges.sum())
        
        return edges.astype(np.uint8)

