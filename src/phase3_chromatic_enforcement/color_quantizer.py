"""GPU-accelerated color quantization with Kornia"""

import numpy as np
import torch
from typing import List, Optional, Union
from src.utils.logger import get_logger
from src.utils.error_handler import ValidationError
from src.utils.palette_manager import PaletteManager
from src.pipeline.config import get_config

logger = get_logger(__name__)

# Try to import Kornia for GPU acceleration
try:
    import kornia
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    logger.warning("kornia_not_available", falling_back_to="cpu")

# Fallback to CPU implementations if Kornia not available
try:
    from scipy.spatial import cKDTree
    from skimage.color import rgb2lab
    CPU_FALLBACK_AVAILABLE = True
except ImportError:
    CPU_FALLBACK_AVAILABLE = False


class ColorQuantizer:
    """CIEDE2000 color quantization to 15-color palette"""
    
    def __init__(
        self,
        palette_manager: Optional[PaletteManager] = None,
        method: str = "auto",
        device: str = "cuda"
    ):
        """
        Initialize color quantizer with GPU acceleration.
        
        Args:
            palette_manager: Optional palette manager
            method: Quantization method - "gpu" (Kornia GPU), "cpu" (CPU fallback), or "auto" (GPU if available)
            device: Device to use for GPU operations
        """
        self.palette_manager = palette_manager or PaletteManager()
        self.palette_rgb = self.palette_manager.get_rgb_colors()
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Determine method
        config = get_config()
        phase_config = config.get_phase_config("phase3")
        config_method = phase_config.get("quantization", {}).get("method", "auto")
        
        if method == "auto":
            method = config_method
        
        # Prefer GPU if available
        if method == "auto" or method == "gpu":
            if KORNIA_AVAILABLE and torch.cuda.is_available():
                self.method = "gpu_kornia"
            elif CPU_FALLBACK_AVAILABLE:
                self.method = "cpu_fallback"
                logger.warning("gpu_quantization_unavailable", falling_back_to="cpu")
            else:
                raise ImportError("Neither Kornia nor CPU fallback available for color quantization")
        elif method == "cpu":
            if CPU_FALLBACK_AVAILABLE:
                self.method = "cpu_fallback"
            else:
                raise ImportError("CPU fallback not available")
        else:
            self.method = method
        
        # Build palette tensors for GPU
        if self.method == "gpu_kornia":
            self._build_gpu_palette()
        else:
            self._build_cpu_palette()
        
        logger.info("color_quantizer_initialized", method=self.method, device=self.device)
    
    def _build_gpu_palette(self):
        """Build GPU palette tensors for Kornia"""
        # Convert palette RGB to tensor (normalized 0-1)
        palette_rgb_norm = self.palette_rgb.astype(np.float32) / 255.0
        # Shape: (15, 3) -> (1, 15, 3) for batch processing
        self.palette_tensor = torch.from_numpy(palette_rgb_norm).unsqueeze(0).to(self.device)
        
        # Convert to LAB using Kornia
        # Kornia expects (B, H, W, 3) format
        palette_lab = kornia.color.rgb_to_lab(self.palette_tensor.unsqueeze(0))  # (1, 1, 15, 3)
        self.palette_lab_tensor = palette_lab.squeeze(0).squeeze(0)  # (15, 3)
        
        logger.info("gpu_palette_built", palette_size=len(self.palette_rgb), device=self.device)
    
    def _build_cpu_palette(self):
        """Build CPU palette for fallback"""
        # Convert RGB to LAB
        rgb_normalized = self.palette_rgb.astype(np.float32) / 255.0
        self.palette_lab = rgb2lab(rgb_normalized.reshape(1, -1, 3)).reshape(-1, 3)
        # Build KDTree for fast nearest neighbor search
        self.kdtree = cKDTree(self.palette_lab)
        logger.info("cpu_palette_built", palette_size=len(self.palette_rgb))
    
    def quantize(
        self,
        image: np.ndarray,
        validate: bool = True
    ) -> np.ndarray:
        """
        Quantize image to 15-color palette using GPU-accelerated LAB Euclidean distance
        
        Args:
            image: Input image as numpy array (RGB, uint8)
            validate: Whether to validate output colors
        
        Returns:
            Quantized image
        """
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        original_shape = image.shape
        
        if self.method == "gpu_kornia":
            quantized = self._quantize_gpu_kornia(image)
        else:
            quantized = self._quantize_cpu_fallback(image)
        
        # Validate output
        if validate:
            self._validate_quantization(quantized)
        
        logger.info("color_quantization_complete", output_shape=quantized.shape, method=self.method)
        
        return quantized.astype(np.uint8)
    
    def _quantize_gpu_kornia(self, image: np.ndarray) -> np.ndarray:
        """GPU-accelerated quantization using Kornia"""
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).to(self.device)
        # Shape: (H, W, 3) -> (1, H, W, 3) for batch processing
        image_tensor = image_tensor.unsqueeze(0)  # (1, H, W, 3)
        
        # Convert to LAB
        image_lab = kornia.color.rgb_to_lab(image_tensor)  # (1, H, W, 3)
        
        # Flatten spatial dimensions: (1, H, W, 3) -> (1, H*W, 3)
        h, w = image_lab.shape[1:3]
        image_lab_flat = image_lab.reshape(1, h * w, 3)  # (1, H*W, 3)
        
        # Compute distances to all palette colors
        # image_lab_flat: (1, H*W, 3), palette_lab_tensor: (15, 3)
        # Expand palette: (15, 3) -> (1, 15, 3)
        palette_expanded = self.palette_lab_tensor.unsqueeze(0)  # (1, 15, 3)
        
        # Compute Euclidean distance: (1, H*W, 3) vs (1, 15, 3)
        # Use broadcasting: (1, H*W, 1, 3) - (1, 1, 15, 3) = (1, H*W, 15, 3)
        image_expanded = image_lab_flat.unsqueeze(2)  # (1, H*W, 1, 3)
        palette_expanded = palette_expanded.unsqueeze(1)  # (1, 1, 15, 3)
        
        # Compute squared distances
        distances = torch.sum((image_expanded - palette_expanded) ** 2, dim=-1)  # (1, H*W, 15)
        distances = distances.squeeze(0)  # (H*W, 15)
        
        # Find nearest palette color for each pixel
        indices = torch.argmin(distances, dim=1)  # (H*W,)
        
        # Map to palette colors
        palette_rgb_tensor = self.palette_tensor.squeeze(0)  # (15, 3)
        quantized_flat = palette_rgb_tensor[indices]  # (H*W, 3)
        
        # Reshape back to image
        quantized = quantized_flat.reshape(h, w, 3)  # (H, W, 3)
        
        # Convert back to numpy and denormalize
        quantized_np = quantized.cpu().numpy()
        quantized_np = (quantized_np * 255.0).astype(np.uint8)
        
        return quantized_np
    
    def _quantize_cpu_fallback(self, image: np.ndarray) -> np.ndarray:
        """CPU fallback quantization"""
        original_shape = image.shape
        image_flat = image.reshape(-1, 3)
        
        # Convert to LAB
        rgb_normalized = image_flat.astype(np.float32) / 255.0
        lab_image = rgb2lab(rgb_normalized.reshape(1, -1, 3)).reshape(-1, 3)
        
        # Use KDTree for nearest neighbor search
        _, indices = self.kdtree.query(lab_image)
        
        # Map to palette colors
        quantized_flat = self.palette_rgb[indices]
        quantized = quantized_flat.reshape(original_shape)
        
        return quantized
    
    
    def _validate_quantization(self, quantized: np.ndarray):
        """Validate that all colors in quantized image are in palette"""
        unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
        
        for color in unique_colors:
            # Check if color matches any palette color
            distances = np.linalg.norm(self.palette_rgb - color, axis=1)
            if np.min(distances) > 1.0:  # Allow 1 unit tolerance
                raise ValidationError(
                    phase="phase3",
                    message=f"Quantization produced invalid color: {color}"
                )
        
        logger.info("quantization_validated", unique_colors=len(unique_colors))

