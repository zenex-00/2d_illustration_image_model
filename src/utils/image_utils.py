"""Image I/O and preprocessing utilities"""

import numpy as np
import os
from PIL import Image
import cv2
from typing import Union, Tuple, Optional
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Input validation constants
MAX_IMAGE_SIZE_MB = 50
MAX_IMAGE_DIMENSION = 4096
ALLOWED_FORMATS = ['JPEG', 'PNG', 'WebP']


def validate_image_file(image_path: Union[str, Path]) -> Image.Image:
    """
    Validate image file before loading.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object if valid
        
    Raises:
        ValueError: If file size, format, or dimensions exceed limits
    """
    image_path = Path(image_path)
    
    # Check file size
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_IMAGE_SIZE_MB:
        raise ValueError(
            f"Image file size ({file_size_mb:.2f}MB) exceeds maximum allowed size "
            f"({MAX_IMAGE_SIZE_MB}MB)"
        )
    
    # Open and check format
    # Use context manager to ensure file is properly closed
    try:
        with Image.open(image_path) as img:
            # Load image data immediately to ensure file handle is released
            img.load()
            # Validate format
            if img.format not in ALLOWED_FORMATS:
                raise ValueError(
                    f"Image format '{img.format}' is not allowed. "
                    f"Allowed formats: {', '.join(ALLOWED_FORMATS)}"
                )
            
            # Check dimensions
            max_dimension = max(img.size)
            if max_dimension > MAX_IMAGE_DIMENSION:
                raise ValueError(
                    f"Image dimension ({max_dimension}px) exceeds maximum allowed dimension "
                    f"({MAX_IMAGE_DIMENSION}px)"
                )
            
            # Convert to RGB (file will be closed when context exits)
            img_rgb = img.convert('RGB')
            # Create a copy since the original will be closed
            # PIL Image.copy() creates a new image with the same data
            return img_rgb.copy()
    except Exception as e:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.error("image_open_failed", path=str(image_path), error=str(e), exc_info=True)
        raise ValueError(f"Failed to open image file: {str(e)}")


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file path and return as numpy array (RGB).
    
    Validates file size, format, and dimensions before loading.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Numpy array (RGB, uint8)
        
    Raises:
        ValueError: If validation fails
    """
    # validate_image_file now returns the image directly
    img = validate_image_file(image_path)
    return np.array(img)


def save_image(image: np.ndarray, output_path: Union[str, Path], format: str = 'PNG') -> None:
    """Save numpy array image to file"""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        img = Image.fromarray(image)
    else:
        img = image
    
    img.save(output_path, format=format)


def resize_image(
    image: np.ndarray,
    target_size: Union[int, Tuple[int, int]],
    maintain_aspect: bool = False
) -> np.ndarray:
    """Resize image to target size"""
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    if maintain_aspect:
        h, w = image.shape[:2]
        # Prevent division by zero
        if h == 0 or target_size[1] == 0:
            logger.warning("invalid_image_dimensions", h=h, w=w, target_size=target_size)
            return image  # Return original if dimensions invalid
        aspect = w / h
        if aspect == 0:
            logger.warning("zero_aspect_ratio", h=h, w=w)
            return image  # Return original if aspect is zero
        if target_size[0] / target_size[1] > aspect:
            new_w = int(target_size[1] * aspect)
            new_h = target_size[1]
        else:
            new_w = target_size[0]
            new_h = int(target_size[0] / aspect)
        target_size = (new_w, new_h)
    
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image to RGB format"""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:
        # Check if BGR and convert to RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def normalize_image(image: np.ndarray, to_float: bool = True) -> np.ndarray:
    """Normalize image to [0, 1] range if to_float, else keep as uint8"""
    if to_float:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)
    else:
        if image.dtype != np.uint8:
            return (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        return image


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 format"""
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)

