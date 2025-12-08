"""Image processing utilities"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image
import io

def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from a path.
    
    Args:
        path: Path to the image file
        
    Returns:
        Image as numpy array (RGB)
        
    Raises:
        FileNotFoundError: If file not found
        ValueError: If image cannot be loaded
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {path}")
        
    # Read using OpenCV
    img = cv2.imread(str(path))
    
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
        
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save an image to a path.
    
    Args:
        img: Image as numpy array (RGB)
        path: Output path
    """
    path_obj = Path(path)
    
    # Create parent directories if needed
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr_img = img
        
    cv2.imwrite(str(path), bgr_img)

def resize_image(img: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Resize image maintaining aspect ratio.
    
    Args:
        img: Input image
        max_dim: Maximum dimension (width or height)
        
    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    
    if max(h, w) <= max_dim:
        return img
        
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def image_to_bytes(img: np.ndarray, format: str = "PNG") -> bytes:
    """
    Convert numpy image to bytes.
    
    Args:
        img: Input image (uint8 format)
        format: Image format (PNG, JPEG, BMP, WEBP)
        
    Returns:
        Image bytes
        
    Raises:
        ValueError: If format is unsupported or image conversion fails
    """
    # Validate format
    valid_formats = ["PNG", "JPEG", "BMP", "WEBP"]
    format_upper = format.upper()
    if format_upper not in valid_formats:
        raise ValueError(f"Unsupported format: {format}. Must be one of {valid_formats}")
    
    try:
        # Convert to PIL
        pil_img = Image.fromarray(img)
        
        # Use context manager for BytesIO to ensure proper cleanup
        with io.BytesIO() as buf:
            pil_img.save(buf, format=format_upper)
            return buf.getvalue()
    except Exception as e:
        raise ValueError(f"Failed to convert image to bytes: {str(e)}") from e
