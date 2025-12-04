"""Quality assurance checks: IoU validation, palette audits, geometric validation"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Union
from src.utils.logger import get_logger
from src.utils.error_handler import ValidationError

logger = get_logger(__name__)


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two binary masks"""
    # Ensure masks are binary
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    
    if union == 0:
        return 1.0  # Both masks are empty
    
    return float(intersection) / float(union)


def validate_geometric_similarity(
    original_mask: np.ndarray,
    generated_mask: np.ndarray,
    threshold: float = 0.85,
    phase: str = "unknown"
) -> Tuple[bool, float]:
    """Validate geometric similarity between original and generated masks"""
    iou = calculate_iou(original_mask, generated_mask)
    is_valid = iou >= threshold
    
    logger.info(
        "geometric_validation",
        phase=phase,
        iou=iou,
        threshold=threshold,
        is_valid=is_valid
    )
    
    if not is_valid:
        raise ValidationError(
            phase=phase,
            message=f"Geometric hallucination detected: IoU {iou:.3f} < threshold {threshold}"
        )
    
    return is_valid, iou


def extract_alpha_mask(image: np.ndarray) -> np.ndarray:
    """Extract alpha channel or create mask from image"""
    if image.shape[2] == 4:
        return image[:, :, 3]
    elif image.shape[2] == 3:
        # Create mask from non-white/transparent regions
        # Assuming white background
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        return mask
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def audit_palette_colors(
    image: np.ndarray,
    palette_hex: List[str],
    tolerance: int = 1
) -> Tuple[bool, List[str]]:
    """Audit image colors against palette, return invalid colors"""
    # Convert hex to RGB
    palette_rgb = []
    for hex_color in palette_hex:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        palette_rgb.append(rgb)
    palette_rgb = np.array(palette_rgb)
    
    # Get unique colors in image
    if len(image.shape) == 3:
        unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    else:
        unique_colors = np.unique(image)
        unique_colors = np.column_stack([unique_colors, unique_colors, unique_colors])
    
    invalid_colors = []
    for color in unique_colors:
        # Check if color matches any palette color within tolerance
        distances = np.linalg.norm(palette_rgb - color[:3], axis=1)
        if np.min(distances) > tolerance:
            invalid_colors.append(f"rgb{tuple(color[:3])}")
    
    is_valid = len(invalid_colors) == 0
    
    if not is_valid:
        logger.warning(
            "palette_audit_failed",
            num_invalid=len(invalid_colors),
            invalid_colors=invalid_colors[:10]  # Log first 10
        )
    
    return is_valid, invalid_colors


def validate_detail_removal(
    original: np.ndarray,
    processed: np.ndarray,
    max_detail_loss: float = 0.03
) -> Tuple[bool, float]:
    """Validate that detail removal is within <3% threshold"""
    # Calculate structural similarity
    from skimage.metrics import structural_similarity as ssim
    
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    similarity = ssim(original_gray, processed_gray)
    detail_loss = 1.0 - similarity
    
    is_valid = detail_loss < max_detail_loss
    
    logger.info(
        "detail_removal_validation",
        similarity=similarity,
        detail_loss=detail_loss,
        max_allowed=max_detail_loss,
        is_valid=is_valid
    )
    
    if not is_valid:
        raise ValidationError(
            phase="quality_check",
            message=f"Detail loss {detail_loss:.3f} exceeds threshold {max_detail_loss}"
        )
    
    return is_valid, detail_loss


def force_snap_to_palette(
    color: Tuple[int, int, int],
    palette_rgb: np.ndarray
) -> Tuple[int, int, int]:
    """Force snap a color to nearest palette color"""
    color_array = np.array(color[:3])
    distances = np.linalg.norm(palette_rgb - color_array, axis=1)
    nearest_idx = np.argmin(distances)
    return tuple(palette_rgb[nearest_idx].astype(int))

