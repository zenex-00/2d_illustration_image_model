"""Security utilities for API: image validation and rate limiting"""

import io
from typing import Optional
from fastapi import HTTPException, UploadFile
from PIL import Image
from src.utils.image_utils import (
    MAX_IMAGE_SIZE_MB,
    MAX_IMAGE_DIMENSION,
    ALLOWED_FORMATS,
    validate_image_file
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Increase PIL's decompression bomb limit to handle large images
# This must be set before any Image.open() calls
Image.MAX_IMAGE_PIXELS = 1_000_000_000  # 1 billion pixels


def validate_uploaded_image(
    file: UploadFile,
    max_size_mb: Optional[float] = None,
    max_dimension: Optional[int] = None
) -> Image.Image:
    """
    Validate uploaded image file.
    
    Args:
        file: FastAPI UploadFile object
        max_size_mb: Optional override for max file size (defaults to MAX_IMAGE_SIZE_MB)
        max_dimension: Optional override for max dimension (defaults to MAX_IMAGE_DIMENSION)
        
    Returns:
        Validated PIL Image object
        
    Raises:
        HTTPException: If validation fails
    """
    max_size = max_size_mb or MAX_IMAGE_SIZE_MB
    max_dim = max_dimension or MAX_IMAGE_DIMENSION
    
    # Read file content
    try:
        content = file.file.read()
        file.file.seek(0)  # Reset file pointer
    except Exception as e:
        logger.error("file_read_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(e)}"
        )
    
    # Check file size
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > max_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image file size ({file_size_mb:.2f}MB) exceeds maximum allowed size "
                f"({max_size}MB)"
            )
        )
    
    # Validate image format and dimensions
    try:
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        logger.error("image_open_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )
    
    # Validate format
    if img.format not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image format '{img.format}' is not allowed. "
                f"Allowed formats: {', '.join(ALLOWED_FORMATS)}"
            )
        )
    
    # Check dimensions
    max_img_dimension = max(img.size)
    if max_img_dimension > max_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image dimension ({max_img_dimension}px) exceeds maximum allowed dimension "
                f"({max_dim}px)"
            )
        )
    
    logger.info("image_validated", format=img.format, size=img.size, file_size_mb=file_size_mb)
    
    return img


def validate_base64_image(
    base64_string: str,
    max_size_mb: Optional[float] = None,
    max_dimension: Optional[int] = None
) -> Image.Image:
    """
    Validate base64-encoded image.
    
    Args:
        base64_string: Base64-encoded image string (with or without data URI prefix)
        max_size_mb: Optional override for max file size
        max_dimension: Optional override for max dimension
        
    Returns:
        Validated PIL Image object
        
    Raises:
        HTTPException: If validation fails
    """
    import base64
    
    max_size = max_size_mb or MAX_IMAGE_SIZE_MB
    max_dim = max_dimension or MAX_IMAGE_DIMENSION
    
    # Remove data URI prefix if present
    if base64_string.startswith("data:image/"):
        base64_string = base64_string.split(",", 1)[1]
    
    # Decode base64
    try:
        image_bytes = base64.b64decode(base64_string)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 encoding: {str(e)}"
        )
    
    # Check file size
    file_size_mb = len(image_bytes) / (1024 * 1024)
    if file_size_mb > max_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image file size ({file_size_mb:.2f}MB) exceeds maximum allowed size "
                f"({max_size}MB)"
            )
        )
    
    # Validate image
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {str(e)}"
        )
    
    # Validate format
    if img.format not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image format '{img.format}' is not allowed. "
                f"Allowed formats: {', '.join(ALLOWED_FORMATS)}"
            )
        )
    
    # Check dimensions
    max_img_dimension = max(img.size)
    if max_img_dimension > max_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image dimension ({max_img_dimension}px) exceeds maximum allowed dimension "
                f"({max_dim}px)"
            )
        )
    
    return img


def get_rate_limit_key(request, user_id: Optional[str] = None) -> str:
    """
    Get rate limit key for request.
    
    Args:
        request: FastAPI Request object
        user_id: Optional user ID for authenticated requests
        
    Returns:
        Rate limit key string
    """
    if user_id:
        return f"user:{user_id}"
    
    # Use client IP address
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


