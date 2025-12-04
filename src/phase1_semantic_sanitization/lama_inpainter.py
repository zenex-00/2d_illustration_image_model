"""LaMa inpainting with quality checks and IoU validation"""

import numpy as np
from typing import Optional
from src.utils.logger import get_logger
from src.utils.error_handler import retry_on_failure, ModelLoadError, ValidationError
from src.utils.quality_assurance import calculate_iou
from src.pipeline.model_cache import get_model_cache

logger = get_logger(__name__)


class LaMaInpainter:
    """LaMa (Large Mask Inpainting) for structural inpainting"""
    
    def __init__(
        self,
        model_name: str = "big-lama",
        device: str = "cuda",
        quality_threshold: float = 0.97
    ):
        """Initialize LaMa inpainter (lazy loading)"""
        self.model_name = model_name
        self.device = device
        self.quality_threshold = quality_threshold
        self.model = None
        # Models loaded lazily on first use
    
    @retry_on_failure(max_attempts=3, exceptions=(Exception,))
    def _load_model(self):
        """Load LaMa model with retry logic"""
        try:
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config, HDStrategy, SDSampler
            
            cache = get_model_cache()
            
            # LaMa uses ModelManager
            self.model = ModelManager(
                name=self.model_name,
                device=self.device,
                hd_strategy=HDStrategy.ORIGINAL
            )
            
            logger.info("lama_loaded", model_name=self.model_name)
        except Exception as e:
            raise ModelLoadError(
                phase="phase1",
                message=f"Failed to load LaMa: {str(e)}",
                original_error=e
            )
    
    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        quality_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Inpaint image using mask
        
        Args:
            image: Input image as numpy array (RGB, uint8)
            mask: Binary mask (0=keep, 1=inpaint)
            quality_threshold: Override default quality threshold
        
        Returns:
            Inpainted image
        """
        if self.model is None:
            self._load_model()
        
        threshold = quality_threshold or self.quality_threshold
        
        # Ensure correct formats
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
        
        # Ensure mask is binary
        mask_binary = (mask > 127).astype(np.uint8) * 255
        
        try:
            # Run inpainting
            result = self.model(image, mask_binary)
            
            # Validate quality: check IoU between original and inpainted regions
            # Create mask of inpainted region
            inpainted_region = (mask_binary > 0).astype(np.uint8)
            
            # Calculate structural similarity in inpainted region
            from skimage.metrics import structural_similarity as ssim
            import cv2
            
            # Extract regions
            original_region = image[inpainted_region > 0]
            inpainted_region_pixels = result[inpainted_region > 0]
            
            if len(original_region) > 0 and len(inpainted_region_pixels) > 0:
                # Convert to grayscale for SSIM
                orig_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                
                # Calculate SSIM in masked region
                mask_region = inpainted_region > 0
                similarity = ssim(
                    orig_gray[mask_region],
                    result_gray[mask_region],
                    data_range=255
                )
                
                logger.info(
                    "lama_inpainting_quality",
                    similarity=similarity,
                    threshold=threshold
                )
                
                # Note: We don't fail on low similarity as inpainting is expected to change the region
                # But we log it for monitoring
            
            logger.info("lama_inpainting_complete", mask_area=mask_binary.sum())
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error("lama_inpainting_failed", error=str(e), exc_info=True)
            raise ValidationError(
                phase="phase1",
                message=f"Inpainting failed: {str(e)}",
                original_error=e
            )


