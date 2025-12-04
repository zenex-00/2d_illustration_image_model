"""BiRefNet background removal"""

import numpy as np
from typing import Optional
from src.utils.logger import get_logger
from src.utils.error_handler import retry_on_failure, ModelLoadError
from src.pipeline.model_cache import get_model_cache

logger = get_logger(__name__)


class BackgroundRemover:
    """BiRefNet for high-quality background removal"""
    
    def __init__(self, model_name: str = "birefnet", device: str = "cuda"):
        """Initialize background remover (lazy loading)"""
        self.model_name = model_name
        self.device = device
        self.session = None
        # Models loaded lazily on first use
    
    @retry_on_failure(max_attempts=3, exceptions=(Exception,))
    def _load_model(self):
        """Load BiRefNet model via rembg library"""
        try:
            from rembg import remove, new_session
            
            # rembg supports multiple models including "birefnet" (BiRefNet)
            # BiRefNet is specified in plan section 4.2.1 for high-quality segmentation
            # that preserves high-frequency details like radio antennas and wheel spokes
            self.session = new_session(self.model_name)
            
            logger.info("background_remover_loaded", model_name=self.model_name)
        except Exception as e:
            raise ModelLoadError(
                phase="phase2",
                message=f"Failed to load background remover: {str(e)}",
                original_error=e
            )
    
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background from image
        
        Args:
            image: Input image as numpy array (RGB, uint8)
        
        Returns:
            Image with alpha channel (RGBA)
        """
        if self.session is None:
            self._load_model()
        
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        try:
            from rembg import remove
            from PIL import Image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Remove background
            result = remove(pil_image, session=self.session)
            
            # Convert back to numpy array
            result_array = np.array(result)
            
            logger.info("background_removed", output_shape=result_array.shape)
            
            return result_array
            
        except Exception as e:
            logger.error("background_removal_failed", error=str(e), exc_info=True)
            raise

