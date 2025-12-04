"""RealESRGAN upscaling with model selection"""

import numpy as np
from typing import Optional
from src.utils.logger import get_logger
from src.utils.error_handler import retry_on_failure, ModelLoadError
from src.pipeline.model_cache import get_model_cache

logger = get_logger(__name__)


class Upscaler:
    """RealESRGAN upscaling"""
    
    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus_anime",
        scale: int = 4,
        device: str = "cuda"
    ):
        """Initialize upscaler (lazy loading)"""
        self.model_name = model_name
        self.scale = scale
        self.device = device
        self.model = None
        # Models loaded lazily on first use
    
    @retry_on_failure(max_attempts=3, exceptions=(Exception,))
    def _load_model(self):
        """Load RealESRGAN model"""
        try:
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            
            cache = get_model_cache()
            
            # Model selection logic
            if "anime" in self.model_name.lower():
                model_path = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/{self.model_name}.pth"
            else:
                model_path = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{self.model_name}.pth"
            
            # Create upsampler
            self.model = RealESRGANer(
                scale=self.scale,
                model_path=model_path,
                model=self.model_name,
                half=True,
                device=self.device
            )
            
            logger.info("upscaler_loaded", model_name=self.model_name, scale=self.scale)
        except Exception as e:
            raise ModelLoadError(
                phase="phase3",
                message=f"Failed to load upscaler: {str(e)}",
                original_error=e
            )
    
    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale image by scale factor
        
        Args:
            image: Input image as numpy array (RGB, uint8)
        
        Returns:
            Upscaled image
        """
        if self.model is None:
            self._load_model()
        
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        try:
            # RealESRGAN expects BGR
            image_bgr = image[:, :, ::-1] if len(image.shape) == 3 else image
            
            # Upscale
            output, _ = self.model.enhance(image_bgr, outscale=self.scale)
            
            # Convert back to RGB
            if len(output.shape) == 3:
                output = output[:, :, ::-1]
            
            logger.info("image_upscaled", input_shape=image.shape, output_shape=output.shape)
            
            return output.astype(np.uint8)
            
        except Exception as e:
            logger.error("upscaling_failed", error=str(e), exc_info=True)
            raise


