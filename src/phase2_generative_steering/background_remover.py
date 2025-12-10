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
        """Load BiRefNet model via rembg library - Primary model is birefnet with fallbacks if unavailable"""
        try:
            from rembg import remove, new_session

            # First try the primary model (birefnet) which is required
            # BiRefNet is a state-of-the-art model that should be available in rembg
            try:
                self.session = new_session(self.model_name)
                logger.info("background_remover_loaded", model_name=self.model_name)
                return
            except ValueError as ve:
                if "No session class found for model" in str(ve):
                    logger.warning(f"Required model {self.model_name} not available. Trying fallback models.", error=str(ve))

                    # Since BiRefNet is required but not available, try fallback models
                    # The pipeline can work with fallback models, but ideally BiRefNet should be available
                    fallback_models = ["u2net", "silueta", "isnet-general-use"]

                    for model in fallback_models:
                        try:
                            self.session = new_session(model)
                            logger.warning("using_fallback_model", model=model)
                            self.model_name = model  # Update to the model that actually worked
                            return
                        except Exception as fallback_error:
                            logger.warning("fallback_model_failed", model=model, error=str(fallback_error))
                            continue

                    # If all models fail, raise an error
                    raise ValueError(f"Primary model {self.model_name} not available and no fallback models worked. Please ensure rembg[birefnet] is installed: pip install rembg[birefnet]")
                else:
                    raise
            except Exception as e:
                logger.error("model_load_general_error", model=self.model_name, error=str(e))
                raise

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

