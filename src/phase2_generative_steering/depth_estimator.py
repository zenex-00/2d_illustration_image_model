"""ZoeDepth depth estimation"""

import numpy as np
from typing import Optional
import torch
from src.utils.logger import get_logger
from src.utils.error_handler import retry_on_failure, ModelLoadError
from src.pipeline.model_cache import get_model_cache

logger = get_logger(__name__)


class DepthEstimator:
    """ZoeDepth for depth map generation"""
    
    def __init__(self, model_type: str = "zoedepth-anywhere", device: str = "cuda"):
        """Initialize depth estimator (lazy loading)"""
        self.model_type = model_type
        self.device = device
        self.model = None
        # Models loaded lazily on first use
    
    @retry_on_failure(max_attempts=3, exceptions=(Exception,))
    def _load_model(self):
        """Load ZoeDepth model"""
        try:
            from zoedepth.models.builder import build_model
            from zoedepth.utils.config import get_config as get_zoedepth_config
            
            cache = get_model_cache()
            
            # Get config (using aliased import to avoid conflict with pipeline config)
            conf = get_zoedepth_config(self.model_type, "infer", pretrained_resource="local")
            
            # Build model
            self.model = build_model(conf)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("depth_estimator_loaded", model_type=self.model_type)
        except Exception as e:
            raise ModelLoadError(
                phase="phase2",
                message=f"Failed to load depth estimator: {str(e)}",
                original_error=e
            )
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from image
        
        Args:
            image: Input image as numpy array (RGB, uint8)
        
        Returns:
            Depth map as numpy array (grayscale, normalized 0-1)
        """
        if self.model is None:
            self._load_model()
        
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        try:
            from zoedepth.utils.misc import colorize
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Convert to PIL and preprocess
            pil_image = Image.fromarray(image).convert('RGB')
            
            # Preprocess
            transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor()
            ])
            
            input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                depth = self.model.infer(input_tensor)
            
            # Convert to numpy
            depth_np = depth.squeeze().cpu().numpy()
            
            # Normalize to 0-1
            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            
            # Resize back to original size
            import cv2
            h, w = image.shape[:2]
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
            
            logger.info("depth_estimated", depth_shape=depth_np.shape)
            
            return depth_np.astype(np.float32)
            
        except Exception as e:
            logger.error("depth_estimation_failed", error=str(e), exc_info=True)
            raise

