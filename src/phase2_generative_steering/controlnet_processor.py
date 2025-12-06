"""Multi-ControlNet processing for SDXL"""

import os
import numpy as np
import torch
from typing import Optional, Tuple, List

# Disable xformers if it's causing import issues (set before importing diffusers)
# Check if already disabled (set by server.py or environment)
if os.getenv("DISABLE_XFORMERS") != "1":
    try:
        import xformers
        try:
            from xformers.ops import fmha  # noqa: F401
            os.environ.setdefault("XFORMERS_DISABLED", "0")
        except Exception:
            os.environ["XFORMERS_DISABLED"] = "1"
            os.environ["DISABLE_XFORMERS"] = "1"
    except Exception:
        os.environ["XFORMERS_DISABLED"] = "1"
        os.environ["DISABLE_XFORMERS"] = "1"

from diffusers import ControlNetModel, UniPCMultistepScheduler
from src.utils.logger import get_logger
from src.utils.error_handler import retry_on_failure, ModelLoadError
from src.pipeline.model_cache import get_model_cache

logger = get_logger(__name__)


class ControlNetProcessor:
    """Multi-ControlNet processor for Depth and Canny"""
    
    def __init__(
        self,
        depth_model_id: str = "diffusers/controlnet-depth-sdxl-1.0",
        canny_model_id: str = "diffusers/controlnet-canny-sdxl-1.0",
        depth_weight: float = 0.6,
        canny_weight: float = 0.4,
        canny_step_off: float = 0.75,
        device: str = "cuda"
    ):
        """Initialize ControlNet processor (lazy loading)"""
        self.depth_model_id = depth_model_id
        self.canny_model_id = canny_model_id
        self.depth_weight = depth_weight
        self.canny_weight = canny_weight
        self.canny_step_off = canny_step_off
        self.device = device
        self.depth_controlnet = None
        self.canny_controlnet = None
        # Models loaded lazily on first use
    
    @retry_on_failure(max_attempts=3, exceptions=(Exception,))
    def _load_models(self):
        """Load ControlNet models"""
        try:
            cache = get_model_cache()
            
            # Load depth ControlNet
            self.depth_controlnet = cache.load_or_cache_model(
                model_id=self.depth_model_id,
                model_loader=lambda model_id, **kwargs: ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    **kwargs
                ),
                cache_key=f"controlnet_depth_{self.depth_model_id}"
            )
            
            # Load canny ControlNet
            self.canny_controlnet = cache.load_or_cache_model(
                model_id=self.canny_model_id,
                model_loader=lambda model_id, **kwargs: ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    **kwargs
                ),
                cache_key=f"controlnet_canny_{self.canny_model_id}"
            )
            
            logger.info("controlnets_loaded", depth_model=self.depth_model_id, canny_model=self.canny_model_id)
        except Exception as e:
            raise ModelLoadError(
                phase="phase2",
                message=f"Failed to load ControlNets: {str(e)}",
                original_error=e
            )
    
    def prepare_control_images(
        self,
        depth_map: np.ndarray,
        edge_map: np.ndarray,
        target_size: Tuple[int, int] = (1024, 1024)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare control images for ControlNet
        
        Args:
            depth_map: Depth map (normalized 0-1)
            edge_map: Edge map (binary)
            target_size: Target size for control images
        
        Returns:
            Tuple of (preprocessed_depth, preprocessed_canny)
        """
        # Ensure models are loaded
        if self.depth_controlnet is None or self.canny_controlnet is None:
            self._load_models()
        
        import cv2
        
        # Resize to target size
        depth_resized = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_LINEAR)
        edge_resized = cv2.resize(edge_map, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize depth to 0-255
        if depth_resized.max() <= 1.0:
            depth_resized = (depth_resized * 255).astype(np.uint8)
        else:
            depth_resized = depth_resized.astype(np.uint8)
        
        # Ensure edge map is uint8
        if edge_resized.max() <= 1.0:
            edge_resized = (edge_resized * 255).astype(np.uint8)
        else:
            edge_resized = edge_resized.astype(np.uint8)
        
        # Convert to 3-channel for ControlNet
        depth_3ch = np.stack([depth_resized] * 3, axis=-1)
        edge_3ch = np.stack([edge_resized] * 3, axis=-1)
        
        return depth_3ch, edge_3ch
    
    def get_controlnet_weights(self, step: int, num_steps: int) -> Tuple[float, float]:
        """
        Get ControlNet weights with canny step-off schedule
        
        Args:
            step: Current step
            num_steps: Total number of steps
        
        Returns:
            Tuple of (depth_weight, canny_weight)
        """
        step_ratio = step / num_steps
        
        if step_ratio >= self.canny_step_off:
            # Turn off canny after step_off
            return self.depth_weight, 0.0
        else:
            # Linear interpolation
            canny_weight = self.canny_weight * (1.0 - step_ratio / self.canny_step_off)
            return self.depth_weight, canny_weight

