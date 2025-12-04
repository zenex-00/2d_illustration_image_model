"""SAM segmentation with error recovery and mask validation"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import torch
from src.utils.logger import get_logger
from src.utils.error_handler import retry_on_failure, ModelLoadError
from src.pipeline.model_cache import get_model_cache

logger = get_logger(__name__)


class SAMSegmenter:
    """Segment Anything Model (SAM) for precise segmentation"""
    
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        dilation_kernel: int = 5
    ):
        """Initialize SAM segmenter (lazy loading)"""
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dilation_kernel = dilation_kernel
        self.predictor = None
        # Models loaded lazily on first use
    
    @retry_on_failure(max_attempts=3, exceptions=(Exception,))
    def _load_model(self):
        """Load SAM model with retry logic"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            if self.checkpoint_path is None:
                # Default checkpoint path
                self.checkpoint_path = f"sam_{self.model_type}_4b8939.pth"
            
            cache = get_model_cache()
            
            # Load SAM model
            # Note: model_id is passed as first argument to model_loader lambda
            sam = cache.load_or_cache_model(
                model_id=self.checkpoint_path,
                model_loader=lambda model_id, **kwargs: sam_model_registry[self.model_type](checkpoint=model_id),
                cache_key=f"sam_{self.model_type}"
            )
            
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            
            logger.info("sam_loaded", model_type=self.model_type, checkpoint=self.checkpoint_path)
        except Exception as e:
            raise ModelLoadError(
                phase="phase1",
                message=f"Failed to load SAM: {str(e)}",
                original_error=e
            )
    
    def generate_masks(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        dilation_kernel: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate segmentation masks from bounding boxes
        
        Args:
            image: Input image as numpy array (RGB)
            boxes: List of bounding boxes as (x1, y1, x2, y2)
            dilation_kernel: Override default dilation kernel size
        
        Returns:
            Combined binary mask
        """
        if self.predictor is None:
            self._load_model()
        
        if len(boxes) == 0:
            logger.warning("sam_no_boxes", message="No boxes provided, returning empty mask")
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Convert RGB to BGR for SAM
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            # Set image
            self.predictor.set_image(image_bgr)
            
            # Convert boxes to SAM format (x1, y1, x2, y2)
            input_boxes = torch.tensor(boxes, device=self.device)
            
            # Generate masks
            masks, scores, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=input_boxes,
                multimask_output=False
            )
            
            # Combine all masks
            combined_mask = masks.sum(dim=0).cpu().numpy().astype(np.uint8)
            combined_mask = np.clip(combined_mask, 0, 1)
            
            # Validate mask quality
            if combined_mask.sum() == 0:
                logger.warning("sam_empty_mask", message="Generated mask is empty")
                return np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Apply dilation for surgical buffer
            kernel_size = dilation_kernel or self.dilation_kernel
            if kernel_size > 0:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
            
            logger.info(
                "sam_masks_generated",
                num_boxes=len(boxes),
                mask_area=combined_mask.sum(),
                dilation_kernel=kernel_size
            )
            
            return combined_mask.astype(np.uint8)
            
        except Exception as e:
            logger.error("sam_segmentation_failed", error=str(e), exc_info=True)
            # Return empty mask on failure
            return np.zeros(image.shape[:2], dtype=np.uint8)

