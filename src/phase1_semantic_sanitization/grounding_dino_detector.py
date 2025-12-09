"""GroundingDINO object detection with retry logic and graceful degradation"""

import numpy as np
from typing import List, Tuple, Optional
import torch
from src.utils.logger import get_logger
from src.utils.error_handler import retry_on_failure, ModelLoadError, PhaseError
from src.pipeline.model_cache import get_model_cache

logger = get_logger(__name__)


class GroundingDINODetector:
    """GroundingDINO detector for open-set object detection"""
    
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda"
    ):
        """Initialize GroundingDINO detector (lazy loading)"""
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.model = None
        self.processor = None
        # Models loaded lazily on first use
    
    @retry_on_failure(max_attempts=3, exceptions=(Exception,))
    def _load_model(self):
        """Load GroundingDINO model with retry logic"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            cache = get_model_cache()
            self.model = cache.load_or_cache_model(
                model_id=self.model_id,
                model_loader=lambda model_id, **kwargs: AutoModelForZeroShotObjectDetection.from_pretrained(
                    model_id,
                    device_map=self.device,
                    **kwargs
                ),
                cache_key=f"grounding_dino_{self.model_id}"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            logger.info("grounding_dino_loaded", model_id=self.model_id)
        except Exception as e:
            raise ModelLoadError(
                phase="phase1",
                message=f"Failed to load GroundingDINO: {str(e)}",
                original_error=e
            )
    
    def detect_objects(
        self,
        image: np.ndarray,
        prompts: List[str],
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect objects in image based on text prompts
        
        Args:
            image: Input image as numpy array (RGB)
            prompts: List of text prompts for detection
            box_threshold: Override default box threshold
            text_threshold: Override default text threshold
        
        Returns:
            List of bounding boxes as (x1, y1, x2, y2)
        """
        if self.model is None:
            self._load_model()
        
        box_thresh = box_threshold or self.box_threshold
        text_thresh = text_threshold or self.text_threshold
        
        # Convert numpy array to PIL Image
        from PIL import Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image
        
        # Combine prompts
        text_prompt = ". ".join(prompts) + "."
        
        try:
            # Process inputs
            inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results - fix API for current transformers version
            # The API has changed - input_ids is not needed in newer versions
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                box_threshold=box_thresh,
                text_threshold=text_thresh,
                target_sizes=[pil_image.size[::-1]]
            )
            
            # Extract bounding boxes
            boxes = []
            if results and len(results) > 0:
                result = results[0]
                for box in result["boxes"]:
                    x1, y1, x2, y2 = box.tolist()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            logger.info(
                "grounding_dino_detection",
                num_detections=len(boxes),
                prompts=prompts,
                box_threshold=box_thresh
            )
            
            return boxes
            
        except Exception as e:
            logger.error("grounding_dino_detection_failed", error=str(e), exc_info=True)
            # Graceful degradation: return empty list if detection fails
            return []

