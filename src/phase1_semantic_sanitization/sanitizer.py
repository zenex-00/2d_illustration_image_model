"""Phase I orchestrator: Semantic Sanitization"""

import numpy as np
from typing import Optional, Tuple
from datetime import datetime
from src.utils.logger import get_logger, set_correlation_id
from src.utils.error_handler import PhaseError
from src.utils.quality_assurance import validate_geometric_similarity, extract_alpha_mask
from src.utils.metrics import MetricsCollector
from src.pipeline.config import get_config
from .grounding_dino_detector import GroundingDINODetector
from .sam_segmenter import SAMSegmenter
from .lama_inpainter import LaMaInpainter

logger = get_logger(__name__)


class Phase1Sanitizer:
    """Orchestrator for Phase I: Semantic Sanitization"""
    
    def __init__(self, config=None):
        """Initialize Phase I components (lazy loading)"""
        self.config = config or get_config()
        phase_config = self.config.get_phase_config("phase1")
        hardware_config = self.config.get_hardware_config()
        
        self.device = hardware_config.get("device", "cuda")
        self.phase_config = phase_config
        
        # Store configuration for lazy initialization
        self.detector = None
        self.segmenter = None
        self.inpainter = None
        
        logger.info("phase1_initialized_lazy")
    
    def sanitize(
        self,
        image: np.ndarray,
        metrics_collector: Optional[MetricsCollector] = None,
        correlation_id: Optional[str] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Perform semantic sanitization: detect, segment, and inpaint prohibited elements
        
        Args:
            image: Input image as numpy array (RGB, uint8)
            metrics_collector: Optional metrics collector
            correlation_id: Optional correlation ID for logging
        
        Returns:
            Tuple of (clean_plate_image, metadata_dict)
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        start_time = datetime.now()
        phase_config = self.config.get_phase_config("phase1")
        
        try:
            logger.info("phase1_start", input_shape=image.shape)
            
            # Lazy load components if not already loaded
            if self.detector is None:
                self.detector = GroundingDINODetector(
                    model_id=self.phase_config["grounding_dino"]["model_id"],
                    box_threshold=self.phase_config["grounding_dino"]["box_threshold"],
                    text_threshold=self.phase_config["grounding_dino"]["text_threshold"],
                    device=self.device
                )
            
            if self.segmenter is None:
                self.segmenter = SAMSegmenter(
                    model_type=self.phase_config["sam"]["model"],
                    checkpoint_path=self.phase_config["sam"].get("checkpoint"),
                    device=self.device,
                    dilation_kernel=self.phase_config["sam"]["dilation_kernel"]
                )
            
            if self.inpainter is None:
                self.inpainter = LaMaInpainter(
                    model_name=self.phase_config["lama"]["model"],
                    device=self.device,
                    quality_threshold=self.phase_config.get("quality_threshold", 0.97)
                )
            
            # Step 1: Detect prohibited objects
            prompts = phase_config["grounding_dino"]["prompts"]
            boxes = self.detector.detect_objects(image, prompts)
            
            if len(boxes) == 0:
                logger.info("phase1_no_detections", message="No prohibited objects detected")
                # Return original image if nothing to remove
                end_time = datetime.now()
                if metrics_collector:
                    metrics_collector.record_phase(
                        phase_name="phase1_semantic_sanitization",
                        start_time=start_time,
                        end_time=end_time,
                        input_shape=image.shape,
                        output_shape=image.shape
                    )
                return image, {"detections": 0, "boxes": []}
            
            # Step 2: Generate precise masks
            mask = self.segmenter.generate_masks(image, boxes)
            
            if mask.sum() == 0:
                logger.warning("phase1_empty_mask", message="Generated mask is empty")
                return image, {"detections": len(boxes), "boxes": boxes, "mask_area": 0}
            
            # Step 3: Inpaint masked regions
            clean_plate = self.inpainter.inpaint(image, mask)
            
            # Step 4: Validate output quality
            # Extract masks for comparison
            original_mask = extract_alpha_mask(image)
            generated_mask = extract_alpha_mask(clean_plate)
            
            try:
                validate_geometric_similarity(
                    original_mask,
                    generated_mask,
                    threshold=phase_config.get("quality_threshold", 0.97),
                    phase="phase1"
                )
            except Exception as e:
                logger.warning("phase1_quality_check_failed", error=str(e), exc_info=True)
                # Continue anyway, but log the issue
            
            end_time = datetime.now()
            
            metadata = {
                "detections": len(boxes),
                "boxes": boxes,
                "mask_area": int(mask.sum()),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000
            }
            
            if metrics_collector:
                metrics_collector.record_phase(
                    phase_name="phase1_semantic_sanitization",
                    start_time=start_time,
                    end_time=end_time,
                    input_shape=image.shape,
                    output_shape=clean_plate.shape
                )
            
            logger.info("phase1_complete", **metadata)
            
            # Flush Phase 1 models from memory
            import gc
            import torch
            from src.pipeline.model_cache import get_model_cache
            
            cache = get_model_cache()
            cache.clear_cache(phase_prefix="grounding_dino")
            cache.clear_cache(phase_prefix="sam_")
            cache.clear_cache(phase_prefix="lama")
            
            # Clear component references
            del self.detector
            del self.segmenter
            del self.inpainter
            self.detector = None
            self.segmenter = None
            self.inpainter = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("phase1_memory_flushed")
            
            return clean_plate, metadata
            
        except Exception as e:
            logger.error("phase1_failed", error=str(e), exc_info=True)
            raise PhaseError(
                phase="phase1",
                message=f"Semantic sanitization failed: {str(e)}",
                original_error=e
            )


