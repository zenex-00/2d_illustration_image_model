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

        # Load training configuration for retry logic
        self.training_config = self.config.get("training", {})
        self.phase1_retry_config = self.training_config.get("phase1_retry", {})

        # Store configuration for lazy initialization
        self.detector = None
        self.segmenter = None
        self.inpainter = None

        logger.info("phase1_initialized_lazy")

    def _try_with_parameters(self, image: np.ndarray, params: dict) -> Tuple[np.ndarray, dict]:
        """Try sanitization with specific parameters."""
        # Reset component references to force reload with new parameters
        self.detector = None
        self.segmenter = None
        self.inpainter = None

        # Lazy load components if not already loaded
        if self.detector is None:
            self.detector = GroundingDINODetector(
                model_id=self.phase_config["grounding_dino"]["model_id"],
                box_threshold=params.get("box_threshold", self.phase_config["grounding_dino"]["box_threshold"]),
                text_threshold=params.get("text_threshold", self.phase_config["grounding_dino"]["text_threshold"]),
                device=self.device
            )

        if self.segmenter is None:
            sam_model = params.get("sam_model", self.phase_config["sam"]["model"])
            checkpoint_path = params.get("checkpoint_path", self.phase_config["sam"].get("checkpoint"))
            dilation_kernel = params.get("dilation_kernel", self.phase_config["sam"]["dilation_kernel"])

            self.segmenter = SAMSegmenter(
                model_type=sam_model,
                checkpoint_path=checkpoint_path,
                device=self.device,
                dilation_kernel=dilation_kernel
            )

        if self.inpainter is None:
            self.inpainter = LaMaInpainter(
                model_name=self.phase_config["lama"]["model"],
                device=self.device,
                quality_threshold=self.phase_config.get("quality_threshold", 0.97)
            )

        # Step 1: Detect prohibited objects
        prompts = self.phase_config["grounding_dino"]["prompts"]
        boxes = self.detector.detect_objects(image, prompts)

        if len(boxes) == 0:
            logger.info("phase1_no_detections", message="No prohibited objects detected")
            # Return original image if nothing to remove
            return image, {"detections": 0, "boxes": [], "mask_area": 0, "parameters_used": params}

        # Step 2: Generate precise masks
        mask = self.segmenter.generate_masks(image, boxes)

        if mask.sum() == 0:
            logger.warning("phase1_empty_mask", message="Generated mask is empty")
            return image, {"detections": len(boxes), "boxes": boxes, "mask_area": 0, "parameters_used": params}

        # Step 3: Inpaint masked regions
        clean_plate = self.inpainter.inpaint(image, mask)

        # Check if output is significantly different from input (avoid no-op results)
        if np.array_equal(image, clean_plate):
            logger.warning("phase1_no_change", message="Output identical to input - no sanitization occurred")
            return image, {"detections": len(boxes), "boxes": boxes, "mask_area": int(mask.sum()), "parameters_used": params, "no_change": True}

        # Additional validation: Check if inpainting actually modified the masked regions
        # If the inpainted regions are very similar to the original, it might indicate a problem
        if len(boxes) > 0 and mask.sum() > 0:
            # Compare the masked regions before and after inpainting
            masked_original = image * mask[..., np.newaxis]
            masked_result = clean_plate * mask[..., np.newaxis]

            # Calculate difference in the masked region
            diff = np.mean(np.abs(masked_original.astype(np.float32) - masked_result.astype(np.float32)))

            if diff < 5.0:  # Threshold for minimal change (adjustable)
                logger.warning("phase1_low_change_in_masked_region",
                              message=f"Low change in masked region (diff: {diff:.2f}), may indicate ineffective inpainting",
                              diff=diff)
                # Still return the result but flag it

        # Step 4: Validate output quality
        # Extract masks for comparison
        original_mask = extract_alpha_mask(image)
        generated_mask = extract_alpha_mask(clean_plate)

        try:
            validate_geometric_similarity(
                original_mask,
                generated_mask,
                threshold=self.phase_config.get("quality_threshold", 0.97),
                phase="phase1"
            )
        except Exception as e:
            logger.warning("phase1_quality_check_failed", error=str(e), exc_info=True)
            # Continue anyway, but log the issue

        metadata = {
            "detections": len(boxes),
            "boxes": boxes,
            "mask_area": int(mask.sum()),
            "parameters_used": params
        }

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

        return clean_plate, metadata

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

        # Get retry configuration
        max_retries = self.phase1_retry_config.get("max_retries", 3)
        retry_on_no_change = self.phase1_retry_config.get("retry_on_no_change", True)
        parameter_variations = self.phase1_retry_config.get("parameter_variations", [])
        fallback_models = self.phase1_retry_config.get("fallback_models", [])

        # Combine parameter variations and fallback models
        all_parameter_combinations = parameter_variations.copy()
        for fallback in fallback_models:
            # Add fallback model combinations with default parameters
            all_parameter_combinations.append(fallback)

        # Add default parameters as first attempt
        default_params = {
            "box_threshold": self.phase_config["grounding_dino"]["box_threshold"],
            "text_threshold": self.phase_config["grounding_dino"]["text_threshold"]
        }
        all_parameter_combinations.insert(0, default_params)

        best_result = None
        best_metadata = None
        last_error = None

        logger.info("phase1_start", input_shape=image.shape, max_retries=max_retries)

        # Try each parameter combination
        for attempt_idx, params in enumerate(all_parameter_combinations[:max_retries]):
            try:
                logger.info(f"phase1_attempt_{attempt_idx+1}", parameters=params)

                result, metadata = self._try_with_parameters(image, params)

                # Check if this result is better than previous attempts
                # Prefer results with actual detections and changes over no-op results
                is_better_result = (
                    best_result is None or  # First successful result
                    (metadata.get("detections", 0) > 0 and metadata.get("mask_area", 0) > 0 and
                     not metadata.get("no_change", False))  # Prefer actual sanitization over no-op
                )

                if is_better_result:
                    best_result = result
                    best_metadata = metadata

                    # If we got actual sanitization (detections and changes), consider it successful
                    if (metadata.get("detections", 0) > 0 and
                        metadata.get("mask_area", 0) > 0 and
                        not metadata.get("no_change", False)):
                        logger.info(f"phase1_successful_sanitization",
                                   attempt=attempt_idx+1,
                                   detections=metadata["detections"],
                                   mask_area=metadata["mask_area"])
                        break

                # If we're not retrying on no-change and we got a good result, exit early
                if (metadata.get("detections", 0) > 0 and
                    metadata.get("mask_area", 0) > 0 and
                    not metadata.get("no_change", False) and
                    not retry_on_no_change):
                    logger.info(f"phase1_successful_sanitization_no_retry",
                               attempt=attempt_idx+1,
                               detections=metadata["detections"],
                               mask_area=metadata["mask_area"])
                    break

            except Exception as e:
                logger.warning(f"phase1_attempt_{attempt_idx+1}_failed", error=str(e), exc_info=True)
                last_error = e
                continue  # Try next parameter combination

        # If no successful attempts were made, raise the last error
        if best_result is None:
            logger.error("phase1_all_attempts_failed", error=str(last_error), exc_info=True)
            raise PhaseError(
                phase="phase1",
                message=f"Semantic sanitization failed after {max_retries} attempts: {str(last_error)}",
                original_error=last_error
            )

        # Log final result
        end_time = datetime.now()
        best_metadata["processing_time_ms"] = (end_time - start_time).total_seconds() * 1000
        best_metadata["attempts_made"] = min(len(all_parameter_combinations), max_retries)

        if metrics_collector:
            metrics_collector.record_phase(
                phase_name="phase1_semantic_sanitization",
                start_time=start_time,
                end_time=end_time,
                input_shape=image.shape,
                output_shape=best_result.shape
            )

        logger.info("phase1_complete", **best_metadata)
        logger.info("phase1_memory_flushed")

        return best_result, best_metadata


