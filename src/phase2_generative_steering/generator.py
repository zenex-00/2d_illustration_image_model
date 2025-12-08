"""Phase II orchestrator: Generative Steering"""

import numpy as np
from typing import Optional, Tuple
from datetime import datetime
from src.utils.logger import get_logger, set_correlation_id
from src.utils.error_handler import PhaseError
from src.utils.metrics import MetricsCollector
from src.pipeline.config import get_config
from .background_remover import BackgroundRemover
from .depth_estimator import DepthEstimator
from .edge_detector import EdgeDetector
from .sdxl_generator import SDXLGenerator

logger = get_logger(__name__)


class Phase2Generator:
    """Orchestrator for Phase II: Generative Steering"""
    
    def __init__(self, config=None):
        """Initialize Phase II components (lazy loading)"""
        self.config = config or get_config()
        phase_config = self.config.get_phase_config("phase2")
        hardware_config = self.config.get_hardware_config()
        
        self.device = hardware_config.get("device", "cuda")
        self.phase_config = phase_config
        
        # Store configuration for lazy initialization
        self.bg_remover = None
        self.depth_estimator = None
        self.edge_detector = None
        self.sdxl_generator = None
        
        logger.info("phase2_initialized_lazy")
        self._last_edge_map = None  # Store edge map for Phase IV
    
    def generate(
        self,
        clean_plate: np.ndarray,
        metrics_collector: Optional[MetricsCollector] = None,
        correlation_id: Optional[str] = None,
        seed: Optional[int] = None,
        controlnet_weights_override: Optional[dict] = None,
        prompt_override: Optional[str] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Perform generative steering: bg removal → depth/edge → SDXL generation
        
        Args:
            clean_plate: Clean plate image from Phase I (RGB, uint8)
            metrics_collector: Optional metrics collector
            correlation_id: Optional correlation ID for logging
            seed: Random seed for reproducibility
            controlnet_weights_override: Optional dict with 'depth_weight' and 'canny_weight' to override config
        
        Returns:
            Tuple of (vector_raster, metadata_dict)
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        start_time = datetime.now()
        phase_config = self.config.get_phase_config("phase2")
        
        try:
            logger.info("phase2_start", input_shape=clean_plate.shape)
            
            # Lazy load components if not already loaded
            if self.bg_remover is None:
                self.bg_remover = BackgroundRemover(
                    model_name=self.phase_config["background_removal"]["model"],
                    device=self.device
                )
            
            if self.depth_estimator is None:
                self.depth_estimator = DepthEstimator(
                    model_type=self.phase_config["depth_estimation"]["model"],
                    device=self.device
                )
            
            if self.edge_detector is None:
                self.edge_detector = EdgeDetector()
            
            if self.sdxl_generator is None:
                self.sdxl_generator = SDXLGenerator(
                    base_model_id=self.phase_config["sdxl"]["base_model"],
                    lora_path=self.phase_config["lora"].get("path"),
                    lora_scale=self.phase_config["lora"].get("scale", 0.8),
                    device=self.device,
                    num_inference_steps=self.phase_config["sdxl"]["num_inference_steps"],
                    guidance_scale=self.phase_config["sdxl"]["guidance_scale"],
                    negative_prompt=self.phase_config["sdxl"]["negative_prompt"]
                )
            
            # Step 1: Remove background
            bg_removed = self.bg_remover.remove_background(clean_plate)
            
            # Handle different image formats with comprehensive shape validation
            if len(bg_removed.shape) == 3:
                if bg_removed.shape[2] == 4:  # RGBA
                    rgb_image = bg_removed[:, :, :3]
                elif bg_removed.shape[2] == 3:  # RGB
                    rgb_image = bg_removed
                else:
                    raise ValueError(f"Unexpected image channels: {bg_removed.shape[2]}. Expected 3 (RGB) or 4 (RGBA)")
            elif len(bg_removed.shape) == 2:  # Grayscale
                # Convert grayscale to RGB by stacking channels
                rgb_image = np.stack([bg_removed] * 3, axis=-1)
            else:
                raise ValueError(f"Unexpected image shape: {bg_removed.shape}. Expected 2D (grayscale) or 3D (RGB/RGBA)")
            
            # Step 2: Generate depth map
            depth_map = self.depth_estimator.estimate_depth(rgb_image)
            
            # Step 3: Detect edges
            edge_map = self.edge_detector.detect_edges(rgb_image)
            
            # Store edge_map for potential use in Phase IV (Strategy B)
            self._last_edge_map = edge_map
            
            # Step 4: Generate vector-style image with SDXL
            # Use prompt override if provided, otherwise use default
            if prompt_override:
                prompt = prompt_override
            else:
                prompt = "flat vector illustration, minimalist style, clean lines, solid colors"
                if phase_config["lora"].get("path"):
                    prompt += " <flt_vctr_style>"
            
            control_images = [depth_map, edge_map]
            
            # Use override weights if provided, otherwise use config
            if controlnet_weights_override:
                control_weights = [
                    controlnet_weights_override.get("depth_weight", phase_config["controlnet"]["depth_weight"]),
                    controlnet_weights_override.get("canny_weight", phase_config["controlnet"]["canny_weight"])
                ]
            else:
                control_weights = [
                    phase_config["controlnet"]["depth_weight"],
                    phase_config["controlnet"]["canny_weight"]
                ]
            
            vector_raster = self.sdxl_generator.generate(
                prompt=prompt,
                control_images=control_images,
                control_weights=control_weights,
                seed=seed or self.config.random_seed
            )
            
            end_time = datetime.now()
            
            metadata = {
                "output_shape": vector_raster.shape,
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000
            }
            
            if metrics_collector:
                metrics_collector.record_phase(
                    phase_name="phase2_generative_steering",
                    start_time=start_time,
                    end_time=end_time,
                    input_shape=clean_plate.shape,
                    output_shape=vector_raster.shape
                )
            
            logger.info("phase2_complete", **metadata)
            
            # Add edge_map to metadata for Phase IV
            metadata["edge_map"] = self._last_edge_map
            
            # Flush Phase 2 models from memory
            import gc
            import torch
            from src.pipeline.model_cache import get_model_cache
            
            cache = get_model_cache()
            cache.clear_cache(phase_prefix="controlnet_")
            cache.clear_cache(phase_prefix="vae_")
            cache.clear_cache(phase_prefix="sdxl")
            
            # Comprehensive cleanup of component references
            if self.bg_remover:
                self.bg_remover = None
            
            if self.depth_estimator:
                self.depth_estimator = None
            
            if self.edge_detector:
                self.edge_detector = None
            
            if self.sdxl_generator:
                # Clear SDXL pipeline internals explicitly
                if hasattr(self.sdxl_generator, 'pipe'):
                    if hasattr(self.sdxl_generator.pipe, 'unet'):
                        self.sdxl_generator.pipe.unet = None
                    if hasattr(self.sdxl_generator.pipe, 'vae'):
                        self.sdxl_generator.pipe.vae = None
                    if hasattr(self.sdxl_generator.pipe, 'text_encoder'):
                        self.sdxl_generator.pipe.text_encoder = None
                    if hasattr(self.sdxl_generator.pipe, 'text_encoder_2'):
                        self.sdxl_generator.pipe.text_encoder_2 = None
                    self.sdxl_generator.pipe = None
                # Clear controlnet_processor reference
                if hasattr(self.sdxl_generator, 'controlnet_processor'):
                    self.sdxl_generator.controlnet_processor = None
                self.sdxl_generator = None
            
            # Clear stored edge map numpy array
            self._last_edge_map = None
            
            # Force garbage collection and CUDA cache clearing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("phase2_memory_flushed")
            
            return vector_raster, metadata
            
        except Exception as e:
            logger.error("phase2_failed", error=str(e), exc_info=True)
            raise PhaseError(
                phase="phase2",
                message=f"Generative steering failed: {str(e)}",
                original_error=e
            )

