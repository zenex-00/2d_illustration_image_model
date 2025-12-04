"""Phase III orchestrator: Chromatic Enforcement"""

import numpy as np
from typing import Optional, Tuple
from datetime import datetime
from src.utils.logger import get_logger, set_correlation_id
from src.utils.error_handler import PhaseError
from src.utils.metrics import MetricsCollector
from src.pipeline.config import get_config
from .upscaler import Upscaler
from .color_quantizer import ColorQuantizer
from .noise_remover import NoiseRemover

logger = get_logger(__name__)


class Phase3Enforcer:
    """Orchestrator for Phase III: Chromatic Enforcement"""
    
    def __init__(self, config=None):
        """Initialize Phase III components (lazy loading)"""
        self.config = config or get_config()
        phase_config = self.config.get_phase_config("phase3")
        hardware_config = self.config.get_hardware_config()
        
        self.device = hardware_config.get("device", "cuda")
        self.phase_config = phase_config
        
        # Store configuration for lazy initialization
        self.upscaler = None
        self.color_quantizer = None
        self.noise_remover = None
        
        logger.info("phase3_initialized_lazy")
    
    def enforce(
        self,
        vector_raster: np.ndarray,
        metrics_collector: Optional[MetricsCollector] = None,
        correlation_id: Optional[str] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Perform chromatic enforcement: upscale → quantize → denoise
        
        Args:
            vector_raster: Vector-style raster from Phase II (RGB, uint8)
            metrics_collector: Optional metrics collector
            correlation_id: Optional correlation ID for logging
        
        Returns:
            Tuple of (quantized_image_4096px, metadata_dict)
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        start_time = datetime.now()
        
        try:
            logger.info("phase3_start", input_shape=vector_raster.shape)
            
            # Lazy load components if not already loaded
            if self.upscaler is None:
                self.upscaler = Upscaler(
                    model_name=self.phase_config["upscaler"]["model"],
                    scale=self.phase_config["upscaler"]["scale"],
                    device=self.device
                )
            
            if self.color_quantizer is None:
                self.color_quantizer = ColorQuantizer()
            
            if self.noise_remover is None:
                self.noise_remover = NoiseRemover(
                    min_area_percent=self.phase_config["noise_removal"]["min_area_percent"]
                )
            
            # Step 1: Upscale to 4096px
            upscaled = self.upscaler.upscale(vector_raster)
            
            # Step 2: Quantize to 15-color palette
            quantized = self.color_quantizer.quantize(upscaled, validate=True)
            
            # Step 3: Remove noise
            denoised = self.noise_remover.remove_noise(quantized)
            
            # Optionally create 2048px version
            from src.utils.image_utils import resize_image
            preview_2048 = resize_image(denoised, 2048)
            
            end_time = datetime.now()
            
            metadata = {
                "output_shape_4096": denoised.shape,
                "output_shape_2048": preview_2048.shape,
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000
            }
            
            if metrics_collector:
                metrics_collector.record_phase(
                    phase_name="phase3_chromatic_enforcement",
                    start_time=start_time,
                    end_time=end_time,
                    input_shape=vector_raster.shape,
                    output_shape=denoised.shape
                )
            
            logger.info("phase3_complete", **metadata)
            
            # Flush Phase 3 models from memory
            import gc
            import torch
            from src.pipeline.model_cache import get_model_cache
            
            cache = get_model_cache()
            cache.clear_cache(phase_prefix="realesrgan")
            cache.clear_cache(phase_prefix="upscaler")
            
            # Clear component references
            del self.upscaler
            del self.color_quantizer
            del self.noise_remover
            self.upscaler = None
            self.color_quantizer = None
            self.noise_remover = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("phase3_memory_flushed")
            
            return denoised, metadata
            
        except Exception as e:
            logger.error("phase3_failed", error=str(e), exc_info=True)
            raise PhaseError(
                phase="phase3",
                message=f"Chromatic enforcement failed: {str(e)}",
                original_error=e
            )


