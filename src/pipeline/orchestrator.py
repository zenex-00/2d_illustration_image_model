"""Main pipeline orchestrator with quality assurance"""

import numpy as np
import torch
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import uuid

from src.utils.logger import get_logger, set_correlation_id, setup_logging
from src.utils.error_handler import PipelineError, PhaseError, ValidationError
from src.utils.metrics import MetricsCollector
from src.utils.quality_assurance import validate_geometric_similarity, audit_palette_colors, extract_alpha_mask
from src.utils.palette_manager import get_palette
from src.utils.image_utils import load_image, save_image, resize_image
from src.pipeline.config import get_config

from src.phase1_semantic_sanitization.sanitizer import Phase1Sanitizer
from src.phase2_generative_steering.generator import Phase2Generator
from src.phase3_chromatic_enforcement.enforcer import Phase3Enforcer
from src.phase4_vector_reconstruction.vectorizer import Phase4Vectorizer

logger = get_logger(__name__)


def extract_iou_from_error(error: Exception) -> Optional[float]:
    """
    Safely extract IoU value from ValidationError exception.
    
    Args:
        error: Exception object that may contain IoU information
        
    Returns:
        IoU value as float if found, None otherwise
    """
    try:
        error_str = str(error)
        if "IoU " not in error_str:
            return None
        parts = error_str.split("IoU ")
        if len(parts) < 2:
            return None
        iou_str = parts[1].split(" <")[0].strip()
        return float(iou_str)
    except (ValueError, IndexError, AttributeError):
        return None


def _validate_config_overrides(config_overrides: Dict[str, Any]) -> None:
    """
    Validate config_overrides structure before merging.
    
    Args:
        config_overrides: Dictionary containing phase configuration overrides
        
    Raises:
        ValueError: If config_overrides structure is invalid
    """
    if not isinstance(config_overrides, dict):
        raise ValueError(f"config_overrides must be a dict, got {type(config_overrides).__name__}")
    
    if "phases" in config_overrides:
        phases = config_overrides["phases"]
        if not isinstance(phases, dict):
            raise ValueError(f"config_overrides['phases'] must be a dict, got {type(phases).__name__}")
        
        # Validate phase keys and enabled flags
        valid_phases = {"phase1", "phase2", "phase3", "phase4"}
        for phase_key, phase_config in phases.items():
            if phase_key not in valid_phases:
                raise ValueError(f"Invalid phase key: {phase_key}. Must be one of {valid_phases}")
            
            if not isinstance(phase_config, dict):
                raise ValueError(f"Phase config for {phase_key} must be a dict, got {type(phase_config).__name__}")
            
            if "enabled" in phase_config and not isinstance(phase_config["enabled"], bool):
                raise ValueError(f"Phase {phase_key} 'enabled' flag must be a bool, got {type(phase_config['enabled']).__name__}")


class Gemini3Pipeline:
    """Main pipeline orchestrator for Gemini 3 Pro architecture"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with all phases (lazy loading)"""
        self.config = get_config(config_path)
        
        # Set random seeds for reproducibility
        seed = self.config.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize phases (lazy loading - models loaded on first use)
        self.phase1 = Phase1Sanitizer(self.config)
        self.phase2 = Phase2Generator(self.config)
        self.phase3 = Phase3Enforcer(self.config)
        self.phase4 = Phase4Vectorizer(self.config)
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()
        
        logger.info("pipeline_initialized_lazy", version=self.config.get("pipeline.version"))
    
    def process_image(
        self,
        input_image_path: str,
        palette_hex_list: Optional[List[str]] = None,
        output_svg_path: Optional[str] = None,
        output_png_path: Optional[str] = None,
        save_intermediates: bool = False,
        intermediate_dir: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process image through full pipeline
        
        Args:
            input_image_path: Path to input image
            palette_hex_list: Optional custom palette (defaults to config)
            output_svg_path: Optional output path for SVG
            output_png_path: Optional output path for PNG preview
            save_intermediates: Whether to save intermediate phase outputs
            intermediate_dir: Directory for intermediate outputs
            config_overrides: Optional dict to override phase enabled flags
        
        Returns:
            Tuple of (svg_xml_string, metadata_dict)
        """
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)
        self.metrics_collector.set_correlation_id(correlation_id)
        
        pipeline_start = datetime.now()
        
        # Validate and merge config overrides
        phase_overrides = {}
        if config_overrides:
            _validate_config_overrides(config_overrides)
            phase_overrides = config_overrides.get("phases", {})
        
        try:
            logger.info("pipeline_start", input_path=input_image_path, correlation_id=correlation_id)
            
            # Load input image
            raw_img = load_image(input_image_path)
            logger.info("image_loaded", shape=raw_img.shape)
            
            # Get palette
            if palette_hex_list:
                from src.utils.palette_manager import PaletteManager
                try:
                    # Create a temporary palette manager instance with custom colors
                    # Use object.__new__ to avoid file loading, but with proper error handling
                    palette = object.__new__(PaletteManager)
                    palette.hex_colors = palette_hex_list
                    palette._validate_palette()  # Validate custom palette
                    palette.rgb_colors = palette._hex_to_rgb(palette_hex_list)
                except (ValueError, TypeError, AttributeError) as e:
                    logger.error(
                        "invalid_palette_fallback",
                        error=str(e),
                        palette_hex_list=palette_hex_list,
                        exc_info=True
                    )
                    # Fallback to default palette
                    palette = get_palette()
            else:
                palette = get_palette()
            
            # Phase I: Semantic Sanitization
            phase1_enabled = phase_overrides.get("phase1", {}).get("enabled", 
                self.config.get_phase_config("phase1").get("enabled", True))
            
            if phase1_enabled:
                logger.info("phase1_starting")
                clean_plate, phase1_metadata = self.phase1.sanitize(
                    raw_img,
                    metrics_collector=self.metrics_collector,
                    correlation_id=correlation_id
                )
            else:
                logger.info("phase1_skipped")
                clean_plate = raw_img
                phase1_metadata = {"skipped": True}
            
            if save_intermediates and intermediate_dir:
                from src.utils.path_validation import validate_intermediate_dir
                validated_dir = validate_intermediate_dir(intermediate_dir)
                save_image(clean_plate, validated_dir / "phase1_clean_plate.png")
            
            # Phase II: Generative Steering with IoU auto-retry
            phase2_enabled = phase_overrides.get("phase2", {}).get("enabled",
                self.config.get_phase_config("phase2").get("enabled", True))
            
            if not phase2_enabled:
                logger.info("phase2_skipped")
                vector_raster = clean_plate
                phase2_metadata = {"skipped": True}
            else:
                logger.info("phase2_starting")
                
                # Extract alpha mask from original for IoU validation
                original_mask = extract_alpha_mask(raw_img)
                
                # Get ControlNet weights from config overrides or config
                phase2_overrides = phase_overrides.get("phase2", {})
                phase2_config = self.config.get_phase_config("phase2")
                
                controlnet_overrides = phase2_overrides.get("controlnet", {})
                depth_weight = controlnet_overrides.get("depth_weight", phase2_config["controlnet"]["depth_weight"])
                canny_weight = controlnet_overrides.get("canny_weight", phase2_config["controlnet"]["canny_weight"])
                
                # Get prompt override if provided
                prompt_override = phase2_overrides.get("prompt_override")
                
                # Get retry configuration from config
                iou_retry_config = phase2_config.get("iou_retry", {})
                max_retries = iou_retry_config.get("max_retries", 2)
                iou_threshold = iou_retry_config.get("iou_threshold", 0.85)
                retry_count = 0
                
                while retry_count <= max_retries:
                    # Build controlnet weights override
                    controlnet_weights_override = {
                        "depth_weight": depth_weight,
                        "canny_weight": canny_weight
                    } if retry_count > 0 or controlnet_overrides else None
                    
                    vector_raster, phase2_metadata = self.phase2.generate(
                        clean_plate,
                        metrics_collector=self.metrics_collector,
                        correlation_id=correlation_id,
                        seed=self.config.random_seed,
                        controlnet_weights_override=controlnet_weights_override,
                        prompt_override=prompt_override
                    )
                    
                    # Check IoU (INSIDE the loop)
                    generated_mask = extract_alpha_mask(vector_raster)
                    try:
                        is_valid, iou = validate_geometric_similarity(
                            original_mask,
                            generated_mask,
                            threshold=iou_threshold,
                            phase="phase2"
                        )
                        
                        if is_valid or retry_count >= max_retries:
                            # IoU is acceptable or we've exhausted retries
                            if retry_count > 0:
                                logger.info(
                                    "iou_retry_success",
                                    retry_count=retry_count,
                                    final_iou=iou,
                                    final_weights={"depth": depth_weight, "canny": canny_weight}
                                )
                            break
                    except ValidationError as e:
                        # IoU below threshold, retry with increased weights
                        iou = extract_iou_from_error(e)
                        if retry_count < max_retries:
                            retry_count += 1
                            # Increase weights (capped at max values)
                            depth_weight = min(depth_weight + 0.1, 0.9)
                            canny_weight = min(canny_weight + 0.1, 0.7)
                            
                            logger.warning(
                                "iou_below_threshold_retrying",
                                iou=iou,
                                retry_count=retry_count,
                                new_weights={"depth": depth_weight, "canny": canny_weight}
                            )
                            continue  # Retry with new weights
                        else:
                            # Max retries reached, log warning but continue
                            logger.warning(
                                "iou_retry_exhausted",
                                final_iou=iou,
                                max_retries=max_retries
                            )
                            break
                
                # Add retry info to metadata
                phase2_metadata["iou_retry_count"] = retry_count
                phase2_metadata["final_controlnet_weights"] = {
                    "depth": depth_weight,
                    "canny": canny_weight
                }
            
            if save_intermediates and intermediate_dir:
                from src.utils.path_validation import validate_intermediate_dir
                validated_dir = validate_intermediate_dir(intermediate_dir)
                save_image(vector_raster, validated_dir / "phase2_vector_raster.png")
            
            # Phase III: Chromatic Enforcement
            phase3_enabled = phase_overrides.get("phase3", {}).get("enabled",
                self.config.get_phase_config("phase3").get("enabled", True))
            
            if not phase3_enabled:
                logger.info("phase3_skipped")
                quantized_image = vector_raster
                phase3_metadata = {"skipped": True}
            else:
                logger.info("phase3_starting")
                quantized_image, phase3_metadata = self.phase3.enforce(
                    vector_raster,
                    metrics_collector=self.metrics_collector,
                    correlation_id=correlation_id
                )
            
            if save_intermediates and intermediate_dir:
                from src.utils.path_validation import validate_intermediate_dir
                validated_dir = validate_intermediate_dir(intermediate_dir)
                save_image(quantized_image, validated_dir / "phase3_quantized.png")
            
            # Phase IV: Vector Reconstruction
            phase4_enabled = phase_overrides.get("phase4", {}).get("enabled",
                self.config.get_phase_config("phase4").get("enabled", True))
            
            if not phase4_enabled:
                logger.info("phase4_skipped")
                # Generate a simple SVG placeholder if phase4 is disabled
                from PIL import Image
                import base64
                import io
                img_pil = Image.fromarray(quantized_image)
                buffer = io.BytesIO()
                img_pil.save(buffer, format='PNG')
                img_data = base64.b64encode(buffer.getvalue()).decode()
                svg_xml = f'<svg xmlns="http://www.w3.org/2000/svg" width="{quantized_image.shape[1]}" height="{quantized_image.shape[0]}"><image href="data:image/png;base64,{img_data}" width="100%" height="100%"/></svg>'
                phase4_metadata = {"skipped": True}
            else:
                logger.info("phase4_starting")
                # Pass edge_map from Phase II for Strategy B centerline tracing
                edge_map = phase2_metadata.get("edge_map") if phase2_enabled else None
                svg_xml, phase4_metadata = self.phase4.vectorize(
                    quantized_image,
                    edge_map=edge_map,
                    output_path=output_svg_path,
                    metrics_collector=self.metrics_collector,
                    correlation_id=correlation_id
                )
            
            # Save PNG preview if requested
            if output_png_path:
                preview = resize_image(quantized_image, 2048)
                save_image(preview, output_png_path)
            
            # Quality assurance checks
            self._quality_assurance(raw_img, quantized_image, svg_xml, palette)
            
            pipeline_end = datetime.now()
            total_time = (pipeline_end - pipeline_start).total_seconds() * 1000
            
            # Get metrics summary
            metrics_summary = self.metrics_collector.get_pipeline_summary()
            
            metadata = {
                "correlation_id": correlation_id,
                "total_processing_time_ms": total_time,
                "phase1": phase1_metadata,
                "phase2": phase2_metadata,
                "phase3": phase3_metadata,
                "phase4": phase4_metadata,
                "metrics": metrics_summary,
                "output_svg_path": output_svg_path,
                "output_png_path": output_png_path
            }
            
            logger.info("pipeline_complete", **{k: v for k, v in metadata.items() if k != "metrics"})
            
            return svg_xml, metadata
            
        except Exception as e:
            logger.error("pipeline_failed", error=str(e), correlation_id=correlation_id, exc_info=True)
            raise PipelineError(f"Pipeline execution failed: {str(e)}") from e
        finally:
            # Reset metrics collector
            self.metrics_collector.reset()
    
    def _quality_assurance(
        self,
        original: np.ndarray,
        quantized: np.ndarray,
        svg_xml: str,
        palette
    ):
        """Perform quality assurance checks"""
        try:
            # Palette audit on quantized image
            is_valid, invalid_colors = audit_palette_colors(
                quantized,
                palette.get_hex_colors()
            )
            
            if not is_valid:
                logger.warning("palette_audit_failed", invalid_colors=invalid_colors[:10])
            
            # SVG palette validation (simplified - would parse SVG properly)
            logger.info("quality_assurance_complete")
            
        except Exception as e:
            logger.warning("quality_assurance_error", error=str(e), exc_info=True)
            # Don't fail pipeline on QA errors, just log them

