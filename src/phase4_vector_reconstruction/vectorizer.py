"""Phase IV orchestrator: Vector Reconstruction"""

import numpy as np
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path
from src.utils.logger import get_logger, set_correlation_id
from src.utils.error_handler import PhaseError
from src.utils.metrics import MetricsCollector
from src.pipeline.config import get_config
from .vtracer_wrapper import VTracerWrapper
from .svg_processor import SVGProcessor
from .centerline_tracer import CenterlineTracer

logger = get_logger(__name__)


class Phase4Vectorizer:
    """Orchestrator for Phase IV: Vector Reconstruction"""
    
    def __init__(self, config=None):
        """Initialize Phase IV components"""
        self.config = config or get_config()
        phase_config = self.config.get_phase_config("phase4")
        
        # Initialize components
        self.vtracer = VTracerWrapper(
            mode=phase_config["vtracer"]["mode"],
            filter_speckle=phase_config["vtracer"]["filter_speckle"],
            corner_threshold=phase_config["vtracer"]["corner_threshold"],
            segment_length=phase_config["vtracer"]["segment_length"],
            timeout_seconds=phase_config["vtracer"]["timeout_seconds"]
        )
        
        self.svg_processor = SVGProcessor(
            stroke_width=phase_config["svg"]["stroke_width"],
            stroke_color=phase_config["svg"]["stroke_color"]
        )
        
        self.centerline_tracer = CenterlineTracer()
        self.strategy = phase_config["svg"].get("strategy", "A")
        
        logger.info("phase4_initialized", strategy=self.strategy)
    
    def vectorize(
        self,
        quantized_image: np.ndarray,
        edge_map: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        correlation_id: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Perform vector reconstruction: VTracer → SVG processing → validation
        
        Args:
            quantized_image: Quantized image from Phase III (RGB, uint8)
            edge_map: Optional edge map for Strategy B centerlines
            output_path: Optional output path for SVG file
            metrics_collector: Optional metrics collector
            correlation_id: Optional correlation ID for logging
        
        Returns:
            Tuple of (svg_xml_string, metadata_dict)
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        start_time = datetime.now()
        
        try:
            logger.info("phase4_start", input_shape=quantized_image.shape)
            
            # Step 1: Vectorize with VTracer
            svg_xml = self.vtracer.vectorize(quantized_image, output_path)
            
            # Step 2: Process SVG (inject strokes, remove duplicates)
            svg_xml = self.svg_processor.inject_strokes(svg_xml)
            svg_xml = self.svg_processor.remove_duplicates(svg_xml)
            svg_xml = self.svg_processor.validate_and_fix_colors(svg_xml)
            
            # Step 3: Add centerlines if Strategy B
            if self.strategy == "B" and edge_map is not None:
                centerline_path = self.centerline_tracer.trace_centerlines(edge_map)
                # Merge centerlines into SVG
                # This is simplified - full implementation would properly merge
                svg_xml = svg_xml.replace("</svg>", f"{centerline_path}</svg>")
            
            # Save if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(svg_xml)
                logger.info("svg_saved", output_path=output_path)
            
            end_time = datetime.now()
            
            metadata = {
                "svg_size_bytes": len(svg_xml),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "strategy": self.strategy
            }
            
            if metrics_collector:
                metrics_collector.record_phase(
                    phase_name="phase4_vector_reconstruction",
                    start_time=start_time,
                    end_time=end_time,
                    input_shape=quantized_image.shape,
                    output_shape=(len(svg_xml),)  # SVG is text
                )
            
            logger.info("phase4_complete", **metadata)
            
            # Flush any remaining GPU memory (Phase 4 is mostly CPU-based)
            import gc
            import torch
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("phase4_memory_flushed")
            
            return svg_xml, metadata
            
        except Exception as e:
            logger.error("phase4_failed", error=str(e), exc_info=True)
            raise PhaseError(
                phase="phase4",
                message=f"Vector reconstruction failed: {str(e)}",
                original_error=e
            )


