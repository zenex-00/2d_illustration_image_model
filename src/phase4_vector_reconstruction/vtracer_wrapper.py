"""VTracer wrapper with timeout and error handling"""

import subprocess
import tempfile
import os
import numpy as np
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger
from src.utils.error_handler import PhaseError

logger = get_logger(__name__)


class VTracerWrapper:
    """Wrapper for VTracer binary with robust error handling"""
    
    def __init__(
        self,
        vtracer_path: Optional[str] = None,
        mode: str = "stacked",
        filter_speckle: int = 4,
        corner_threshold: int = 60,
        segment_length: int = 4,
        timeout_seconds: int = 300
    ):
        """Initialize VTracer wrapper"""
        self.vtracer_path = vtracer_path or self._find_vtracer()
        self.mode = mode
        self.filter_speckle = filter_speckle
        self.corner_threshold = corner_threshold
        self.segment_length = segment_length
        self.timeout_seconds = timeout_seconds
    
    def _find_vtracer(self) -> str:
        """Find VTracer binary in PATH or common locations"""
        # Check PATH
        import shutil
        vtracer = shutil.which("vtracer")
        if vtracer:
            return vtracer
        
        # Check common locations
        common_paths = [
            "/usr/local/bin/vtracer",
            "/usr/bin/vtracer",
            "./vtracer",
            "./bin/vtracer"
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        raise FileNotFoundError(
            "VTracer binary not found. Please install VTracer or specify path."
        )
    
    def vectorize(
        self,
        image: np.ndarray,
        output_path: Optional[str] = None
    ) -> str:
        """
        Vectorize image using VTracer
        
        Args:
            image: Input image as numpy array (RGB, uint8)
            output_path: Optional output path for SVG
        
        Returns:
            SVG XML string
        """
        import numpy as np
        from PIL import Image
        
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_input:
            input_path = tmp_input.name
            Image.fromarray(image).save(input_path, 'PNG')
        
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False, mode='w') as tmp_output:
                output_path = tmp_output.name
        else:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Build VTracer command
            cmd = [
                self.vtracer_path,
                "--input", input_path,
                "--output", output_path,
                "--colormode", "color",
                "--mode", self.mode,
                "--filter_speckle", str(self.filter_speckle),
                "--corner_threshold", str(self.corner_threshold),
                "--segment_length", str(self.segment_length)
            ]
            
            logger.info("vtracer_starting", cmd=" ".join(cmd))
            
            # Run VTracer with timeout
            result = subprocess.run(
                cmd,
                timeout=self.timeout_seconds,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                raise PhaseError(
                    phase="phase4",
                    message=f"VTracer failed: {error_msg}"
                )
            
            # Read SVG output
            with open(output_path, 'r') as f:
                svg_xml = f.read()
            
            logger.info("vtracer_complete", output_size=len(svg_xml))
            
            return svg_xml
            
        except subprocess.TimeoutExpired:
            raise PhaseError(
                phase="phase4",
                message=f"VTracer execution timed out after {self.timeout_seconds} seconds"
            )
        except Exception as e:
            logger.error("vtracer_failed", error=str(e), exc_info=True)
            raise PhaseError(
                phase="phase4",
                message=f"VTracer execution failed: {str(e)}",
                original_error=e
            )
        finally:
            # Clean up temporary input file
            if os.path.exists(input_path):
                os.unlink(input_path)

