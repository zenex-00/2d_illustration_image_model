"""AutoTrace centerline tracing (Strategy B)"""

import os
import subprocess
import tempfile
import numpy as np
from typing import Optional
from skimage.morphology import skeletonize
from src.utils.logger import get_logger
from src.utils.error_handler import PhaseError

logger = get_logger(__name__)


class CenterlineTracer:
    """Centerline tracing for technical detail lines (Strategy B)"""
    
    def __init__(self, autotrace_path: Optional[str] = None):
        """
        Initialize centerline tracer.
        
        Args:
            autotrace_path: Optional path to AutoTrace binary
        """
        self.autotrace_path = autotrace_path or self._find_autotrace()
    
    def _find_autotrace(self) -> Optional[str]:
        """Find AutoTrace binary in PATH or common locations"""
        import shutil
        
        # Check PATH
        autotrace = shutil.which("autotrace")
        if autotrace:
            return autotrace
        
        # Check common locations
        common_paths = [
            "/usr/local/bin/autotrace",
            "/usr/bin/autotrace",
            "./autotrace",
            "./bin/autotrace"
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        logger.warning("autotrace_not_found", falling_back_to="skeleton_tracing")
        return None
    
    def trace_centerlines(self, edge_map: np.ndarray) -> str:
        """
        Trace centerlines from edge map.
        
        Uses AutoTrace if available, otherwise falls back to skeleton-based tracing.
        
        Args:
            edge_map: Binary edge map
        
        Returns:
            SVG path string for centerlines
        """
        # Skeletonize edges first
        skeleton = skeletonize(edge_map > 0)
        
        logger.info("centerlines_traced", skeleton_pixels=skeleton.sum())
        
        # Try AutoTrace if available
        if self.autotrace_path:
            try:
                return self._trace_with_autotrace(skeleton)
            except Exception as e:
                logger.warning("autotrace_failed", error=str(e), falling_back_to="skeleton", exc_info=True)
        
        # Fallback to skeleton-based tracing
        return self._trace_from_skeleton(skeleton)
    
    def _trace_with_autotrace(self, skeleton: np.ndarray) -> str:
        """Trace centerlines using AutoTrace binary"""
        from PIL import Image
        
        # Save skeleton as PNG
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_input:
            input_path = tmp_input.name
            Image.fromarray((skeleton * 255).astype(np.uint8)).save(input_path, 'PNG')
        
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False, mode='w') as tmp_output:
            output_path = tmp_output.name
        
        try:
            # Run AutoTrace with centerline mode
            cmd = [
                self.autotrace_path,
                "--centerline",
                "--output-file", output_path,
                input_path
            ]
            
            result = subprocess.run(
                cmd,
                timeout=60,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise PhaseError(
                    phase="phase4",
                    message=f"AutoTrace failed: {result.stderr}"
                )
            
            # Read SVG output
            with open(output_path, 'r') as f:
                svg_xml = f.read()
            
            # Add stroke attributes
            svg_xml = self._add_stroke_attributes(svg_xml)
            
            logger.info("autotrace_complete", output_size=len(svg_xml))
            
            return svg_xml
            
        finally:
            # Clean up temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def _trace_from_skeleton(self, skeleton: np.ndarray) -> str:
        """
        Trace centerlines from skeleton (fallback method).
        
        Converts skeleton pixels to simple SVG paths.
        """
        import cv2
        
        # Find contours in skeleton
        contours, _ = cv2.findContours(
            (skeleton * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Build SVG paths
        paths = []
        for contour in contours:
            if len(contour) < 2:
                continue
            
            # Convert contour to SVG path
            path_data = "M "
            for i, point in enumerate(contour):
                x, y = point[0]
                if i == 0:
                    path_data += f"{x},{y} "
                else:
                    path_data += f"L {x},{y} "
            
            paths.append(f'<path d="{path_data}" fill="none" stroke="black" stroke-width="2" vector-effect="non-scaling-stroke"/>')
        
        if not paths:
            return '<path d="" fill="none" stroke="black" stroke-width="2"/>'
        
        svg_content = "\n".join(paths)
        logger.info("skeleton_tracing_complete", num_paths=len(paths))
        
        return svg_content
    
    def _add_stroke_attributes(self, svg_xml: str) -> str:
        """Add stroke attributes to SVG paths"""
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(svg_xml)
            
            # Find all path elements
            paths = root.findall(".//{http://www.w3.org/2000/svg}path")
            if not paths:
                paths = root.findall(".//path")
            
            for path in paths:
                path.set("fill", "none")
                path.set("stroke", "black")
                path.set("stroke-width", "2")
                path.set("vector-effect", "non-scaling-stroke")
            
            return ET.tostring(root, encoding='unicode')
        except Exception as e:
            logger.warning("svg_attribute_injection_failed", error=str(e), exc_info=True)
            return svg_xml

