"""SVG processing: stroke injection, duplicate removal, validation"""

import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
import numpy as np
from src.utils.logger import get_logger
from src.utils.error_handler import ValidationError
from src.utils.palette_manager import PaletteManager
from src.utils.quality_assurance import force_snap_to_palette

logger = get_logger(__name__)


class SVGProcessor:
    """Process SVG: add strokes, remove duplicates, validate colors"""
    
    def __init__(
        self,
        stroke_width: int = 2,
        stroke_color: str = "black",
        palette_manager: Optional[PaletteManager] = None
    ):
        """Initialize SVG processor"""
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.palette_manager = palette_manager or PaletteManager()
        self.palette_rgb = self.palette_manager.get_rgb_colors()
    
    def inject_strokes(self, svg_xml: str) -> str:
        """
        Inject stroke attributes into all path elements
        
        Args:
            svg_xml: Input SVG XML string
        
        Returns:
            Modified SVG XML string
        """
        root = ET.fromstring(svg_xml)
        
        # Find all path elements (handle both namespaced and non-namespaced SVG)
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        if not paths:
            paths = root.findall(".//path")
        
        for path in paths:
            path.set("stroke", self.stroke_color)
            path.set("stroke-width", str(self.stroke_width))
            path.set("vector-effect", "non-scaling-stroke")
        
        # Convert back to string
        return ET.tostring(root, encoding='unicode')
    
    def remove_duplicates(self, svg_xml: str) -> str:
        """
        Remove duplicate and redundant paths
        
        Args:
            svg_xml: Input SVG XML string
        
        Returns:
            Modified SVG XML string
        """
        root = ET.fromstring(svg_xml)
        
        # Get all paths (handle both namespaced and non-namespaced SVG)
        # Try namespaced first, fallback to non-namespaced
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        if not paths:
            paths = root.findall(".//path")
        
        # Track paths to remove
        to_remove = []
        path_data = {}
        
        for i, path in enumerate(paths):
            d_attr = path.get("d", "")
            fill = path.get("fill", "none")
            
            # Check for identical paths
            key = (d_attr, fill)
            if key in path_data:
                to_remove.append(path)
                continue
            
            path_data[key] = path
            
            # Check for fully contained paths (occlusion culling)
            # This is simplified - full implementation would use shapely
            bbox = self._get_path_bbox(d_attr)
            if bbox:
                for j, other_path in enumerate(paths):
                    if i != j and other_path not in to_remove:
                        other_d = other_path.get("d", "")
                        other_bbox = self._get_path_bbox(other_d)
                        if other_bbox and self._is_contained(bbox, other_bbox):
                            # Check if same color
                            if path.get("fill") == other_path.get("fill"):
                                to_remove.append(path)
                                break
        
        # Remove duplicate paths
        for path in to_remove:
            parent = path.getparent()
            if parent is not None:
                parent.remove(path)
        
        logger.info("duplicates_removed", num_removed=len(to_remove))
        
        return ET.tostring(root, encoding='unicode')
    
    def validate_and_fix_colors(self, svg_xml: str) -> str:
        """
        Validate and fix colors to match palette
        
        Args:
            svg_xml: Input SVG XML string
        
        Returns:
            Modified SVG XML string
        """
        root = ET.fromstring(svg_xml)
        
        fixed_count = 0
        
        # Check all fill and stroke colors (handle both namespaced and non-namespaced)
        # Use iter() which works with both
        for element in root.iter():
            # Check fill
            fill = element.get("fill")
            if fill and fill not in ["none", "transparent"]:
                if not self._is_valid_color(fill):
                    fixed_color = self._snap_to_palette(fill)
                    element.set("fill", fixed_color)
                    fixed_count += 1
            
            # Check stroke
            stroke = element.get("stroke")
            if stroke and stroke not in ["none", "transparent"]:
                if not self._is_valid_color(stroke):
                    fixed_color = self._snap_to_palette(stroke)
                    element.set("stroke", fixed_color)
                    fixed_count += 1
        
        if fixed_count > 0:
            logger.warning("colors_fixed", num_fixed=fixed_count)
        
        return ET.tostring(root, encoding='unicode')
    
    def _is_valid_color(self, color: str) -> bool:
        """Check if color is in palette"""
        # Convert hex to RGB
        if color.startswith("#"):
            try:
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                return self.palette_manager.validate_color(rgb)
            except (ValueError, IndexError, TypeError) as e:
                logger.debug("invalid_color_format", color=color, error=str(e))
                return False
        return False
    
    def _snap_to_palette(self, color: str) -> str:
        """Snap color to nearest palette color"""
        if color.startswith("#"):
            try:
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                nearest = force_snap_to_palette(rgb, self.palette_rgb)
                return f"#{nearest[0]:02x}{nearest[1]:02x}{nearest[2]:02x}"
            except (ValueError, IndexError, TypeError) as e:
                logger.warning("color_snap_failed", color=color, error=str(e), exc_info=True)
                return self.stroke_color  # Fallback
        return self.stroke_color  # Fallback
    
    def _get_path_bbox(self, d_attr: str) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box of path (simplified)"""
        # This is a simplified implementation
        # Full implementation would parse path data properly
        try:
            # Extract numbers from path data
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', d_attr)
            if len(numbers) >= 4:
                coords = [float(n) for n in numbers[:4]]
                return (min(coords[0], coords[2]), min(coords[1], coords[3]),
                        max(coords[0], coords[2]), max(coords[1], coords[3]))
        except (ValueError, IndexError, TypeError) as e:
            logger.debug("path_bbox_extraction_failed", d_attr=d_attr[:50], error=str(e))
            return None
        return None
    
    def _is_contained(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Check if bbox1 is fully contained in bbox2"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        return (x1_min >= x2_min and y1_min >= y2_min and
                x1_max <= x2_max and y1_max <= y2_max)

