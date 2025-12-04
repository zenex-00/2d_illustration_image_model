"""15-color palette management"""

import yaml
from pathlib import Path
from typing import List, Optional, Union, Tuple
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PaletteManager:
    """Manages the 15-color palette for vector illustrations"""
    
    def __init__(self, palette_path: Optional[str] = None):
        """Initialize palette from YAML file"""
        if palette_path is None:
            palette_path = Path(__file__).parent.parent.parent / "configs" / "palette.yaml"
        
        with open(palette_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.hex_colors = config.get('palette', [])
        self._validate_palette()
        self.rgb_colors = self._hex_to_rgb(self.hex_colors)
        logger.info("palette_loaded", num_colors=len(self.hex_colors))
    
    def _validate_palette(self):
        """Validate palette has exactly 15 colors and valid hex codes"""
        if len(self.hex_colors) != 15:
            raise ValueError(f"Palette must contain exactly 15 colors, got {len(self.hex_colors)}")
        
        for i, hex_color in enumerate(self.hex_colors):
            if not hex_color.startswith('#'):
                raise ValueError(f"Color {i} must start with #: {hex_color}")
            if len(hex_color) != 7:
                raise ValueError(f"Color {i} must be 7 characters (e.g., #FF0000): {hex_color}")
            try:
                int(hex_color[1:], 16)
            except ValueError:
                raise ValueError(f"Color {i} is not a valid hex code: {hex_color}")
    
    def _hex_to_rgb(self, hex_colors: List[str]) -> np.ndarray:
        """Convert hex colors to RGB numpy array"""
        rgb_list = []
        for hex_color in hex_colors:
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
            rgb_list.append(rgb)
        return np.array(rgb_list, dtype=np.uint8)
    
    def get_hex_colors(self) -> List[str]:
        """Get list of hex color codes"""
        return self.hex_colors.copy()
    
    def get_rgb_colors(self) -> np.ndarray:
        """Get RGB colors as numpy array (N, 3)"""
        return self.rgb_colors.copy()
    
    def validate_color(self, color: Union[str, Tuple[int, int, int]]) -> bool:
        """Check if a color is in the palette"""
        if isinstance(color, str):
            return color.upper() in [c.upper() for c in self.hex_colors]
        elif isinstance(color, (tuple, list, np.ndarray)):
            rgb = np.array(color[:3])
            # Check if color matches any palette color (with small tolerance)
            distances = np.linalg.norm(self.rgb_colors - rgb, axis=1)
            return np.any(distances < 1.0)  # Allow 1 unit tolerance
        return False
    
    def find_nearest_color(self, color: Union[str, Tuple[int, int, int]]) -> str:
        """Find nearest palette color to given color"""
        if isinstance(color, str):
            if color.startswith('#'):
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            else:
                raise ValueError(f"Invalid color format: {color}")
        else:
            rgb = tuple(color[:3])
        
        rgb_array = np.array(rgb)
        distances = np.linalg.norm(self.rgb_colors - rgb_array, axis=1)
        nearest_idx = np.argmin(distances)
        return self.hex_colors[nearest_idx]


# Global palette instance
_palette_instance: Optional[PaletteManager] = None


def get_palette(palette_path: Optional[str] = None) -> PaletteManager:
    """Get or create global palette instance"""
    global _palette_instance
    if _palette_instance is None:
        _palette_instance = PaletteManager(palette_path)
    return _palette_instance

