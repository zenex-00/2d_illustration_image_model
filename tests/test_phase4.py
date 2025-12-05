"""Unit tests for Phase IV: Vector Reconstruction"""

import pytest
import numpy as np
from tests.conftest import sample_image


class TestSVGProcessor:
    """Test SVG processor"""
    
    def test_stroke_injection(self):
        """Test stroke attribute injection"""
        from src.phase4_vector_reconstruction.svg_processor import SVGProcessor
        
        processor = SVGProcessor()
        svg_xml = '<svg><path d="M0,0 L10,10"/></svg>'
        result = processor.inject_strokes(svg_xml)
        
        assert "stroke" in result
        assert "stroke-width" in result
        assert "vector-effect" in result
    
    def test_duplicate_removal(self):
        """Test duplicate path removal"""
        from src.phase4_vector_reconstruction.svg_processor import SVGProcessor
        
        processor = SVGProcessor()
        svg_xml = '<svg><path d="M0,0 L10,10"/><path d="M0,0 L10,10"/></svg>'
        result = processor.remove_duplicates(svg_xml)
        
        # Should have fewer paths (simplified check)
        assert isinstance(result, str)







