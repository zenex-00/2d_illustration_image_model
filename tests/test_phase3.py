"""Unit tests for Phase III: Chromatic Enforcement"""

import pytest
import numpy as np
from tests.conftest import sample_image, mock_palette


class TestColorQuantizer:
    """Test color quantizer"""
    
    def test_quantizer_initialization(self, mock_palette):
        """Test quantizer initialization"""
        from src.phase3_chromatic_enforcement.color_quantizer import ColorQuantizer
        
        quantizer = ColorQuantizer(palette_manager=mock_palette)
        assert quantizer.palette_rgb is not None
        assert quantizer.kdtree is not None
    
    def test_quantization(self, sample_image, mock_palette):
        """Test color quantization"""
        from src.phase3_chromatic_enforcement.color_quantizer import ColorQuantizer
        
        quantizer = ColorQuantizer(palette_manager=mock_palette)
        quantized = quantizer.quantize(sample_image, validate=False)
        
        assert quantized.shape == sample_image.shape
        assert quantized.dtype == np.uint8


class TestNoiseRemover:
    """Test noise remover"""
    
    def test_noise_removal(self, sample_image):
        """Test noise removal"""
        from src.phase3_chromatic_enforcement.noise_remover import NoiseRemover
        
        remover = NoiseRemover(min_area_percent=0.001)
        denoised = remover.remove_noise(sample_image)
        
        assert denoised.shape == sample_image.shape
        assert denoised.dtype == np.uint8







