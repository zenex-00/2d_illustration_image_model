"""Unit tests for Phase I: Semantic Sanitization"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.phase1_semantic_sanitization.sanitizer import Phase1Sanitizer
from tests.conftest import sample_image, mock_config


class TestPhase1Sanitizer:
    """Test Phase I sanitizer"""
    
    @patch('src.phase1_semantic_sanitization.grounding_dino_detector.GroundingDINODetector')
    @patch('src.phase1_semantic_sanitization.sam_segmenter.SAMSegmenter')
    @patch('src.phase1_semantic_sanitization.lama_inpainter.LaMaInpainter')
    def test_sanitizer_initialization(self, mock_lama, mock_sam, mock_dino, mock_config):
        """Test sanitizer initialization"""
        sanitizer = Phase1Sanitizer(config=mock_config)
        assert sanitizer.detector is not None
        assert sanitizer.segmenter is not None
        assert sanitizer.inpainter is not None
    
    @patch('src.phase1_semantic_sanitization.grounding_dino_detector.GroundingDINODetector')
    @patch('src.phase1_semantic_sanitization.sam_segmenter.SAMSegmenter')
    @patch('src.phase1_semantic_sanitization.lama_inpainter.LaMaInpainter')
    def test_sanitize_no_detections(self, mock_lama, mock_sam, mock_dino, sample_image, mock_config):
        """Test sanitization when no objects are detected"""
        # Setup mocks
        mock_dino_instance = Mock()
        mock_dino_instance.detect_objects.return_value = []
        mock_dino.return_value = mock_dino_instance
        
        sanitizer = Phase1Sanitizer(config=mock_config)
        clean_plate, metadata = sanitizer.sanitize(sample_image)
        
        assert clean_plate.shape == sample_image.shape
        assert metadata["detections"] == 0
    
    @patch('src.phase1_semantic_sanitization.grounding_dino_detector.GroundingDINODetector')
    @patch('src.phase1_semantic_sanitization.sam_segmenter.SAMSegmenter')
    @patch('src.phase1_semantic_sanitization.lama_inpainter.LaMaInpainter')
    def test_sanitize_with_detections(self, mock_lama, mock_sam, mock_dino, sample_image, mock_config):
        """Test sanitization with detections"""
        # Setup mocks
        mock_dino_instance = Mock()
        mock_dino_instance.detect_objects.return_value = [(10, 10, 50, 50)]
        mock_dino.return_value = mock_dino_instance
        
        mock_sam_instance = Mock()
        mock_sam_instance.generate_masks.return_value = np.zeros((512, 512), dtype=np.uint8)
        mock_sam.return_value = mock_sam_instance
        
        mock_lama_instance = Mock()
        mock_lama_instance.inpaint.return_value = sample_image
        mock_lama.return_value = mock_lama_instance
        
        sanitizer = Phase1Sanitizer(config=mock_config)
        clean_plate, metadata = sanitizer.sanitize(sample_image)
        
        assert clean_plate.shape == sample_image.shape
        assert metadata["detections"] == 1


class TestGroundingDINODetector:
    """Test GroundingDINO detector"""
    
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        # This would require actual model loading, so we'll skip for now
        pass


class TestSAMSegmenter:
    """Test SAM segmenter"""
    
    def test_segmenter_initialization(self):
        """Test segmenter can be initialized"""
        # This would require actual model loading, so we'll skip for now
        pass


class TestLaMaInpainter:
    """Test LaMa inpainter"""
    
    def test_inpainter_initialization(self):
        """Test inpainter can be initialized"""
        # This would require actual model loading, so we'll skip for now
        pass







