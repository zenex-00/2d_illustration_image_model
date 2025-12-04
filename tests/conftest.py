"""Pytest fixtures and test configuration"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_1024():
    """Create a larger sample test image"""
    return np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)


@pytest.fixture
def mock_config():
    """Create a mock config object"""
    config = Mock()
    config.get = Mock(return_value="cuda")
    config.get_phase_config = Mock(return_value={
        "grounding_dino": {
            "model_id": "test-model",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "prompts": ["test"]
        },
        "sam": {
            "model": "vit_h",
            "dilation_kernel": 5
        },
        "lama": {
            "model": "big-lama"
        }
    })
    config.get_hardware_config = Mock(return_value={"device": "cuda"})
    config.random_seed = 42
    return config


@pytest.fixture
def mock_palette():
    """Create a mock palette"""
    from src.utils.palette_manager import PaletteManager
    palette = Mock(spec=PaletteManager)
    palette.get_hex_colors = Mock(return_value=["#000000", "#FFFFFF"] * 7 + ["#FF0000"])
    palette.get_rgb_colors = Mock(return_value=np.array([[0, 0, 0], [255, 255, 255]] * 7 + [[255, 0, 0]]))
    palette.validate_color = Mock(return_value=True)
    return palette


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test outputs"""
    return tmp_path






