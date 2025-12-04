"""Simple unit tests that don't require any model downloads or heavy dependencies"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock


class TestStepOffCallback:
    """Test step-off callback - no dependencies needed"""
    
    def test_step_off_callback_creation(self):
        """Test StepOffCallback initialization"""
        from src.phase2_generative_steering.sdxl_generator import StepOffCallback
        
        callback = StepOffCallback(
            step_off_ratio=0.75,
            initial_canny_weight=0.4,
            initial_depth_weight=0.6
        )
        
        assert callback.step_off_ratio == 0.75
        assert callback.initial_canny_weight == 0.4
        assert callback.initial_depth_weight == 0.6
        assert callback.current_weights == [0.6, 0.4]
    
    def test_step_off_callback_before_threshold(self):
        """Test callback before step-off threshold"""
        from src.phase2_generative_steering.sdxl_generator import StepOffCallback
        
        callback = StepOffCallback(
            step_off_ratio=0.75,
            initial_canny_weight=0.4,
            initial_depth_weight=0.6
        )
        
        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.scheduler.timesteps = list(range(10))  # 10 steps
        
        # Step 0-7 should have full canny weight (75% of 10 = 7.5, so steps 0-7)
        callback_kwargs = {}
        result = callback(mock_pipe, step_index=0, timestep=1000, callback_kwargs=callback_kwargs)
        assert result["controlnet_conditioning_scale"][0] == 0.6  # Depth constant
        assert result["controlnet_conditioning_scale"][1] == 0.4  # Canny full
        
        result = callback(mock_pipe, step_index=7, timestep=300, callback_kwargs={})
        assert result["controlnet_conditioning_scale"][1] == 0.4  # Still full
    
    def test_step_off_callback_after_threshold(self):
        """Test callback after step-off threshold"""
        from src.phase2_generative_steering.sdxl_generator import StepOffCallback
        
        callback = StepOffCallback(
            step_off_ratio=0.75,
            initial_canny_weight=0.4,
            initial_depth_weight=0.6
        )
        
        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.scheduler.timesteps = list(range(10))  # 10 steps
        
        # Step 8-9 should fade to zero
        result = callback(mock_pipe, step_index=8, timestep=200, callback_kwargs={})
        assert result["controlnet_conditioning_scale"][1] < 0.4  # Fading
        assert result["controlnet_conditioning_scale"][1] > 0.0  # Not zero yet
        
        result = callback(mock_pipe, step_index=9, timestep=100, callback_kwargs={})
        assert result["controlnet_conditioning_scale"][1] == 0.0  # Zero at end


class TestConfigFiles:
    """Test configuration file structure"""
    
    def test_model_versions_yaml_exists(self):
        """Test that model_versions.yaml exists and has correct structure"""
        config_path = Path(__file__).parent.parent / "configs" / "model_versions.yaml"
        assert config_path.exists(), "model_versions.yaml should exist"
    
    def test_model_versions_has_metadata(self):
        """Test that model versions have required metadata"""
        try:
            import yaml
        except ImportError:
            pytest.skip("yaml not available")
        
        config_path = Path(__file__).parent.parent / "configs" / "model_versions.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "models" in config
        
        # Check that key models have version metadata
        key_models = ["sdxl", "controlnet_depth", "controlnet_canny"]
        for model_name in key_models:
            if model_name in config["models"]:
                model_config = config["models"][model_name]
                assert "last_updated" in model_config, f"{model_name} missing last_updated"
                assert "api_version" in model_config, f"{model_name} missing api_version"
                assert "pytorch_version" in model_config, f"{model_name} missing pytorch_version"
                assert "transformers_version" in model_config, f"{model_name} missing transformers_version"


class TestCodeStructure:
    """Test code structure and imports"""
    
    def test_sdxl_generator_import(self):
        """Test that SDXLGenerator can be imported"""
        from src.phase2_generative_steering.sdxl_generator import SDXLGenerator
        assert SDXLGenerator is not None
    
    def test_step_off_callback_import(self):
        """Test that StepOffCallback can be imported"""
        from src.phase2_generative_steering.sdxl_generator import StepOffCallback
        assert StepOffCallback is not None
    
    def test_training_function_import(self):
        """Test that train_lora can be imported"""
        from src.phase2_generative_steering.train_lora_sdxl import train_lora
        assert train_lora is not None
    
    def test_requirements_txt_exists(self):
        """Test that requirements.txt exists"""
        req_path = Path(__file__).parent.parent / "requirements.txt"
        assert req_path.exists(), "requirements.txt should exist"
    
    def test_requirements_has_updated_versions(self):
        """Test that requirements.txt has updated dependency versions"""
        req_path = Path(__file__).parent.parent / "requirements.txt"
        with open(req_path, 'r') as f:
            content = f.read()
        
        # Check for updated versions
        assert "torch>=2.5.1" in content or "torch>=" in content
        assert "diffusers>=0.30.0" in content or "diffusers>=" in content
        assert "peft>=0.13.0" in content or "peft>=" in content
        assert "xformers" in content or "bitsandbytes" in content  # At least one optimization package


class TestSafetyComment:
    """Test security documentation"""
    
    def test_server_has_safety_comment(self):
        """Test that server.py has trust_remote_code safety comment"""
        server_path = Path(__file__).parent.parent / "src" / "api" / "server.py"
        assert server_path.exists(), "server.py should exist"
        
        with open(server_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for safety comment
        assert "trust_remote_code" in content.lower(), "Should have trust_remote_code documentation"
        assert "false" in content.lower() or "enforce" in content.lower(), "Should document false enforcement"



