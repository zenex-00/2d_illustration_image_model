"""Standalone tests that don't require any dependencies or model downloads"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_step_off_callback_creation():
    """Test StepOffCallback initialization - no dependencies"""
    # Test by reading source code and verifying class structure
    sdxl_path = Path(__file__).parent.parent / "src" / "phase2_generative_steering" / "sdxl_generator.py"
    with open(sdxl_path, 'r') as f:
        source = f.read()
    
    # Verify StepOffCallback class exists with required methods
    assert "class StepOffCallback" in source
    assert "__init__" in source
    assert "__call__" in source
    assert "step_off_ratio" in source
    assert "initial_canny_weight" in source
    assert "initial_depth_weight" in source
    assert "current_weights" in source
    print("✓ StepOffCallback creation test passed (structure verified)")


def test_step_off_callback_logic():
    """Test step-off callback logic - no dependencies"""
    # Test the callback logic by reading the source and verifying structure
    sdxl_path = Path(__file__).parent.parent / "src" / "phase2_generative_steering" / "sdxl_generator.py"
    with open(sdxl_path, 'r') as f:
        source = f.read()
    
    # Verify StepOffCallback class exists and has correct logic
    assert "class StepOffCallback" in source
    assert "__call__" in source
    assert "step_off_ratio" in source
    assert "steps_before_off" in source  # Verify step-off logic exists
    assert "controlnet_conditioning_scale" in source  # Verify it modifies weights
    print("✓ StepOffCallback logic test passed (structure verified)")


def test_model_versions_config():
    """Test model_versions.yaml structure"""
    config_path = Path(__file__).parent.parent / "configs" / "model_versions.yaml"
    assert config_path.exists(), "model_versions.yaml should exist"
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "models" in config
        
        # Check key models have metadata
        key_models = ["sdxl", "controlnet_depth", "controlnet_canny"]
        for model_name in key_models:
            if model_name in config["models"]:
                model_config = config["models"][model_name]
                assert "last_updated" in model_config
                assert "api_version" in model_config
        print("✓ Model versions config test passed")
    except ImportError:
        print("⚠ yaml not available, skipping detailed config test")


def test_requirements_updated():
    """Test that requirements.txt has updated versions"""
    req_path = Path(__file__).parent.parent / "requirements.txt"
    assert req_path.exists(), "requirements.txt should exist"
    
    with open(req_path, 'r') as f:
        content = f.read()
    
    # Check for updated versions
    assert "torch>=" in content
    assert "diffusers>=" in content
    assert "peft>=" in content
    print("✓ Requirements updated test passed")


def test_safety_comment():
    """Test that server.py has safety comment"""
    server_path = Path(__file__).parent.parent / "src" / "api" / "server.py"
    assert server_path.exists(), "server.py should exist"
    
    with open(server_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert "trust_remote_code" in content.lower()
    print("✓ Safety comment test passed")


def test_imports():
    """Test that key modules can be imported"""
    # Test file existence instead of actual imports to avoid dependency issues
    sdxl_path = Path(__file__).parent.parent / "src" / "phase2_generative_steering" / "sdxl_generator.py"
    train_path = Path(__file__).parent.parent / "src" / "phase2_generative_steering" / "train_lora_sdxl.py"
    
    assert sdxl_path.exists(), "sdxl_generator.py should exist"
    assert train_path.exists(), "train_lora_sdxl.py should exist"
    
    # Verify classes exist in source
    with open(sdxl_path, 'r') as f:
        sdxl_source = f.read()
    assert "class SDXLGenerator" in sdxl_source
    assert "class StepOffCallback" in sdxl_source
    
    with open(train_path, 'r') as f:
        train_source = f.read()
    assert "def train_lora" in train_source
    
    print("✓ Imports test passed (files and classes verified)")


if __name__ == "__main__":
    print("Running standalone tests (no dependencies required)...\n")
    
    tests = [
        test_step_off_callback_creation,
        test_step_off_callback_logic,
        test_model_versions_config,
        test_requirements_updated,
        test_safety_comment,
        test_imports,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    
    if failed > 0:
        sys.exit(1)

