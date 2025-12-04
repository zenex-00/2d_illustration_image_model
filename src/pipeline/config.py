"""Configuration management for the pipeline"""

import os
import threading
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Pipeline configuration loader and manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file and environment variables"""
        # Load environment variables
        load_dotenv()
        
        # Determine config path
        if config_path is None:
            config_path = os.path.join(
                Path(__file__).parent.parent.parent,
                "configs",
                "default_config.yaml"
            )
        
        # Load YAML config
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        if os.getenv("DEVICE"):
            self._config["hardware"]["device"] = os.getenv("DEVICE")
        if os.getenv("PRECISION"):
            self._config["hardware"]["precision"] = os.getenv("PRECISION")
        if os.getenv("MAX_VRAM_GB"):
            self._config["hardware"]["max_vram_gb"] = int(os.getenv("MAX_VRAM_GB"))
        if os.getenv("LOG_LEVEL"):
            self._config.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path"""
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_phase_config(self, phase: str) -> Dict[str, Any]:
        """Get configuration for a specific phase"""
        phase_key = f"phase{phase}" if isinstance(phase, int) else phase
        return self._config.get(phase_key, {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self._config.get("hardware", {})
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline-level configuration"""
        return self._config.get("pipeline", {})
    
    def get_serverless_config(self) -> Dict[str, Any]:
        """Get serverless configuration"""
        return self._config.get("serverless", {})
    
    @property
    def random_seed(self) -> int:
        """Get random seed for reproducibility"""
        return self.get("pipeline.random_seed", 42)
    
    @property
    def device(self) -> str:
        """Get device (cuda/cpu)"""
        return self.get("hardware.device", "cuda")
    
    @property
    def precision(self) -> str:
        """Get precision (float16/float32)"""
        return self.get("hardware.precision", "float16")
    
    @property
    def enable_xformers(self) -> bool:
        """Check if xformers should be enabled"""
        return self.get("hardware.enable_xformers", True)
    
    @property
    def enable_attention_slicing(self) -> bool:
        """Check if attention slicing should be enabled"""
        return self.get("hardware.enable_attention_slicing", True)


# Global config instance
_config_instance: Optional[Config] = None
_config_lock = threading.Lock()


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create global config instance (thread-safe singleton)"""
    global _config_instance
    
    # Fast path: if already initialized, return immediately
    if _config_instance is not None:
        return _config_instance
    
    # Acquire lock for initialization
    with _config_lock:
        # Double-check pattern: another thread may have initialized while we waited
        if _config_instance is None:
            _config_instance = Config(config_path)
        return _config_instance


