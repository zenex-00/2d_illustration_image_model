"""Enhanced model loader with network volume support"""

import os
from pathlib import Path
from typing import Optional, Callable, Any
from src.utils.logger import get_logger
from src.pipeline.config import get_config

logger = get_logger(__name__)


class ModelLoader:
    """Model loader that checks network volume before downloading"""
    
    def __init__(self, volume_path: Optional[str] = None):
        """
        Initialize model loader
        
        Args:
            volume_path: Path to network volume (defaults to config)
        """
        self.config = get_config()
        serverless_config = self.config.get("serverless", {})
        
        # Get volume path from config or environment
        self.volume_path = (
            volume_path or 
            serverless_config.get("model_volume_path") or
            os.getenv("MODEL_VOLUME_PATH", "/models")
        )
        
        self.volume_dir = Path(self.volume_path)
        self.volume_available = self.volume_dir.exists() and self.volume_dir.is_dir()
        
        logger.info(
            "model_loader_initialized",
            volume_path=self.volume_path,
            volume_available=self.volume_available
        )
    
    def get_model_path(
        self,
        model_id: str,
        model_type: str = "huggingface",
        subdir: Optional[str] = None
    ) -> Optional[Path]:
        """
        Get model path from volume if available
        
        Args:
            model_id: Model identifier (HuggingFace ID or filename)
            model_type: Type of model ("huggingface", "direct", etc.)
            subdir: Subdirectory within volume
        
        Returns:
            Path to model if found, None otherwise
        """
        if not self.volume_available:
            return None
        
        if model_type == "huggingface":
            # HuggingFace models are stored by repo_id
            # Extract repo name from model_id (e.g., "stabilityai/stable-diffusion-xl-base-1.0" -> "stable-diffusion-xl-base-1.0")
            repo_name = model_id.split("/")[-1]
            model_path = self.volume_dir / (subdir or repo_name)
        else:
            # Direct file models
            model_path = self.volume_dir / (subdir or model_id)
        
        if model_path.exists():
            logger.info("model_found_on_volume", model_id=model_id, path=str(model_path))
            return model_path
        
        logger.debug("model_not_on_volume", model_id=model_id, path=str(model_path))
        return None
    
    def load_from_volume_or_download(
        self,
        model_id: str,
        model_loader: Callable,
        model_type: str = "huggingface",
        subdir: Optional[str] = None,
        **loader_kwargs
    ) -> Any:
        """
        Load model from volume if available, otherwise download
        
        Args:
            model_id: Model identifier
            model_loader: Function to load model (takes model_id or path)
            model_type: Type of model
            subdir: Subdirectory within volume
            **loader_kwargs: Additional arguments for model loader
        
        Returns:
            Loaded model
        """
        # Try to load from volume first
        volume_path = self.get_model_path(model_id, model_type, subdir)
        
        if volume_path and volume_path.exists():
            try:
                # Try loading from volume path
                if model_type == "huggingface":
                    # For HuggingFace, use the path as local_dir
                    model = model_loader(model_id, local_dir=str(volume_path), **loader_kwargs)
                else:
                    # For direct files, use the path directly
                    model = model_loader(str(volume_path), **loader_kwargs)
                
                logger.info("model_loaded_from_volume", model_id=model_id, path=str(volume_path))
                return model
            except Exception as e:
                logger.warning(
                    "volume_load_failed_falling_back",
                    model_id=model_id,
                    error=str(e),
                    exc_info=True
                )
                # Fall through to download
        
        # Fallback to standard download
        logger.info("downloading_model", model_id=model_id)
        return model_loader(model_id, **loader_kwargs)


# Global loader instance
_loader_instance: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create global model loader instance"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ModelLoader()
    return _loader_instance

