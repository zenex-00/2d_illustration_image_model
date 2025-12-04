"""Model caching and GPU memory management"""

import torch
import threading
from typing import Dict, Any, Optional, Callable
from functools import lru_cache
import gc
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelCache:
    """Singleton model cache with GPU memory management"""
    
    _instance: Optional['ModelCache'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.cache: Dict[str, Any] = {}
            self.memory_usage: Dict[str, float] = {}
            self._initialized = True
    
    def load_or_cache_model(
        self,
        model_id: str,
        model_loader: Callable,
        cache_key: Optional[str] = None,
        **loader_kwargs
    ) -> Any:
        """Load model or return cached version"""
        cache_key = cache_key or model_id
        
        if cache_key in self.cache:
            logger.info("model_cache_hit", model_id=model_id, cache_key=cache_key)
            return self.cache[cache_key]
        
        logger.info("model_cache_miss", model_id=model_id, cache_key=cache_key)
        
        # Track memory before loading
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
        
        # Load model
        model = model_loader(model_id, **loader_kwargs)
        
        # Track memory after loading
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_used = memory_after - memory_before
            self.memory_usage[cache_key] = memory_used
            logger.info(
                "model_loaded",
                model_id=model_id,
                memory_used_gb=memory_used,
                total_memory_gb=memory_after
            )
        
        # Cache model
        self.cache[cache_key] = model
        return model
    
    def clear_cache(self, model_key: Optional[str] = None, phase_prefix: Optional[str] = None) -> None:
        """
        Clear model cache, optionally for a specific model or phase
        
        Args:
            model_key: Specific model key to clear
            phase_prefix: Clear all models with this prefix (e.g., "phase1", "grounding_dino", "sam")
        """
        if phase_prefix:
            # Clear all models matching phase prefix
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(phase_prefix)]
            for key in keys_to_remove:
                model = self.cache.pop(key)
                self._move_to_cpu_and_delete(model)
                if key in self.memory_usage:
                    del self.memory_usage[key]
            logger.info("model_cache_cleared_by_phase", phase_prefix=phase_prefix, cleared_count=len(keys_to_remove))
        elif model_key:
            if model_key in self.cache:
                model = self.cache.pop(model_key)
                self._move_to_cpu_and_delete(model)
                if model_key in self.memory_usage:
                    del self.memory_usage[model_key]
                logger.info("model_cache_cleared", model_key=model_key)
        else:
            # Clear all
            for model in self.cache.values():
                self._move_to_cpu_and_delete(model)
            self.cache.clear()
            self.memory_usage.clear()
            logger.info("model_cache_cleared_all")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _move_to_cpu_and_delete(self, model: Any) -> None:
        """Move model to CPU and delete"""
        try:
            if hasattr(model, 'to'):
                model.to('cpu')
            elif hasattr(model, 'cpu'):
                model = model.cpu()
            # Disable training mode before deletion
            if hasattr(model, 'eval'):
                model.eval()
            del model
        except Exception as e:
            logger.warning("error_clearing_model", error=str(e), exc_info=True)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage for all cached models"""
        return self.memory_usage.copy()
    
    def get_total_memory_used(self) -> float:
        """Get total memory used by cached models"""
        return sum(self.memory_usage.values())
    
    def enable_cpu_offload(self, model: Any) -> None:
        """Enable CPU offload for a model (if supported)"""
        if hasattr(model, 'enable_model_cpu_offload'):
            model.enable_model_cpu_offload()
            logger.info("cpu_offload_enabled", model_type=type(model).__name__)


# Global cache instance
_cache_instance: Optional[ModelCache] = None
_cache_lock = threading.Lock()


def get_model_cache() -> ModelCache:
    """Get or create global model cache instance (thread-safe singleton)"""
    global _cache_instance
    
    # Fast path: if already initialized, return immediately
    if _cache_instance is not None:
        return _cache_instance
    
    # Acquire lock for initialization
    with _cache_lock:
        # Double-check pattern: another thread may have initialized while we waited
        if _cache_instance is None:
            _cache_instance = ModelCache()
        return _cache_instance


