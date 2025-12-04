"""Health check utilities for liveness and readiness probes"""

import torch
import shutil
from typing import Dict, Optional, Tuple
from src.pipeline.config import get_config
from src.pipeline.model_cache import get_model_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HealthChecker:
    """Health checker for liveness and readiness probes"""
    
    def __init__(self, require_gpu: bool = True):
        """
        Initialize health checker.
        
        Args:
            require_gpu: Whether GPU is required for readiness
        """
        self.require_gpu = require_gpu
        self.config = get_config()
    
    def check_liveness(self) -> bool:
        """
        Check if service is alive (liveness probe).
        Should be fast (<5ms) and always return True if server is running.
        
        Returns:
            True if service is alive
        """
        # Simple check - if we can execute this, server is alive
        return True
    
    def check_readiness(self) -> Tuple[bool, Dict[str, str]]:
        """
        Check if service is ready to handle requests (readiness probe).
        
        Returns:
            Tuple of (is_ready, checks_dict)
        """
        checks = {}
        is_ready = True
        
        # Check GPU availability
        if self.require_gpu:
            gpu_available = torch.cuda.is_available()
            checks["gpu"] = "available" if gpu_available else "unavailable"
            if not gpu_available:
                is_ready = False
                logger.warning("gpu_unavailable")
        else:
            checks["gpu"] = "not_required"
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
                
                checks["gpu_memory"] = f"{memory_allocated:.2f}GB/{memory_total:.2f}GB allocated"
                
                # Check if memory usage is reasonable (<95%)
                memory_usage_percent = (memory_reserved / memory_total) * 100
                if memory_usage_percent > 95:
                    checks["gpu_memory"] = f"critical ({memory_usage_percent:.1f}% used)"
                    logger.warning("gpu_memory_critical", usage_percent=memory_usage_percent)
            except Exception as e:
                checks["gpu_memory"] = f"error: {str(e)}"
                logger.warning("gpu_memory_check_failed", error=str(e), exc_info=True)
        
        # Check model cache status
        try:
            cache = get_model_cache()
            cache_size = len(cache.cache)
            checks["model_cache"] = f"ready ({cache_size} models cached)"
        except Exception as e:
            checks["model_cache"] = f"error: {str(e)}"
            logger.warning("model_cache_check_failed", error=str(e), exc_info=True)
        
        # Check disk space (optional, but useful)
        try:
            disk = shutil.disk_usage("/")
            free_gb = disk.free / (1024 ** 3)
            total_gb = disk.total / (1024 ** 3)
            free_percent = (disk.free / disk.total) * 100
            
            checks["disk_space"] = f"{free_gb:.2f}GB free ({free_percent:.1f}%)"
            
            # Warn if disk space is low (<10%)
            if free_percent < 10:
                checks["disk_space"] = f"low ({free_percent:.1f}% free)"
                logger.warning("disk_space_low", free_percent=free_percent)
        except Exception as e:
            checks["disk_space"] = f"error: {str(e)}"
            logger.warning("disk_space_check_failed", error=str(e), exc_info=True)
        
        # Check memory availability (optional)
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            total_gb = memory.total / (1024 ** 3)
            available_percent = (memory.available / memory.total) * 100
            
            checks["system_memory"] = f"{available_gb:.2f}GB available ({available_percent:.1f}%)"
            
            if available_percent < 10:
                checks["system_memory"] = f"low ({available_percent:.1f}% available)"
                logger.warning("system_memory_low", available_percent=available_percent)
        except ImportError:
            checks["system_memory"] = "check_unavailable (psutil not installed)"
        except Exception as e:
            checks["system_memory"] = f"error: {str(e)}"
            logger.warning("system_memory_check_failed", error=str(e), exc_info=True)
        
        if not is_ready:
            logger.warning("readiness_check_failed", checks=checks)
        else:
            logger.info("readiness_check_passed", checks=checks)
        
        return is_ready, checks
    
    def get_component_status(self) -> Dict[str, str]:
        """
        Get detailed status of all components.
        
        Returns:
            Dictionary of component statuses
        """
        _, checks = self.check_readiness()
        return checks


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker(require_gpu: bool = True) -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(require_gpu=require_gpu)
    return _health_checker

