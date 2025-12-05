"""Error handling and resilience patterns for the pipeline"""

import torch
from typing import Optional, Callable, Any
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)


class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass


class PhaseError(PipelineError):
    """Error in a specific phase"""
    def __init__(self, phase: str, message: str, original_error: Optional[Exception] = None):
        self.phase = phase
        self.original_error = original_error
        super().__init__(f"Phase {phase} failed: {message}")


class ModelLoadError(PhaseError):
    """Error loading a model"""
    pass


class GPUOOMError(PhaseError):
    """GPU out of memory error"""
    pass


class ValidationError(PhaseError):
    """Validation error (quality checks failed)"""
    pass


def handle_gpu_oom(func: Callable) -> Callable:
    """Decorator to handle GPU OOM errors with automatic recovery"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            # Clear cache and retry once
            torch.cuda.empty_cache()
            try:
                return func(*args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                raise GPUOOMError(
                    phase=func.__name__,
                    message="GPU out of memory after cache clear",
                    original_error=e
                )
    return wrapper


def retry_on_failure(
    max_attempts: int = 3,
    wait_min: float = 2.0,
    wait_max: float = 10.0,
    exceptions: tuple = (Exception,)
):
    """Decorator factory for retry logic with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=2, min=wait_min, max=wait_max),
            retry=retry_if_exception_type(exceptions),
            reraise=True
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def recover_from_error(
    fallback_func: Optional[Callable] = None,
    error_message: str = "Operation failed"
):
    """Decorator to provide fallback behavior on error"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                raise PhaseError(
                    phase=func.__name__,
                    message=f"{error_message}: {str(e)}",
                    original_error=e
                )
        return wrapper
    return decorator







