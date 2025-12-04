"""Rate limiting for API endpoints"""

import os
from typing import Callable
from functools import wraps
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request, HTTPException
from src.api.security import get_rate_limit_key
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default rate limits (configurable via environment)
DEFAULT_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10/minute")
DEFAULT_RATE_LIMIT_PER_IP = os.getenv("API_RATE_LIMIT_PER_IP", "10/minute")

# Create limiter instance
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[DEFAULT_RATE_LIMIT]
)


def get_rate_limit_key_func(request: Request) -> str:
    """
    Get rate limit key for request.
    Uses IP address by default, can be extended for user-based limiting.
    """
    return get_remote_address(request)


def rate_limit(
    calls: str = DEFAULT_RATE_LIMIT_PER_IP,
    key_func: Callable = None
):
    """
    Decorator for rate limiting endpoints.
    
    Args:
        calls: Rate limit string (e.g., "10/minute", "100/hour")
        key_func: Optional function to get rate limit key (defaults to IP address)
    
    Example:
        @app.post("/api/v1/process")
        @rate_limit(calls="5/minute")
        async def process_image(...):
            ...
    """
    if key_func is None:
        key_func = get_rate_limit_key_func
    
    def decorator(func):
        # Apply slowapi rate limit decorator
        return limiter.limit(calls, key_func=key_func)(func)
    
    return decorator


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded"""
    logger.warning(
        "rate_limit_exceeded",
        client_ip=get_remote_address(request),
        limit=exc.detail
    )
    
    # Calculate retry after seconds
    retry_after = int(exc.retry_after) if hasattr(exc, 'retry_after') else 60
    
    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded: {exc.detail}. Retry after {retry_after} seconds.",
        headers={"Retry-After": str(retry_after)}
    )


def setup_rate_limiting(app):
    """
    Setup rate limiting middleware and exception handler.
    
    Args:
        app: FastAPI application instance
    """
    # Add slowapi middleware
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    
    logger.info("rate_limiting_configured", default_limit=DEFAULT_RATE_LIMIT)






