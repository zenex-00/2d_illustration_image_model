"""Error response handlers for FastAPI (RFC 7807 format)"""

from datetime import datetime
from typing import Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from src.api.schemas import ErrorResponse
from src.utils.error_handler import (
    PipelineError,
    PhaseError,
    ModelLoadError,
    ValidationError,
    GPUOOMError
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Error type URIs (RFC 7807)
ERROR_TYPE_BASE = "https://api.gemini3pro.example.com/errors"
ERROR_TYPES = {
    "validation": f"{ERROR_TYPE_BASE}/validation-error",
    "pipeline": f"{ERROR_TYPE_BASE}/pipeline-error",
    "model_load": f"{ERROR_TYPE_BASE}/model-load-error",
    "gpu_oom": f"{ERROR_TYPE_BASE}/gpu-out-of-memory",
    "internal": f"{ERROR_TYPE_BASE}/internal-server-error"
}


def create_error_response(
    error: Exception,
    status_code: int,
    error_type: str,
    request: Optional[Request] = None
) -> JSONResponse:
    """
    Create RFC 7807 Problem Details error response.
    
    Args:
        error: Exception object
        status_code: HTTP status code
        error_type: Error type key (maps to ERROR_TYPES)
        request: Optional FastAPI Request object
        
    Returns:
        JSONResponse with error details
    """
    error_type_uri = ERROR_TYPES.get(error_type, ERROR_TYPES["internal"])
    
    # Get error title from exception class name
    title = error.__class__.__name__.replace("Error", " Error")
    
    # Get detail message
    detail = str(error) if str(error) else "An error occurred"
    
    # Get instance path
    instance = request.url.path if request else None
    
    error_response = ErrorResponse(
        type=error_type_uri,
        title=title,
        status=status_code,
        detail=detail,
        instance=instance,
        timestamp=datetime.utcnow()
    )
    
    logger.error(
        "api_error",
        error_type=error_type,
        status_code=status_code,
        detail=detail,
        instance=instance
    )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors"""
    errors = exc.errors()
    detail = "; ".join([f"{err['loc']}: {err['msg']}" for err in errors])
    
    return create_error_response(
        exc,
        status.HTTP_400_BAD_REQUEST,
        "validation",
        request
    )


async def validation_error_handler(
    request: Request,
    exc: ValidationError
) -> JSONResponse:
    """Handle ValidationError from pipeline"""
    return create_error_response(
        exc,
        status.HTTP_400_BAD_REQUEST,
        "validation",
        request
    )


async def model_load_error_handler(
    request: Request,
    exc: ModelLoadError
) -> JSONResponse:
    """Handle ModelLoadError"""
    return create_error_response(
        exc,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "model_load",
        request
    )


async def gpu_oom_error_handler(
    request: Request,
    exc: GPUOOMError
) -> JSONResponse:
    """Handle GPU OOM errors"""
    return create_error_response(
        exc,
        status.HTTP_507_INSUFFICIENT_STORAGE,  # Or 503
        "gpu_oom",
        request
    )


async def pipeline_error_handler(
    request: Request,
    exc: PipelineError
) -> JSONResponse:
    """Handle PipelineError"""
    return create_error_response(
        exc,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "pipeline",
        request
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle generic exceptions"""
    # Don't expose internal error details in production
    detail = "An internal server error occurred"
    
    # Log full error for debugging
    logger.exception("unhandled_exception", exc_info=exc)
    
    # Create sanitized error
    class SanitizedError(Exception):
        pass
    
    sanitized = SanitizedError(detail)
    
    return create_error_response(
        sanitized,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal",
        request
    )


def register_error_handlers(app):
    """
    Register all error handlers with FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_error_handler)
    app.add_exception_handler(ModelLoadError, model_load_error_handler)
    app.add_exception_handler(GPUOOMError, gpu_oom_error_handler)
    app.add_exception_handler(PipelineError, pipeline_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("error_handlers_registered")






