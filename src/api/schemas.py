"""Pydantic schemas for API requests and responses"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl


class ProcessImageRequest(BaseModel):
    """Request schema for image processing"""
    
    palette_hex_list: Optional[List[str]] = Field(
        None,
        description="Optional custom 15-color palette (hex codes). If not provided, uses default palette.",
        example=["#FFFFFF", "#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#808080", "#800000", "#008000", "#000080", "#808000", "#800080", "#008080"]
    )
    
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional configuration overrides for pipeline parameters"
    )
    
    @validator('palette_hex_list')
    def validate_palette(cls, v):
        """Validate palette has exactly 15 colors and valid hex codes"""
        if v is None:
            return v
        
        if len(v) != 15:
            raise ValueError("Palette must contain exactly 15 colors")
        
        for hex_color in v:
            if not hex_color.startswith('#'):
                raise ValueError(f"Invalid hex color format: {hex_color}. Must start with '#'")
            if len(hex_color) != 7:
                raise ValueError(f"Invalid hex color length: {hex_color}. Must be 7 characters (e.g., #FF0000)")
            try:
                int(hex_color[1:], 16)
            except ValueError:
                raise ValueError(f"Invalid hex color: {hex_color}. Must be valid hexadecimal")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "palette_hex_list": ["#FFFFFF", "#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#808080", "#800000", "#008000", "#000080", "#808000", "#800080", "#008080"],
                "config_overrides": {
                    "phase2": {
                        "sdxl": {
                            "num_inference_steps": 30
                        }
                    }
                }
            }
        }


class ProcessImageResponse(BaseModel):
    """Response schema for image processing"""
    
    status: str = Field(
        "success",
        description="Processing status"
    )
    
    svg_url: Optional[HttpUrl] = Field(
        None,
        description="URL to generated SVG file"
    )
    
    png_preview_url: Optional[HttpUrl] = Field(
        None,
        description="URL to PNG preview (2048px)"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    
    model_versions: Dict[str, str] = Field(
        ...,
        description="Model versions used in processing"
    )
    
    correlation_id: str = Field(
        ...,
        description="Correlation ID for tracing this request"
    )
    
    phase_timings: Optional[Dict[str, float]] = Field(
        None,
        description="Processing time per phase (milliseconds)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "svg_url": "https://example.com/output.svg",
                "png_preview_url": "https://example.com/preview.png",
                "processing_time_ms": 45230.5,
                "model_versions": {
                    "sdxl": "1.0",
                    "grounding_dino": "1.0"
                },
                "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                "phase_timings": {
                    "phase1": 8500.0,
                    "phase2": 28000.0,
                    "phase3": 12000.0,
                    "phase4": 5000.0
                }
            }
        }


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details error response"""
    
    type: str = Field(
        ...,
        description="URI reference that identifies the problem type",
        example="https://api.example.com/errors/validation-error"
    )
    
    title: str = Field(
        ...,
        description="Short, human-readable summary of the problem type"
    )
    
    status: int = Field(
        ...,
        description="HTTP status code"
    )
    
    detail: str = Field(
        ...,
        description="Human-readable explanation specific to this occurrence"
    )
    
    instance: Optional[str] = Field(
        None,
        description="URI reference that identifies the specific occurrence"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when error occurred"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "type": "https://api.example.com/errors/validation-error",
                "title": "Validation Error",
                "status": 400,
                "detail": "Image file size (75.5MB) exceeds maximum allowed size (50MB)",
                "instance": "/api/v1/process",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(
        ...,
        description="Health status",
        example="healthy"
    )
    
    version: str = Field(
        ...,
        description="API version"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Current server time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "3.0.0",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class ReadyResponse(BaseModel):
    """Readiness probe response"""
    
    status: str = Field(
        ...,
        description="Readiness status",
        example="ready"
    )
    
    checks: Dict[str, str] = Field(
        ...,
        description="Component health checks",
        example={
            "gpu": "available",
            "model_cache": "ready",
            "disk_space": "sufficient"
        }
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Current server time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ready",
                "checks": {
                    "gpu": "available",
                    "model_cache": "ready",
                    "disk_space": "sufficient"
                },
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class PhaseRequest(BaseModel):
    """Request schema for individual phase processing"""
    
    image_url: Optional[HttpUrl] = Field(
        None,
        description="URL to input image"
    )
    
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional configuration overrides"
    )


class JobStatusResponse(BaseModel):
    """Response schema for async job status"""
    
    job_id: str = Field(
        ...,
        description="Job correlation ID"
    )
    
    status: str = Field(
        ...,
        description="Job status: pending, processing, completed, failed"
    )
    
    progress: Optional[float] = Field(
        None,
        description="Progress percentage (0-100)"
    )
    
    result_url: Optional[HttpUrl] = Field(
        None,
        description="URL to result when completed"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message if failed"
    )
    
    created_at: datetime = Field(
        ...,
        description="Job creation timestamp"
    )
    
    updated_at: datetime = Field(
        ...,
        description="Last update timestamp"
    )






