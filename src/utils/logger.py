"""Structured logging setup with JSON format and correlation IDs"""

import logging
import sys
import uuid
from typing import Optional
import structlog
from datetime import datetime


class CorrelationIDFilter(logging.Filter):
    """Add correlation ID to log records"""
    def __init__(self):
        super().__init__()
        self.correlation_id = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set the correlation ID for this request"""
        self.correlation_id = correlation_id
    
    def filter(self, record):
        record.correlation_id = self.correlation_id or str(uuid.uuid4())
        return True


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> structlog.BoundLogger:
    """Setup structured logging with JSON output"""
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Configure structlog
    if log_format == "json":
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance with optional name"""
    logger = structlog.get_logger(name or "gemini3")
    return logger


def set_correlation_id(correlation_id: str):
    """Set correlation ID for request tracing"""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)






