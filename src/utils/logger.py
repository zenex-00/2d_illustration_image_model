"""Structured logging setup with JSON format and correlation IDs"""

import logging
import sys
import uuid
import io
import atexit
from typing import Optional
import structlog
from datetime import datetime

_log_handlers = []  # Track handlers for cleanup


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


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None
) -> structlog.BoundLogger:
    """Setup structured logging with JSON output and proper flushing"""
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add stdout handler with line buffering
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(getattr(logging, log_level.upper()))
    stdout_handler.flush = lambda: sys.stdout.flush()  # Force flush
    root_logger.addHandler(stdout_handler)
    _log_handlers.append(stdout_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(
                log_file,
                mode='a',
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            root_logger.addHandler(file_handler)
            _log_handlers.append(file_handler)
            logging.info(f"Logging to file: {log_file}")
        except Exception as e:
            logging.warning(f"Could not setup file logging: {e}")
    
    # Configure structlog
    if log_format == "json":
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer() if log_level.upper() == "DEBUG" else structlog.processors.JSONRenderer()
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
    
    # Register flush on exit
    atexit.register(_flush_handlers)
    
    return structlog.get_logger()


def _flush_handlers():
    """Flush all handlers on exit"""
    for handler in _log_handlers:
        try:
            handler.flush()
        except Exception:
            pass  # Ignore flush errors on shutdown


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance with optional name"""
    logger = structlog.get_logger(name or "gemini3")
    return logger


def set_correlation_id(correlation_id: str):
    """Set correlation ID for request tracing"""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)







