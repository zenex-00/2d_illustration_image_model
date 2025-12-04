"""Performance metrics collection"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client_not_available", metrics_export="disabled")


@dataclass
class PhaseMetrics:
    """Metrics for a single pipeline phase"""
    phase_name: str
    start_time: datetime
    end_time: datetime
    input_shape: tuple
    output_shape: tuple
    memory_used_mb: float
    errors: Optional[List[str]] = None
    
    @property
    def duration_ms(self) -> float:
        """Calculate duration in milliseconds"""
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "phase": self.phase_name,
            "duration_ms": self.duration_ms,
            "memory_mb": self.memory_used_mb,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "status": "success" if not self.errors else "failed",
            "errors": self.errors or []
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    pipeline_requests_total = Counter(
        'pipeline_requests_total',
        'Total number of pipeline requests',
        ['status']
    )
    
    pipeline_duration_seconds = Histogram(
        'pipeline_duration_seconds',
        'Pipeline processing duration in seconds',
        ['phase'],
        buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
    )
    
    pipeline_errors_total = Counter(
        'pipeline_errors_total',
        'Total number of pipeline errors',
        ['phase', 'error_type']
    )
    
    gpu_memory_used_bytes = Gauge(
        'gpu_memory_used_bytes',
        'GPU memory used in bytes'
    )
    
    model_load_time_seconds = Histogram(
        'model_load_time_seconds',
        'Model loading time in seconds',
        ['model_name']
    )


class MetricsCollector:
    """Collects and aggregates metrics across pipeline phases"""
    
    def __init__(self):
        self.phases: List[PhaseMetrics] = []
        self.correlation_id: Optional[str] = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for this request"""
        self.correlation_id = correlation_id
    
    def record_phase(
        self,
        phase_name: str,
        start_time: datetime,
        end_time: datetime,
        input_shape: tuple,
        output_shape: tuple,
        memory_used_mb: Optional[float] = None,
        errors: Optional[List[str]] = None
    ):
        """Record metrics for a phase"""
        if memory_used_mb is None and torch.cuda.is_available():
            # Try to get memory from CUDA
            memory_used_mb = torch.cuda.max_memory_allocated() / 1024**2
        
        metric = PhaseMetrics(
            phase_name=phase_name,
            start_time=start_time,
            end_time=end_time,
            input_shape=input_shape,
            output_shape=output_shape,
            memory_used_mb=memory_used_mb or 0.0,
            errors=errors
        )
        
        self.phases.append(metric)
        
        # Export to Prometheus if available
        if PROMETHEUS_AVAILABLE:
            # Record duration
            pipeline_duration_seconds.labels(phase=phase_name).observe(
                metric.duration_ms / 1000.0
            )
            
            # Record errors if any
            if errors:
                for error in errors:
                    pipeline_errors_total.labels(
                        phase=phase_name,
                        error_type=type(error).__name__ if hasattr(error, '__name__') else "unknown"
                    ).inc()
            
            # Update GPU memory gauge
            if torch.cuda.is_available():
                gpu_memory_used_bytes.set(torch.cuda.memory_allocated())
        
        # Log metrics
        logger.info(
            "phase_completed",
            correlation_id=self.correlation_id,
            **metric.to_dict()
        )
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of all phases"""
        total_duration = sum(p.duration_ms for p in self.phases)
        total_memory = sum(p.memory_used_mb for p in self.phases)
        
        return {
            "correlation_id": self.correlation_id,
            "total_duration_ms": total_duration,
            "total_memory_mb": total_memory,
            "num_phases": len(self.phases),
            "phases": [p.to_dict() for p in self.phases],
            "status": "success" if all(not p.errors for p in self.phases) else "failed"
        }
    
    def reset(self):
        """Reset metrics collector"""
        self.phases.clear()
        self.correlation_id = None
    
    @staticmethod
    def record_request(status: str = "success"):
        """Record a pipeline request"""
        if PROMETHEUS_AVAILABLE:
            pipeline_requests_total.labels(status=status).inc()
    
    @staticmethod
    def record_model_load(model_name: str, load_time_seconds: float):
        """Record model load time"""
        if PROMETHEUS_AVAILABLE:
            model_load_time_seconds.labels(model_name=model_name).observe(load_time_seconds)

