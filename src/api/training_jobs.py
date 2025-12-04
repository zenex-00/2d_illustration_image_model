"""Training job management for LoRA training"""

import uuid
import threading
from datetime import datetime
from typing import Dict, Optional, List, Any
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingJobStatus(str, Enum):
    """Training job status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    """Training job data structure"""
    job_id: str
    status: TrainingJobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    params: Dict[str, Any] = field(default_factory=dict)
    log_buffer: List[str] = field(default_factory=list)
    error: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)  # e.g., {"weights": "/path/to/weights.safetensors"}
    progress: float = 0.0  # 0.0 to 100.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_log(self, message: str, level: str = "info"):
        """Add a log message to the buffer"""
        with self._lock:
            timestamp = datetime.utcnow().isoformat()
            log_entry = f"[{timestamp}] [{level.upper()}] {message}"
            self.log_buffer.append(log_entry)
            # Keep only last 1000 log lines (in-place deletion for efficiency)
            if len(self.log_buffer) > 1000:
                del self.log_buffer[:-1000]

    def get_logs(self, tail: int = 100) -> List[str]:
        """Get last N log lines"""
        with self._lock:
            return self.log_buffer[-tail:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status.value,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "params": self.params,
                "error": self.error,
                "artifacts": self.artifacts,
                "progress": self.progress,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "log_count": len(self.log_buffer)
            }


class TrainingJobRegistry:
    """Registry for training jobs (thread-safe singleton)"""
    
    _instance: Optional['TrainingJobRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.jobs: Dict[str, TrainingJob] = {}
            self.max_jobs = 100
            self._instance_lock = threading.Lock()  # Instance-level lock for operations
            self._initialized = True
            logger.info("training_job_registry_initialized")

    def create_job(self, params: Dict[str, Any]) -> str:
        """Create a new training job"""
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            job_id=job_id,
            status=TrainingJobStatus.PENDING,
            created_at=datetime.utcnow(),
            params=params
        )
        
        with self._instance_lock:
            self.jobs[job_id] = job
            
            # Cleanup old jobs if needed
            if len(self.jobs) > self.max_jobs:
                self._cleanup_old_jobs()
        
        logger.info("training_job_created", job_id=job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID"""
        with self._instance_lock:
            return self.jobs.get(job_id)

    def update_job_status(
        self,
        job_id: str,
        status: TrainingJobStatus,
        error: Optional[str] = None
    ) -> bool:
        """Update job status"""
        job = self.get_job(job_id)
        if not job:
            logger.warning("training_job_not_found", job_id=job_id)
            return False
        
        with job._lock:
            job.status = status
            
            if status == TrainingJobStatus.PROCESSING and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in (TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED) and not job.completed_at:
                job.completed_at = datetime.utcnow()
            
            if error:
                job.error = error
        
        logger.info("training_job_status_updated", job_id=job_id, status=status.value)
        return True

    def update_job_progress(
        self,
        job_id: str,
        progress: Optional[float] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None
    ) -> bool:
        """Update job progress"""
        job = self.get_job(job_id)
        if not job:
            return False
        
        with job._lock:
            if progress is not None:
                job.progress = min(max(progress, 0.0), 100.0)
            if epoch is not None:
                job.current_epoch = epoch
            if step is not None:
                job.current_step = step
            if train_loss is not None:
                job.train_loss = train_loss
            if val_loss is not None:
                job.val_loss = val_loss
        
        return True

    def add_job_log(self, job_id: str, message: str, level: str = "info"):
        """Add log message to job"""
        job = self.get_job(job_id)
        if job:
            job.add_log(message, level)

    def set_job_artifacts(self, job_id: str, artifacts: Dict[str, str]):
        """Set job artifacts (e.g., trained weights path)"""
        job = self.get_job(job_id)
        if job:
            with job._lock:
                job.artifacts.update(artifacts)

    def _cleanup_old_jobs(self):
        """Remove oldest completed/failed jobs (thread-safe)"""
        # Note: This method should be called with _instance_lock already held
        sorted_jobs = sorted(
            self.jobs.items(),
            key=lambda x: x[1].created_at
        )
        
        removed = 0
        for job_id, job in sorted_jobs:
            # Check existence before deletion to prevent KeyError
            if job_id in self.jobs and job.status in (TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED):
                del self.jobs[job_id]
                removed += 1
                if len(self.jobs) <= self.max_jobs:
                    break
        
        if removed > 0:
            logger.info("old_training_jobs_cleaned_up", removed_count=removed)


# Global registry instance
_training_registry_instance: Optional[TrainingJobRegistry] = None
_training_registry_lock = threading.Lock()


def get_training_registry() -> TrainingJobRegistry:
    """Get or create global training job registry (thread-safe singleton)"""
    global _training_registry_instance
    
    # Fast path: if already initialized, return immediately
    if _training_registry_instance is not None:
        return _training_registry_instance
    
    # Acquire lock for initialization
    with _training_registry_lock:
        # Double-check pattern: another thread may have initialized while we waited
        if _training_registry_instance is None:
            _training_registry_instance = TrainingJobRegistry()
        return _training_registry_instance


