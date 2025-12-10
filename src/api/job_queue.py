"""Simple in-memory job queue for async processing"""

import uuid
import threading
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from src.utils.logger import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Job data structure"""
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: list = field(default_factory=list)
    phase_status: Dict[str, Any] = field(default_factory=dict)
    metrics_history: Dict[str, list] = field(default_factory=dict)  # Store historical metrics for charts
    phase_details: Dict[str, Any] = field(default_factory=dict)  # Store detailed phase information


class JobQueue:
    """Simple in-memory job queue"""
    
    def __init__(self, max_jobs: int = 1000):
        """
        Initialize job queue
        
        Args:
            max_jobs: Maximum number of jobs to keep in memory
        """
        self.jobs: Dict[str, Job] = {}
        self.max_jobs = max_jobs
        self._lock = threading.Lock()
        logger.info("job_queue_initialized", max_jobs=max_jobs)
    
    def create_job(self, job_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new job
        
        Args:
            job_id: Optional specific Job ID (generated if None)
            metadata: Optional job metadata
        
        Returns:
            Job ID
        """
        # Acquire lock before any job creation logic to prevent race conditions
        with self._lock:
            # Handle case where job_id was passed as first arg but might be dict (legacy compat check not strictly needed if valid types used)
            if isinstance(job_id, dict) and metadata is None:
                metadata = job_id
                job_id = None
            
            if job_id is None:
                job_id = str(uuid.uuid4())
            
            # Check if job_id already exists (race condition protection)
            if job_id in self.jobs:
                logger.warning("job_id_collision", job_id=job_id)
                # Generate new UUID if collision detected
                job_id = str(uuid.uuid4())

            job = Job(
                job_id=job_id,
                status=JobStatus.PENDING,
                created_at=datetime.utcnow(),
                metadata=metadata or {},
                progress=0.0,
                current_epoch=0,
                total_epochs=0
            )
            
            self.jobs[job_id] = job
            
            # Cleanup old jobs if we exceed max (while holding lock)
            if len(self.jobs) > self.max_jobs:
                self._cleanup_old_jobs_unlocked()
        
        logger.info("job_created", job_id=job_id)
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    # Kept for backward compatibility if needed, but update_job is preferred
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        success = self.update_job(job_id, status=status, result=result, error=error)
        if not success:
            logger.warning("job_update_failed", job_id=job_id, status=status)
        return success

    def update_job(
        self,
        job_id: str,
        status: Optional[Union[JobStatus, str]] = None,
        progress: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update job details (flexible update)
        
        Args:
            job_id: Job ID
            status: New status (string or Enum)
            progress: New progress (0-100, float)
            result: Optional result data
            error: Optional error message
            metadata: Optional metadata dict (will be merged with existing)
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                logger.warning("job_not_found", job_id=job_id)
                return False
            
            if status:
                # Normalize string input to JobStatus enum
                if isinstance(status, str):
                    try:
                        status = JobStatus(status.lower())
                    except ValueError:
                        logger.warning("invalid_status_string", job_id=job_id, status=status)
                        return False
                
                # Now status is guaranteed to be JobStatus enum
                job.status = status
                
                if status == JobStatus.PROCESSING:
                    if not job.started_at:
                        job.started_at = datetime.utcnow()
                elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    if not job.completed_at:
                        job.completed_at = datetime.utcnow()
            
            if progress is not None:
                job.progress = progress
            
            if result is not None:
                job.result = result
            
            if error is not None:
                job.error = error
            
            if metadata is not None:
                job.metadata.update(metadata)
            
            # Log status value (status is JobStatus enum after normalization, or None)
            status_value = status.value if status is not None else None
            logger.info("job_updated", job_id=job_id, status=status_value, progress=progress)
            return True
    
    def update_phase_status(
        self,
        job_id: str,
        phase: str,
        status: str,
        progress: Optional[str] = None
    ) -> bool:
        """
        Update phase-specific status for a job

        Args:
            job_id: Job ID
            phase: Phase name ("phase1" or "phase2")
            status: Phase status ("not_started" | "processing" | "completed" | "failed")
            progress: Optional progress string (e.g., "3/10")

        Returns:
            True if updated, False if job not found
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning("job_not_found", job_id=job_id)
            return False

        if not hasattr(job, 'phase_status'):
            job.phase_status = {}

        job.phase_status[f"{phase}_status"] = status

        if progress is not None:
            job.phase_status[f"{phase}_progress"] = progress

        logger.debug("phase_status_updated", job_id=job_id, phase=phase, status=status, progress=progress)
        return True

    def update_job_metrics(
        self,
        job_id: str,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        progress: Optional[float] = None,
        phase_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update job metrics for training progress tracking

        Args:
            job_id: Job ID
            train_loss: Training loss value
            val_loss: Validation loss value
            current_epoch: Current epoch number
            total_epochs: Total number of epochs
            progress: Progress percentage (0-100)
            phase_details: Detailed phase information

        Returns:
            True if updated, False if job not found
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning("job_not_found", job_id=job_id)
            return False

        # Update metrics
        if train_loss is not None:
            job.train_loss = train_loss
            # Add to metrics history for charts
            if "train_loss_history" not in job.metrics_history:
                job.metrics_history["train_loss_history"] = []
            job.metrics_history["train_loss_history"].append({
                "value": train_loss,
                "timestamp": datetime.utcnow().isoformat()
            })

        if val_loss is not None:
            job.val_loss = val_loss
            # Add to metrics history for charts
            if "val_loss_history" not in job.metrics_history:
                job.metrics_history["val_loss_history"] = []
            job.metrics_history["val_loss_history"].append({
                "value": val_loss,
                "timestamp": datetime.utcnow().isoformat()
            })

        if current_epoch is not None:
            job.current_epoch = current_epoch

        if total_epochs is not None:
            job.total_epochs = total_epochs

        if progress is not None:
            job.progress = progress

        if phase_details is not None:
            job.phase_details.update(phase_details)

        # Calculate progress percentage if epochs are available
        if job.total_epochs and job.current_epoch and job.total_epochs > 0:
            calculated_progress = (job.current_epoch / job.total_epochs) * 100
            if calculated_progress > job.progress:  # Don't decrease progress
                job.progress = calculated_progress

        logger.debug(
            "job_metrics_updated",
            job_id=job_id,
            train_loss=train_loss,
            val_loss=val_loss,
            current_epoch=current_epoch,
            progress=job.progress
        )
        return True

    def add_job_log(self, job_id: str, log_entry: str, log_type: str = "info") -> bool:
        """
        Add a log entry to a job

        Args:
            job_id: Job ID
            log_entry: Log message
            log_type: Type of log ("info", "warning", "error", "debug")

        Returns:
            True if updated, False if job not found
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning("job_not_found", job_id=job_id)
            return False

        timestamp = datetime.utcnow().isoformat()
        job.logs.append({
            "timestamp": timestamp,
            "type": log_type,
            "message": log_entry
        })

        # Limit logs to last 100 entries to prevent memory issues
        if len(job.logs) > 100:
            job.logs = job.logs[-100:]

        logger.debug("job_log_added", job_id=job_id, log_type=log_type)
        return True
    
    def _cleanup_old_jobs(self):
        """Remove oldest completed/failed jobs (thread-safe)"""
        with self._lock:
            self._cleanup_old_jobs_unlocked()
    
    def _cleanup_old_jobs_unlocked(self):
        """Remove oldest completed/failed jobs (assumes lock is already held)"""
        # Sort jobs by creation time
        sorted_jobs = sorted(
            self.jobs.items(),
            key=lambda x: x[1].created_at
        )
        
        # Remove oldest completed/failed jobs
        removed = 0
        for job_id, job in sorted_jobs:
            # Check existence before deletion to prevent KeyError
            if job_id in self.jobs and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                del self.jobs[job_id]
                removed += 1
                if len(self.jobs) <= self.max_jobs:
                    break
        
        if removed > 0:
            logger.info("old_jobs_cleaned_up", removed_count=removed)


# Global job queue instance
_job_queue_instance: Optional[JobQueue] = None
_job_queue_lock = threading.Lock()


def get_job_queue() -> JobQueue:
    """Get or create global job queue instance (thread-safe singleton)"""
    global _job_queue_instance
    
    # Fast path: if already initialized, return immediately
    if _job_queue_instance is not None:
        return _job_queue_instance
    
    # Acquire lock for initialization
    with _job_queue_lock:
        # Double-check pattern: another thread may have initialized while we waited
        if _job_queue_instance is None:
            _job_queue_instance = JobQueue()
        return _job_queue_instance



