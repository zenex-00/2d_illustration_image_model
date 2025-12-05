"""Simple in-memory job queue for async processing"""

import uuid
import threading
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
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
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    
    def create_job(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new job
        
        Args:
            metadata: Optional job metadata
        
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
            metadata=metadata or {},
            progress=0
        )
        
        self.jobs[job_id] = job
        
        # Cleanup old jobs if we exceed max
        if len(self.jobs) > self.max_jobs:
            self._cleanup_old_jobs()
        
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
        return self.update_job(job_id, status=status, result=result, error=error)

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update job details (flexible update)
        
        Args:
            job_id: Job ID
            status: New status (string or Enum)
            progress: New progress (0-100)
            result: Optional result data
            error: Optional error message
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning("job_not_found", job_id=job_id)
            return False
        
        if status:
            # Handle string input for status
            # If it's a string that matches our Enum values, use it.
            # If it's already an Enum, use it.
            job.status = status
            
            if status == JobStatus.PROCESSING or status == "processing":
                if not job.started_at:
                    job.started_at = datetime.utcnow()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED) or status in ("completed", "failed"):
                if not job.completed_at:
                    job.completed_at = datetime.utcnow()
        
        if progress is not None:
            job.progress = progress
        
        if result is not None:
            job.result = result
        
        if error is not None:
            job.error = error
        
        logger.info("job_updated", job_id=job_id, status=str(status), progress=progress)
        return True
    
    def _cleanup_old_jobs(self):
        """Remove oldest completed/failed jobs (thread-safe)"""
        with self._lock:
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



