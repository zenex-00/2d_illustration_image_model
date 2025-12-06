"""Training job management API endpoints"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from src.api.schemas import JobStatusResponse
from src.api.job_queue import get_job_queue, JobStatus
from src.api.security import get_api_key

router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["jobs"],
    dependencies=[Depends(get_api_key)]
)

@router.get("/", response_model=List[JobStatusResponse])
async def list_jobs(limit: int = 50):
    """List recent jobs (simple in-memory implementation)"""
    queue = get_job_queue()
    # Convert dict values to list and simple sort
    # Note: In production this would query a DB
    jobs = sorted(
        queue.jobs.values(), 
        key=lambda x: x.created_at, 
        reverse=True
    )
    
    return [
        JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.metadata.get("progress"),
            result_url=f"/api/v1/results/{job.job_id}" if job.status == JobStatus.COMPLETED else None,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.completed_at or job.started_at or job.created_at
        ) 
        for job in jobs[:limit]
    ]

@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a running job.
    Note: Thread cancellation is not guaranteed in this simple implementation.
    """
    queue = get_job_queue()
    job = queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        return {"message": "Job already finished"}
        
    # Mark as failed/cancelled
    queue.update_job_status(job_id, JobStatus.FAILED, error="Cancelled by user")
    return {"message": "Job marked for cancellation"}
