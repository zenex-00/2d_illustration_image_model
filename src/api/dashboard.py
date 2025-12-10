"""Dashboard API endpoints for real-time monitoring"""
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.api.schemas import JobStatusResponse
from src.api.job_queue import get_job_queue, JobStatus
from src.api.security import get_api_key

router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["dashboard"],
    dependencies=[Depends(get_api_key)]
)

# Initialize templates
templates = Jinja2Templates(directory="templates")

@router.get("/metrics", response_class=HTMLResponse)
async def get_dashboard_metrics_html(request: Request):
    """Get dashboard metrics as HTML for HTMX"""
    queue = get_job_queue()

    all_jobs = list(queue.jobs.values())

    # Count active jobs
    active_jobs = len([job for job in all_jobs if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]])

    # Count completed jobs (today)
    today = datetime.utcnow().date()
    completed_jobs = len([
        job for job in all_jobs
        if job.status == JobStatus.COMPLETED and job.completed_at and job.completed_at.date() == today
    ])

    # Calculate success rate
    total_jobs = len(all_jobs)
    if total_jobs > 0:
        successful_jobs = len([job for job in all_jobs if job.status == JobStatus.COMPLETED])
        success_rate = round((successful_jobs / total_jobs) * 100, 1)
    else:
        success_rate = 0

    # Calculate average processing time
    processing_times = []
    for job in all_jobs:
        if job.status == JobStatus.COMPLETED and job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
            processing_times.append(processing_time)

    avg_time = round(sum(processing_times) / len(processing_times)) if processing_times else 0

    context = {
        "request": request,
        "active_jobs_count": active_jobs,
        "completed_jobs_count": completed_jobs,
        "success_rate": f"{success_rate}%",
        "avg_time": f"{avg_time}s",
        "total_jobs": total_jobs,
        "failed_jobs": len([job for job in all_jobs if job.status == JobStatus.FAILED])
    }

    return templates.TemplateResponse("dashboard_metrics.html", context)

@router.get("/job-status", response_class=HTMLResponse)
async def get_job_status_overview_html(request: Request):
    """Get job status overview as HTML for HTMX"""
    queue = get_job_queue()

    jobs = []
    for job in queue.jobs.values():
        # Calculate progress percentage
        progress = job.progress
        if job.status == JobStatus.COMPLETED:
            progress = 100
        elif job.status == JobStatus.FAILED:
            progress = 0

        # Format timestamps
        created_at = job.created_at.isoformat() if job.created_at else None
        started_at = job.started_at.isoformat() if job.started_at else None
        completed_at = job.completed_at.isoformat() if job.completed_at else None

        jobs.append({
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": progress,
            "created_at": created_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "error": job.error,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "train_loss": job.train_loss,
            "val_loss": job.val_loss,
            "phase_status": job.phase_status
        })

    # Sort by creation time, most recent first
    jobs.sort(key=lambda x: datetime.fromisoformat(x["created_at"]) if x["created_at"] else datetime.min, reverse=True)

    context = {
        "request": request,
        "jobs": jobs[:10]  # Return top 10 jobs
    }

    return templates.TemplateResponse("dashboard_job_status.html", context)

@router.get("/job-list", response_class=HTMLResponse)
async def get_recent_jobs_html(request: Request):
    """Get recent jobs as HTML for HTMX"""
    queue = get_job_queue()

    jobs = []
    for job in queue.jobs.values():
        # Format timestamps
        created_at = job.created_at.isoformat() if job.created_at else None
        started_at = job.started_at.isoformat() if job.started_at else None
        completed_at = job.completed_at.isoformat() if job.completed_at else None

        jobs.append({
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "created_at": created_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "error": job.error,
            "type": "training" if "training" in (job.metadata.get("type", "") or "") else "inference"
        })

    # Sort by creation time, most recent first
    jobs.sort(key=lambda x: datetime.fromisoformat(x["created_at"]) if x["created_at"] else datetime.min, reverse=True)

    context = {
        "request": request,
        "jobs": jobs[:20]  # Return top 20 jobs
    }

    return templates.TemplateResponse("dashboard_job_list.html", context)

@router.get("/metrics-data")
async def get_dashboard_metrics():
    """Get dashboard metrics: active jobs, completed jobs, success rate, avg time"""
    queue = get_job_queue()

    all_jobs = list(queue.jobs.values())

    # Count active jobs
    active_jobs = len([job for job in all_jobs if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]])

    # Count completed jobs (today)
    today = datetime.utcnow().date()
    completed_jobs = len([
        job for job in all_jobs
        if job.status == JobStatus.COMPLETED and job.completed_at and job.completed_at.date() == today
    ])

    # Calculate success rate
    total_jobs = len(all_jobs)
    if total_jobs > 0:
        successful_jobs = len([job for job in all_jobs if job.status == JobStatus.COMPLETED])
        success_rate = round((successful_jobs / total_jobs) * 100, 1)
    else:
        success_rate = 0

    # Calculate average processing time
    processing_times = []
    for job in all_jobs:
        if job.status == JobStatus.COMPLETED and job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
            processing_times.append(processing_time)

    avg_time = round(sum(processing_times) / len(processing_times)) if processing_times else 0

    return {
        "active_jobs_count": active_jobs,
        "completed_jobs_count": completed_jobs,
        "success_rate": f"{success_rate}%",
        "avg_time": f"{avg_time}s",
        "total_jobs": total_jobs,
        "failed_jobs": len([job for job in all_jobs if job.status == JobStatus.FAILED])
    }

@router.get("/job-status-data")
async def get_job_status_overview():
    """Get overview of all jobs with their status"""
    queue = get_job_queue()

    jobs = []
    for job in queue.jobs.values():
        # Calculate progress percentage
        progress = job.progress
        if job.status == JobStatus.COMPLETED:
            progress = 100
        elif job.status == JobStatus.FAILED:
            progress = 0

        # Format timestamps
        created_at = job.created_at.isoformat() if job.created_at else None
        started_at = job.started_at.isoformat() if job.started_at else None
        completed_at = job.completed_at.isoformat() if job.completed_at else None

        jobs.append({
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": progress,
            "created_at": created_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "error": job.error,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "train_loss": job.train_loss,
            "val_loss": job.val_loss,
            "phase_status": job.phase_status
        })

    # Sort by creation time, most recent first
    jobs.sort(key=lambda x: datetime.fromisoformat(x["created_at"]) if x["created_at"] else datetime.min, reverse=True)

    return jobs[:10]  # Return top 10 jobs

@router.get("/job-list-data")
async def get_recent_jobs():
    """Get recent jobs for the dashboard list"""
    queue = get_job_queue()

    jobs = []
    for job in queue.jobs.values():
        # Format timestamps
        created_at = job.created_at.isoformat() if job.created_at else None
        started_at = job.started_at.isoformat() if job.started_at else None
        completed_at = job.completed_at.isoformat() if job.completed_at else None

        jobs.append({
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "created_at": created_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "error": job.error,
            "type": "training" if "training" in (job.metadata.get("type", "") or "") else "inference"
        })

    # Sort by creation time, most recent first
    jobs.sort(key=lambda x: datetime.fromisoformat(x["created_at"]) if x["created_at"] else datetime.min, reverse=True)

    return jobs[:20]  # Return top 20 jobs

@router.get("/training-progress/{job_id}")
async def get_training_progress(job_id: str):
    """Get detailed training progress for a specific job"""
    queue = get_job_queue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return training-specific metrics
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "train_loss": job.train_loss,
        "val_loss": job.val_loss,
        "artifacts": job.artifacts,
        "logs": job.logs[-10:] if job.logs else [],  # Last 10 log entries
        "phase_status": job.phase_status,
        "metrics_history": job.metrics_history,
        "phase_details": job.phase_details
    }

@router.get("/job-metrics-history/{job_id}")
async def get_job_metrics_history(job_id: str):
    """Get historical metrics for a job for charting"""
    queue = get_job_queue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return metrics history for charting
    return {
        "job_id": job.job_id,
        "train_loss_history": job.metrics_history.get("train_loss_history", []),
        "val_loss_history": job.metrics_history.get("val_loss_history", []),
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs
    }

@router.get("/pipeline-status")
async def get_pipeline_status():
    """Get current pipeline status across all phases"""
    # This would integrate with the actual pipeline to get real status
    # For now, return example data
    return {
        "overall_progress": 45,
        "phases": {
            "phase1": {
                "status": "processing",
                "progress": 75,
                "details": "Processing semantic sanitization",
                "active_jobs": 2
            },
            "phase2": {
                "status": "processing",
                "progress": 30,
                "details": "Running generative steering",
                "active_jobs": 1
            },
            "phase3": {
                "status": "pending",
                "progress": 0,
                "details": "Waiting for phase 2 completion",
                "active_jobs": 0
            },
            "phase4": {
                "status": "pending",
                "progress": 0,
                "details": "Waiting for phase 3 completion",
                "active_jobs": 0
            }
        },
        "performance": {
            "phase1_time": 2450,
            "phase2_time": 4200,
            "phase3_time": 1800,
            "phase4_time": 3100
        }
    }

@router.get("/pipeline-metrics", response_class=HTMLResponse)
async def get_pipeline_metrics_html(request: Request):
    """Get pipeline metrics as HTML for HTMX"""
    # For now, return example data - in a real implementation this would fetch actual metrics
    context = {
        "request": request,
        "active_pipelines": 3,
        "avg_pipeline_time": "125s",
        "success_rate": "94%",
        "gpu_memory": "18.7"
    }

    # Create a simple template response manually since we don't have a specific template
    html = f"""
    <div class="pipeline-metrics">
        <div class="metric-card">
            <div class="metric-label">Active Pipelines</div>
            <div class="metric-value">{context['active_pipelines']}</div>
            <div class="metric-label">Currently Processing</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg. Pipeline Time</div>
            <div class="metric-value">{context['avg_pipeline_time']}</div>
            <div class="metric-label">Per Image</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value">{context['success_rate']}</div>
            <div class="metric-label">Overall</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Active Memory</div>
            <div class="metric-value">{context['gpu_memory']} GB</div>
            <div class="metric-label">GPU Usage</div>
        </div>
    </div>
    """
    return HTMLResponse(content=html)

@router.get("/pipeline-phases", response_class=HTMLResponse)
async def get_pipeline_phases_html(request: Request):
    """Get pipeline phase details as HTML for HTMX"""
    # Get actual phase status from the pipeline system
    pipeline_status = await get_pipeline_status()
    phases = pipeline_status.get("phases", {})

    phase_html = ""

    # Phase 1: Semantic Sanitization
    phase1 = phases.get("phase1", {})
    status1 = phase1.get("status", "pending")
    phase_html += f"""
    <div class="phase-card">
        <div class="phase-header">
            <div class="phase-title">Phase 1: Semantic Sanitization</div>
            <div class="phase-status {status1}">{status1.title()}</div>
        </div>
        <div class="phase-content">
            <p>Processing: {phase1.get('details', 'Detecting and removing prohibited objects')}</p>
            <p>Active Jobs: {phase1.get('active_jobs', 0)}</p>
            <p>Progress: {phase1.get('progress', 0)}%</p>
        </div>
    </div>
    """

    # Phase 2: Generative Steering
    phase2 = phases.get("phase2", {})
    status2 = phase2.get("status", "pending")
    phase_html += f"""
    <div class="phase-card">
        <div class="phase-header">
            <div class="phase-title">Phase 2: Generative Steering</div>
            <div class="phase-status {status2}">{status2.title()}</div>
        </div>
        <div class="phase-content">
            <p>Processing: {phase2.get('details', 'Background removal and depth estimation')}</p>
            <p>Active Jobs: {phase2.get('active_jobs', 0)}</p>
            <p>Progress: {phase2.get('progress', 0)}%</p>
        </div>
    </div>
    """

    # Phase 3: Chromatic Enforcement
    phase3 = phases.get("phase3", {})
    status3 = phase3.get("status", "pending")
    phase_html += f"""
    <div class="phase-card">
        <div class="phase-header">
            <div class="phase-title">Phase 3: Chromatic Enforcement</div>
            <div class="phase-status {status3}">{status3.title()}</div>
        </div>
        <div class="phase-content">
            <p>Processing: {phase3.get('details', 'Color quantization and palette enforcement')}</p>
            <p>Active Jobs: {phase3.get('active_jobs', 0)}</p>
            <p>Progress: {phase3.get('progress', 0)}%</p>
        </div>
    </div>
    """

    # Phase 4: Vector Reconstruction
    phase4 = phases.get("phase4", {})
    status4 = phase4.get("status", "pending")
    phase_html += f"""
    <div class="phase-card">
        <div class="phase-header">
            <div class="phase-title">Phase 4: Vector Reconstruction</div>
            <div class="phase-status {status4}">{status4.title()}</div>
        </div>
        <div class="phase-content">
            <p>Processing: {phase4.get('details', 'SVG generation and optimization')}</p>
            <p>Active Jobs: {phase4.get('active_jobs', 0)}</p>
            <p>Progress: {phase4.get('progress', 0)}%</p>
        </div>
    </div>
    """

    return HTMLResponse(content=phase_html)

@router.get("/pipeline-images", response_class=HTMLResponse)
async def get_pipeline_images_html(request: Request):
    """Get pipeline image previews as HTML for HTMX"""
    # This would return actual image previews from the pipeline
    # For now, return placeholder HTML
    html = """
    <div class="preview-item">
        <div class="preview-image" style="background: #e3f2fd; display: flex; align-items: center; justify-content: center;">📄</div>
        <div class="preview-label">Input Image</div>
    </div>
    <div class="preview-item">
        <div class="preview-image" style="background: #e8f5e9; display: flex; align-items: center; justify-content: center;">🔍</div>
        <div class="preview-label">Phase 1 Output</div>
    </div>
    <div class="preview-item">
        <div class="preview-image" style="background: #fff3e0; display: flex; align-items: center; justify-content: center;">🎨</div>
        <div class="preview-label">Phase 2 Output</div>
    </div>
    <div class="preview-item">
        <div class="preview-image" style="background: #f3e5f5; display: flex; align-items: center; justify-content: center;">🎨</div>
        <div class="preview-label">Phase 3 Output</div>
    </div>
    <div class="preview-item">
        <div class="preview-image" style="background: #e0f2f1; display: flex; align-items: center; justify-content: center;">📐</div>
        <div class="preview-label">Final SVG</div>
    </div>
    """
    return HTMLResponse(content=html)

@router.get("/system-stats")
async def get_system_stats():
    """Get system-level statistics"""
    import psutil
    import torch

    stats = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_stats()
        stats["gpu"] = {
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
            "utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        }

    return stats