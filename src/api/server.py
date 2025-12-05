import os
import time
import uuid
import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.api.schemas import (
    ProcessImageRequest, 
    ProcessImageResponse, 
    ErrorResponse, 
    HealthResponse, 
    ReadyResponse,
    JobStatusResponse
)
from src.pipeline.orchestrator import Gemini3Pipeline
from src.utils.logger import get_logger, setup_logging
from src.api.job_queue import JobQueue, Job

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global state
pipeline: Optional[Gemini3Pipeline] = None
job_queue = JobQueue()

# Initialize API
app = FastAPI(
    title="Gemini 3 Pro Vehicle-to-Vector API",
    description="High-fidelity vector graphic generation pipeline",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
API_OUTPUT_DIR = os.getenv("API_OUTPUT_DIR", "/tmp/gemini3_output")
os.makedirs(API_OUTPUT_DIR, exist_ok=True)

# Dependency for pipeline lazy loading
def get_pipeline():
    global pipeline
    if pipeline is None:
        logger.info("loading_pipeline")
        pipeline = Gemini3Pipeline()
    return pipeline

@app.on_event("startup")
async def startup_event():
    """Server startup event"""
    logger.info("server_starting")
    # We delay pipeline loading until first request to speed up container start

@app.on_event("shutdown")
async def shutdown_event():
    """Server shutdown event"""
    logger.info("server_shutdown")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe"""
    return HealthResponse(
        status="healthy", 
        version="3.0.0"
    )

@app.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """Readiness probe"""
    checks = {
        "disk_space": "sufficient", # Simplified check
        "gpu": "available" if os.path.exists("/dev/nvidia0") else "unavailable"
    }
    
    global pipeline
    checks["model_cache"] = "ready" if pipeline else "lazy_loaded"
    
    return ReadyResponse(
        status="ready",
        checks=checks
    )

@app.post("/api/v1/process", response_model=JobStatusResponse)
async def process_image_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    palette_hex_list: Optional[str] = Form(None), # Comma separated if passed as form
    config_overrides: Optional[str] = Form(None)  # JSON string
):
    """
    Submit an image processing job.
    Returns immediately with a job ID.
    """
    job_id = str(uuid.uuid4())
    logger.info("job_received", job_id=job_id, filename=file.filename)
    
    # Save input file
    input_path = os.path.join(API_OUTPUT_DIR, f"{job_id}_input.png")
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    # Create job
    # Note: Using a simplified dictionary until we fix imports if needed
    job_queue.create_job(job_id)
    
    # Process in background
    background_tasks.add_task(
        run_pipeline_task, 
        job_id, 
        input_path, 
        palette_hex_list,
        config_overrides
    )
    
    return JobStatusResponse(
        job_id=job_id,
        status="pending",
        created_at=time.time(), # Simplified timestamp
        updated_at=time.time()
    )

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status"""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result_url=f"/api/v1/results/{job_id}" if job.status == "completed" else None,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at
    )

@app.get("/api/v1/results/{job_id}")
async def get_job_result(job_id: str):
    """Download job result (SVG)"""
    # Simply looking for the SVG file
    svg_path = os.path.join(API_OUTPUT_DIR, f"{job_id}.svg")
    if not os.path.exists(svg_path):
         raise HTTPException(status_code=404, detail="Result not found")
         
    return FileResponse(svg_path, media_type="image/svg+xml", filename=f"vector_{job_id}.svg")

def run_pipeline_task(job_id: str, input_path: str, palette: Any, config: Any):
    """Background task wrapper"""
    try:
        job_queue.update_job(job_id, status="processing", progress=10)
        
        # Load pipeline
        pipe = get_pipeline()
        
        output_svg = os.path.join(API_OUTPUT_DIR, f"{job_id}.svg")
        output_png = os.path.join(API_OUTPUT_DIR, f"{job_id}_preview.png")
        
        # Parse optional args
        # (Handling logic for palette/config parsing would go here)
        
        svg_xml, metadata = pipe.process_image(
            input_image_path=input_path,
            output_svg_path=output_svg,
            output_png_path=output_png
        )
        
        job_queue.update_job(job_id, status="completed", progress=100)
        
    except Exception as e:
        logger.error("job_failed", job_id=job_id, error=str(e), exc_info=True)
        job_queue.update_job(job_id, status="failed", error=str(e))


# UI Mounts (if templates exist)
if os.path.exists("templates"):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
    
    @app.get("/ui")
    async def ui_home(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

