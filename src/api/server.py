import os
import time
import uuid
import logging
from typing import Dict, Any, Optional, List

# Disable xformers early if it's causing import issues
# This prevents RuntimeError when xformers is installed but incompatible with PyTorch/CUDA
# The error manifests as: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib

# For RTX 5090, always disable xformers (uses native SDPA which is faster and more compatible)
# RTX 5090 uses Blackwell (sm_120) architecture and benefits from PyTorch's native SDPA
# xformers may not be compiled for Blackwell, causing compatibility issues
gpu_model = os.getenv("GPU_MODEL", "")
if gpu_model == "RTX_5090" or os.getenv("DISABLE_XFORMERS") == "1":
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["DISABLE_XFORMERS"] = "1"
    if gpu_model == "RTX_5090":
        print("INFO: xformers disabled for RTX_5090 (using native SDPA)")
else:
    try:
        # Try to import xformers - this may fail with ImportError or RuntimeError
        # if the compiled extension is incompatible with the current PyTorch/CUDA version
        import xformers
        # If import succeeds, try to use a feature to verify it actually works
        try:
            # Try to access a module that would trigger the symbol loading
            from xformers.ops import fmha  # noqa: F401
            os.environ.setdefault("XFORMERS_DISABLED", "0")
            print("INFO: xformers enabled and compatible")
        except Exception as e:
            # xformers is installed but incompatible (undefined symbol error)
            os.environ["XFORMERS_DISABLED"] = "1"
            os.environ["DISABLE_XFORMERS"] = "1"
            print(f"WARNING: xformers is installed but incompatible, disabling: {e}")
    except Exception as e:
        # xformers is not available or incompatible, disable it
        os.environ["XFORMERS_DISABLED"] = "1"
        os.environ["DISABLE_XFORMERS"] = "1"
        print(f"INFO: xformers not available, using PyTorch SDPA instead: {e}")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse, Response, FileResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio

from src.api.schemas import (
    ProcessImageRequest, 
    ProcessImageResponse, 
    ErrorResponse, 
    HealthResponse, 
    ReadyResponse,
    JobStatusResponse
)

# Import orchestrator - this will trigger diffusers import which may fail if xformers is incompatible
# We catch the error here to provide a helpful message
try:
    from src.pipeline.orchestrator import Gemini3Pipeline
except RuntimeError as e:
    error_str = str(e).lower()
    if "xformers" in error_str or "flash_attn" in error_str or "undefined symbol" in error_str:
        import sys
        print("\n" + "="*60, file=sys.stderr)
        print("ERROR: xformers is installed but incompatible with your PyTorch/CUDA version", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"\nError details: {e}", file=sys.stderr)
        print("\nTo fix this, run:", file=sys.stderr)
        print("  pip uninstall xformers", file=sys.stderr)
        print("\nThe application will use PyTorch's built-in SDPA instead.", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
        sys.exit(1)
    raise
from src.utils.logger import get_logger, setup_logging
from src.api.job_queue import JobQueue, Job
from src.api import training_jobs

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

# Include routers
app.include_router(training_jobs.router)

# Correlation ID middleware
class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to propagate correlation IDs through the request lifecycle
    and into subprocess calls.
    """
    async def dispatch(self, request: Request, call_next):
        # Get or create correlation ID
        correlation_id = request.headers.get(
            "X-Correlation-ID",
            str(uuid.uuid4())
        )
        
        # Set in context
        from src.utils.logger import set_correlation_id
        set_correlation_id(correlation_id)
        
        # Pass to subprocesses via environment variable
        os.environ["CORRELATION_ID"] = correlation_id
        os.environ["REQUEST_PATH"] = request.url.path
        os.environ["REQUEST_METHOD"] = request.method
        
        # Log request start
        logger.info(
            "request_start",
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path
        )
        
        try:
            response = await call_next(request)
            
            logger.info(
                "request_complete",
                correlation_id=correlation_id,
                status_code=response.status_code
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
        
        except Exception as e:
            logger.error(
                "request_error",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            raise

# Timeout middleware
class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to timeout long-running requests"""
    def __init__(self, app, timeout: int = 300):
        super().__init__(app)
        self.timeout = timeout
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
            return response
        except asyncio.TimeoutError:
            logger.error(
                "request_timeout",
                path=request.url.path,
                method=request.method,
                timeout=self.timeout
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "message": f"Request exceeded {self.timeout} seconds",
                    "path": request.url.path
                }
            )

# Add correlation ID middleware (first, so it's available to all other middleware)
app.add_middleware(CorrelationIDMiddleware)

# Add timeout middleware (5 minutes default)
request_timeout = int(os.getenv("REQUEST_TIMEOUT", "300"))
app.add_middleware(TimeoutMiddleware, timeout=request_timeout)

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

# Initialize templates
templates = Jinja2Templates(directory="templates")

@app.on_event("shutdown")
async def shutdown_event():
    """Server shutdown event"""
    logger.info("server_shutdown")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main UI page (Home)"""
    return templates.TemplateResponse("home.html", {"request": request})

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
    
    @app.get("/ui/training", response_class=HTMLResponse)
    async def ui_training(request: Request):
        return templates.TemplateResponse("training.html", {"request": request})

    @app.get("/ui/inference", response_class=HTMLResponse)
    async def ui_inference(request: Request):
        return templates.TemplateResponse("inference.html", {"request": request})

    @app.get("/ui/training/jobs/{job_id}", response_class=HTMLResponse)
    async def ui_training_job(request: Request, job_id: str):
        job = job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return templates.TemplateResponse("training_job.html", {"request": request, "job": job})

    @app.get("/ui/inference/jobs/{job_id}", response_class=HTMLResponse)
    async def ui_inference_job(request: Request, job_id: str):
        job = job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return templates.TemplateResponse("inference_job.html", {"request": request, "job": job})

    @app.post("/ui/inference", response_class=RedirectResponse)
    async def ui_inference_post(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        palette_hex_list: Optional[str] = Form(None),
        request: Request = None
    ):
        """Handle inference form submission"""
        job_id = str(uuid.uuid4())
        logger.info("ui_inference_submitted", job_id=job_id)
        
        # Save input file
        input_path = os.path.join(API_OUTPUT_DIR, f"{job_id}_input.png")
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Create job
        job_queue.create_job(job_id)
        
        # Process in background
        background_tasks.add_task(
            run_pipeline_task, 
            job_id, 
            input_path, 
            palette_hex_list,
            None # config overrides
        )
        
        return RedirectResponse(url=f"/ui/inference/jobs/{job_id}", status_code=303)

    @app.post("/ui/training", response_class=RedirectResponse)
    async def ui_training_post(
        background_tasks: BackgroundTasks,
        input_files: List[UploadFile] = File(...),
        target_files: List[UploadFile] = File(...),
        learning_rate: float = Form(...),
        batch_size: int = Form(...),
        num_epochs: int = Form(...),
        rank: int = Form(...),
        alpha: int = Form(...),
        validation_split: float = Form(...),
        seed: int = Form(...),
        request: Request = None
    ):
        """Handle training form submission (Mock implementation)"""
        job_id = str(uuid.uuid4())
        logger.info("ui_training_submitted", job_id=job_id, num_inputs=len(input_files))
        
        # In a real impl, we would save files and start training
        # For now, just create a job record so the UI doesn't 404
        job_queue.create_job(job_id)
        job_queue.update_job(job_id, status="pending", progress=0)
        
        return RedirectResponse(url=f"/ui/training/jobs/{job_id}", status_code=303)

