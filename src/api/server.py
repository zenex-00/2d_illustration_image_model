import os
import time
import uuid
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

# xformers configuration for RTX 4090 (Ada Lovelace, sm_89)
# RTX 4090 benefits from xformers for memory-efficient attention
# This prevents RuntimeError when xformers is installed but incompatible with PyTorch/CUDA
# The error manifests as: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib

# Check if xformers should be explicitly disabled via environment variable
gpu_model = os.getenv("GPU_MODEL", "")
if os.getenv("DISABLE_XFORMERS") == "1":
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["DISABLE_XFORMERS"] = "1"
    print("INFO: xformers explicitly disabled via DISABLE_XFORMERS=1")
else:
    # For RTX 4090, try to enable xformers (it's compatible with Ada Lovelace architecture)
    try:
        # Try to import xformers - this may fail with ImportError or RuntimeError
        # if the compiled extension is incompatible with the current PyTorch/CUDA version
        import xformers
        # If import succeeds, try to use a feature to verify it actually works
        try:
            # Try to access a module that would trigger the symbol loading
            from xformers.ops import fmha  # noqa: F401
            os.environ.setdefault("XFORMERS_DISABLED", "0")
            print("INFO: xformers enabled and compatible (RTX 4090 optimized)")
        except Exception as e:
            # xformers is installed but incompatible (undefined symbol error)
            os.environ["XFORMERS_DISABLED"] = "1"
            os.environ["DISABLE_XFORMERS"] = "1"
            print(f"WARNING: xformers is installed but incompatible, disabling: {e}")
            print("INFO: Falling back to PyTorch SDPA")
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
from concurrent.futures import ThreadPoolExecutor

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
from src.api.job_queue import JobQueue, Job, get_job_queue
from src.api import training_jobs, dashboard

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global state
pipeline: Optional[Gemini3Pipeline] = None
# Use get_job_queue() to ensure we use the same singleton instance
# that training_runner.py uses, preventing "training_job_not_found" errors
job_queue = get_job_queue()

# Thread pool executor for long-running blocking tasks (like training)
# This prevents blocking the event loop
training_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="training")

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
app.include_router(dashboard.router)

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

# Add timeout middleware with validation
try:
    request_timeout = int(os.getenv("REQUEST_TIMEOUT", "600"))
    if request_timeout < 30 or request_timeout > 3600:
        logger.warning(
            "request_timeout_out_of_range",
            value=request_timeout,
            using_default=600
        )
        request_timeout = 600
except ValueError:
    logger.warning("invalid_request_timeout_config", using_default=600)
    request_timeout = 600

app.add_middleware(TimeoutMiddleware, timeout=request_timeout)

# CORS configuration
origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
origins = [origin.strip() for origin in origins_str.split(",") if origin.strip()]

# Validate CORS configuration
# Cannot use "*" with allow_credentials=True (browser security restriction)
if "*" in origins:
    allow_credentials = False
    logger.warning("cors_wildcard_with_credentials", message="Using '*' origin with credentials disabled for browser compatibility")
else:
    allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Correlation-ID"],
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
    global training_executor
    # Shutdown thread pool executor gracefully
    if training_executor:
        training_executor.shutdown(wait=True, timeout=30)
        logger.info("training_executor_shutdown")
    logger.info("server_shutdown")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main UI page (Dashboard)"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/ui", response_class=HTMLResponse)
async def ui_home(request: Request):
    """Serve the UI home page (Dashboard)"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

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
    
    # Compute updated_at from available timestamps
    updated_at = job.completed_at or job.started_at or job.created_at
        
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result_url=f"/api/v1/results/{job_id}" if job.status == "completed" else None,
        error=job.error,
        created_at=job.created_at,
        updated_at=updated_at
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
        
        # Parse config_overrides JSON string if provided
        config_overrides_dict = None
        if config:
            try:
                import json
                config_overrides_dict = json.loads(config)
                # Validate config_overrides before passing to pipeline
                from src.pipeline.orchestrator import _validate_config_overrides
                _validate_config_overrides(config_overrides_dict)
            except json.JSONDecodeError as e:
                logger.error("invalid_config_overrides_json", job_id=job_id, error=str(e))
                job_queue.update_job(job_id, status="failed", error=f"Invalid config_overrides JSON: {str(e)}")
                return
            except ValueError as e:
                logger.error("invalid_config_overrides", job_id=job_id, error=str(e))
                job_queue.update_job(job_id, status="failed", error=f"Invalid config_overrides: {str(e)}")
                return
        
        svg_xml, metadata = pipe.process_image(
            input_image_path=input_path,
            output_svg_path=output_svg,
            output_png_path=output_png,
            config_overrides=config_overrides_dict
        )
        
        job_queue.update_job(job_id, status="completed", progress=100)
        
    except Exception as e:
        logger.error("job_failed", job_id=job_id, error=str(e), exc_info=True)
        job_queue.update_job(job_id, status="failed", error=str(e))


def run_training_background_with_files(
    job_id: str,
    input_file_paths: List[str],
    target_file_paths: List[str],
    training_params: Dict[str, Any],
    temp_dir: str
):
    """Wrapper to handle file paths for training background task"""
    try:
        from src.phase2_generative_steering.training_runner import run_training_background
        
        # Pass file paths directly (training_runner now handles paths)
        run_training_background(job_id, input_file_paths, target_file_paths, training_params)
        
    except Exception as e:
        logger.error("training_wrapper_failed", job_id=job_id, error=str(e), exc_info=True)
        job_queue.update_job(job_id, status="failed", error=str(e))
    finally:
        # Cleanup temp directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning("temp_cleanup_failed", temp_dir=temp_dir, error=str(e))


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
        return templates.TemplateResponse("training_job_detailed.html", {"request": request, "job": job})
    
    @app.get("/ui/training/jobs/{job_id}/status", response_class=HTMLResponse)
    async def ui_training_job_status(request: Request, job_id: str):
        """HTMX endpoint for polling training job status"""
        job = job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return templates.TemplateResponse("training_job_partial.html", {"request": request, "job": job})
    
    @app.get("/api/training/jobs/{job_id}/images/{image_type}/{image_name}")
    async def get_training_image(job_id: str, image_type: str, image_name: str):
        """Serve intermediate training images"""
        job = job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get intermediate images directory from metadata
        intermediate_dir = job.metadata.get("intermediate_images_dir")
        if not intermediate_dir:
            raise HTTPException(status_code=404, detail="Intermediate images directory not found")
        
        # Map image types to directories
        type_to_dir = {
            "raw": "inputs",
            "phase1": "processed_inputs",
            "phase2_depth": "control_images",
            "phase2_edge": "control_images",
            "target": "targets"
        }
        
        if image_type not in type_to_dir:
            raise HTTPException(status_code=400, detail=f"Invalid image type: {image_type}")
        
        # Construct file path
        base_dir = Path(intermediate_dir) / type_to_dir[image_type]
        
        # For phase2_depth and phase2_edge, modify filename
        if image_type == "phase2_depth":
            if not image_name.startswith("depth_"):
                # Extract index from input format (e.g., "0000_image.jpg")
                try:
                    idx = image_name.split("_")[0]
                    image_name = f"depth_{idx}.png"
                except (ValueError, IndexError):
                    raise HTTPException(status_code=400, detail=f"Invalid image name format for depth: {image_name}")
        elif image_type == "phase2_edge":
            if not image_name.startswith("edge_"):
                # Extract index from input format
                try:
                    idx = image_name.split("_")[0]
                    image_name = f"edge_{idx}.png"
                except (ValueError, IndexError):
                    raise HTTPException(status_code=400, detail=f"Invalid image name format for edge: {image_name}")
        
        image_path = base_dir / image_name
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")
        
        # Determine content type
        content_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        
        return FileResponse(str(image_path), media_type=content_type)
    
    @app.get("/api/training/jobs/{job_id}/preprocessing-gallery")
    async def get_preprocessing_gallery(job_id: str):
        """Get list of available preprocessing images"""
        job = job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        intermediate_dir = job.metadata.get("intermediate_images_dir")
        if not intermediate_dir:
            return JSONResponse({
                "raw_inputs": [],
                "phase1_outputs": [],
                "phase2_depth": [],
                "phase2_edge": [],
                "targets": []
            })
        
        base_path = Path(intermediate_dir)
        gallery = {
            "raw_inputs": [],
            "phase1_outputs": [],
            "phase2_depth": [],
            "phase2_edge": [],
            "targets": []
        }
        
        # Scan directories and build image lists
        dirs = {
            "raw_inputs": base_path / "inputs",
            "phase1_outputs": base_path / "processed_inputs",
            "targets": base_path / "targets"
        }
        
        for key, dir_path in dirs.items():
            if dir_path.exists():
                for img_file in sorted(dir_path.glob("*.{jpg,jpeg,png}")):
                    gallery[key].append({
                        "name": img_file.name,
                        "url": f"/api/training/jobs/{job_id}/images/{key.replace('_', '/')}/{img_file.name}"
                    })
        
        # Handle control images separately
        control_dir = base_path / "control_images"
        if control_dir.exists():
            for depth_file in sorted(control_dir.glob("depth_*.png")):
                idx = depth_file.stem.replace("depth_", "")
                # Use a placeholder name that matches the expected format
                placeholder_name = f"{idx}_depth.png"
                gallery["phase2_depth"].append({
                    "name": depth_file.name,
                    "url": f"/api/training/jobs/{job_id}/images/phase2_depth/{placeholder_name}"
                })
            
            for edge_file in sorted(control_dir.glob("edge_*.png")):
                idx = edge_file.stem.replace("edge_", "")
                placeholder_name = f"{idx}_edge.png"
                gallery["phase2_edge"].append({
                    "name": edge_file.name,
                    "url": f"/api/training/jobs/{job_id}/images/phase2_edge/{placeholder_name}"
                })
        
        return JSONResponse(gallery)

    @app.get("/ui/pipeline", response_class=HTMLResponse)
    async def ui_pipeline_monitor(request: Request):
        return templates.TemplateResponse("pipeline_monitor.html", {"request": request})

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
        """Handle training form submission"""
        job_id = str(uuid.uuid4())
        logger.info("ui_training_submitted", job_id=job_id, num_inputs=len(input_files))
        
        # Validate file counts match
        if len(input_files) != len(target_files):
            raise HTTPException(
                status_code=400,
                detail=f"Input and target file counts must match: {len(input_files)} inputs vs {len(target_files)} targets"
            )
        
        if len(input_files) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 10 image pairs, got {len(input_files)}"
            )
        
        # Store training parameters in metadata
        training_metadata = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "rank": rank,
            "alpha": alpha,
            "validation_split": validation_split,
            "seed": seed,
            "num_input_files": len(input_files),
            "num_target_files": len(target_files)
        }
        
        # Create job with training metadata
        job_queue.create_job(job_id, metadata=training_metadata)
        job_queue.update_job(job_id, status="pending", progress=0)
        
        # Store file objects for background task (read them now before async context ends)
        # We need to read the files into memory or save them temporarily
        # Since FastAPI UploadFile objects need to be read before the request ends,
        # we'll save them to a temp location and pass paths to the background task
        
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix=f"training_{job_id}_")
        
        # Save files temporarily (background task will move them to final location)
        saved_input_files = []
        saved_target_files = []
        
        for idx, (input_file, target_file) in enumerate(zip(input_files, target_files)):
            # Save input
            input_temp_path = os.path.join(temp_dir, f"input_{idx}_{input_file.filename}")
            with open(input_temp_path, "wb") as f:
                content = await input_file.read()
                f.write(content)
            saved_input_files.append(input_temp_path)
            
            # Save target
            target_temp_path = os.path.join(temp_dir, f"target_{idx}_{target_file.filename}")
            with open(target_temp_path, "wb") as f:
                content = await target_file.read()
                f.write(content)
            saved_target_files.append(target_temp_path)
        
        # Prepare training parameters
        training_params = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "rank": rank,
            "alpha": alpha,
            "validation_split": validation_split,
            "seed": seed,
        }
        
        # Start training in background using thread pool executor
        # This ensures the long-running training task doesn't block the event loop
        # and actually executes properly
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            training_executor,
            run_training_background_with_files,
            job_id,
            saved_input_files,
            saved_target_files,
            training_params,
            temp_dir
        )
        
        return RedirectResponse(url=f"/ui/training/jobs/{job_id}", status_code=303)

