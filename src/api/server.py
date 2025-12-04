"""FastAPI server for Gemini 3 Pro Vehicle-to-Vector API"""

# Security Note: All model loading operations in this codebase enforce trust_remote_code=False
# by default. This prevents remote code execution vulnerabilities. If trust_remote_code=True
# is ever needed, it must be explicitly set and validated for each specific model.

import os
import tempfile
import base64
import threading
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Depends, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from PIL import Image

from src.api.schemas import (
    ProcessImageRequest,
    ProcessImageResponse,
    HealthResponse,
    ReadyResponse,
    ErrorResponse,
    JobStatusResponse
)
from src.api.job_queue import get_job_queue, JobStatus
from src.api.security import validate_uploaded_image, validate_base64_image
from src.api.error_responses import register_error_handlers
from src.api.rate_limiting import setup_rate_limiting, rate_limit
from src.pipeline.orchestrator import Gemini3Pipeline
from src.pipeline.config import get_config
from src.pipeline.health_checks import get_health_checker
from src.utils.image_utils import load_image, save_image
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)

# Initialize FastAPI app
config = get_config()
app = FastAPI(
    title="Gemini 3 Pro Vehicle-to-Vector API",
    version=config.get("pipeline.version", "3.0.0"),
    description="Production-grade pipeline for converting vehicle photos to vector illustrations",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup CORS
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup rate limiting
setup_rate_limiting(app)

# Setup CSRF protection (optional - can be disabled for API-only deployments)
# Uncomment to enable CSRF protection for UI forms
# from src.api.csrf import CSRFMiddleware
# app.add_middleware(CSRFMiddleware)

# Register error handlers
register_error_handlers(app)

# Global pipeline instance (lazy initialization)
_pipeline: Optional[Gemini3Pipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> Gemini3Pipeline:
    """Dependency to get pipeline instance (thread-safe singleton)"""
    global _pipeline
    
    # Fast path: if already initialized, return immediately
    if _pipeline is not None:
        return _pipeline
    
    # Acquire lock for initialization
    with _pipeline_lock:
        # Double-check pattern: another thread may have initialized while we waited
        if _pipeline is None:
            logger.info("initializing_pipeline")
            _pipeline = Gemini3Pipeline()
            logger.info("pipeline_initialized")
        return _pipeline


# Output directory for API responses
OUTPUT_DIR = Path(os.getenv("API_OUTPUT_DIR", "/tmp/gemini3_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training data directories
TRAIN_DATA_ROOT = Path(os.getenv("TRAIN_DATA_ROOT", "/tmp/gemini3_training_data"))
TRAIN_DATA_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN_OUTPUT_ROOT = Path(os.getenv("TRAIN_OUTPUT_ROOT", "/tmp/gemini3_training"))
TRAIN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Setup Jinja2 templates with auto-escaping enabled for XSS prevention
templates_dir = Path(__file__).parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir), autoescape=True)

# Mount static files (if static directory exists)
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Liveness probe endpoint.
    Returns 200 if server is running.
    """
    config = get_config()
    return HealthResponse(
        status="healthy",
        version=config.get("pipeline.version", "3.0.0"),
        timestamp=datetime.utcnow()
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness_check():
    """
    Readiness probe endpoint.
    Returns 200 if service is ready to handle requests.
    """
    health_checker = get_health_checker(require_gpu=True)
    is_ready, checks = health_checker.check_readiness()
    
    if not is_ready:
        return JSONResponse(
            status_code=503,
            content=ReadyResponse(
                status="not_ready",
                checks=checks,
                timestamp=datetime.utcnow()
            ).dict()
        )
    
    return ReadyResponse(
        status="ready",
        checks=checks,
        timestamp=datetime.utcnow()
    )


@app.post("/api/v1/jobs", tags=["Jobs"])
@rate_limit(calls="10/minute")
async def create_job(
    request: Request,
    file: UploadFile = File(..., description="Image file to process"),
    palette_hex_list: Optional[str] = Form(None, description="Comma-separated list of 15 hex colors"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Create a new processing job (async).
    
    Returns job_id for status polling.
    """
    import uuid
    from fastapi import BackgroundTasks
    job_queue = get_job_queue()
    
    try:
        # Validate uploaded image
        img = validate_uploaded_image(file)
        
        # Parse palette if provided
        palette_list = None
        if palette_hex_list:
            palette_list = [c.strip() for c in palette_hex_list.split(",")]
            if len(palette_list) != 15:
                raise HTTPException(
                    status_code=400,
                    detail="Palette must contain exactly 15 colors"
                )
        
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_input:
            img.save(tmp_input.name, 'PNG')
            input_path = tmp_input.name
        
        # Create job
        job_id = job_queue.create_job(metadata={
            "input_file": file.filename,
            "input_path": input_path,
            "palette_hex_list": palette_list
        })
        
        # Start background processing
        background_tasks.add_task(process_job_background, job_id, input_path, palette_list, None)
        
        logger.info("job_created", job_id=job_id)
        
        return {
            "job_id": job_id,
            "status": "pending",
            "status_url": f"/api/v1/jobs/{job_id}"
        }
        
    except Exception as e:
        logger.error("job_creation_failed", error=str(e), exc_info=True)
        raise


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    Get job status and results.
    
    Poll this endpoint to check job completion.
    """
    job_queue = get_job_queue()
    job = job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response_data = {
        "job_id": job.job_id,
        "status": job.status.value,
        "created_at": job.created_at,
        "updated_at": job.completed_at or job.started_at or job.created_at
    }
    
    if job.status == JobStatus.COMPLETED and job.result:
        response_data["result_url"] = job.result.get("svg_url")
        response_data["png_preview_url"] = job.result.get("png_preview_url")
    
    if job.status == JobStatus.FAILED and job.error:
        response_data["error"] = job.error
    
    return response_data


async def process_job_background(job_id: str, input_path: str, palette_list: Optional[List[str]], config_overrides: Optional[Dict] = None):
    """Background task to process job"""
    job_queue = get_job_queue()
    pipeline = get_pipeline()
    
    try:
        # Update status to processing
        job_queue.update_job_status(job_id, JobStatus.PROCESSING)
        
        # Generate output paths
        output_id = job_id[:8]
        svg_path = OUTPUT_DIR / f"{output_id}.svg"
        png_path = OUTPUT_DIR / f"{output_id}.png"
        
        # Process image
        logger.info("job_processing_start", job_id=job_id)
        
        svg_xml, metadata = pipeline.process_image(
            input_image_path=input_path,
            palette_hex_list=palette_list,
            output_svg_path=str(svg_path),
            output_png_path=str(png_path),
            config_overrides=config_overrides
        )
        
        # Clean up temp input file
        if os.path.exists(input_path):
            os.unlink(input_path)
        
        # Update job with results
        result = {
            "svg_url": f"/api/v1/download/{output_id}.svg",
            "png_preview_url": f"/api/v1/download/{output_id}.png",
            "processing_time_ms": metadata.get("total_processing_time_ms", 0),
            "correlation_id": metadata.get("correlation_id", job_id),
            "phase_timings": {
                "phase1": metadata.get("phase1", {}).get("processing_time_ms", 0),
                "phase2": metadata.get("phase2", {}).get("processing_time_ms", 0),
                "phase3": metadata.get("phase3", {}).get("processing_time_ms", 0),
                "phase4": metadata.get("phase4", {}).get("processing_time_ms", 0)
            }
        }
        
        job_queue.update_job_status(job_id, JobStatus.COMPLETED, result=result)
        logger.info("job_completed", job_id=job_id)
        
    except Exception as e:
        logger.error("job_processing_failed", job_id=job_id, error=str(e), exc_info=True)
        job_queue.update_job_status(job_id, JobStatus.FAILED, error=str(e))
    finally:
        # Always clean up temp file, even on error
        if os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception as cleanup_error:
                logger.warning("temp_file_cleanup_failed", path=input_path, error=str(cleanup_error))


@app.post("/api/v1/process", response_model=ProcessImageResponse, tags=["Processing"])
@rate_limit(calls="10/minute")
async def process_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to process"),
    palette_hex_list: Optional[str] = Form(None, description="Comma-separated list of 15 hex colors"),
    pipeline: Gemini3Pipeline = Depends(get_pipeline)
):
    """
    Process image through full pipeline (all 4 phases) - synchronous.
    
    For async processing, use POST /api/v1/jobs instead.
    
    Returns SVG and PNG preview URLs.
    """
    import uuid
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    try:
        # Validate uploaded image
        img = validate_uploaded_image(file)
        
        # Parse palette if provided
        palette_list = None
        if palette_hex_list:
            palette_list = [c.strip() for c in palette_hex_list.split(",")]
            if len(palette_list) != 15:
                raise HTTPException(
                    status_code=400,
                    detail="Palette must contain exactly 15 colors"
                )
        
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_input:
            img.save(tmp_input.name, 'PNG')
            input_path = tmp_input.name
        
        # Generate output paths
        output_id = correlation_id[:8]
        svg_path = OUTPUT_DIR / f"{output_id}.svg"
        png_path = OUTPUT_DIR / f"{output_id}.png"
        
        # Process image
        logger.info("api_process_start", correlation_id=correlation_id, input_file=file.filename)
        
        svg_xml, metadata = pipeline.process_image(
            input_image_path=input_path,
            palette_hex_list=palette_list,
            output_svg_path=str(svg_path),
            output_png_path=str(png_path)
        )
        
        # Get model versions from config
        from src.pipeline.config import get_config
        config = get_config()
        model_versions = {
            "pipeline": config.get("pipeline.version", "3.0.0")
        }
        
        # Build response
        response = ProcessImageResponse(
            status="success",
            svg_url=f"/api/v1/download/{output_id}.svg",
            png_preview_url=f"/api/v1/download/{output_id}.png",
            processing_time_ms=metadata.get("total_processing_time_ms", 0),
            model_versions=model_versions,
            correlation_id=correlation_id,
            phase_timings={
                "phase1": metadata.get("phase1", {}).get("processing_time_ms", 0),
                "phase2": metadata.get("phase2", {}).get("processing_time_ms", 0),
                "phase3": metadata.get("phase3", {}).get("processing_time_ms", 0),
                "phase4": metadata.get("phase4", {}).get("processing_time_ms", 0)
            }
        )
        
        logger.info("api_process_complete", correlation_id=correlation_id)
        return response
        
    except Exception as e:
        logger.error("api_process_failed", correlation_id=correlation_id, error=str(e), exc_info=True)
        raise
    finally:
        # Always clean up temp input file
        if 'input_path' in locals() and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception as cleanup_error:
                logger.warning("temp_file_cleanup_failed", path=input_path, error=str(cleanup_error))


@app.get("/api/v1/download/{filename}", tags=["Download"])
async def download_file(filename: str):
    """
    Download processed files (SVG or PNG).
    """
    from src.utils.path_validation import sanitize_filename, validate_path_within_directory
    
    try:
        # Sanitize filename first
        sanitized_filename = sanitize_filename(filename)
        
        # Check extension before path resolution
        if not sanitized_filename.endswith(('.svg', '.png')):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Construct path
        file_path = OUTPUT_DIR / sanitized_filename
        
        # Validate path is within OUTPUT_DIR (prevents path traversal)
        try:
            validated_path = validate_path_within_directory(
                file_path,
                OUTPUT_DIR,
                must_exist=True
            )
        except ValueError as e:
            logger.warning("path_traversal_attempt", filename=filename, error=str(e))
            raise HTTPException(status_code=403, detail="Access denied")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=validated_path,
            filename=sanitized_filename,
            media_type="image/svg+xml" if sanitized_filename.endswith('.svg') else "image/png"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("download_file_error", filename=filename, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/phase1", tags=["Processing"])
@rate_limit(calls="20/minute")
async def process_phase1(
    request: Request,
    file: UploadFile = File(...),
    pipeline: Gemini3Pipeline = Depends(get_pipeline)
):
    """Process Phase I: Semantic Sanitization only"""
    import uuid
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    try:
        img = validate_uploaded_image(file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_input:
            img.save(tmp_input.name, 'PNG')
            input_path = tmp_input.name
        
        # Load image and process Phase I
        raw_img = load_image(input_path)
        clean_plate, metadata = pipeline.phase1.sanitize(
            raw_img,
            correlation_id=correlation_id
        )
        
        # Save output
        output_id = correlation_id[:8]
        output_path = OUTPUT_DIR / f"{output_id}_phase1.png"
        save_image(clean_plate, output_path)
        
        return {
            "status": "success",
            "output_url": f"/api/v1/download/{output_id}_phase1.png",
            "correlation_id": correlation_id,
            "metadata": metadata
        }
    except Exception as e:
        logger.error("api_phase1_failed", correlation_id=correlation_id, error=str(e), exc_info=True)
        raise
    finally:
        # Always clean up temp input file
        if 'input_path' in locals() and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception as cleanup_error:
                logger.warning("temp_file_cleanup_failed", path=input_path, error=str(cleanup_error))


@app.post("/api/v1/phase2", tags=["Processing"])
@rate_limit(calls="20/minute")
async def process_phase2(
    request: Request,
    file: UploadFile = File(...),
    pipeline: Gemini3Pipeline = Depends(get_pipeline)
):
    """Process Phase II: Generative Steering only"""
    import uuid
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    try:
        img = validate_uploaded_image(file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_input:
            img.save(tmp_input.name, 'PNG')
            input_path = tmp_input.name
        
        raw_img = load_image(input_path)
        vector_raster, metadata = pipeline.phase2.generate(
            raw_img,
            correlation_id=correlation_id
        )
        
        output_id = correlation_id[:8]
        output_path = OUTPUT_DIR / f"{output_id}_phase2.png"
        save_image(vector_raster, output_path)
        
        return {
            "status": "success",
            "output_url": f"/api/v1/download/{output_id}_phase2.png",
            "correlation_id": correlation_id,
            "metadata": metadata
        }
    except Exception as e:
        logger.error("api_phase2_failed", correlation_id=correlation_id, error=str(e), exc_info=True)
        raise
    finally:
        # Always clean up temp input file
        if 'input_path' in locals() and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception as cleanup_error:
                logger.warning("temp_file_cleanup_failed", path=input_path, error=str(cleanup_error))


@app.post("/api/v1/phase3", tags=["Processing"])
@rate_limit(calls="20/minute")
async def process_phase3(
    request: Request,
    file: UploadFile = File(...),
    pipeline: Gemini3Pipeline = Depends(get_pipeline)
):
    """Process Phase III: Chromatic Enforcement only"""
    import uuid
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    try:
        img = validate_uploaded_image(file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_input:
            img.save(tmp_input.name, 'PNG')
            input_path = tmp_input.name
        
        raw_img = load_image(input_path)
        quantized, metadata = pipeline.phase3.enforce(
            raw_img,
            correlation_id=correlation_id
        )
        
        output_id = correlation_id[:8]
        output_path = OUTPUT_DIR / f"{output_id}_phase3.png"
        save_image(quantized, output_path)
        
        return {
            "status": "success",
            "output_url": f"/api/v1/download/{output_id}_phase3.png",
            "correlation_id": correlation_id,
            "metadata": metadata
        }
    except Exception as e:
        logger.error("api_phase3_failed", correlation_id=correlation_id, error=str(e), exc_info=True)
        raise
    finally:
        # Always clean up temp input file
        if 'input_path' in locals() and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception as cleanup_error:
                logger.warning("temp_file_cleanup_failed", path=input_path, error=str(cleanup_error))


@app.post("/api/v1/phase4", tags=["Processing"])
@rate_limit(calls="20/minute")
async def process_phase4(
    request: Request,
    file: UploadFile = File(...),
    pipeline: Gemini3Pipeline = Depends(get_pipeline)
):
    """Process Phase IV: Vector Reconstruction only"""
    import uuid
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    try:
        img = validate_uploaded_image(file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_input:
            img.save(tmp_input.name, 'PNG')
            input_path = tmp_input.name
        
        raw_img = load_image(input_path)
        svg_xml, metadata = pipeline.phase4.vectorize(
            raw_img,
            output_path=None,
            correlation_id=correlation_id
        )
        
        output_id = correlation_id[:8]
        output_path = OUTPUT_DIR / f"{output_id}_phase4.svg"
        with open(output_path, 'w') as f:
            f.write(svg_xml)
        
        return {
            "status": "success",
            "output_url": f"/api/v1/download/{output_id}_phase4.svg",
            "correlation_id": correlation_id,
            "metadata": metadata
        }
    except Exception as e:
        logger.error("api_phase4_failed", correlation_id=correlation_id, error=str(e), exc_info=True)
        raise
    finally:
        # Always clean up temp input file
        if 'input_path' in locals() and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception as cleanup_error:
                logger.warning("temp_file_cleanup_failed", path=input_path, error=str(cleanup_error))


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    logger.info("api_startup")
    # Pipeline will be initialized lazily on first request


@app.get("/metrics", tags=["Monitoring"])
async def metrics_endpoint():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus format.
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={"error": "Prometheus client not available"}
        )


# ==================== UI Routes ====================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root(request: Request):
    """Root route - redirects to UI"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui", status_code=302)


@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
async def ui_home(request: Request):
    """Main UI dashboard"""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/ui/training", response_class=HTMLResponse, tags=["UI"])
async def ui_training(request: Request):
    """Training UI page"""
    from src.api.training_jobs import get_training_registry
    registry = get_training_registry()
    
    # Get list of recent jobs
    recent_jobs = []
    for job_id, job in list(registry.jobs.items())[-10:]:
        recent_jobs.append(job.to_dict())
    
    # Get list of available trained models
    trained_models = []
    if TRAIN_OUTPUT_ROOT.exists():
        for job_dir in TRAIN_OUTPUT_ROOT.iterdir():
            if job_dir.is_dir():
                weights_file = list(job_dir.glob("*.safetensors"))
                if weights_file:
                    trained_models.append({
                        "job_id": job_dir.name,
                        "path": str(weights_file[0]),
                        "name": weights_file[0].name
                    })
    
    return templates.TemplateResponse("training.html", {
        "request": request,
        "recent_jobs": recent_jobs,
        "trained_models": trained_models
    })


@app.post("/ui/training", tags=["UI"])
async def ui_training_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    input_files: List[UploadFile] = File(...),
    target_files: List[UploadFile] = File(...),
    learning_rate: float = Form(1e-4),
    batch_size: int = Form(1),
    num_epochs: int = Form(10),
    rank: int = Form(32),
    alpha: int = Form(16),
    validation_split: float = Form(0.2),
    seed: int = Form(42)
):
    """Submit training job"""
    from src.api.training_jobs import get_training_registry
    from src.phase2_generative_steering.training_runner import run_training_background
    
    registry = get_training_registry()
    
    try:
        # Validate file pairs
        if len(input_files) != len(target_files):
            raise HTTPException(status_code=400, detail="Number of input and target files must match")
        
        if len(input_files) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient image pairs for training. Minimum required: 10 pairs. Provided: {len(input_files)} pairs. Please upload at least 10 matching input/output image pairs."
            )
        
        # Create job
        params = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "rank": rank,
            "alpha": alpha,
            "validation_split": validation_split,
            "seed": seed,
            "output_path": "vector_style_lora.safetensors"
        }
        
        job_id = registry.create_job(params)
        
        # Save uploaded files to job-specific directory
        job_data_dir = TRAIN_DATA_ROOT / job_id
        inputs_dir = job_data_dir / "inputs"
        targets_dir = job_data_dir / "targets"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        targets_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (input_file, target_file) in enumerate(zip(input_files, target_files)):
            # Validate images - allow very large dimensions for training (max 65536px)
            # Training will resize to 1024x1024 anyway, so we allow large uploads
            # We'll resize down to 2048px max after validation to save storage
            input_img = validate_uploaded_image(input_file, max_dimension=65536)
            target_img = validate_uploaded_image(target_file, max_dimension=65536)
            
            # Resize very large images to reduce storage/processing time
            # Keep aspect ratio, max dimension 2048px for storage efficiency
            MAX_TRAINING_DIMENSION = 2048
            input_max_dim = max(input_img.size)
            if input_max_dim > MAX_TRAINING_DIMENSION:
                original_size = input_img.size
                scale = MAX_TRAINING_DIMENSION / input_max_dim
                new_size = (int(input_img.size[0] * scale), int(input_img.size[1] * scale))
                input_img = input_img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info("resized_input_image", original_size=original_size, new_size=new_size)
            
            target_max_dim = max(target_img.size)
            if target_max_dim > MAX_TRAINING_DIMENSION:
                original_size = target_img.size
                scale = MAX_TRAINING_DIMENSION / target_max_dim
                new_size = (int(target_img.size[0] * scale), int(target_img.size[1] * scale))
                target_img = target_img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info("resized_target_image", original_size=original_size, new_size=new_size)
            
            # Convert RGBA to RGB if needed (for JPEG compatibility), otherwise keep original format
            # Always save training images as PNG to preserve quality and support transparency
            if input_img.mode == 'RGBA':
                # Convert RGBA to RGB with white background for better training compatibility
                rgb_img = Image.new('RGB', input_img.size, (255, 255, 255))
                rgb_img.paste(input_img, mask=input_img.split()[3])  # Use alpha channel as mask
                input_img = rgb_img
            elif input_img.mode not in ('RGB', 'L'):
                input_img = input_img.convert('RGB')
            
            if target_img.mode == 'RGBA':
                # Convert RGBA to RGB with white background
                rgb_img = Image.new('RGB', target_img.size, (255, 255, 255))
                rgb_img.paste(target_img, mask=target_img.split()[3])
                target_img = rgb_img
            elif target_img.mode not in ('RGB', 'L'):
                target_img = target_img.convert('RGB')
            
            # Save files as PNG (lossless, supports all modes)
            input_path = inputs_dir / f"{i:04d}_{Path(input_file.filename).stem}.png"
            target_path = targets_dir / f"{i:04d}_{Path(target_file.filename).stem}.png"
            
            input_img.save(input_path, 'PNG')
            target_img.save(target_path, 'PNG')
        
        registry.add_job_log(job_id, f"Saved {len(input_files)} image pairs", level="info")
        
        # Start background training
        background_tasks.add_task(
            run_training_background,
            job_id,
            str(job_data_dir),
            params
        )
        
        # Redirect to job status page
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/ui/training/jobs/{job_id}", status_code=303)
        
    except Exception as e:
        logger.error("training_submit_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ui/training/jobs/{job_id}", response_class=HTMLResponse, tags=["UI"])
async def ui_training_job(request: Request, job_id: str):
    """Training job status page"""
    from src.api.training_jobs import get_training_registry
    registry = get_training_registry()
    
    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Get logs (last 100 lines)
    logs = job.get_logs(tail=100)
    
    return templates.TemplateResponse("training_job.html", {
        "request": request,
        "job": job.to_dict(),
        "logs": logs
    })


@app.get("/ui/training/jobs/{job_id}/status", tags=["UI"])
async def ui_training_job_status(request: Request, job_id: str):
    """HTMX endpoint for training job status updates"""
    from src.api.training_jobs import get_training_registry
    registry = get_training_registry()
    
    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    logs = job.get_logs(tail=50)
    
    # Use template to render properly
    return templates.TemplateResponse("training_job_partial.html", {
        "request": request,
        "job": job.to_dict(),
        "logs": logs
    })


@app.get("/ui/inference", response_class=HTMLResponse, tags=["UI"])
async def ui_inference(request: Request):
    """Inference UI page"""
    from src.api.training_jobs import get_training_registry
    registry = get_training_registry()
    
    # Get list of available trained models
    trained_models = []
    if TRAIN_OUTPUT_ROOT.exists():
        for job_dir in TRAIN_OUTPUT_ROOT.iterdir():
            if job_dir.is_dir():
                weights_file = list(job_dir.glob("*.safetensors"))
                if weights_file:
                    trained_models.append({
                        "job_id": job_dir.name,
                        "path": str(weights_file[0]),
                        "name": weights_file[0].name
                    })
    
    return templates.TemplateResponse("inference.html", {
        "request": request,
        "trained_models": trained_models
    })


@app.post("/ui/inference", tags=["UI"])
async def ui_inference_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    lora_checkpoint: Optional[str] = Form(None),
    phase1_enabled: bool = Form(True),
    phase2_enabled: bool = Form(True),
    phase2_background_removal: bool = Form(True),
    phase2_depth_estimation: bool = Form(True),
    phase2_edge_detection: bool = Form(True),
    phase3_enabled: bool = Form(True),
    phase3_upscaler: bool = Form(True),
    phase3_quantization: bool = Form(True),
    phase3_noise_removal: bool = Form(True),
    phase4_enabled: bool = Form(True),
    phase4_centerline: bool = Form(True),
    palette_hex_list: Optional[str] = Form(None),
    prompt_override: Optional[str] = Form(None),
    depth_weight: Optional[float] = Form(None),
    canny_weight: Optional[float] = Form(None)
):
    """Submit inference job with preprocessing toggles"""
    try:
        # Validate uploaded image
        img = validate_uploaded_image(file)
        
        # Parse palette if provided
        palette_list = None
        if palette_hex_list:
            palette_list = [c.strip() for c in palette_hex_list.split(",")]
            if len(palette_list) != 15:
                raise HTTPException(status_code=400, detail="Palette must contain exactly 15 colors")
        
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_input:
            img.save(tmp_input.name, 'PNG')
            input_path = tmp_input.name
        
        # Create config overrides from toggles
        config_overrides = {
            "phases": {
                "phase1": {"enabled": phase1_enabled},
                "phase2": {
                    "enabled": phase2_enabled,
                    "background_removal": {"enabled": phase2_background_removal},
                    "depth_estimation": {"enabled": phase2_depth_estimation},
                    "edge_detection": {"enabled": phase2_edge_detection},
                    "controlnet": {}
                },
                "phase3": {
                    "enabled": phase3_enabled,
                    "upscaler": {"enabled": phase3_upscaler},
                    "quantization": {"enabled": phase3_quantization},
                    "noise_removal": {"enabled": phase3_noise_removal}
                },
                "phase4": {
                    "enabled": phase4_enabled,
                    "centerline": {"enabled": phase4_centerline}
                }
            }
        }
        
        # Add ControlNet weights if provided
        if depth_weight is not None:
            config_overrides["phases"]["phase2"]["controlnet"]["depth_weight"] = depth_weight
        if canny_weight is not None:
            config_overrides["phases"]["phase2"]["controlnet"]["canny_weight"] = canny_weight
        
        # Add prompt override if provided
        if prompt_override:
            config_overrides["phases"]["phase2"]["prompt_override"] = prompt_override
        
        # Create job using existing job queue
        job_queue = get_job_queue()
        job_id = job_queue.create_job(metadata={
            "input_file": file.filename,
            "input_path": input_path,
            "palette_hex_list": palette_list,
            "config_overrides": config_overrides,
            "lora_checkpoint": lora_checkpoint
        })
        
        # Start background processing (need to modify process_job_background to accept config_overrides)
        background_tasks.add_task(process_job_background_ui, job_id, input_path, palette_list, config_overrides)
        
        # Redirect to job status page
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/ui/inference/jobs/{job_id}", status_code=303)
        
    except Exception as e:
        logger.error("inference_submit_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ui/inference/jobs/{job_id}", response_class=HTMLResponse, tags=["UI"])
async def ui_inference_job(request: Request, job_id: str):
    """Inference job status page"""
    job_queue = get_job_queue()
    job = job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Convert job to dict for template
    job_dict = {
        "job_id": job.job_id,
        "status": job.status.value,  # Enum always has .value attribute
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at.isoformat() if hasattr(job.created_at, "isoformat") else str(job.created_at),
        "started_at": job.started_at.isoformat() if job.started_at and hasattr(job.started_at, "isoformat") else (str(job.started_at) if job.started_at else None),
        "completed_at": job.completed_at.isoformat() if job.completed_at and hasattr(job.completed_at, "isoformat") else (str(job.completed_at) if job.completed_at else None)
    }
    
    return templates.TemplateResponse("inference_job.html", {
        "request": request,
        "job": job_dict
    })


@app.get("/ui/inference/jobs/{job_id}/status", response_class=HTMLResponse, tags=["UI"])
async def ui_inference_job_status(request: Request, job_id: str):
    """HTMX endpoint for inference job status updates"""
    job_queue = get_job_queue()
    job = job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Convert job to dict for template
    job_dict = {
        "job_id": job.job_id,
        "status": job.status.value,  # Enum always has .value attribute
        "result": job.result,
        "error": job.error
    }
    
    # Return partial HTML for HTMX
    return templates.TemplateResponse("inference_job_partial.html", {
        "request": request,
        "job": job_dict
    })


async def process_job_background_ui(job_id: str, input_path: str, palette_list: Optional[List[str]], config_overrides: Dict):
    """Background task for UI inference jobs with config overrides"""
    job_queue = get_job_queue()
    pipeline = get_pipeline()
    
    try:
        job_queue.update_job_status(job_id, JobStatus.PROCESSING)
        
        output_id = job_id[:8]
        svg_path = OUTPUT_DIR / f"{output_id}.svg"
        png_path = OUTPUT_DIR / f"{output_id}.png"
        
        logger.info("ui_job_processing_start", job_id=job_id)
        
        svg_xml, metadata = pipeline.process_image(
            input_image_path=input_path,
            palette_hex_list=palette_list,
            output_svg_path=str(svg_path),
            output_png_path=str(png_path),
            config_overrides=config_overrides
        )
        
        if os.path.exists(input_path):
            os.unlink(input_path)
        
        result = {
            "svg_url": f"/api/v1/download/{output_id}.svg",
            "png_preview_url": f"/api/v1/download/{output_id}.png",
            "processing_time_ms": metadata.get("total_processing_time_ms", 0),
            "correlation_id": metadata.get("correlation_id", job_id)
        }
        
        job_queue.update_job_status(job_id, JobStatus.COMPLETED, result=result)
        logger.info("ui_job_completed", job_id=job_id)
        
    except Exception as e:
        logger.error("ui_job_processing_failed", job_id=job_id, error=str(e), exc_info=True)
        job_queue.update_job_status(job_id, JobStatus.FAILED, error=str(e))
    finally:
        if os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception as cleanup_error:
                logger.warning("temp_file_cleanup_failed", path=input_path, error=str(cleanup_error))


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("api_shutdown")
    global _pipeline
    if _pipeline is not None:
        # Cleanup if needed
        _pipeline = None

