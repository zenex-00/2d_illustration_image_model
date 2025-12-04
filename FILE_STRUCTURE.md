# File Structure & Component Functions

## Overview

This document provides a comprehensive guide to the codebase structure, explaining the purpose and function of each directory, module, and key file in the Gemini 3 Pro Vehicle-to-Vector Pipeline.

---

## Root Directory Structure

```
image_generation/
├── src/                    # Main source code
├── cli/                    # Command-line interface
├── templates/              # HTML templates for web UI
├── tests/                  # Test suite
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── run.py                  # Main server entry point
├── run.bat                 # Windows startup script
├── run.sh                  # Linux/Mac startup script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container definition
├── README.md               # Project documentation
└── [documentation files]   # Various .md files
```

---

## Source Code Directory (`src/`)

### `src/api/` - FastAPI Web Server & API Layer

**Purpose**: HTTP API server, request handling, job management, security

#### `server.py` (Main API Server)
- **Function**: FastAPI application definition and route handlers
- **Key Responsibilities**:
  - Initialize FastAPI app with CORS, rate limiting, error handlers
  - Define REST API endpoints (`/api/v1/process`, `/api/v1/jobs`, etc.)
  - Handle file uploads and validation
  - Manage background job processing
  - Serve web UI templates
  - Coordinate with pipeline orchestrator
- **Key Endpoints**:
  - `POST /api/v1/process` - Synchronous image processing
  - `POST /api/v1/jobs` - Create async processing job
  - `GET /api/v1/jobs/{job_id}` - Get job status
  - `GET /api/v1/download/{filename}` - Download results
  - `GET /ui`, `/ui/inference`, `/ui/training` - Web UI pages
- **Dependencies**: FastAPI, pipeline orchestrator, job queue, security modules

#### `job_queue.py` (Job Management)
- **Function**: In-memory job queue for async processing
- **Key Classes**:
  - `JobStatus` (Enum): PENDING, PROCESSING, COMPLETED, FAILED
  - `Job` (Dataclass): Job metadata, status, results, timestamps
  - `JobQueue`: Thread-safe job storage and retrieval
- **Key Methods**:
  - `create_job()`: Create new job with metadata
  - `get_job()`: Retrieve job by ID
  - `update_job_status()`: Update job status and results
  - `_cleanup_old_jobs()`: Remove old completed/failed jobs
- **Thread Safety**: Uses `threading.Lock` for concurrent access
- **Storage**: In-memory dictionary (max 1000 jobs by default)

#### `training_jobs.py` (Training Job Management)
- **Function**: Management of LoRA training jobs
- **Key Classes**:
  - `TrainingJobStatus` (Enum): Job status enumeration
  - `TrainingJob`: Training job with progress tracking, logs, metrics
  - `TrainingJobRegistry`: Thread-safe registry for training jobs
- **Key Features**:
  - Progress tracking (epoch, step, loss values)
  - Log buffer (last 1000 lines)
  - Artifact management (trained model paths)
  - Real-time status updates

#### `security.py` (Security & Validation)
- **Function**: Image validation and security utilities
- **Key Functions**:
  - `validate_uploaded_image()`: Validate FastAPI UploadFile
    - File size limits (50MB default)
    - Format validation (JPEG, PNG, WebP)
    - Dimension limits (4096px max)
  - `validate_base64_image()`: Validate base64-encoded images
  - `get_rate_limit_key()`: Generate rate limit keys from requests
- **Security Features**: Prevents malicious file uploads, enforces limits

#### `rate_limiting.py` (Rate Limiting)
- **Function**: API rate limiting using SlowAPI
- **Key Components**:
  - `limiter`: SlowAPI Limiter instance
  - `rate_limit()`: Decorator for endpoint rate limiting
  - `setup_rate_limiting()`: Configure rate limiting middleware
- **Default Limits**: 10 requests/minute per IP
- **Configurable**: Via environment variables

#### `error_responses.py` (Error Handling)
- **Function**: Standardized error response formatting (RFC 7807)
- **Key Functions**:
  - `create_error_response()`: Create Problem Details JSON response
  - `register_error_handlers()`: Register FastAPI exception handlers
- **Error Types**: Validation, Pipeline, Model Load, GPU OOM, Internal
- **Format**: RFC 7807 Problem Details for HTTP APIs

#### `schemas.py` (API Schemas)
- **Function**: Pydantic models for API request/response validation
- **Key Models**:
  - `ProcessImageRequest`: Image processing request schema
  - `ProcessImageResponse`: Processing response with URLs and metadata
  - `JobStatusResponse`: Job status response schema
  - `HealthResponse`, `ReadyResponse`: Health check responses
  - `ErrorResponse`: Error response schema

#### `csrf.py` (CSRF Protection)
- **Function**: CSRF token middleware (optional, commented out by default)
- **Usage**: Can be enabled for UI form protection

---

### `src/phase1_semantic_sanitization/` - Phase I Components

**Purpose**: Remove prohibited elements (logos, mirrors, text) from vehicle photos

#### `sanitizer.py` (Phase I Orchestrator)
- **Function**: Coordinates Phase I processing
- **Key Class**: `Phase1Sanitizer`
- **Key Method**: `sanitize()` - Main processing function
- **Process**: Detection → Segmentation → Inpainting → Validation
- **Dependencies**: GroundingDINO detector, SAM segmenter, LaMa inpainter

#### `grounding_dino_detector.py` (Object Detection)
- **Function**: Detect prohibited elements using GroundingDINO
- **Key Class**: `GroundingDINODetector`
- **Model**: `IDEA-Research/grounding-dino-base`
- **Features**: Zero-shot detection with text prompts
- **Output**: Bounding boxes for detected elements

#### `sam_segmenter.py` (Segmentation)
- **Function**: Precise pixel-level segmentation using SAM
- **Key Class**: `SAMSegmenter`
- **Model**: SAM (Segment Anything Model) `vit_h`
- **Features**: Converts bounding boxes to precise masks
- **Output**: Binary masks for inpainting

#### `lama_inpainter.py` (Inpainting)
- **Function**: Fill masked regions using LaMa inpainting
- **Key Class**: `LaMaInpainter`
- **Model**: LaMa (Large Mask Inpainting) `big-lama`
- **Features**: High-quality inpainting for large masked areas
- **Output**: Clean plate image without prohibited elements

---

### `src/phase2_generative_steering/` - Phase II Components

**Purpose**: Generate vector-style imagery using SDXL with ControlNet guidance

#### `generator.py` (Phase II Orchestrator)
- **Function**: Coordinates Phase II processing
- **Key Class**: `Phase2Generator`
- **Key Method**: `generate()` - Main processing function
- **Process**: Background removal → Depth/Edge → SDXL generation → IoU validation
- **Features**: Auto-retry with adjusted ControlNet weights
- **Dependencies**: Background remover, depth estimator, edge detector, SDXL generator

#### `sdxl_generator.py` (Image Generation)
- **Function**: Generate images using Stable Diffusion XL
- **Key Class**: `SDXLGenerator`
- **Model**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Features**:
  - Multi-ControlNet guidance (depth + canny)
  - LoRA support for custom styles
  - Configurable inference steps, guidance scale
  - Prompt override support
- **Output**: Vector-style raster image

#### `controlnet_processor.py` (ControlNet Processing)
- **Function**: Process ControlNet inputs (depth maps, edge maps)
- **Key Models**:
  - Depth ControlNet: `diffusers/controlnet-depth-sdxl-1.0`
  - Canny ControlNet: `diffusers/controlnet-canny-sdxl-1.0`
- **Features**: Weight adjustment, step scheduling

#### `depth_estimator.py` (Depth Estimation)
- **Function**: Estimate depth maps using ZoeDepth
- **Key Class**: `DepthEstimator`
- **Model**: `zoedepth-anywhere`
- **Output**: Depth map for ControlNet guidance

#### `edge_detector.py` (Edge Detection)
- **Function**: Extract Canny edges for geometric guidance
- **Key Class**: `EdgeDetector`
- **Algorithm**: Canny edge detection
- **Output**: Edge map for ControlNet guidance

#### `background_remover.py` (Background Removal)
- **Function**: Remove background using BiRefNet
- **Key Class**: `BackgroundRemover`
- **Model**: BiRefNet
- **Output**: Vehicle-only image with alpha channel

#### `train_lora_sdxl.py` (LoRA Training)
- **Function**: Fine-tune SDXL with LoRA adapters
- **Key Class**: LoRA training implementation
- **Features**: Custom style training from image pairs
- **Output**: `.safetensors` LoRA checkpoint

#### `training_runner.py` (Training Execution)
- **Function**: Execute LoRA training jobs
- **Key Features**: Background training, progress tracking, checkpoint saving

---

### `src/phase3_chromatic_enforcement/` - Phase III Components

**Purpose**: Enforce 15-color palette through quantization

#### `enforcer.py` (Phase III Orchestrator)
- **Function**: Coordinates Phase III processing
- **Key Class**: `Phase3Enforcer`
- **Key Method**: `enforce()` - Main processing function
- **Process**: Upscaling → Quantization → Noise removal → Validation
- **Dependencies**: Upscaler, color quantizer, noise remover

#### `upscaler.py` (Image Upscaling)
- **Function**: Upscale images using RealESRGAN
- **Key Class**: `Upscaler`
- **Model**: `RealESRGAN_x4plus_anime`
- **Scale Factor**: 4x (configurable)
- **Output**: High-resolution image for better quantization

#### `color_quantizer.py` (Color Quantization)
- **Function**: Reduce colors to 15-color palette using CIEDE2000
- **Key Class**: `ColorQuantizer`
- **Algorithm**: CIEDE2000 color distance (perceptually accurate)
- **Features**:
  - Exact quantization (if colorspacious available)
  - Approximate quantization (fallback)
  - Palette validation
- **Output**: Quantized image with exactly 15 colors

#### `noise_remover.py` (Noise Removal)
- **Function**: Remove small artifacts and speckles
- **Key Class**: `NoiseRemover`
- **Algorithm**: Connected component analysis
- **Threshold**: Removes components < 0.1% of image area
- **Output**: Clean quantized image

---

### `src/phase4_vector_reconstruction/` - Phase IV Components

**Purpose**: Convert quantized raster to SVG vector format

#### `vectorizer.py` (Phase IV Orchestrator)
- **Function**: Coordinates Phase IV processing
- **Key Class**: `Phase4Vectorizer`
- **Key Method**: `vectorize()` - Main processing function
- **Process**: VTracer conversion → Centerline tracing (optional) → SVG processing
- **Strategies**: Strategy A (standard) or Strategy B (centerlines)
- **Dependencies**: VTracer wrapper, SVG processor, centerline tracer

#### `vtracer_wrapper.py` (VTracer Integration)
- **Function**: Interface to VTracer binary for raster-to-vector conversion
- **Key Class**: `VTracerWrapper`
- **Features**:
  - Mode: "stacked" (layered approach)
  - Speckle filtering
  - Corner threshold
  - Segment length control
  - Timeout handling
- **Output**: Raw SVG XML

#### `svg_processor.py` (SVG Post-Processing)
- **Function**: Process and optimize SVG output
- **Key Class**: `SVGProcessor`
- **Features**:
  - Stroke injection (width, color)
  - Path optimization
  - Structure validation
- **Output**: Optimized SVG XML

#### `centerline_tracer.py` (Centerline Extraction)
- **Function**: Extract centerlines for Strategy B
- **Key Class**: `CenterlineTracer`
- **Usage**: Optional, uses edge maps from Phase II
- **Output**: Centerline-based SVG paths

---

### `src/pipeline/` - Pipeline Infrastructure

**Purpose**: Core pipeline orchestration, configuration, model management

#### `orchestrator.py` (Main Pipeline Orchestrator)
- **Function**: Main pipeline coordinator
- **Key Class**: `Gemini3Pipeline`
- **Key Method**: `process_image()` - Execute full 4-phase pipeline
- **Responsibilities**:
  - Coordinate all phases
  - Handle config overrides
  - Manage correlation IDs
  - Collect metrics
  - Quality assurance checks
  - Error handling and retry logic
- **Features**:
  - Phase enable/disable
  - Custom palette support
  - Intermediate output saving
  - IoU validation with auto-retry
- **Dependencies**: All phase modules, config, utilities

#### `config.py` (Configuration Management)
- **Function**: Load and manage configuration
- **Key Class**: `Config`
- **Features**:
  - YAML config loading
  - Environment variable overrides
  - Dot-notation access (`config.get("phase1.enabled")`)
  - Phase-specific config access
  - Hardware config access
- **Thread Safety**: Singleton pattern with locking
- **Config Files**: `configs/default_config.yaml`

#### `model_loader.py` (Model Loading)
- **Function**: Load models from volume or download
- **Key Class**: `ModelLoader`
- **Features**:
  - Network volume support (serverless deployments)
  - Automatic download fallback
  - HuggingFace model support
  - Direct file model support
- **Volume Path**: Configurable via `MODEL_VOLUME_PATH`

#### `model_cache.py` (Model Caching)
- **Function**: Cache loaded models in memory
- **Features**: Reduce model loading overhead
- **Memory Management**: GPU memory tracking

#### `health_checks.py` (Health Monitoring)
- **Function**: System health and readiness checks
- **Key Class**: `HealthChecker`
- **Checks**:
  - GPU availability
  - Model availability
  - Disk space
  - Memory status
- **Endpoints**: `/health` (liveness), `/ready` (readiness)

---

### `src/utils/` - Utility Modules

**Purpose**: Shared utilities used across the pipeline

#### `logger.py` (Structured Logging)
- **Function**: Structured JSON logging with correlation IDs
- **Key Functions**:
  - `get_logger()`: Get logger instance
  - `set_correlation_id()`: Set request correlation ID
  - `setup_logging()`: Configure logging
- **Format**: JSON structured logs
- **Library**: Structlog

#### `image_utils.py` (Image I/O & Processing)
- **Function**: Image loading, saving, preprocessing utilities
- **Key Functions**:
  - `load_image()`: Load image from file (with validation)
  - `save_image()`: Save numpy array to file
  - `resize_image()`: Resize with aspect ratio preservation
  - `convert_to_rgb()`: Convert to RGB format
  - `normalize_image()`: Normalize pixel values
  - `validate_image_file()`: Validate before loading
- **Constants**: `MAX_IMAGE_SIZE_MB`, `MAX_IMAGE_DIMENSION`, `ALLOWED_FORMATS`

#### `palette_manager.py` (Palette Management)
- **Function**: Manage 15-color palette
- **Key Class**: `PaletteManager`
- **Features**:
  - Load palette from YAML
  - Validate palette (exactly 15 colors)
  - Hex to RGB conversion
  - Color matching and nearest color finding
- **Config**: `configs/palette.yaml`

#### `error_handler.py` (Error Classes)
- **Function**: Custom exception classes
- **Key Classes**:
  - `PipelineError`: General pipeline errors
  - `PhaseError`: Phase-specific errors
  - `ValidationError`: Validation failures
  - `ModelLoadError`: Model loading failures
  - `GPUOOMError`: GPU out-of-memory errors

#### `metrics.py` (Metrics Collection)
- **Function**: Collect and aggregate performance metrics
- **Key Class**: `MetricsCollector`
- **Features**:
  - Per-phase timing
  - GPU memory usage
  - Pipeline summary generation
  - Correlation ID tracking

#### `path_validation.py` (Path Security)
- **Function**: Prevent path traversal attacks
- **Key Functions**:
  - `validate_path_within_directory()`: Ensure path is within allowed directory
  - `sanitize_filename()`: Sanitize filenames
  - `validate_intermediate_dir()`: Validate intermediate output directories
- **Security**: Prevents directory traversal attacks

#### `quality_assurance.py` (QA Checks)
- **Function**: Quality validation checks
- **Key Functions**:
  - `validate_geometric_similarity()`: IoU validation
  - `audit_palette_colors()`: Palette compliance check
  - `extract_alpha_mask()`: Extract alpha channel for validation

#### `secrets.py` (Secret Management)
- **Function**: Handle API keys and secrets (if needed)
- **Features**: Environment variable access for sensitive data

---

## CLI Directory (`cli/`)

### `cli/main.py` (Command-Line Interface)
- **Function**: CLI entry point using Click
- **Commands**:
  - `process`: Full pipeline processing
  - `phase1`, `phase2`, `phase3`, `phase4`: Individual phase execution
  - `batch`: Batch processing
- **Options**:
  - `--config`: Custom config file
  - `--verbose`: Verbose logging
  - `--output-png`: PNG preview output
  - `--palette`: Custom palette file
  - `--save-intermediates`: Save intermediate outputs
- **Usage**: `python -m cli.main [command] [options]`

---

## Templates Directory (`templates/`)

**Purpose**: HTML templates for web UI (Jinja2)

### `layout.html` (Base Template)
- **Function**: Base template with navigation and styling
- **Features**: Responsive design, HTMX integration, loading indicators

### `home.html` (Home Page)
- **Function**: Dashboard/home page
- **Content**: Links to training and inference interfaces

### `inference.html` (Inference Interface)
- **Function**: Image processing form
- **Features**:
  - File upload
  - Phase enable/disable toggles
  - Custom palette input
  - Advanced options (prompt override, ControlNet weights)
  - LoRA checkpoint selection

### `inference_job.html` (Job Status Page)
- **Function**: Display inference job status
- **Features**: Real-time status updates via HTMX polling

### `inference_job_partial.html` (HTMX Partial)
- **Function**: Partial template for HTMX updates
- **Usage**: Refreshed via HTMX polling

### `training.html` (Training Interface)
- **Function**: LoRA training form
- **Features**: File pair upload, hyperparameter configuration

### `training_job.html` (Training Job Status)
- **Function**: Display training job status and logs
- **Features**: Real-time log streaming, progress tracking

### `training_job_partial.html` (HTMX Partial)
- **Function**: Partial template for training job updates

---

## Tests Directory (`tests/`)

**Purpose**: Test suite using pytest

### Test Files:
- `test_phase1.py`: Phase I unit tests
- `test_phase2.py`: Phase II unit tests
- `test_phase3.py`: Phase III unit tests
- `test_phase4.py`: Phase IV unit tests
- `test_pipeline.py`: Pipeline integration tests
- `test_security.py`: Security and validation tests
- `test_simple_no_models.py`: Tests without model loading
- `test_standalone.py`: Standalone component tests
- `test_training_no_models.py`: Training tests without models
- `conftest.py`: Pytest configuration and fixtures

---

## Configs Directory (`configs/`)

### `default_config.yaml` (Main Configuration)
- **Function**: Default pipeline configuration
- **Sections**:
  - `pipeline`: Version, name, random seed
  - `hardware`: Device, precision, VRAM limits
  - `serverless`: Model volume, timeouts
  - `phase1`, `phase2`, `phase3`, `phase4`: Phase-specific settings
  - `output`: Output format settings

### `palette.yaml` (Color Palette)
- **Function**: Default 15-color palette definition
- **Format**: YAML list of hex colors

### `model_versions.yaml` (Model Versioning)
- **Function**: Track model versions for reproducibility

---

## Scripts Directory (`scripts/`)

### `setup_model_volume.py` (Model Setup)
- **Function**: Download and verify models on network volumes
- **Usage**: Run on serverless platform startup
- **Features**: Model verification, download if missing

---

## Root-Level Files

### `run.py` (Server Entry Point)
- **Function**: Start FastAPI server with uvicorn
- **Features**:
  - Environment variable configuration
  - Worker configuration
  - Reload mode for development
  - Log level configuration
- **Usage**: `python run.py`

### `run.bat` (Windows Startup)
- **Function**: Windows batch script to start server
- **Usage**: Double-click or `run.bat` from command line

### `run.sh` (Linux/Mac Startup)
- **Function**: Shell script to start server
- **Usage**: `./run.sh` (after `chmod +x run.sh`)

### `requirements.txt` (Dependencies)
- **Function**: Python package dependencies
- **Categories**:
  - Core ML libraries (PyTorch, Transformers, Diffusers)
  - Computer vision (OpenCV, Pillow, SAM)
  - API (FastAPI, Uvicorn)
  - Utilities (NumPy, YAML, logging)
  - Optional (Prometheus, colorspacious)

### `Dockerfile` (Container Definition)
- **Function**: Docker container for deployment
- **Base Image**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
- **Features**: Model volume mounting, health checks, VTracer installation

### `modal_deploy.py` (Modal Deployment)
- **Function**: Modal.com deployment configuration
- **Features**: Serverless function definition

### `runpod_test.py` (RunPod Configuration)
- **Function**: RunPod deployment configuration
- **Features**: GPU instance setup, volume mounting

---

## File Dependencies & Relationships

### Import Hierarchy

```
run.py
  └── src.api.server (FastAPI app)
      ├── src.pipeline.orchestrator (Gemini3Pipeline)
      │   ├── src.phase1_semantic_sanitization.sanitizer
      │   ├── src.phase2_generative_steering.generator
      │   ├── src.phase3_chromatic_enforcement.enforcer
      │   └── src.phase4_vector_reconstruction.vectorizer
      ├── src.api.job_queue (JobQueue)
      ├── src.api.security (validate_uploaded_image)
      └── src.api.training_jobs (TrainingJobRegistry)
```

### Data Flow Dependencies

1. **API Layer** (`src/api/`) depends on:
   - Pipeline orchestrator
   - Job queue
   - Security utilities
   - Configuration

2. **Pipeline** (`src/pipeline/`) depends on:
   - All phase modules
   - Configuration
   - Utilities (logging, metrics, error handling)

3. **Phase Modules** depend on:
   - Configuration
   - Utilities (image utils, logging, error handling)
   - Model loaders

4. **Utilities** (`src/utils/`) are mostly independent:
   - Some depend on configuration
   - Image utils depend on PIL/OpenCV
   - Logger depends on structlog

---

## Module Responsibilities Summary

| Module | Primary Responsibility | Key Dependencies |
|--------|----------------------|------------------|
| `api/server.py` | HTTP request handling, routing | Pipeline, Job Queue, Security |
| `api/job_queue.py` | Async job management | Threading |
| `pipeline/orchestrator.py` | Pipeline coordination | All phases, Config |
| `phase1/sanitizer.py` | Remove prohibited elements | GroundingDINO, SAM, LaMa |
| `phase2/generator.py` | Generate vector-style images | SDXL, ControlNet, Depth/Edge |
| `phase3/enforcer.py` | Enforce color palette | RealESRGAN, Quantizer |
| `phase4/vectorizer.py` | Convert to SVG | VTracer, SVG Processor |
| `utils/logger.py` | Structured logging | Structlog |
| `utils/image_utils.py` | Image I/O operations | PIL, OpenCV |
| `pipeline/config.py` | Configuration management | PyYAML |

---

## Key Design Patterns

### Singleton Pattern
- **Config**: `get_config()` - Single config instance
- **Pipeline**: `get_pipeline()` - Single pipeline instance
- **Job Queue**: `get_job_queue()` - Single queue instance
- **Model Loader**: `get_model_loader()` - Single loader instance

### Lazy Loading
- **Models**: Loaded on first use, not at initialization
- **Pipeline Phases**: Components initialized lazily
- **Benefits**: Faster startup, lower memory usage

### Factory Pattern
- **Phase Initialization**: Each phase creates its sub-components
- **Model Loading**: ModelLoader creates model instances

### Observer Pattern
- **Metrics Collection**: MetricsCollector observes pipeline execution
- **Logging**: Structured logging throughout pipeline

### Strategy Pattern
- **Phase IV Strategies**: Strategy A (standard) vs Strategy B (centerlines)
- **Quantization Methods**: Exact vs approximate CIEDE2000

---

## Thread Safety

### Thread-Safe Components
- `JobQueue`: Uses `threading.Lock` for concurrent access
- `TrainingJobRegistry`: Uses locks for job management
- `Config`: Singleton with lock for initialization
- `Pipeline`: Singleton with lock for initialization

### Thread-Unsafe Components
- Phase modules: Not designed for concurrent use (single GPU)
- Model instances: Shared but not thread-safe (single-threaded processing)

---

## Related Documentation

- **[APPLICATION_OVERVIEW.md](APPLICATION_OVERVIEW.md)** - What the application does
- **[DATAFLOW.md](DATAFLOW.md)** - How data flows through the system
- **[README.md](README.md)** - Quick start guide

---

**Last Updated**: Generated from codebase analysis  
**Version**: 3.0.0

