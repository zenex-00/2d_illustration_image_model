# Application Overview: Gemini 3 Pro Vehicle-to-Vector Pipeline

## Executive Summary

The **Gemini 3 Pro Vehicle-to-Vector Pipeline** is a production-grade machine learning system that transforms high-resolution vehicle photographs into minimalist, high-fidelity vector illustrations. The system employs a sophisticated 4-phase modular architecture that combines state-of-the-art computer vision models, generative AI, and vector graphics processing to produce scalable SVG outputs suitable for professional design applications.

---

## Core Functionality

### Primary Purpose

The application converts raster automotive photography into vector illustrations through an automated pipeline that:

1. **Removes unwanted elements** (logos, mirrors, text, stickers) from vehicle photos
2. **Generates vector-style imagery** using generative AI with geometric constraints
3. **Enforces a strict 15-color palette** for consistent visual style
4. **Converts to scalable SVG format** suitable for print and digital media

### Use Cases

- **Automotive Marketing**: Create consistent vehicle illustrations for brochures, websites, and advertisements
- **Design Systems**: Generate standardized vehicle graphics for design libraries
- **E-commerce**: Produce clean product images without branding or text
- **Artistic Applications**: Transform photographs into stylized vector art
- **Batch Processing**: Process large catalogs of vehicle images automatically

---

## The 4-Phase Pipeline Architecture

### Phase I: Semantic Sanitization

**Purpose**: Remove prohibited elements from vehicle photographs

**Technology Stack**:
- **GroundingDINO**: Zero-shot object detection to identify prohibited elements (logos, mirrors, license plates, text, stickers)
- **SAM (Segment Anything Model)**: Precise segmentation of detected elements
- **LaMa (Large Mask Inpainting)**: High-quality inpainting to fill removed regions

**Process Flow**:
1. Input image is analyzed using GroundingDINO with text prompts for prohibited elements
2. Detected bounding boxes are refined using SAM for pixel-perfect masks
3. Masked regions are inpainted using LaMa to create a "clean plate" image
4. Quality assurance validates geometric similarity (IoU > 0.97) between original and cleaned image

**Output**: Clean vehicle image without logos, mirrors, text, or other prohibited elements

**Configuration**: Located in `configs/default_config.yaml` under `phase1` section

---

### Phase II: Generative Steering

**Purpose**: Generate vector-style imagery from clean vehicle photos using generative AI

**Technology Stack**:
- **SDXL (Stable Diffusion XL)**: Base generative model for high-resolution image generation
- **Multi-ControlNet**: Dual ControlNet guidance using depth maps and Canny edge detection
- **BiRefNet**: Background removal for cleaner vehicle isolation
- **ZoeDepth**: Depth estimation for 3D structure preservation
- **Canny Edge Detection**: Edge maps for geometric constraint

**Process Flow**:
1. Background removal isolates the vehicle from its environment
2. Depth estimation creates a depth map preserving 3D structure
3. Edge detection extracts geometric features (contours, lines)
4. SDXL generates vector-style image using:
   - Depth ControlNet (weight: 0.6) for structure preservation
   - Canny ControlNet (weight: 0.4) for geometric accuracy
   - Custom prompt: "minimalist vector illustration, clean lines, flat colors"
5. IoU validation ensures geometric similarity (threshold: 0.85)
   - Auto-retry mechanism adjusts ControlNet weights if IoU is too low
   - Up to 2 retries with increased weights

**Output**: Vector-style raster image with preserved geometry and structure

**Configuration**: Located in `configs/default_config.yaml` under `phase2` section

**Advanced Features**:
- **LoRA Support**: Custom fine-tuned models can be loaded for specific styles
- **Prompt Override**: Custom generation prompts can be provided via API
- **ControlNet Weight Tuning**: Depth and Canny weights adjustable per request

---

### Phase III: Chromatic Enforcement

**Purpose**: Enforce a strict 15-color palette on the generated image

**Technology Stack**:
- **RealESRGAN**: 4x upscaling for higher resolution processing
- **CIEDE2000 Color Quantization**: Perceptually accurate color reduction
- **Noise Removal**: Removes small artifacts and speckles

**Process Flow**:
1. **Upscaling**: Image is upscaled 4x using RealESRGAN for better quantization quality
2. **Color Quantization**: 
   - Uses CIEDE2000 color distance metric (perceptually accurate)
   - Maps all pixels to nearest palette color
   - Supports exact quantization (if colorspacious library available) or approximate
3. **Noise Removal**: Small connected components (< 0.1% of image area) are removed
4. **Palette Audit**: Validates that all colors in output match the 15-color palette

**Output**: Quantized image using exactly 15 colors from the configured palette

**Configuration**: Located in `configs/default_config.yaml` under `phase3` section

**Palette Management**:
- Default palette defined in `configs/palette.yaml`
- Custom palettes can be provided via API (15 hex colors required)
- Palette validation ensures exactly 15 colors with valid hex codes

---

### Phase IV: Vector Reconstruction

**Purpose**: Convert quantized raster image to scalable SVG vector format

**Technology Stack**:
- **VTracer**: Raster-to-vector conversion engine
- **Centerline Tracing**: Optional centerline extraction for cleaner strokes
- **SVG Processing**: Post-processing for stroke injection and optimization

**Process Flow**:
1. **VTracer Conversion**: 
   - Converts quantized PNG to SVG using VTracer binary
   - Mode: "stacked" (layered approach for better quality)
   - Filters small speckles (< 4 pixels)
   - Corner threshold: 60 degrees
   - Segment length: 4 pixels
   - Timeout: 300 seconds
2. **Centerline Tracing** (Strategy B, optional):
   - Extracts centerlines from edge maps (from Phase II)
   - Creates cleaner stroke-based representation
3. **SVG Post-Processing**:
   - Injects stroke attributes (width: 2px, color: black)
   - Optimizes paths and removes redundant elements
   - Validates SVG structure

**Output**: Final SVG file ready for use in design applications

**Configuration**: Located in `configs/default_config.yaml` under `phase4` section

**Strategies**:
- **Strategy A**: Standard VTracer conversion (default)
- **Strategy B**: Centerline-based tracing for cleaner strokes

---

## Three Access Methods

### 1. Command-Line Interface (CLI)

**Location**: `cli/main.py`

**Usage Examples**:
```bash
# Full pipeline
python -m cli.main process input.jpg output.svg --output-png preview.png

# Individual phases
python -m cli.main phase1 input.jpg clean_plate.png
python -m cli.main phase2 clean_plate.png vector_raster.png
python -m cli.main phase3 vector_raster.png quantized.png
python -m cli.main phase4 quantized.png output.svg

# Batch processing
python -m cli.main batch input_dir/ output_dir/

# Custom configuration
python -m cli.main --config custom_config.yaml process input.jpg output.svg
```

**Features**:
- Direct file-based processing
- Batch processing support
- Intermediate output saving
- Custom palette support via YAML files

---

### 2. Python API

**Location**: `src/pipeline/orchestrator.py`

**Usage Example**:
```python
from src.pipeline.orchestrator import Gemini3Pipeline

# Initialize pipeline
pipeline = Gemini3Pipeline()

# Process image
svg_xml, metadata = pipeline.process_image(
    input_image_path="input.jpg",
    output_svg_path="output.svg",
    output_png_path="preview.png",
    palette_hex_list=["#FF0000", "#00FF00", ...],  # 15 colors
    config_overrides={
        "phases": {
            "phase1": {"enabled": True},
            "phase2": {"enabled": True}
        }
    }
)

print(f"Processing time: {metadata['total_processing_time_ms']}ms")
print(f"Phase timings: {metadata['phase1']}, {metadata['phase2']}, ...")
```

**Features**:
- Programmatic control
- Custom palette support
- Phase enable/disable
- Config overrides
- Metadata access (timings, metrics, correlation IDs)

---

### 3. Web API & UI

**Location**: `src/api/server.py` (FastAPI server)

**API Endpoints**:

**Synchronous Processing**:
- `POST /api/v1/process` - Process image and return results immediately
- `POST /api/v1/phase1` - Run Phase I only
- `POST /api/v1/phase2` - Run Phase II only
- `POST /api/v1/phase3` - Run Phase III only
- `POST /api/v1/phase4` - Run Phase IV only

**Asynchronous Processing**:
- `POST /api/v1/jobs` - Create processing job (returns job_id)
- `GET /api/v1/jobs/{job_id}` - Get job status and results

**Web UI**:
- `GET /ui` - Home dashboard
- `GET /ui/inference` - Inference interface with phase toggles
- `GET /ui/training` - LoRA training interface
- `GET /ui/inference/jobs/{job_id}` - Job status page with real-time updates

**Features**:
- RESTful API with OpenAPI documentation (`/docs`)
- Rate limiting (10 requests/minute default)
- Image validation (size, format, dimensions)
- Background job processing
- Real-time status updates via HTMX
- Custom palette support
- Phase enable/disable via UI toggles
- LoRA checkpoint selection

---

## Key Features

### Modular Design

Each phase can run independently, allowing for:
- **Selective Processing**: Skip phases that aren't needed
- **Debugging**: Process individual phases to isolate issues
- **Custom Workflows**: Chain phases in different orders
- **Performance Optimization**: Disable expensive phases when not needed

### Error Handling & Resilience

- **Comprehensive Retry Logic**: IoU validation with auto-retry in Phase II
- **Graceful Degradation**: Phases can be disabled if models fail to load
- **GPU OOM Recovery**: Automatic memory management and recovery
- **Structured Error Responses**: RFC 7807 Problem Details format
- **Correlation IDs**: Track requests through entire pipeline

### Structured Logging

- **JSON Logs**: Machine-readable log format
- **Correlation IDs**: Link all log entries for a single request
- **Structured Fields**: Consistent log structure across all components
- **Log Levels**: DEBUG, INFO, WARNING, ERROR with appropriate verbosity

### GPU Memory Management

- **Lazy Model Loading**: Models loaded only when needed
- **Model Caching**: Reuse loaded models across requests
- **Memory Monitoring**: Track VRAM usage and prevent OOM
- **Automatic Cleanup**: Release GPU memory after processing

### Quality Assurance

- **IoU Validation**: Geometric similarity checks between phases
- **Palette Audits**: Verify color compliance
- **SVG Validation**: Structure and format validation
- **Metrics Collection**: Performance and quality metrics

### Reproducibility

- **Fixed Random Seeds**: Consistent results across runs
- **Model Versioning**: Track model versions in metadata
- **Config Versioning**: Pipeline version tracking
- **Deterministic Processing**: Same input produces same output

---

## Technical Requirements

### Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (16GB+ VRAM for optimal performance)
- **CPU**: Multi-core processor for preprocessing and post-processing
- **RAM**: 32GB+ recommended for large batch processing
- **Storage**: SSD recommended for model storage and temporary files

### Software Requirements

- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **Operating System**: Linux, macOS, or Windows
- **VTracer Binary**: Required for Phase IV (must be in PATH or specified in config)

### Dependencies

**Core ML Libraries**:
- PyTorch 2.5.1+
- Transformers 4.56.2+
- Diffusers 0.30.0+
- Accelerate 0.33.0+

**Computer Vision**:
- OpenCV 4.8.0+
- Pillow 10.0.0+
- scikit-image 0.21.0+
- Segment Anything (SAM)

**Specialized Models**:
- GroundingDINO (object detection)
- LaMa (inpainting)
- ControlNet Aux (depth/canny processing)
- RealESRGAN (upscaling)
- ZoeDepth (depth estimation)
- BiRefNet (background removal)

**API & Web**:
- FastAPI 0.104.0+
- Uvicorn 0.24.0+
- Jinja2 3.1.2+ (for templates)
- SlowAPI (rate limiting)

**Utilities**:
- NumPy, SciPy
- PyYAML
- Structlog (structured logging)
- Tenacity (retry logic)

See `requirements.txt` for complete dependency list.

---

## Model Requirements

The pipeline requires the following models, which are downloaded automatically on first use:

### Phase I Models
- **GroundingDINO**: `IDEA-Research/grounding-dino-base` (HuggingFace)
- **SAM**: `vit_h` checkpoint (`sam_vit_h_4b8939.pth`)
- **LaMa**: `big-lama` model

### Phase II Models
- **SDXL Base**: `stabilityai/stable-diffusion-xl-base-1.0`
- **ControlNet Depth**: `diffusers/controlnet-depth-sdxl-1.0`
- **ControlNet Canny**: `diffusers/controlnet-canny-sdxl-1.0`
- **BiRefNet**: Background removal model
- **ZoeDepth**: `zoedepth-anywhere` depth estimation

### Phase III Models
- **RealESRGAN**: `RealESRGAN_x4plus_anime` (4x upscaling)

### Phase IV
- **VTracer**: Binary executable (must be installed separately)

### Optional Models
- **LoRA Checkpoints**: Custom fine-tuned models (`.safetensors` format)
- Can be trained using the built-in LoRA training interface

**Model Storage**:
- Models are cached locally after first download
- Can be stored on network volumes for serverless deployments
- Configurable via `MODEL_VOLUME_PATH` environment variable

---

## Configuration

### Configuration Files

**Main Config**: `configs/default_config.yaml`
- Pipeline version and settings
- Hardware configuration (device, precision, VRAM limits)
- Phase-specific parameters
- Model paths and checkpoints

**Palette Config**: `configs/palette.yaml`
- Default 15-color palette definition

**Model Versions**: `configs/model_versions.yaml`
- Model version tracking

### Environment Variables

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEVICE`: Compute device (`cuda` or `cpu`)
- `PRECISION`: Model precision (`float16` or `float32`)
- `MAX_VRAM_GB`: Maximum VRAM limit
- `LOG_LEVEL`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `MODEL_VOLUME_PATH`: Path to model storage volume
- `API_OUTPUT_DIR`: Directory for API output files
- `CORS_ORIGINS`: CORS allowed origins (comma-separated)

---

## Output Formats

### SVG (Scalable Vector Graphics)
- **Format**: XML-based vector format
- **Resolution**: Scalable (no fixed resolution)
- **Use Case**: Print, web, design applications
- **Features**: Path-based, stroke-injected, optimized

### PNG Preview
- **Format**: Raster preview image
- **Resolution**: 2048px (configurable)
- **Use Case**: Quick preview, thumbnails, web display
- **Features**: Quantized to 15-color palette

### Intermediate Outputs
- Phase I: Clean plate PNG
- Phase II: Vector-style raster PNG
- Phase III: Quantized PNG
- Phase IV: SVG (final output)

---

## Performance Characteristics

### Processing Times (Approximate)

- **Phase I**: 5-15 seconds (depends on number of detected elements)
- **Phase II**: 30-60 seconds (SDXL generation is most time-consuming)
- **Phase III**: 10-20 seconds (upscaling and quantization)
- **Phase IV**: 5-30 seconds (depends on image complexity)
- **Total Pipeline**: 50-125 seconds per image

### Resource Usage

- **GPU Memory**: 8-16GB VRAM (depending on image size and models)
- **CPU**: Multi-threaded preprocessing, single-threaded generation
- **Disk I/O**: Moderate (model loading, temporary files)

### Scalability

- **Concurrent Requests**: Limited by GPU memory (typically 1-2 concurrent)
- **Batch Processing**: Sequential processing recommended
- **Serverless**: Supports cold start optimization and model caching

---

## Security Features

- **Image Validation**: File size, format, and dimension limits
- **Path Traversal Protection**: Sanitized file paths
- **Rate Limiting**: Prevents abuse (10 requests/minute default)
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Error Sanitization**: No sensitive information in error messages
- **Input Validation**: Strict validation of all user inputs

---

## Monitoring & Observability

- **Health Endpoints**: `/health` (liveness), `/ready` (readiness)
- **Metrics Endpoint**: `/metrics` (Prometheus format)
- **Structured Logging**: JSON logs with correlation IDs
- **Performance Metrics**: Per-phase timing and resource usage
- **Error Tracking**: Comprehensive error logging with stack traces

---

## Future Enhancements

Potential areas for improvement:
- Multi-GPU support for parallel processing
- WebSocket support for real-time progress updates
- Additional vectorization strategies
- Custom model fine-tuning UI
- Batch API endpoints
- Cloud storage integration
- Advanced palette customization
- Style transfer options

---

## Related Documentation

- **[FILE_STRUCTURE.md](FILE_STRUCTURE.md)** - Detailed file structure and component functions
- **[DATAFLOW.md](DATAFLOW.md)** - Dataflow through the application
- **[COMPREHENSIVE_BUG_REPORT.md](COMPREHENSIVE_BUG_REPORT.md)** - Known issues and fixes
- **[README.md](README.md)** - Quick start guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment instructions

---

**Last Updated**: Generated from codebase analysis  
**Version**: 3.0.0

