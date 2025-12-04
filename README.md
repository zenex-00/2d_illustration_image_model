# Gemini 3 Pro Vehicle-to-Vector Pipeline

Production-grade pipeline for converting vehicle photographs into high-fidelity vector illustrations.

## Overview

The Gemini 3 Pro architecture transforms raster automotive photography into minimalist vector illustrations through a 4-phase modular pipeline:

1. **Phase I: Semantic Sanitization** - Removes prohibited elements (logos, mirrors, text) using GroundingDINO, SAM, and LaMa
2. **Phase II: Generative Steering** - Generates vector-style images using SDXL with Multi-ControlNet guidance
3. **Phase III: Chromatic Enforcement** - Enforces 15-color palette using CIEDE2000 quantization
4. **Phase IV: Vector Reconstruction** - Converts to SVG using VTracer with stroke injection and validation

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- VTracer binary (for Phase IV)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd image_generation

# Install dependencies
pip install -r requirements.txt

# Install VTracer (see VTracer documentation)
# VTracer binary should be in PATH or specify path in config
```

## Running the Application

### Quick Start

The easiest way to run the application:

**Python (Cross-platform):**
```bash
python run.py
```

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

### Configuration

You can configure the server using environment variables:

```bash
# Set host and port
export HOST=0.0.0.0
export PORT=8000

# Enable auto-reload for development
export RELOAD=true

# Set log level
export LOG_LEVEL=info

# Run with multiple workers (production)
export WORKERS=4

# Then run
python run.py
```

### Access Points

Once running, access the application at:

- **API Documentation:**
  - Swagger UI: http://localhost:8000/docs
  - ReDoc: http://localhost:8000/redoc

- **Web UI:**
  - Home: http://localhost:8000/ui
  - Training: http://localhost:8000/ui/training
  - Inference: http://localhost:8000/ui/inference

- **API Endpoints:**
  - Health: http://localhost:8000/health
  - Process Image: http://localhost:8000/api/v1/process

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and component details
- **[API.md](API.md)** - API documentation and endpoints
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment guide for cloud platforms
- **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration reference
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

## Configuration

Edit `configs/default_config.yaml` to customize:

- Model paths and checkpoints
- Phase-specific parameters (thresholds, weights, steps)
- Hardware settings (device, precision)
- Output settings (resolutions, formats)

See [CONFIGURATION.md](CONFIGURATION.md) for detailed configuration options.

## Usage

### CLI

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

# With custom config
python -m cli.main --config custom_config.yaml process input.jpg output.svg
```

### Python API

```python
from src.pipeline.orchestrator import Gemini3Pipeline

# Initialize pipeline
pipeline = Gemini3Pipeline()

# Process image
svg_xml, metadata = pipeline.process_image(
    input_image_path="input.jpg",
    output_svg_path="output.svg",
    output_png_path="preview.png"
)

print(f"Processing time: {metadata['total_processing_time_ms']}ms")
```

### Web API

Start the FastAPI server:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Access API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

See [API.md](API.md) for complete API documentation.

## Project Structure

```
image_generation/
├── src/
│   ├── pipeline/          # Main orchestrator
│   ├── phase1_*/          # Phase I modules
│   ├── phase2_*/          # Phase II modules
│   ├── phase3_*/          # Phase III modules
│   ├── phase4_*/          # Phase IV modules
│   └── utils/             # Utility modules
├── cli/                   # CLI interface
├── tests/                 # Test suite
├── configs/               # Configuration files
└── requirements.txt       # Dependencies
```

## Features

- **Modular Design**: Each phase can run independently
- **Error Handling**: Comprehensive retry logic and graceful degradation
- **Structured Logging**: JSON logs with correlation IDs
- **GPU Memory Management**: Automatic OOM recovery and model caching
- **Quality Assurance**: IoU validation, palette audits, geometric checks
- **Reproducibility**: Fixed random seeds and model versioning

## Model Requirements

The pipeline requires the following models (downloaded automatically on first use):

- GroundingDINO (object detection)
- SAM (segmentation)
- LaMa (inpainting)
- SDXL (generation)
- ControlNet Depth & Canny (guidance)
- RealESRGAN (upscaling)
- ZoeDepth (depth estimation)
- BiRefNet (background removal)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## License

[Add license information]

## Citation

If you use this pipeline, please cite the original Gemini 3 Pro architecture document.

