# Deployment Guide

## Overview

This guide covers deployment options for the Gemini 3 Pro Vehicle-to-Vector pipeline, including cloud platforms (Modal, RunPod) and local deployment.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- VTracer binary installed
- 50GB+ storage for models

## Environment Variables

Create a `.env` file or set environment variables:

```bash
# API Configuration
API_RATE_LIMIT=10/minute
CORS_ORIGINS=*
API_OUTPUT_DIR=/tmp/gemini3_output

# Hardware
CUDA_VISIBLE_DEVICES=0
MAX_VRAM_GB=16

# Model Paths (optional)
MODEL_CACHE_DIR=/path/to/models

# Secrets (if needed)
HUGGINGFACE_TOKEN=your_token_here
```

## Modal Deployment

### Setup

1. Install Modal:
```bash
pip install modal
```

2. Create `modal_deploy.py`:
```python
import modal

stub = modal.Stub("gemini3-pipeline")

image = (
    modal.Image.debian_slim()
    .pip_install_from_pyproject("requirements.txt")
    .apt_install("vtracer")
)

@stub.function(
    image=image,
    gpu="A10G",
    timeout=300
)
def process_image(image_bytes: bytes):
    from src.pipeline.orchestrator import Gemini3Pipeline
    import tempfile
    
    pipeline = Gemini3Pipeline()
    
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
        tmp.write(image_bytes)
        svg_xml, metadata = pipeline.process_image(tmp.name)
        return svg_xml, metadata
```

3. Deploy:
```bash
modal deploy modal_deploy.py
```

### API Endpoint

Modal automatically creates HTTP endpoints. Access via:
```bash
curl -X POST https://your-username--gemini3-pipeline-process-image.modal.run \
  -F "file=@car.jpg"
```

## RunPod Deployment

### Setup

1. Create RunPod template:
```yaml
name: gemini3-pipeline
image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
env:
  - API_RATE_LIMIT=10/minute
ports:
  - 8000:8000
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start API server:
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### Health Checks

Configure RunPod health checks:
- **Liveness:** `GET /health`
- **Readiness:** `GET /ready`

## Local Deployment

### Development Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

### Production Server

```bash
# Using gunicorn with uvicorn workers
gunicorn src.api.server:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t gemini3-pipeline .
docker run --gpus all -p 8000:8000 gemini3-pipeline
```

## Model Download

Models are downloaded automatically on first use. To pre-download:

```python
from src.pipeline.model_cache import get_model_cache

cache = get_model_cache()
# Models will be downloaded and cached
```

Or manually:
```bash
# Download models to cache directory
python -c "from src.pipeline.orchestrator import Gemini3Pipeline; Gemini3Pipeline()"
```

## Scaling Considerations

### Horizontal Scaling

- API is stateless (except model cache)
- Use load balancer with sticky sessions for model caching
- Consider model server (e.g., TensorRT Inference Server)

### Vertical Scaling

- Increase GPU VRAM for larger batch sizes
- Increase system RAM for model caching
- Use faster storage (NVMe SSD) for model loading

## Monitoring

### Health Checks

Configure health checks in your deployment platform:

```yaml
# Kubernetes example
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 5
```

### Metrics

Prometheus metrics available at `/metrics`:

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'gemini3-pipeline'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

## Performance Tuning

### Model Caching

Enable model caching to reduce load times:
```python
# In config
model_cache:
  enabled: true
  max_size_gb: 20
```

### GPU Memory

Adjust batch sizes and precision:
```yaml
hardware:
  precision: "float16"  # or "float32"
  enable_xformers: true
  enable_attention_slicing: true
```

### API Timeouts

Set appropriate timeouts:
- **Request timeout:** 300 seconds (5 minutes)
- **Keep-alive:** 60 seconds







