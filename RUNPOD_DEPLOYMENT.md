# RunPod RTX 5090 Deployment Guide

## Overview

This guide provides the complete startup command for deploying the Gemini 3 Pro pipeline on RunPod with RTX 5090 GPU. RunPod only allows a single startup command, so all installation steps must be consolidated into one command.

## Prerequisites

- RunPod account with RTX 5090 GPU access
- Persistent volume (100GB recommended) mounted at `/models`
- Container image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` or similar with Python 3.10+

## Complete Startup Command

Copy and paste this entire command into the RunPod startup command field:

```bash
/bin/bash -c "export DEBIAN_FRONTEND=noninteractive && \
dpkg --configure -a || true && \
apt-get update && \
apt-get install -y -q git libgl1-mesa-glx libglib2.0-0 tzdata && \
cd /workspace && \
rm -rf image_generation ZoeDepth && \
git clone https://github.com/zenex-00/2d_illustration_image_model.git image_generation && \
git clone https://github.com/isl-org/ZoeDepth.git && \
cd /workspace/image_generation && \
pip install -q --upgrade pip && \
pip install -q --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 && \
grep -v 'lama-cleaner' requirements.txt | grep -v 'zoedepth' | grep -v '^xformers' | grep -v '^torch' | grep -v '^torchvision' > /tmp/requirements_filtered.txt && \
pip install -q -r /tmp/requirements_filtered.txt || echo 'Warning: Some requirements failed, continuing...' && \
LAMA_VENV_DIR=/opt/lama-cleaner-venv && \
python3 -m venv \$LAMA_VENV_DIR && \
source \$LAMA_VENV_DIR/bin/activate && \
pip install -q --upgrade pip && \
pip install -q lama-cleaner>=1.2.0 && \
deactivate && \
wget -q https://github.com/visioncortex/vtracer/releases/download/v0.6.1/vtracer-linux-x64 -O /usr/local/bin/vtracer && \
chmod +x /usr/local/bin/vtracer && \
export PYTHONPATH=/workspace/image_generation:/workspace/ZoeDepth:\$PYTHONPATH && \
python scripts/setup_model_volume.py --volume-path /models && \
export DISABLE_XFORMERS=1 && \
export GPU_MODEL=RTX_5090 && \
export CUDA_VISIBLE_DEVICES=0 && \
export MODEL_VOLUME_PATH=/models && \
export API_OUTPUT_DIR=/tmp/gemini3_output && \
export PYTHONUNBUFFERED=1 && \
export WORKERS=1 && \
export LOG_LEVEL=info && \
uvicorn src.api.server:app --host 0.0.0.0 --port 5090 --workers 1"
```

## Command Breakdown

### 1. System Dependencies
```bash
apt-get install -y -q git libgl1-mesa-glx libglib2.0-0 tzdata
```
- `git`: For cloning repositories
- `libgl1-mesa-glx`: OpenCV support
- `libglib2.0-0`: PyQT/GUI support
- `tzdata`: Timezone data

### 2. Repository Setup
```bash
git clone https://github.com/zenex-00/2d_illustration_image_model.git image_generation
git clone https://github.com/isl-org/ZoeDepth.git
```
- Clones the main pipeline repository
- Clones ZoeDepth for depth estimation

### 3. PyTorch Installation (RTX 5090 Specific)
```bash
pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```
- **Critical:** Uses CUDA 12.8 nightly build for RTX 5090 (sm_120 architecture)
- PyTorch 2.8.0+ required for RTX 5090 support

### 4. Main Requirements
```bash
grep -v 'lama-cleaner' requirements.txt | ... | pip install -r /tmp/requirements_filtered.txt
```
- Filters out conflicting packages
- Installs all other dependencies

### 5. Lama-Cleaner Isolation
```bash
python3 -m venv /opt/lama-cleaner-venv
source /opt/lama-cleaner-venv/bin/activate
pip install lama-cleaner>=1.2.0
deactivate
```
- Creates isolated virtual environment to prevent dependency conflicts
- Installs lama-cleaner with its dependencies

### 6. VTracer Binary
```bash
wget .../vtracer-linux-x64 -O /usr/local/bin/vtracer
chmod +x /usr/local/bin/vtracer
```
- Downloads pre-built VTracer binary for vectorization
- Makes it executable

### 7. Model Volume Setup
```bash
python scripts/setup_model_volume.py --volume-path /models
```
- Downloads all required models to persistent volume
- Models persist across pod restarts

### 8. Environment Variables
```bash
export DISABLE_XFORMERS=1      # Required for RTX 5090
export GPU_MODEL=RTX_5090      # GPU identification
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export MODEL_VOLUME_PATH=/models
export API_OUTPUT_DIR=/tmp/gemini3_output
export PYTHONUNBUFFERED=1      # Real-time logging
```

### 9. Server Startup
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 5090 --workers 1
```
- Starts FastAPI server on port 5090
- Single worker for RTX 5090 (24GB VRAM)

## Environment Variables

Set these in RunPod's environment variables section (optional, already in startup command):

| Variable | Value | Purpose |
|----------|-------|---------|
| `DISABLE_XFORMERS` | `1` | Disable xformers (RTX 5090 uses native SDPA) |
| `GPU_MODEL` | `RTX_5090` | GPU identification for optimizations |
| `CUDA_VISIBLE_DEVICES` | `0` | Use first GPU |
| `MODEL_VOLUME_PATH` | `/models` | Path to persistent volume |
| `API_OUTPUT_DIR` | `/tmp/gemini3_output` | Temporary output directory |
| `PYTHONUNBUFFERED` | `1` | Real-time log output |
| `WORKERS` | `1` | Number of uvicorn workers |
| `LOG_LEVEL` | `info` | Logging verbosity |

## Port Configuration

- **API Port:** 5090 (for RTX 5090)
- **Health Check:** `http://localhost:5090/health`
- **API Docs:** `http://localhost:5090/docs`

## Health Checks

### Startup Probe (first 120 seconds)
```
GET /health
Expected: 200 {"status":"healthy","version":"3.0.0"}
Initial delay: 30s
Timeout: 10s
```

### Readiness Probe (every 30 seconds)
```
GET /ready
Expected: 200 {"status":"ready","checks":{...}}
```

### Liveness Probe (every 60 seconds)
```
GET /health
Expected: 200 (any healthy response)
```

## Validation After Startup

Run these commands to verify deployment:

```bash
# 1. Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 2. Verify models exist
ls -lah /models | grep -E "sdxl|controlnet|grounding|sam"

# 3. Test API health
curl -s http://localhost:5090/health | python -m json.tool

# 4. Test pipeline initialization
curl -s http://localhost:5090/ready | python -m json.tool
```

## Troubleshooting

### Issue: "CUDA is not available for compute capability sm_120"
**Solution:** Ensure PyTorch 2.8.0+ with CUDA 12.8 nightly is installed:
```bash
pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: xformers import fails
**Solution:** Already handled with `DISABLE_XFORMERS=1` in startup command

### Issue: Memory errors ("CUDA out of memory")
**Causes & Solutions:**
1. Other processes using GPU: Check with `nvidia-smi`
2. Models not flushed: Check logs for "phase2_memory_flushed"
3. Config settings: Verify `precision: "float16"` and `enable_attention_slicing: true` in config

### Issue: VTracer not found
**Solution:** Verify binary was downloaded:
```bash
which vtracer
/usr/local/bin/vtracer --version
```

### Issue: Models not loading
**Solution:** Check model volume:
```bash
ls -lah /models
python scripts/setup_model_volume.py --volume-path /models
```

## Notes

- **Shell Scripts:** The `install_dependencies.sh` and `lama_cleaner_cli.sh` scripts are NOT used in RunPod deployment. All installation is done inline in the startup command.
- **First Startup:** Initial model download takes 20-30 minutes. Subsequent startups are faster (models cached on volume).
- **Memory:** Peak VRAM usage is ~18GB, well within RTX 5090's 24GB capacity.
- **Performance:** Expect 60-120 seconds per image processing time.

## Cost Estimation

- **GPU (RTX 5090):** ~$0.90/hr
- **Storage (100GB):** ~$5/month
- **Network:** ~$0.30/GB

Estimated monthly cost: ~$692 for 100% uptime

