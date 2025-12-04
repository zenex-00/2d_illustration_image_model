"""RunPod testing/training configuration for Gemini 3 Pro pipeline"""

import os
from pathlib import Path

# RunPod template configuration
# This file is used to configure RunPod pods for testing and training

RUNPOD_CONFIG = {
    "name": "gemini3-pipeline-test",
    "image": "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
    "gpu_type": "NVIDIA A10G",
    "gpu_count": 1,
    "volume_mounts": [
        {
            "path": "/models",
            "volume_id": os.getenv("RUNPOD_MODEL_VOLUME_ID", ""),
            "description": "Persistent volume for model weights"
        }
    ],
    "env": {
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "MODEL_VOLUME_PATH": "/models",
        "API_OUTPUT_DIR": "/tmp/gemini3_output",
        "LOG_LEVEL": "INFO"
    },
    "ports": [
        {
            "container_port": 8000,
            "public_port": 8000,
            "type": "tcp"
        }
    ],
    "health_check": {
        "path": "/health",
        "interval": 30,
        "timeout": 10,
        "retries": 3
    },
    "startup_command": "python scripts/setup_model_volume.py --volume-path /models && uvicorn src.api.server:app --host 0.0.0.0 --port 8000"
}


def get_runpod_template():
    """Get RunPod template configuration"""
    return RUNPOD_CONFIG


def print_runpod_setup_instructions():
    """Print instructions for setting up RunPod"""
    print("""
RunPod Testing/Training Setup Instructions
==========================================

1. Create a RunPod Pod with the following settings:
   - GPU Type: NVIDIA A10G (24GB VRAM)
   - Container Image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
   - Volume: Create a persistent volume (50GB+) and mount at /models

2. Environment Variables:
   - MODEL_VOLUME_PATH=/models
   - API_OUTPUT_DIR=/tmp/gemini3_output
   - CUDA_VISIBLE_DEVICES=0

3. Startup Command:
   python scripts/setup_model_volume.py --volume-path /models && \\
   uvicorn src.api.server:app --host 0.0.0.0 --port 8000
   
   Note: Dependencies should be pre-installed in the Docker image.
   The Dockerfile already installs all dependencies, so no pip install is needed.

4. Health Check:
   - Path: /health
   - Interval: 30s
   - Timeout: 10s

5. Testing:
   - Full pipeline testing: POST /api/v1/jobs with test images
   - Memory profiling: Monitor VRAM usage during execution
   - Performance testing: Measure cold start and execution times
   - LoRA training: Use train_lora_sdxl.py for fine-tuning

6. After successful testing, deploy to Modal for production.
""")


if __name__ == "__main__":
    print_runpod_setup_instructions()




