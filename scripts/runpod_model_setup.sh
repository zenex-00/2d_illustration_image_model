#!/bin/bash
# Alternative model setup for RunPod - download models one by one with space checking

set -e

VOLUME_PATH="${1:-/models}"
MODEL_TYPE="${2:-all}"

echo "RunPod Model Setup - Space-Conscious Download"
echo "Target volume: $VOLUME_PATH"
echo "Model type: $MODEL_TYPE"
echo ""

# Check available space
echo "Checking available space..."
AVAILABLE_SPACE=$(df "$VOLUME_PATH" | tail -1 | awk '{print $4}' | sed 's/G$//')
echo "Available space: ${AVAILABLE_SPACE}GB"

# Warn if space is low
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    echo "WARNING: Less than 10GB available. This may not be sufficient for all models."
    echo "Consider cleaning up space or using a larger volume."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create the volume directory
mkdir -p "$VOLUME_PATH"

# Function to download a single model
download_model() {
    local model_name=$1
    local volume_path=$2
    echo "Downloading $model_name..."

    # Check space before each download
    CURRENT_AVAILABLE=$(df "$volume_path" | tail -1 | awk '{print $4}')
    echo "Space before download: $CURRENT_AVAILABLE"

    if python -c "
import sys
sys.path.insert(0, '/workspace/image_generation')
from scripts.setup_model_volume import setup_model_volume
setup_model_volume('$volume_path', models=['$model_name'])
";
    then
        echo "✓ Successfully downloaded $model_name"
    else
        echo "✗ Failed to download $model_name"
        return 1
    fi
}

# Download models based on type
if [ "$MODEL_TYPE" = "all" ] || [ "$MODEL_TYPE" = "essential" ]; then
    echo "Downloading essential models first..."

    # Download smaller models first
    download_model "grounding_dino" "$VOLUME_PATH" || echo "Grounding DINO failed"
    download_model "sdxl" "$VOLUME_PATH" || echo "SDXL failed"
    download_model "controlnet_depth" "$VOLUME_PATH" || echo "ControlNet Depth failed"
    download_model "controlnet_canny" "$VOLUME_PATH" || echo "ControlNet Canny failed"
fi

if [ "$MODEL_TYPE" = "all" ] || [ "$MODEL_TYPE" = "sam" ]; then
    echo "Downloading SAM model (largest model)..."
    download_model "sam" "$VOLUME_PATH" || echo "SAM failed"
fi

if [ "$MODEL_TYPE" = "all" ] || [ "$MODEL_TYPE" = "realesrgan" ]; then
    echo "Setting up RealESRGAN..."
    download_model "realesrgan" "$VOLUME_PATH" || echo "RealESRGAN failed"
fi

echo ""
echo "Model setup completed. Final space check:"
df -h "$VOLUME_PATH"