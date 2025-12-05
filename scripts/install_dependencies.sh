#!/bin/bash
# Gemini 3 Pro Vehicle-to-Vector Pipeline - Dependency Installation Script
# This script handles the installation of all dependencies, including resolving
# conflicts between lama-cleaner and diffusers versions.

set -e  # Exit on error

echo "=========================================="
echo "Gemini 3 Pro Pipeline - Installing Dependencies"
echo "=========================================="

# Upgrade pip first
echo "[1/5] Upgrading pip..."
pip install --upgrade pip

# Install main requirements (excluding lama-cleaner to avoid conflict)
echo "[2/5] Installing main requirements..."
pip install -r requirements.txt || {
    echo "Warning: Some requirements failed, attempting workaround..."
    
    grep -v "lama-cleaner" requirements.txt | grep -v "zoedepth" > /tmp/requirements_filtered.txt
    pip install -r /tmp/requirements_filtered.txt
}

# Install lama-cleaner without its dependencies to avoid diffusers version conflict
# lama-cleaner requires diffusers==0.16.1 but we need diffusers>=0.30.0 for SDXL ControlNet
echo "[3/5] Installing lama-cleaner (without conflicting dependencies)..."
pip install --no-deps lama-cleaner>=1.2.0 || echo "Warning: lama-cleaner installation skipped"

# Install lama-cleaner's required dependencies (except diffusers)
echo "[4/5] Installing lama-cleaner dependencies..."
pip install pydantic rich yacs omegaconf safetensors piexif loguru || true

# Install ZoeDepth manually since it's not on PyPI
echo "[4.5/5] Installing ZoeDepth from GitHub..."
if [ ! -d "ZoeDepth" ]; then
    git clone https://github.com/isl-org/ZoeDepth.git
    echo "ZoeDepth cloned."
else
    echo "ZoeDepth already exists."
fi
# Add ZoeDepth to PYTHONPATH in current session (user needs to add it to their env persistently)
export PYTHONPATH=$PYTHONPATH:$(pwd)/ZoeDepth

# Verify critical packages
echo "[5/5] Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

echo "=========================================="
echo "✅ Dependencies installed successfully!"
echo "=========================================="
