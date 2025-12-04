#!/bin/bash
# Installation script that handles dependency conflicts
# Specifically handles lama-cleaner which conflicts with diffusers version requirements

set -e  # Exit on error

echo "Installing system libraries required for OpenCV..."
apt-get update -qq
apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1 || true

echo "Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Installing lama-cleaner with --no-deps (to avoid diffusers conflict)..."
echo "Note: lama-cleaner requires diffusers==0.16.1, but we use diffusers>=0.30.0 for SDXL"
pip install --no-deps lama-cleaner>=1.2.0

echo ""
echo "Installing lama-cleaner runtime dependencies (most already installed)..."
# These are the minimal dependencies needed for ModelManager to work
# Most are already installed via requirements.txt
pip install pydantic rich yacs omegaconf safetensors piexif loguru || true

echo ""
echo "Verifying installation..."
python -c "import diffusers; print(f'diffusers: {diffusers.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "from lama_cleaner.model_manager import ModelManager; print('lama-cleaner: OK')" || echo "Warning: lama-cleaner import failed - may need additional dependencies"

echo ""
echo "Installation complete!"

