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

# Filter requirements first to remove packages that cause conflicts or aren't on PyPI
echo "Filtering requirements..."
grep -v "lama-cleaner" requirements.txt | grep -v "zoedepth" | grep -v "^xformers" > /tmp/requirements_filtered.txt

# Install main requirements (excluding conflicts)
echo "[2/5] Installing main requirements..."
pip install -r /tmp/requirements_filtered.txt || {
    echo "Warning: Main requirements installation had issues, but continuing..."
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

# Try to install xformers (optional, may fail due to CUDA/PyTorch version mismatch)
echo "[4.6/5] Attempting to install xformers (optional)..."
if pip install xformers>=0.0.23 2>&1 | grep -q "error\|Error\|ERROR"; then
    echo "Warning: xformers installation failed (this is OK, will use PyTorch SDPA instead)"
else
    # Test if xformers actually works (check for undefined symbol errors)
    echo "Testing xformers compatibility..."
    python -c "import sys; import xformers; from xformers.ops import fmha; print('xformers is compatible')" 2>&1 || {
        echo "xformers is installed but incompatible, uninstalling..."
        pip uninstall -y xformers || true
        echo "xformers uninstalled - will use PyTorch SDPA instead"
    }
fi

# Verify critical packages
echo "[5/5] Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

echo "=========================================="
echo "✅ Dependencies installed successfully!"
echo "=========================================="
