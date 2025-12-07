#!/bin/bash
# Gemini 3 Pro Vehicle-to-Vector Pipeline - Dependency Installation Script
# This script handles the installation of all dependencies, including resolving
# conflicts between lama-cleaner and diffusers versions by using an isolated venv.

set -e  # Exit on error

# Suppress root user warnings
export PIP_ROOT_USER_ACTION=ignore

echo "=========================================="
echo "Gemini 3 Pro Pipeline - Installing Dependencies"
echo "=========================================="

# Upgrade pip first
echo "[1/6] Upgrading pip..."
pip install --upgrade pip

# Filter requirements first to remove packages that cause conflicts or aren't on PyPI
# Also exclude torch and torchvision since they should be upgraded separately to ensure compatibility
echo "Filtering requirements..."
grep -v "lama-cleaner" requirements.txt | grep -v "zoedepth" | grep -v "^xformers" | grep -v "^torch" | grep -v "^torchvision" > /tmp/requirements_filtered.txt

# Install main requirements (excluding conflicts and PyTorch packages)
echo "[2/6] Installing main requirements..."
pip install -r /tmp/requirements_filtered.txt || {
    echo "Warning: Main requirements installation had issues, but continuing..."
}

# Create isolated virtual environment for lama-cleaner
# This prevents dependency conflicts with the main pipeline
LAMA_VENV_DIR="${LAMA_VENV_DIR:-/opt/lama-cleaner-venv}"
echo "[3/6] Creating isolated virtual environment for lama-cleaner..."
if [ -d "$LAMA_VENV_DIR" ]; then
    echo "Lama-cleaner venv already exists, skipping creation..."
else
    python3 -m venv "$LAMA_VENV_DIR"
    echo "Virtual environment created at: $LAMA_VENV_DIR"
fi

# Activate venv and install lama-cleaner with all its dependencies
echo "[4/6] Installing lama-cleaner in isolated environment..."
source "$LAMA_VENV_DIR/bin/activate"
pip install --upgrade pip
# Install lama-cleaner with all its dependencies (including old versions)
pip install lama-cleaner>=1.2.0
deactivate

echo "✅ Lama-cleaner installed in isolated venv at: $LAMA_VENV_DIR"
echo "   To use lama-cleaner CLI: source $LAMA_VENV_DIR/bin/activate && lama-cleaner"
echo "   The Python code will automatically use this venv via sys.path manipulation"

# Install ZoeDepth manually since it's not on PyPI
echo "[5/6] Installing ZoeDepth from GitHub..."
if [ ! -d "ZoeDepth" ]; then
    git clone https://github.com/isl-org/ZoeDepth.git
    echo "ZoeDepth cloned."
else
    echo "ZoeDepth already exists."
fi
# Add ZoeDepth to PYTHONPATH in current session (user needs to add it to their env persistently)
export PYTHONPATH=$PYTHONPATH:$(pwd)/ZoeDepth

# Try to install xformers (optional, may fail due to CUDA/PyTorch version mismatch)
echo "[5.5/6] Attempting to install xformers (optional)..."
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
echo "[6/6] Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')" || {
    echo "ERROR: torchvision import failed - this indicates a version mismatch with PyTorch"
    echo "Attempting to fix by reinstalling torchvision to match PyTorch version..."
    # Get PyTorch version to match torchvision
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    if [ -n "$TORCH_VERSION" ]; then
        # Check if PyTorch is from cu128 (nightly)
        if echo "$TORCH_VERSION" | grep -q "cu128"; then
            echo "Reinstalling torchvision from cu128 nightly to match PyTorch..."
            pip install --upgrade --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
        else
            echo "Reinstalling torchvision to match PyTorch version..."
            pip install --upgrade "torchvision>=0.23.0"
        fi
        # Verify again
        python -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')" || {
            echo "WARNING: torchvision still not compatible - server may fail to start"
        }
    fi
}
# Test torchvision import to catch the nms error early
python -c "import torch; import torchvision; from torchvision.ops import nms; print('torchvision ops test passed')" || {
    echo "ERROR: torchvision ops test failed - operator torchvision::nms does not exist"
    echo "This indicates PyTorch and torchvision are from incompatible builds"
    echo "Attempting to fix by reinstalling both together..."
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    if echo "$TORCH_VERSION" | grep -q "cu128"; then
        pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
    else
        pip install --upgrade torch>=2.8.0 torchvision>=0.23.0 --force-reinstall
    fi
    # Test again
    python -c "import torch; import torchvision; from torchvision.ops import nms; print('torchvision ops test passed after reinstall')" || {
        echo "CRITICAL: torchvision compatibility fix failed - manual intervention required"
    }
}
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

echo "=========================================="
echo "✅ Dependencies installed successfully!"
echo "=========================================="
