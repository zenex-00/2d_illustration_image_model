# Lama-Cleaner Isolated Environment Setup

## Overview

Lama-cleaner has been isolated in a separate Python virtual environment to prevent dependency conflicts with the main pipeline. The main pipeline uses modern versions of `diffusers` (>=0.30.0) and `transformers` (>=4.56.2), while lama-cleaner requires older versions (`diffusers==0.16.1`, `transformers==4.27.4`).

## Installation

The installation script (`scripts/install_dependencies.sh`) automatically creates an isolated virtual environment for lama-cleaner at `/opt/lama-cleaner-venv` (or the path specified by `LAMA_VENV_DIR` environment variable).

### During Installation

The script will:
1. Install all main pipeline dependencies (excluding lama-cleaner)
2. Create a separate virtual environment for lama-cleaner
3. Install lama-cleaner and all its dependencies in the isolated environment
4. Suppress root user warnings with `PIP_ROOT_USER_ACTION=ignore`

## Usage

### From Python Code

The `LaMaInpainter` class automatically uses the isolated environment. No code changes are needed - it will dynamically add the venv's site-packages to `sys.path` when importing `lama_cleaner`.

```python
from src.phase1_semantic_sanitization.lama_inpainter import LaMaInpainter

inpainter = LaMaInpainter()
result = inpainter.inpaint(image, mask)
```

### From Command Line

#### Option 1: Using the Bash Helper Script (Linux/macOS)

```bash
./scripts/lama_cleaner_cli.sh [arguments...]
```

#### Option 2: Using the Python Helper Script (Cross-platform)

```bash
python scripts/lama_cleaner_cli.py [arguments...]
```

#### Option 3: Manual Activation

```bash
# Linux/macOS
source /opt/lama-cleaner-venv/bin/activate
lama-cleaner [arguments...]
deactivate

# Windows
C:\opt\lama-cleaner-venv\Scripts\activate
lama-cleaner [arguments...]
deactivate
```

## Environment Variable

You can customize the venv location by setting `LAMA_VENV_DIR`:

```bash
export LAMA_VENV_DIR=/custom/path/to/venv
./scripts/install_dependencies.sh
```

## Troubleshooting

### Venv Not Found

If you see a warning about the venv not being found:
1. Ensure `scripts/install_dependencies.sh` has been run
2. Check that `LAMA_VENV_DIR` points to the correct location
3. Verify the venv exists: `ls -la $LAMA_VENV_DIR` (or `dir %LAMA_VENV_DIR%` on Windows)

### Import Errors

If you encounter import errors when using `LaMaInpainter`:
1. Verify the venv was created successfully
2. Check that lama-cleaner is installed in the venv:
   ```bash
   source /opt/lama-cleaner-venv/bin/activate
   python -c "import lama_cleaner; print(lama_cleaner.__version__)"
   deactivate
   ```

### Reinstalling Lama-Cleaner

To reinstall lama-cleaner in the isolated environment:

```bash
source /opt/lama-cleaner-venv/bin/activate
pip install --upgrade lama-cleaner>=1.2.0
deactivate
```

## Architecture

```
Main Pipeline Environment:
├── diffusers>=0.30.0
├── transformers>=4.56.2
├── controlnet-aux>=0.0.10
└── ... (other modern dependencies)

Isolated Lama-Cleaner Environment (/opt/lama-cleaner-venv):
├── diffusers==0.16.1
├── transformers==4.27.4
├── controlnet-aux==0.0.3
├── lama-cleaner>=1.2.0
└── ... (lama-cleaner's dependencies)
```

The Python code dynamically imports from the isolated environment when needed, ensuring no conflicts occur.


