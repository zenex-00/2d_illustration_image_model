#!/usr/bin/env python3
"""
Helper script to invoke lama-cleaner CLI from the isolated virtual environment.
This works cross-platform (Windows, Linux, macOS).

Usage:
    python scripts/lama_cleaner_cli.py [lama-cleaner arguments...]
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Path to isolated lama-cleaner virtual environment
LAMA_VENV_DIR = os.getenv("LAMA_VENV_DIR", "/opt/lama-cleaner-venv")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_venv_python():
    """Get the Python executable from the lama-cleaner venv"""
    if sys.platform == "win32":
        python_exe = Path(LAMA_VENV_DIR) / "Scripts" / "python.exe"
    else:
        python_exe = Path(LAMA_VENV_DIR) / "bin" / "python"
    
    return python_exe

def get_lama_executable():
    """Get lama-cleaner executable path"""
    if sys.platform == "win32":
        return Path(LAMA_VENV_DIR) / "Scripts" / "lama-cleaner.exe"
    else:
        return Path(LAMA_VENV_DIR) / "bin" / "lama-cleaner"

def main():
    if not Path(LAMA_VENV_DIR).exists():
        logger.error(f"Lama-cleaner venv not found at: {LAMA_VENV_DIR}")
        logger.error("Please run: python scripts/install_dependencies.sh")
        sys.exit(1)
    
    python_exe = get_venv_python()
    if not python_exe.exists():
        logger.error(f"Python executable not found: {python_exe}")
        logger.error("Virtual environment may be corrupted.")
        sys.exit(1)
    
    lama_exe = get_lama_executable()
    
    if not lama_exe.exists():
        logger.info("Using python -m lama_cleaner.app")
        cmd = [str(python_exe), "-m", "lama_cleaner.app"] + sys.argv[1:]
    else:
        logger.info(f"Using lama-cleaner executable: {lama_exe}")
        cmd = [str(lama_exe)] + sys.argv[1:]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        # Use subprocess.run with proper error handling
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            logger.error(f"Lama-cleaner failed with exit code {result.returncode}")
            logger.error("Check the output above for details.")
        
        sys.exit(result.returncode)
    
    except FileNotFoundError as e:
        logger.error(f"Executable not found: {e}")
        logger.error("Make sure lama-cleaner is installed in the venv.")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


