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
from pathlib import Path

# Path to isolated lama-cleaner virtual environment
LAMA_VENV_DIR = os.getenv("LAMA_VENV_DIR", "/opt/lama-cleaner-venv")

def get_venv_python():
    """Get the Python executable from the lama-cleaner venv"""
    if sys.platform == "win32":
        python_exe = Path(LAMA_VENV_DIR) / "Scripts" / "python.exe"
    else:
        python_exe = Path(LAMA_VENV_DIR) / "bin" / "python"
    
    return python_exe

def main():
    if not Path(LAMA_VENV_DIR).exists():
        print(f"Error: Lama-cleaner virtual environment not found at: {LAMA_VENV_DIR}")
        print("Please run scripts/install_dependencies.sh first to create the venv.")
        sys.exit(1)
    
    python_exe = get_venv_python()
    if not python_exe.exists():
        print(f"Error: Python executable not found at: {python_exe}")
        print(f"Virtual environment may be corrupted. Please reinstall.")
        sys.exit(1)
    
    # Get lama-cleaner executable
    if sys.platform == "win32":
        lama_exe = Path(LAMA_VENV_DIR) / "Scripts" / "lama-cleaner.exe"
    else:
        lama_exe = Path(LAMA_VENV_DIR) / "bin" / "lama-cleaner"
    
    # If executable doesn't exist, try running via python -m
    if not lama_exe.exists():
        cmd = [str(python_exe), "-m", "lama_cleaner.app"] + sys.argv[1:]
    else:
        cmd = [str(lama_exe)] + sys.argv[1:]
    
    # Execute lama-cleaner
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()


