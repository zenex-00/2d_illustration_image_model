#!/bin/bash
# Helper script to invoke lama-cleaner CLI from the isolated virtual environment
# Usage: ./scripts/lama_cleaner_cli.sh [lama-cleaner arguments...]

set -e

# Path to isolated lama-cleaner virtual environment
LAMA_VENV_DIR="${LAMA_VENV_DIR:-/opt/lama-cleaner-venv}"

if [ ! -d "$LAMA_VENV_DIR" ]; then
    echo "Error: Lama-cleaner virtual environment not found at: $LAMA_VENV_DIR"
    echo "Please run scripts/install_dependencies.sh first to create the venv."
    exit 1
fi

# Activate venv and run lama-cleaner
source "$LAMA_VENV_DIR/bin/activate"
exec lama-cleaner "$@"


