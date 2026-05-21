#!/bin/bash
# Install prc_custom_ops independently
# Usage: ./install.sh [--editable]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$1" == "--editable" ] || [ "$1" == "-e" ]; then
    echo "Installing prc_custom_ops in editable mode..."
    pip install -e .
else
    echo "Installing prc_custom_ops..."
    pip install .
fi

echo "prc_custom_ops installed successfully!"
