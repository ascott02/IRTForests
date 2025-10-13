#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/init_venv.sh [venv_dir]
# Creates a Python virtual environment, upgrades pip, and installs project requirements.

VENVDIR=${1:-.venv}
python3 -m venv "$VENVDIR"
source "$VENVDIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment ready at $VENVDIR"
echo "To activate later: source $VENVDIR/bin/activate"
