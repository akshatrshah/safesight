#!/usr/bin/env bash
# SafeSight setup script.
# Creates a virtual environment, installs dependencies, and verifies
# everything imports correctly. Safe to re-run — it won't recreate the
# venv if one already exists.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e  # exit immediately if any command fails

VENV_DIR="venv"
PYTHON_BIN="python3"

echo "== SafeSight setup =="

# 1. Check python3 exists
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first (e.g. 'brew install python3' on macOS)."
    exit 1
fi

echo "Using: $($PYTHON_BIN --version)"

# 2. Create virtual environment if it doesn't already exist
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at ./$VENV_DIR — reusing it."
else
    echo "Creating virtual environment at ./$VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# 3. Activate it for the rest of this script
source "$VENV_DIR/bin/activate"

# 4. Upgrade pip, then install project dependencies
echo "Installing dependencies from requirements.txt ..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 5. Verify the core libraries actually import
echo "Verifying installation ..."
python3 -c "
import ultralytics, cv2, numpy, yaml
print(f'  ultralytics {ultralytics.__version__}')
print(f'  opencv      {cv2.__version__}')
print(f'  numpy       {numpy.__version__}')
"

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  source venv/bin/activate     # activate the environment in your shell"
echo "  pytest tests/ -v             # run the test suite"
echo "  python scripts/run_detection.py --source sample_data --out outputs/annotated"
echo "  python scripts/benchmark_latency.py --source sample_data/bus.jpg"
echo "  python perception/detection/evaluate.py"