#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==============================================================="
echo "cache_prediction environment setup"
echo "==============================================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Virtual env: ${VENV_DIR}"
echo

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found."
  echo "Set PYTHON_BIN to a valid interpreter if needed."
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists. Reusing it."
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

echo
echo "Environment setup complete."
echo
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python -m tokenizer.test_tokenizer"
echo "  python -m train.train"
