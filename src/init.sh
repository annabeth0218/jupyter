#!/usr/bin/env bash
# init.sh — install dependencies into the existing `conch` conda environment.
# Run this ONCE on the JupyterLab server, from inside Anna_CONCH/CONCH.
#
# Usage:
#   bash init.sh
#
# Optional overrides:
#   CONDA_ENV=myenv  bash init.sh          # use a different env name
#   TORCH_INDEX_URL=... bash init.sh       # use a different CUDA wheel index
#   SKIP_TORCH=1 bash init.sh              # skip torch install (already installed)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-conch}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
SKIP_TORCH="${SKIP_TORCH:-0}"

# --- activate the existing conda env ---
# `conda activate` only works after sourcing conda's shell hook in non-interactive shells.
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found on PATH. Open a terminal where conda is initialized, or run:"
  echo "  source /opt/conda/etc/profile.d/conda.sh   # adjust path to your conda install"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "ERROR: conda env '$CONDA_ENV' does not exist. Create it first, e.g.:"
  echo "  conda create -n $CONDA_ENV python=3.10 -y"
  exit 1
fi

echo "Activating conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

echo "Python: $(which python)"
echo "Pip:    $(which pip)"

echo "Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

if [[ "$SKIP_TORCH" != "1" ]]; then
  echo "Installing PyTorch CUDA wheels from: $TORCH_INDEX_URL"
  pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
  echo "Skipping torch install (SKIP_TORCH=1)"
fi

echo "Installing core model dependencies"
pip install transformers tqdm accelerate safetensors sentencepiece protobuf pillow numpy

echo "Installing optional A100 / image-format helpers"
pip install bitsandbytes pydicom openslide-python || \
  echo "WARNING: one or more optional packages failed to install; continuing."

echo "Installing CONCH"
pip install git+https://github.com/Mahmoodlab/CONCH.git

echo
echo "Install complete in conda env: $CONDA_ENV"
echo
echo "For every new shell, activate with:"
echo "  cd Anna_CONCH/CONCH"
echo "  conda activate $CONDA_ENV"
echo
echo "Before first run, export your Hugging Face token:"
echo "  export HF_TOKEN=\"hf_your_token_here\""