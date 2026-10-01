#!/usr/bin/env bash
# setup_env.sh: create the conda env used for TITAN embedding (run.sh step 1).
#
# Usage:
#   bash setup_env.sh              # env name "titan", CUDA 11.8 wheels
#   bash setup_env.sh myenv        # custom env name
#   CUDA=cu117 bash setup_env.sh   # other CUDA wheel tag (cu117 / cu118)
#
# The existing "conch" env (projector + Qwen) is NOT touched: TITAN pins
# torch 2.0.1 and transformers 4.46.0, which conflict with the newer
# transformers used by train.py, so the two steps use two envs.

set -euo pipefail

ENV_NAME="${1:-titan}"
CUDA="${CUDA:-cu118}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found on PATH" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Env '$ENV_NAME' already exists; installing/updating packages in it."
else
  conda create -n "$ENV_NAME" python=3.9 -y
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url "https://download.pytorch.org/whl/${CUDA}"
python -m pip install -r "$HERE/requirements-titan.txt"

echo
echo "Checking the install..."
python - <<'PY'
import torch, transformers, timm, openslide, h5py
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("transformers", transformers.__version__, "| timm", timm.__version__)
print("openslide", openslide.__version__, "| library", openslide.__library_version__)
PY

cat <<EOF

Done. Next:
  1. Request access to https://huggingface.co/MahmoodLab/TITAN (use the same HF account as your token).
  2. export HF_TOKEN="hf_..."
  3. bash smoke_test.sh /path/to/one_slide.ndpi
EOF
