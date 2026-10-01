#!/usr/bin/env bash
# smoke_test.sh: check the real TITAN pipeline on ONE slide, on the GPU server.
#
# Usage:
#   export HF_TOKEN="hf_..."
#   bash smoke_test.sh /path/to/slide.ndpi [PROJECTOR.pt]
#
# Steps:
#   1. titan env: imports, GPU, OpenSlide can read the slide (mpp, levels)
#   2. tissue / tiling preview  -> smoke/qc/<slide>_tiles.png   (look at it!)
#   3. TITAN download + embedding (checks gated HF access) -> smoke/cache.pt
#   4. conch env: cache is 768-d; if a TITAN projector exists, generate a report
#
# Env overrides: EMBED_ENV=titan CONDA_ENV=conch DEVICE=cuda

set -euo pipefail

SLIDE="${1:?usage: bash smoke_test.sh /path/to/slide.ndpi [PROJECTOR.pt]}"
PROJ="${2:-}"
EMBED_ENV="${EMBED_ENV:-titan}"
CONDA_ENV="${CONDA_ENV:-conch}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../src"
OUT="$HERE/smoke"
mkdir -p "$OUT"
SLIDE="$(cd "$(dirname "$SLIDE")" && pwd)/$(basename "$SLIDE")"

[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: export HF_TOKEN first" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "== 1. environment ($EMBED_ENV) =="
conda activate "$EMBED_ENV"
python - "$SLIDE" <<'PY'
import sys, torch, transformers, openslide
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
s = openslide.OpenSlide(sys.argv[1])
print("slide", s.dimensions, "levels", s.level_count, [round(d, 2) for d in s.level_downsamples])
print("mpp-x", s.properties.get("openslide.mpp-x"), "| objective", s.properties.get("openslide.objective-power"))
PY

echo; echo "== 2. tiling preview =="
python "$SRC/tile_preview.py" "$SLIDE" -o "$OUT/qc" --save-patches 4

echo; echo "== 3. TITAN embedding =="
T0=$(date +%s)
python "$SRC/embed-s.py" "$SLIDE" -o "$OUT/cache.pt" --feat-dir "$OUT/titan_feats" ${DEVICE:+--device "$DEVICE"}
echo "embedding took $(( $(date +%s) - T0 ))s"

echo; echo "== 4. projector side ($CONDA_ENV) =="
conda activate "$CONDA_ENV"
cd "$SRC"
echo "cache dim: $(python info.py --cache-dim "$OUT/cache.pt")  (expected 768)"
if [[ -z "$PROJ" ]]; then PROJ="$(python info.py --latest ../checkpoints 2>/dev/null || true)"; fi
if [[ -n "$PROJ" ]]; then
  echo "generating with $PROJ"
  CACHE="$OUT/cache.pt" SKIP_CONDA=1 MAX_NEW_TOKENS=120 bash run.sh "$SLIDE" -c "$PROJ" -o "$OUT" -n smoke
  cat "$OUT/smoke.json"
else
  echo "No TITAN projector yet: build a training cache and run train.py (see README), then re-run."
fi

echo; echo "Smoke test OK. Check $OUT/qc/ for the tissue mask / patch grid."
