#!/usr/bin/env bash
# run.sh — embed pathology images then generate interpretations.
#
# Usage:
#   bash run.sh <source> [-o OUTPUT_DIR] [-p PROMPT_FILE] [-c PROJECTOR]
#               [-l LLM] [-n NAME] [-- extra args passed to eval.py]
#
# Positional:
#   <source>    Image file, folder, glob, URL, CSV, JSON, or JSONL manifest.
#
# Options:
#   -o, --output-dir DIR     Where to write predictions.
#                            Default: outputs/run_<timestamp>
#   -n, --name NAME          Friendly name; writes <output-dir>/<name>.json
#                            instead of <output-dir>/predictions.json.
#   -p, --prompt-file FILE   Path to a .txt file used as the generation prompt.
#   -c, --projector FILE     Projector .pt checkpoint.
#                            Default: $PROJECTOR or checkpoints/proj-arvo-llama.pt
#   -l, --llm NAME           HF model id.
#                            Default: $LLM or Qwen/Qwen2-7B-Instruct
#   -h, --help               Show this help.
#
# Environment overrides (CLI flags win):
#   CONDA_ENV=conch
#   PROJECTOR=checkpoints/proj-arvo-llama.pt
#   LLM=Qwen/Qwen2-7B-Instruct
#   ID_KEY=id
#   IMAGE_KEY=image
#   RUN_4BIT=1
#   MAX_NEW_TOKENS=200
#   TEMPERATURE=0.2
#   TOP_P=0.9
#   SAMPLE=1
#   DEVICE=cuda
#   KEEP_CACHE=1             # keep the embedding cache after the run (for debugging)
#   CACHE=/path/to/cache.pt  # use an existing cache and skip embedding entirely
#
# Examples:
#   bash run.sh manifest.jsonl -c checkpoints/proj.pt -p prompts/few_shot.txt -o outputs/s109
#   bash run.sh manifest.jsonl -n s109_run1
#   KEEP_CACHE=1 bash run.sh manifest.jsonl          # inspect cache afterwards
#   CACHE=outputs/prev/cache.pt bash run.sh manifest.jsonl   # reuse existing cache

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

print_help() {
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Defaults from env
# ---------------------------------------------------------------------------

SOURCE=""
OUT_DIR=""
NAME=""
PROMPT_FILE=""
PROJECTOR="${PROJECTOR:-checkpoints/proj-arvo-llama.pt}"
LLM="${LLM:-Qwen/Qwen2-7B-Instruct}"

CONDA_ENV="${CONDA_ENV:-conch}"
#ID_KEY="${ID_KEY:-id}"
#IMAGE_KEY="${IMAGE_KEY:-image}"
RUN_4BIT="${RUN_4BIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
SAMPLE="${SAMPLE:-0}"
DEVICE="${DEVICE:-}"
KEEP_CACHE="${KEEP_CACHE:-0}"
CACHE="${CACHE:-}"   # if set, skip embedding and use this cache directly

# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)       print_help; exit 0 ;;
    -o|--output-dir) OUT_DIR="$2";    shift 2 ;;
    -n|--name)       NAME="$2";       shift 2 ;;
    -p|--prompt-file)PROMPT_FILE="$2";shift 2 ;;
    -c|--projector)  PROJECTOR="$2";  shift 2 ;;
    -l|--llm)        LLM="$2";        shift 2 ;;
    --)              shift; EXTRA_ARGS+=("$@"); break ;;
    -*) die "Unknown option: $1" ;;
    *)
      [[ -z "$SOURCE" ]] || die "Unexpected positional argument: $1"
      SOURCE="$1"; shift ;;
  esac
done

[[ -n "$SOURCE" ]] || { print_help >&2; die "missing <source> argument."; }

OUT_DIR="${OUT_DIR:-outputs/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Activate conda
# ---------------------------------------------------------------------------

command -v conda >/dev/null 2>&1 || die "'conda' not found on PATH."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "Using conda env: $CONDA_ENV  ($(which python))"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

[[ -n "${HF_TOKEN:-}" ]] || die \
  "HF_TOKEN is not set. Create a read token at https://huggingface.co/settings/tokens and run: export HF_TOKEN=\"hf_...\""

[[ -f "$PROJECTOR" ]] || die \
  "Projector checkpoint not found: $PROJECTOR"

[[ -e "$SOURCE" || "$SOURCE" == http* ]] || die \
  "Input source not found: $SOURCE"

if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || die "Prompt file not found: $PROMPT_FILE"
fi

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

if [[ -n "$NAME" ]]; then
  PRED="$OUT_DIR/${NAME}.json"
else
  PRED="$OUT_DIR/predictions.json"
fi

# ---------------------------------------------------------------------------
# Cache lifecycle
# ---------------------------------------------------------------------------
# If CACHE is already set (env var), use it directly and never delete it.
# Otherwise create a temp file, embed into it, and delete it on exit.

OWNS_CACHE=0
if [[ -z "$CACHE" ]]; then
  CACHE="$(mktemp --suffix=.pt)"
  OWNS_CACHE=1
  # Always clean up on exit unless KEEP_CACHE=1
  trap '[[ "$KEEP_CACHE" == "1" ]] && echo "Cache kept at: $CACHE" || rm -f "$CACHE"' EXIT
fi

# ---------------------------------------------------------------------------
# Print config
# ---------------------------------------------------------------------------

echo
echo "Run configuration:"
echo "  source        : $SOURCE"
echo "  output dir    : $OUT_DIR"
echo "  predictions   : $PRED"
echo "  projector     : $PROJECTOR"
echo "  llm           : $LLM"
echo "  prompt file   : ${PROMPT_FILE:-<eval.py default>}"
echo "  device        : ${DEVICE:-auto}"
echo "  4-bit load    : $RUN_4BIT"
echo "  sample        : $SAMPLE"
echo "  keep cache    : $KEEP_CACHE"
echo "  cache path    : $CACHE"
echo

# ---------------------------------------------------------------------------
# Step 1: embed (skip if CACHE was provided externally)
# ---------------------------------------------------------------------------

if [[ "$OWNS_CACHE" == "1" ]]; then
  echo "Step 1/2: embedding images from $SOURCE"
  EMBED_ARGS=(
    "$SOURCE"
#    --image-key "$IMAGE_KEY"
#    --id-key    "$ID_KEY"
    --output    "$CACHE"
  )
  [[ -n "$DEVICE" ]] && EMBED_ARGS+=(--device "$DEVICE")
  python embed.py "${EMBED_ARGS[@]}"
else
  echo "Step 1/2: SKIPPED — using existing cache: $CACHE"
  [[ -f "$CACHE" ]] || die "Provided CACHE not found: $CACHE"
fi

# ---------------------------------------------------------------------------
# Step 2: generate
# ---------------------------------------------------------------------------

echo
echo "Step 2/2: generating interpretations with $LLM"

INFER_ARGS=(
  --cache          "$CACHE"
  --projector      "$PROJECTOR"
  --output         "$PRED"
  --llm            "$LLM"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature    "$TEMPERATURE"
  --top-p          "$TOP_P"
)
[[ -n "$DEVICE" ]]       && INFER_ARGS+=(--device "$DEVICE")
[[ "$RUN_4BIT" == "1" ]] && INFER_ARGS+=(--load-4bit)
[[ "$SAMPLE"   == "1" ]] && INFER_ARGS+=(--sample)
[[ -n "$PROMPT_FILE" ]]  && INFER_ARGS+=(--prompt-file "$PROMPT_FILE")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && INFER_ARGS+=("${EXTRA_ARGS[@]}")

python eval.py "${INFER_ARGS[@]}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo
echo "Done."
echo "Predictions: $PRED"
