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
# Options (all optional; long forms also accepted):
#   -o, --output-dir DIR     Where to write cache + predictions.
#                            Default: outputs/run_<timestamp>
#   -n, --name NAME          Friendly name used as the predictions filename
#                            (writes <output-dir>/<name>.json instead of
#                            <output-dir>/predictions.json).
#   -p, --prompt-file FILE   Path to a .txt file whose contents are passed to
#                            eval.py as the generation prompt.
#   -c, --projector FILE     Projector .pt checkpoint.
#                            Default: $PROJECTOR or checkpoints/proj-arvo-llama.pt
#   -l, --llm NAME           HF model id for the LLM (e.g. Qwen/Qwen2-7B-Instruct,
#                            meta-llama/Llama-3.1-8B). Default: $LLM or
#                            meta-llama/Llama-3.1-8B
#   -h, --help               Show this help.
#
# Environment overrides (still honored, CLI flags win):
#   CONDA_ENV=conch
#   PROJECTOR=checkpoints/proj-arvo-llama.pt
#   LLM=meta-llama/Llama-3.1-8B
#   ID_KEY=id
#   IMAGE_KEY=image
#   RUN_4BIT=1                 # load LLM with 4-bit quantization (bitsandbytes)
#   MAX_NEW_TOKENS=300
#   TEMPERATURE=0.2
#   TOP_P=0.9
#   SAMPLE=1                   # sampling instead of greedy
#   DEVICE=cuda                # cuda | cpu | mps (empty = auto)
#   SKIP_EMBED=1               # reuse an existing cache; pair with CACHE=...
#   CACHE=/path/to/conch_cache.pt
#   HF_TOKEN=...               # required to download CONCH weights
#
# Examples:
#   bash run.sh case_s109_03375_manifest.jsonl
#   bash run.sh case_s109_03375_manifest.jsonl -o outputs/s109 -n s109_run1
#   bash run.sh manifest.jsonl -p prompts/few_shot.txt -l Qwen/Qwen2-7B-Instruct
#   bash run.sh manifest.jsonl -c checkpoints/proj-qwen.pt -l Qwen/Qwen2-7B-Instruct

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

print_help() {
  # Print the leading comment block (everything from line 2 up to the first
  # blank/non-# line). This keeps `--help` aligned with the doc above.
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

# --- defaults from env (CLI flags below override these) ---
SOURCE=""
OUT_DIR=""
NAME=""
PROMPT_FILE=""
PROJECTOR="${PROJECTOR:-checkpoints/proj-arvo-3.pt}"
LLM="${LLM:-Qwen/Qwen2-7B-Instruct}"

CONDA_ENV="${CONDA_ENV:-conch}"
ID_KEY="${ID_KEY:-id}"
IMAGE_KEY="${IMAGE_KEY:-image}"
RUN_4BIT="${RUN_4BIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
SAMPLE="${SAMPLE:-0}"
DEVICE="${DEVICE:-}"
SKIP_EMBED="${SKIP_EMBED:-0}"

# --- parse CLI ---
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help; exit 0 ;;
    -o|--output-dir)
      OUT_DIR="$2"; shift 2 ;;
    -n|--name)
      NAME="$2"; shift 2 ;;
    -p|--prompt-file)
      PROMPT_FILE="$2"; shift 2 ;;
    -c|--projector)
      PROJECTOR="$2"; shift 2 ;;
    -l|--llm)
      LLM="$2"; shift 2 ;;
    --)
      shift; EXTRA_ARGS+=("$@"); break ;;
    -*)
      echo "Unknown option: $1" >&2
      print_help >&2
      exit 1 ;;
    *)
      if [[ -z "$SOURCE" ]]; then
        SOURCE="$1"
      else
        echo "Unexpected positional argument: $1" >&2
        print_help >&2
        exit 1
      fi
      shift ;;
  esac
done

if [[ -z "$SOURCE" ]]; then
  echo "ERROR: missing <source> argument." >&2
  print_help >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-outputs/run_$(date +%Y%m%d_%H%M%S)}"

# --- activate conda env ---
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found on PATH. Source conda first, e.g.:"
  echo "  source /opt/conda/etc/profile.d/conda.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "Using conda env: $CONDA_ENV  ($(which python))"

# --- preflight checks ---
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set."
  echo "Create a read token at https://huggingface.co/settings/tokens, accept CONCH access, then run:"
  echo "  export HF_TOKEN=\"hf_your_token_here\""
  exit 1
fi

if [[ ! -f "$PROJECTOR" ]]; then
  echo "Projector checkpoint not found: $PROJECTOR" >&2
  echo "Pass one with -c/--projector or set PROJECTOR=/path/to/projector.pt" >&2
  exit 1
fi

if [[ ! -e "$SOURCE" ]] && [[ "$SOURCE" != http* ]]; then
  echo "Input source not found: $SOURCE" >&2
  exit 1
fi

# Read the prompt file if provided. We prefer this over the PROMPT env var
# because it is more reproducible and easier to share alongside checkpoints.
PROMPT_TEXT=""
if [[ -n "$PROMPT_FILE" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Prompt file not found: $PROMPT_FILE" >&2
    exit 1
  fi
  PROMPT_TEXT="$(cat "$PROMPT_FILE")"
  echo "Using prompt file: $PROMPT_FILE  ($(wc -c <"$PROMPT_FILE" | tr -d ' ') bytes)"
elif [[ -n "${PROMPT:-}" ]]; then
  PROMPT_TEXT="$PROMPT"
  echo "Using PROMPT env var ($(printf '%s' "$PROMPT_TEXT" | wc -c | tr -d ' ') bytes)"
else
  echo "Using eval.py's built-in DEFAULT_PROMPT."
fi

mkdir -p "$OUT_DIR"
CACHE="${CACHE:-$OUT_DIR/conch_cache.pt}"
if [[ -n "$NAME" ]]; then
  PRED="$OUT_DIR/${NAME}.json"
else
  PRED="$OUT_DIR/predictions.json"
fi

echo
echo "Run configuration:"
echo "  source       : $SOURCE"
echo "  output dir   : $OUT_DIR"
echo "  predictions  : $PRED"
echo "  cache        : $CACHE"
echo "  projector    : $PROJECTOR"
echo "  llm          : $LLM"
echo "  device       : ${DEVICE:-auto}"
echo "  4-bit load   : $RUN_4BIT"
echo "  sample       : $SAMPLE"
echo

# --- step 1: embed ---
if [[ "$SKIP_EMBED" == "1" ]]; then
  echo "Step 1/2: SKIPPED (using existing cache: $CACHE)"
  if [[ ! -f "$CACHE" ]]; then
    echo "ERROR: SKIP_EMBED=1 but cache not found: $CACHE" >&2
    exit 1
  fi
else
  echo "Step 1/2: embedding images from $SOURCE"
  EMBED_ARGS=(
    "$SOURCE"
    --image-key "$IMAGE_KEY"
    --id-key    "$ID_KEY"
    --output    "$CACHE"
  )
  [[ -n "$DEVICE" ]] && EMBED_ARGS+=(--device "$DEVICE")
  python embed.py "${EMBED_ARGS[@]}"
fi

# --- step 2: generate ---
echo
echo "Step 2/2: generating pathology interpretations with $LLM"
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
[[ -n "$PROMPT_TEXT" ]]  && INFER_ARGS+=(--prompt "$PROMPT_TEXT")

# Pass through anything after `--` directly to eval.py.
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  INFER_ARGS+=("${EXTRA_ARGS[@]}")
fi

python eval.py "${INFER_ARGS[@]}"

echo
echo "Done."
echo "Cache:       $CACHE"
echo "Predictions: $PRED"
