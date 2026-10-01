# OPathLM

### Basic usage

Before run:

```bash
cd Anna/CONCH
conda activate conch
export HF_TOKEN="hf_..."
```

Use `python ../src/train.py` to train new projector. Use `../src/run.sh` to run the full pipeline on custom source:

```bash
bash src/run.sh <source> [-o OUTPUT_DIR] [-p PROMPT_FILE] [-c PROJECTOR] [-l LLM] [-n NAME] [-- extra args passed to eval.py]
```

`<source>` can be an image file, folder, glob, URL, CSV, JSON, or JSONL manifest.
- `-o, --output-dir DIR`: directory for outputs. Default: `outputs/run_<timestamp>`
- `-n, --name NAME`: write predictions to `<output-dir>/<name>.json`
- `-p, --prompt-file FILE`: path to a prompt `.txt` file
- `-c, --projector FILE`: projector checkpoint path
- `-l, --llm NAME`: Hugging Face model id
- `--`: pass any remaining arguments directly to `eval.py`

### Environment overrides

* `CONDA_ENV=conch` :  Conda environment for step 2 (projector + LLM).
* `EMBED_ENV=titan` :  Conda environment for step 1 (TITAN embedding).
* `PROJECTOR=../checkpoints/proj_xxxxx.pt` :  Path to the projector checkpoint file.
* `LLM=Qwen/Qwen2.5-7B-Instruct` :  Large Language Model identifier or path.
* `ID_KEY=id` :  Key name used for unique IDs in the dataset.
* `IMAGE_KEY=image` :  Key name used for images in the dataset.
* `RUN_4BIT=1` :  Flag to enable 4-bit quantization model loading.
* `MAX_NEW_TOKENS=200` :  Maximum number of new tokens to generate during inference.
* `TEMPERATURE=0.2` :  Sampling temperature for generation randomness.
* `TOP_P=0.9` :  Top-p (nucleus) sampling threshold.
* `SAMPLE=1` :  Flag to enable sampling mode.
* `DEVICE=cuda` :  Hardware device to run computation on (e.g., `cuda`, `cpu`).
* `KEEP_CACHE=1` :  Keep the embedding cache after the run (useful for debugging).
* `CACHE=/path/to/cache.pt` :  Path to an existing cache file to skip the embedding step entirely.
* `TITAN_FEAT_DIR=DIR` :  Directory for per-slide patch features, reused across runs (Default: `<output-dir>/titan_feats`).
* `EMBED_ARGS="..."` :  Extra arguments passed to `embed-s.py` (e.g., `--batch-size 128 --min-tissue 0.1`).
* `SKIP_CONDA=1` :  Do not activate conda environments; use the currently active Python interpreter.

### Examples

Keep the embedding cache for debugging:

```bash
KEEP_CACHE=1 bash src/run.sh manifest.jsonl
```

Reuse an existing cache and skip embedding:

```bash
CACHE=outputs/prev/cache.pt bash src/run.sh manifest.jsonl
```
