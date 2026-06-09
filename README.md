# OPathLM

### Basic usage

Before run:

```bash
cd Anna_CONCH/CONCH
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

### Environment variables

These can be used to override defaults:

- `CONDA_ENV=conch`
- `PROJECTOR=checkpoints/proj-arvo-llama.pt`
- `LLM=Qwen/Qwen2-7B-Instruct`
- `ID_KEY=id`
- `IMAGE_KEY=image`
- `RUN_4BIT=1`
- `MAX_NEW_TOKENS=200`
- `TEMPERATURE=0.2`
- `TOP_P=0.9`
- `SAMPLE=1`
- `DEVICE=cuda`
- `KEEP_CACHE=1`
- `CACHE=/path/to/cache.pt`

### Examples

Keep the embedding cache for debugging:

```bash
KEEP_CACHE=1 bash src/run.sh manifest.jsonl
```

Reuse an existing cache and skip embedding:

```bash
CACHE=outputs/prev/cache.pt bash src/run.sh manifest.jsonl
```
