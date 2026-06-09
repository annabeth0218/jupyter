# OPathLM

## Using `run.sh`

The repository includes `src/run.sh` to run the full pipeline: embed pathology images and then generate interpretations.

### Basic usage

```bash
bash src/run.sh <source>
```

`<source>` can be an image file, folder, glob, URL, CSV, JSON, or JSONL manifest.

### Common options

```bash
bash src/run.sh <source> [-o OUTPUT_DIR] [-p PROMPT_FILE] [-c PROJECTOR] [-l LLM] [-n NAME] [-- extra args passed to eval.py]
```

- `-o, --output-dir DIR`: directory for outputs. Default: `outputs/run_<timestamp>`
- `-n, --name NAME`: write predictions to `<output-dir>/<name>.json`
- `-p, --prompt-file FILE`: path to a prompt `.txt` file
- `-c, --projector FILE`: projector checkpoint path
- `-l, --llm NAME`: Hugging Face model id
- `--`: pass any remaining arguments directly to `eval.py`

### Requirements

Before running the script:

1. Make sure `conda` is installed and available in your `PATH`
2. Set a Hugging Face token:

```bash
export HF_TOKEN="hf_..."
```

3. Ensure the projector checkpoint exists. By default the script uses:

```bash
checkpoints/proj-arvo-llama.pt
```

4. The default conda environment is `conch`. You can override it with:

```bash
export CONDA_ENV=your_env_name
```

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

Run with a manifest file:

```bash
bash src/run.sh manifest.jsonl -c checkpoints/proj.pt -p prompts/few_shot.txt -o outputs/s109
```

Write predictions to a custom file name:

```bash
bash src/run.sh manifest.jsonl -n s109_run1
```

Keep the embedding cache for debugging:

```bash
KEEP_CACHE=1 bash src/run.sh manifest.jsonl
```

Reuse an existing cache and skip embedding:

```bash
CACHE=outputs/prev/cache.pt bash src/run.sh manifest.jsonl
```
