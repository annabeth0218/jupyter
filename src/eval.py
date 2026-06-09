from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

from projector import DEFAULT_PROMPT, format_reference_caption, generate_from_embedding, load_projector


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pathology interpretations from a CONCH embedding cache. "
            "Use run.sh to drive the full embed → generate pipeline."
        )
    )
    parser.add_argument("--cache",          required=True,  help="Embedding cache .pt produced by embed.py.")
    parser.add_argument("--projector",      required=True,  help="Projector checkpoint .pt path.")
    parser.add_argument("--output", "-o",   required=True,  help="Output predictions JSON path.")
    parser.add_argument("--llm",            default="Qwen/Qwen2-7B-Instruct", help="HF model id for the LLM.")
    parser.add_argument("--prompt-file",    default=None,   help="Path to a .txt file whose contents are used as the generation prompt.")
    parser.add_argument("--prompt",         default=None,   help="Prompt string (overridden by --prompt-file if both given).")
    parser.add_argument("--max-new-tokens", type=int,   default=128)
    parser.add_argument("--temperature",    type=float, default=0.2)
    parser.add_argument("--top-p",          type=float, default=0.9)
    parser.add_argument("--sample",         action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--load-4bit",      action="store_true", help="Load the LLM with bitsandbytes 4-bit quantisation.")
    parser.add_argument("--device",         default=default_device())
    return parser.parse_args()

def _shift_digits(text: str, shift: int = 2) -> str:
    return "".join(str((int(c) + shift) % 10) if c.isdigit() else c for c in text)

# ---------------------------------------------------------------------------
# Prompt resolution
# ---------------------------------------------------------------------------

def resolve_prompt(args: argparse.Namespace) -> str:
    """--prompt-file beats --prompt beats DEFAULT_PROMPT."""
    if args.prompt_file:
        path = Path(args.prompt_file)
        if not path.exists():
            raise SystemExit(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8")
        print(f"Using prompt file: {path}  ({len(text)} chars)", flush=True)
        return text
    if args.prompt:
        return args.prompt
    print("Using DEFAULT_PROMPT from projector.py", flush=True)
    return DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    prompt = resolve_prompt(args)

    # --- load cache ---
    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise SystemExit(f"Cache not found: {cache_path}")

    print(f"Loading embedding cache: {cache_path}", flush=True)
    cache = torch.load(cache_path, map_location="cpu")
    image_embeddings: torch.Tensor = cache["embeddings"].float()
    meta: Dict[str, List[Any]] = cache.get("meta", {})

    if image_embeddings.numel() == 0:
        raise SystemExit(f"No embeddings found in cache: {cache_path}")
    print(f"Loaded {image_embeddings.shape[0]} embeddings (dim {image_embeddings.shape[1]}).", flush=True)

    # --- tokenizer ---
    print(f"Loading tokenizer: {args.llm}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- LLM ---
    if args.device == "cpu":
        print(
            "Warning: running on CPU — loading a 7B model will be very slow.",
            flush=True,
        )
    model_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if args.load_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    print(f"Loading LLM: {args.llm}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.llm, **model_kwargs)
    device = next(model.parameters()).device
    print(f"LLM on device: {device}", flush=True)

    # --- projector ---
    print(f"Loading projector: {args.projector}", flush=True)
    projector = load_projector(args.projector, device)
    print("Projector loaded.", flush=True)

    # --- generate ---
    results: List[Dict[str, Any]] = []
    for idx in tqdm(range(image_embeddings.shape[0]), desc="Generating"):
        caption = generate_from_embedding(
            image_embedding=image_embeddings[idx],
            projector=projector,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.sample,
        )
        record: Dict[str, Any] = {
            "idx":    idx,
            "id":     _shift_digits(_meta_at(meta, "id", idx),2),
            "image":  Path(_meta_at(meta, "image_paths", idx)).name,
            "model":  args.projector,
            "caption": f"Microscopic findings: {caption}",
        }
        reference = format_reference_caption(meta, idx)
        if reference and reference != "Pathologic diagnosis:.\nMicroscopic findings:":
            record["reference"] = reference
        results.append(record)

    # --- save ---
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(results)} predictions to {output}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta_at(meta: Dict[str, List[Any]], key: str, idx: int) -> str:
    values = meta.get(key, [])
    if idx >= len(values):
        return ""
    value = values[idx]
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()
