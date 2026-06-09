from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from projector import DEFAULT_PROMPT, format_reference_caption, generate_from_embedding, load_projector


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pathology interpretations from a cache or flexible image inputs.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--cache", help="Existing embedding cache produced by embed_images.py.")
    input_group.add_argument("--source", help="Raw image input; will be embedded first using embed_images.py.")
    parser.add_argument("--projector", required=True, help="Projector checkpoint path.")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path.")
    parser.add_argument("--llm", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of deterministic decoding.")
    parser.add_argument("--load-4bit", action="store_true", help="Load the LLM with bitsandbytes 4-bit quantization.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--image-key", default="image", help="Manifest column/key for --source mode.")
    parser.add_argument("--id-key", default=None, help="Manifest id column/key for --source mode.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for --source mode. Defaults to HF_TOKEN env var.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.cache or _build_temporary_cache(args)

    print(f"Loading embedding cache: {cache_path}", flush=True)
    cache = torch.load(cache_path, map_location="cpu")
    image_embeddings = cache["embeddings"].float()
    meta = cache.get("meta", {})
    if image_embeddings.numel() == 0:
        raise SystemExit(f"No embeddings found in cache: {cache_path}")
    print(f"Loaded {image_embeddings.shape[0]} embeddings with dimension {image_embeddings.shape[1]}.", flush=True)

    print(f"Loading tokenizer: {args.llm}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.llm, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Preparing LLM load: {args.llm}", flush=True)
    if args.device == "cpu":
        print(
            "No GPU/MPS device was selected. Loading Qwen2-7B on CPU can take several minutes and inference will be slow.",
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
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("Loading LLM weights. The first run may download several GB from Hugging Face.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.llm, **model_kwargs)
    device = next(model.parameters()).device
    print(f"LLM loaded on device: {device}", flush=True)
    print(f"Loading projector: {args.projector}", flush=True)
    projector = load_projector(args.projector, device)
    print("Projector loaded.", flush=True)

    results: List[Dict[str, Any]] = []
    for idx in tqdm(range(image_embeddings.shape[0]), desc="Generating"):
        caption = generate_from_embedding(
            image_embedding=image_embeddings[idx],
            projector=projector,
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.sample,
        )
        record = {
            "idx": idx,
            "id": _meta_at(meta, "id", idx),
            "image": _meta_at(meta, "image_paths", idx),
            "model": args.projector,
            "caption": caption,
        }
        reference = format_reference_caption(meta, idx)
        if reference and reference != "Pathologic diagnosis:.\nMicroscopic findings:":
            record["reference"] = reference
        results.append(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(results)} interpretations to {output}.")


def _build_temporary_cache(args: argparse.Namespace) -> str:
    from embed_images import main as embed_main
    import sys

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    old_argv = sys.argv
    argv = [
        "embed_images.py",
        args.source,
        "--output",
        tmp.name,
        "--image-key",
        args.image_key,
    ]
    if args.id_key:
        argv.extend(["--id-key", args.id_key])
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        argv.extend(["--hf-token", hf_token])
    sys.argv = argv
    try:
        embed_main()
    finally:
        sys.argv = old_argv
    return tmp.name


def _meta_at(meta: Dict[str, List[Any]], key: str, idx: int) -> str:
    values = meta.get(key, [])
    if idx >= len(values):
        return ""
    value = values[idx]
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()
