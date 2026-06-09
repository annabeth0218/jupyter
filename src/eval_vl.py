"""
eval_vl.py

Generate pathology image captions using a Vision-Language model and save
a predictions.json compatible with scoring.py.

Supported models (pass via --model):
    Qwen/Qwen2.5-VL-7B-Instruct          (default)
    Qwen/Qwen2-VL-7B-Instruct
    llava-hf/llava-1.5-7b-hf
    llava-hf/llava-v1.6-mistral-7b-hf    (LLaVA-NeXT)
    microsoft/Phi-3.5-vision-instruct
    openbmb/MiniCPM-V-2_6
    HuggingFaceM4/Idefics3-8B-Llama3

The script auto-detects which loader / inference path to use based on the
model family, so you only need to change --model.

Input JSONL format (one record per line):
    {"image": "path/to/img.jpg", "caption": "...", "disease": "..."}
    or the richer format produced by caption_io.py.

Output predictions.json format (identical to eval.py):
    [
      {
        "idx":       0,
        "id":        "...",
        "image":     "img.jpg",
        "model":     "Qwen/Qwen2.5-VL-7B-Instruct",
        "caption":   "<model output>",
        "reference": "<ground truth from JSONL>"
      },
      ...
    ]

Usage:
    python eval_vl.py \\
        --input  manifest.jsonl \\
        --output outputs/vl_predictions.json \\
        --model  Qwen/Qwen2.5-VL-7B-Instruct \\
        --prompt "Describe the microscopic pathological findings in this image."

    # 4-bit quantised (saves ~12 GB VRAM):
    python eval_vl.py --input manifest.jsonl --output out.json --load-4bit

    # Specific image root if paths in JSONL are relative:
    python eval_vl.py --input manifest.jsonl --output out.json \\
        --image-root /data/who4e

Requirements:
    pip install transformers>=4.45 accelerate pillow tqdm
    pip install bitsandbytes          # for --load-4bit
    pip install qwen-vl-utils         # for Qwen2.5-VL / Qwen2-VL only
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Default prompt
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = (
    "You are an expert pathologist. "
    "Examine this pathology image and describe the microscopic findings in detail. "
    "Include the key morphological features, cell types, tissue architecture, "
    "and any pathological changes observed. "
    "Then state the pathological diagnosis."
)


# ---------------------------------------------------------------------------
# JSONL manifest reader
# ---------------------------------------------------------------------------

def _resolve_image_path(
    image_field: str,
    manifest_dir: Path,
    image_root: Optional[Path],
) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(image_field)))
    if p.is_absolute():
        return p
    if image_root is not None:
        candidate = (image_root / p).resolve()
        if candidate.exists():
            return candidate
    candidate = (manifest_dir / p).resolve()
    if candidate.exists():
        return candidate
    # Return best-effort path even if it doesn't exist yet (error raised later)
    return candidate


def iter_jsonl(
    path: Path,
    image_root: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield enriched records from a JSONL manifest."""
    manifest_dir = path.parent.resolve()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

            if "image" not in rec or not rec["image"]:
                raise ValueError(
                    f"Record at line {line_no} is missing required 'image' field: {rec}"
                )

            resolved = _resolve_image_path(rec["image"], manifest_dir, image_root)
            rec["_resolved_image"] = resolved
            rec["_line_no"] = line_no
            yield rec


def build_reference(rec: Dict[str, Any]) -> str:
    """
    Reproduce the same reference format as projector.py:format_reference_caption,
    so scores are comparable against the CONCH-based pipeline.
    """
    disease = (rec.get("disease") or "").strip()
    caption = (rec.get("caption") or rec.get("captions") or "").strip()
    title   = (rec.get("title") or "").strip()
    subcls  = (rec.get("subcls") or "").strip()
    cls     = (rec.get("cls") or "").strip()

    def low_first(t: str) -> str:
        return t[:1].lower() + t[1:] if t else ""

    diagnosis = f"Pathologic diagnosis: {disease}" if disease else "Pathologic diagnosis:"
    if subcls or cls:
        if subcls and cls:
            diagnosis += (
                f", classified as {low_first(subcls)} "
                f"within the broader category of {low_first(cls)}"
            )
        elif subcls:
            diagnosis += f", classified as {low_first(subcls)}"
        else:
            diagnosis += f", classified as {low_first(cls)}"

    finding_prefix = (
        f"{title} is shown. "
        if title and disease.lower() not in title.lower()
        else ""
    )
    ref = f"{diagnosis}.\nMicroscopic findings: {finding_prefix}{caption}".strip()
    # Return None-equivalent if there's genuinely nothing
    if ref == "Pathologic diagnosis:.\nMicroscopic findings:":
        return ""
    return ref


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def detect_family(model_id: str) -> str:
    mid = model_id.lower()
    if "qwen2.5-vl" in mid or "qwen2_5_vl" in mid:
        return "qwen2_5_vl"
    if "qwen2-vl" in mid or "qwen2_vl" in mid:
        return "qwen2_vl"
    if "llava-1.5" in mid or "llava1.5" in mid:
        return "llava15"
    if "llava" in mid:
        return "llava"          # LLaVA-NeXT / LLaVA-1.6
    if "phi-3" in mid or "phi3" in mid:
        return "phi3_vision"
    if "minicpm" in mid:
        return "minicpm"
    if "idefics" in mid:
        return "idefics"
    return "generic"            # fallback: try AutoModelForVision2Seq


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_processor(
    model_id: str,
    family: str,
    load_4bit: bool = False,
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """
    Returns (model, processor).
    `processor` may be an AutoProcessor or AutoTokenizer depending on the family.
    """
    quant_kwargs: Dict[str, Any] = (
        {"quantization_config": _bnb_config()} if load_4bit else {}
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model  : {model_id}")
    print(f"Family         : {family}")
    print(f"4-bit quant    : {load_4bit}")

    # ---- Qwen2.5-VL --------------------------------------------------------
    if family == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            **quant_kwargs,
        )
        return model, processor

    # ---- Qwen2-VL ----------------------------------------------------------
    if family == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            **quant_kwargs,
        )
        return model, processor

    # ---- LLaVA-1.5 ---------------------------------------------------------
    if family == "llava15":
        from transformers import LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            **quant_kwargs,
        )
        return model, processor

    # ---- LLaVA-NeXT (1.6) --------------------------------------------------
    if family == "llava":
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            **quant_kwargs,
        )
        return model, processor

    # ---- Phi-3.5 Vision ----------------------------------------------------
    if family == "phi3_vision":
        from transformers import AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, num_crops=4
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            **quant_kwargs,
        )
        return model, processor

    # ---- MiniCPM-V ---------------------------------------------------------
    if family == "minicpm":
        from transformers import AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            **quant_kwargs,
        )
        model.eval()
        return model, tokenizer

    # ---- Idefics3 ----------------------------------------------------------
    if family == "idefics":
        from transformers import AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            **quant_kwargs,
        )
        return model, processor

    # ---- Generic fallback --------------------------------------------------
    from transformers import AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        **quant_kwargs,
    )
    return model, processor


# ---------------------------------------------------------------------------
# Inference — one image at a time
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_caption(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    family: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    do_sample: bool = False,
) -> str:
    """Dispatch to the right inference path for each model family."""

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    # ---- Qwen2.5-VL and Qwen2-VL ------------------------------------------
    if family in ("qwen2_5_vl", "qwen2_vl"):
        from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, **gen_kwargs)
        # Strip the prompt tokens from the output
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    # ---- LLaVA-1.5 ---------------------------------------------------------
    if family == "llava15":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=text_prompt, images=image, return_tensors="pt"
        ).to(model.device)
        output = model.generate(**inputs, **gen_kwargs)
        full = processor.decode(output[0], skip_special_tokens=True)
        # LLaVA-1.5 echoes the prompt; strip it
        if "ASSISTANT:" in full:
            return full.split("ASSISTANT:")[-1].strip()
        return full.strip()

    # ---- LLaVA-NeXT --------------------------------------------------------
    if family == "llava":
        from transformers import LlavaNextProcessor
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=text_prompt, images=image, return_tensors="pt"
        ).to(model.device)
        output = model.generate(**inputs, **gen_kwargs)
        full = processor.decode(output[0], skip_special_tokens=True)
        if "[/INST]" in full:
            return full.split("[/INST]")[-1].strip()
        return full.strip()

    # ---- Phi-3.5 Vision ----------------------------------------------------
    if family == "phi3_vision":
        # Phi-3.5 uses <|image_1|> placeholder syntax
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]
        text_prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text_prompt, images=[image], return_tensors="pt").to(
            model.device
        )
        output = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **gen_kwargs)
        # Decode only new tokens
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

    # ---- MiniCPM-V ---------------------------------------------------------
    if family == "minicpm":
        # MiniCPM-V has its own .chat() interface
        msgs = [{"role": "user", "content": [image, prompt]}]
        result = model.chat(image=None, msgs=msgs, tokenizer=processor)
        return result.strip()

    # ---- Idefics3 / generic ------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=text_prompt, images=[image], return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, **gen_kwargs)
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    family = detect_family(args.model)
    image_root = Path(args.image_root).resolve() if args.image_root else None

    # Resolve prompt
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise SystemExit(f"Prompt file not found: {prompt_path}")
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"Using prompt file: {prompt_path}")
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT
    print(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}\n")

    # Load model
    model, processor = load_model_and_processor(
        args.model, family, load_4bit=args.load_4bit
    )
    model.eval()
    print("Model loaded.\n")

    # Read manifest
    manifest_path = Path(args.input)
    if not manifest_path.exists():
        raise SystemExit(f"Input JSONL not found: {manifest_path}")

    records = list(iter_jsonl(manifest_path, image_root=image_root))
    print(f"Found {len(records)} records in {manifest_path.name}\n")

    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, rec in enumerate(tqdm(records, desc="Generating")):
        image_path: Path = rec["_resolved_image"]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            msg = f"idx={idx}: cannot open image {image_path}: {exc}"
            print(f"\n  ⚠  {msg}")
            errors.append(msg)
            continue

        # Generate
        try:
            caption = generate_caption(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                family=family,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.sample,
            )
        except Exception as exc:
            msg = f"idx={idx}: generation error for {image_path.name}: {exc}"
            print(f"\n  ⚠  {msg}")
            errors.append(msg)
            caption = ""

        reference = build_reference(rec)

        result: Dict[str, Any] = {
            "idx":   idx,
            "id":    str(rec.get("id") or rec.get("number") or ""),
            "image": image_path.name,
            "model": args.model,
            "caption": caption,
        }
        # Only include reference if there's meaningful ground truth
        # (mirrors the behaviour in eval.py so scoring.py skips empty ones)
        if reference:
            result["reference"] = reference

        results.append(result)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSaved {len(results)} predictions → {output_path}")
    if errors:
        print(f"Errors / skipped: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Quick preview
    if results:
        print("\n--- Sample output (idx=0) ---")
        r = results[0]
        print(f"Image    : {r['image']}")
        print(f"Caption  : {r['caption'][:200]}{'...' if len(r['caption']) > 200 else ''}")
        if "reference" in r:
            print(f"Reference: {r['reference'][:200]}{'...' if len(r['reference']) > 200 else ''}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a Vision-Language model on a JSONL manifest and produce "
            "a predictions.json compatible with scoring.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Qwen2.5-VL (default)
  python eval_vl.py -i manifest.jsonl -o outputs/vl_preds.json

  # Different model, 4-bit, custom prompt
  python eval_vl.py -i manifest.jsonl -o outputs/llava_preds.json \\
      --model llava-hf/llava-v1.6-mistral-7b-hf --load-4bit

  # With image root and prompt file
  python eval_vl.py -i manifest.jsonl -o outputs/preds.json \\
      --image-root /data/who4e --prompt-file prompts/pathology.txt

  # Then score exactly like the CONCH pipeline:
  python scoring.py --input outputs/vl_preds.json --output outputs/vl_scores.json
""",
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Input JSONL manifest (same format as used by embed.py).")
    p.add_argument("--output", "-o", required=True,
                   help="Output predictions JSON path.")
    p.add_argument("--model",  "-m",
                   default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HuggingFace model ID. Default: Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--prompt-file",
                   default=None,
                   help="Path to a .txt file to use as the generation prompt.")
    p.add_argument("--prompt",
                   default=None,
                   help="Prompt string. Overridden by --prompt-file if both given.")
    p.add_argument("--image-root",
                   default=None,
                   help="Root directory to resolve relative image paths in the JSONL.")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Maximum tokens to generate per image. Default: 256")
    p.add_argument("--temperature",    type=float, default=0.2,
                   help="Sampling temperature (only used with --sample). Default: 0.2")
    p.add_argument("--sample",         action="store_true",
                   help="Use sampling instead of greedy decoding.")
    p.add_argument("--load-4bit",      action="store_true",
                   help="Load model in 4-bit (bitsandbytes NF4). Saves ~12 GB VRAM.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
