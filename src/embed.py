from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from image_io import iter_records, load_image


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CONCH embedding cache from flexible image inputs.")
    parser.add_argument("source", help="Image file, folder, glob, URL, CSV, JSON, or JSONL manifest.")
    parser.add_argument("--output", "-o", required=True, help="Output .pt cache path.")
    parser.add_argument("--image-key", default="image", help="Manifest column/key containing the image path.")
    parser.add_argument("--id-key", default=None, help="Optional manifest column/key to use as the record id.")
    parser.add_argument("--encoder", default="conch_ViT-B-16", help="CONCH encoder name.")
    parser.add_argument("--pretrained", default="hf_hub:MahmoodLab/conch", help="CONCH checkpoint reference.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max-pixels", type=int, default=16_000_000, help="Resize very large inputs before preprocessing.")
    parser.add_argument("--allow-empty", action="store_true", help="Write an empty cache instead of failing on no valid images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(iter_records(args.source, image_key=args.image_key, id_key=args.id_key))
    if not records and not args.allow_empty:
        raise SystemExit("No image records found.")

    from conch.open_clip_custom import create_model_from_pretrained

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    model, preprocess = create_model_from_pretrained(args.encoder, args.pretrained, hf_auth_token=hf_token)
    model = model.to(args.device).eval()

    embeddings: List[torch.Tensor] = []
    meta: Dict[str, List[str]] = {"id": [], "image_paths": []}
    errors: List[Dict[str, str]] = []

    with torch.inference_mode():
        for record in tqdm(records, desc="Embedding images"):
            try:
                image = load_image(record.resolved_image, max_pixels=args.max_pixels)
                image_tensor = preprocess(image).unsqueeze(0).to(args.device)
                emb = model.encode_image(image_tensor, proj_contrast=False, normalize=False)
            except Exception as exc:
                errors.append({"image": record.resolved_image, "error": str(exc)})
                continue

            embeddings.append(emb.squeeze(0).cpu())
            meta["id"].append(record.id or Path(record.resolved_image).stem)
            meta["image_paths"].append(record.resolved_image)
            for key, value in record.metadata.items():
                meta.setdefault(key, []).append("" if value is None else str(value))
            for key in set(meta) - set(record.metadata) - {"id", "image_paths"}:
                if len(meta[key]) < len(embeddings):
                    meta[key].append("")

    if not embeddings and not args.allow_empty:
        first_error = f" First error: {errors[0]}" if errors else ""
        raise SystemExit(f"No images could be embedded.{first_error}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stacked = torch.stack(embeddings) if embeddings else torch.empty((0, 0))
    torch.save(
        {
            "embeddings": stacked,
            "meta": meta,
            "errors": errors,
            "encoder": f"CONCH {args.encoder}",
            "source": str(args.source),
        },
        output,
    )
    print(f"Saved {len(embeddings)} embeddings to {output} with shape {tuple(stacked.shape)}.")
    if errors:
        print(f"Skipped {len(errors)} unreadable inputs. See cache['errors'] for details.")


if __name__ == "__main__":
    main()
