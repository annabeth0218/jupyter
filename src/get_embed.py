"""
get_embed.py

Embed pathology images with the CONCH ViT-B/16 encoder and save a cache
that is consumed directly by `projector_train_0502.py`.

Inputs
------
One or more .jsonl manifests, OR one or more directories that contain
.jsonl files (the directory may contain other unrelated files; only the
.jsonl files within are picked up).  Each manifest line should contain
at minimum:
    - "image"    : path to the image file (relative to --image-root or
                   absolute / resolvable as-is)
    - "caption"  : training caption (already cleaned by caption_io.py)
Optionally:
    - "disease"  : diagnostic label string

Anything else in each row is ignored; the cleaning step is the
responsibility of `caption_io.py`.

Output
------
A torch .pt file with this structure:

    {
        "embeddings":  FloatTensor [N, D],
        "meta": {
            "image_paths": [str, ...],
            "captions":    [str, ...],
            "disease":     [str, ...],     # "" when missing
        },
        "encoder": "CONCH ViT-B/16",
        "sources": [str, ...],             # input manifest paths
    }

Usage
-----
    # Single manifest
    python get_embed.py manifest_who4e.clean.jsonl -o cache.pt

    # Multiple manifests
    python get_embed.py m1.jsonl m2.jsonl m3.jsonl -o combined_cache.pt \\
        --image-root /data/who4e

    # A folder of manifests (other files in the folder are ignored)
    python get_embed.py /path/to/manifests_dir -o cache.pt
    python ../../train/get_embed.py data -o cache.pt

    # Mix manifests and folders
    python get_embed.py extra.jsonl /path/to/manifests_dir -o cache.pt

    # Recurse into subdirectories of a folder
    python get_embed.py /path/to/manifests_dir --recursive -o cache.pt

Set HF_TOKEN in the environment (or pass --hf-token) so the CONCH weights
can be downloaded from MahmoodLab/conch on first use.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Input expansion (files + folders)
# ---------------------------------------------------------------------------

def _iter_jsonl_in_dir(directory: Path, recursive: bool) -> List[Path]:
    """Return sorted .jsonl files inside `directory`. Other files are ignored."""
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def expand_manifest_inputs(inputs: Iterable[str], recursive: bool) -> List[Path]:
    """
    Expand a mixed list of CLI inputs into a flat list of .jsonl manifest paths.

    Each input may be:
      * a path to a .jsonl file -> kept as-is
      * a directory             -> all .jsonl files inside are collected
                                   (non-.jsonl files in the directory are ignored)

    Anything else (missing path, non-.jsonl file) raises SystemExit.
    Order is preserved, duplicates are removed.
    """
    resolved: List[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        p = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
        if not p.exists():
            raise SystemExit(f"Input not found: {raw}")

        if p.is_dir():
            found = _iter_jsonl_in_dir(p, recursive=recursive)
            if not found:
                where = "recursively " if recursive else ""
                raise SystemExit(f"No .jsonl files found {where}in directory: {p}")
            for f in found:
                if f not in seen:
                    seen.add(f)
                    resolved.append(f)
        elif p.is_file():
            if p.suffix.lower() != ".jsonl":
                raise SystemExit(
                    f"Expected a .jsonl file or a directory, got: {p}"
                )
            if p not in seen:
                seen.add(p)
                resolved.append(p)
        else:
            raise SystemExit(f"Unsupported input (not a file or directory): {p}")

    return resolved


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def _resolve_image(image: str, manifest_dir: Path, image_root: Path | None) -> Path:
    """
    Resolve a manifest's "image" string against a root directory.

    Resolution order:
      1. If the path is absolute, use it.
      2. If --image-root is given, try image_root / image.
      3. Try manifest_dir / image (the directory holding the .jsonl).
      4. Fall back to the literal string.
    """
    p = Path(os.path.expandvars(os.path.expanduser(image)))
    if p.is_absolute():
        return p
    if image_root is not None:
        candidate = (image_root / p).resolve()
        if candidate.exists():
            return candidate
    candidate = (manifest_dir / p).resolve()
    if candidate.exists():
        return candidate
    if image_root is not None:
        return (image_root / p).resolve()
    return candidate


def collect_records(
    manifest_paths: Iterable[Path],
    image_root: Path | None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[Path, str]]]:
    """
    Load every record from every manifest and validate that the image file
    exists and can be opened.  Returns (good_records, bad_records).
    """
    good: List[Dict[str, Any]] = []
    bad: List[Tuple[Path, str]] = []

    for mpath in manifest_paths:
        manifest_dir = mpath.parent.resolve()
        for rec in _iter_jsonl(mpath):
            image_field = rec.get("image", "")
            if not image_field:
                bad.append((mpath, f"missing 'image' field in {rec}"))
                continue
            if "caption" not in rec:
                bad.append((mpath, f"missing 'caption' field for image {image_field}"))
                continue

            resolved = _resolve_image(image_field, manifest_dir, image_root)
            if not resolved.exists():
                bad.append((mpath, f"image file not found: {resolved}"))
                continue

            try:
                with Image.open(resolved) as im:
                    im.verify()
            except Exception as exc:  # PIL throws many flavors here
                bad.append((mpath, f"image not readable: {resolved} ({exc})"))
                continue

            good.append(
                {
                    "image_path": str(resolved),
                    "caption": str(rec.get("caption", "")),
                    "disease": str(rec.get("disease", "") or ""),
                    "source_manifest": str(mpath),
                }
            )

    return good, bad


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_cache(
    records: List[Dict[str, Any]],
    *,
    encoder: str = "conch_ViT-B-16",
    pretrained: str = "hf_hub:MahmoodLab/conch",
    hf_token: str | None = None,
    device: str | None = None,
    sources: List[str] | None = None,
) -> Dict[str, Any]:
    """Run CONCH on every record and return a cache dict."""
    device = device or default_device()

    # Local import so the file can be inspected without CONCH installed.
    from conch.open_clip_custom import create_model_from_pretrained

    print(f"Loading CONCH encoder: {encoder}", flush=True)
    model, preprocess = create_model_from_pretrained(
        encoder, pretrained, hf_auth_token=hf_token
    )
    model = model.to(device).eval()

    embeddings: List[torch.Tensor] = []
    meta: Dict[str, List[str]] = {"image_paths": [], "captions": [], "disease": []}

    with torch.inference_mode():
        for rec in tqdm(records, desc="CONCH embedding"):
            img = Image.open(rec["image_path"]).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            emb = model.encode_image(tensor, proj_contrast=False, normalize=False)
            embeddings.append(emb.squeeze(0).cpu())
            meta["image_paths"].append(rec["image_path"])
            meta["captions"].append(rec["caption"])
            meta["disease"].append(rec["disease"])

    stacked = torch.stack(embeddings) if embeddings else torch.empty((0, 0))
    return {
        "embeddings": stacked,
        "meta": meta,
        "encoder": "CONCH ViT-B/16",
        "sources": sources or [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a CONCH embedding cache from .jsonl manifests. "
            "Inputs may be individual .jsonl files, directories containing "
            ".jsonl files (other files in the directory are ignored), or any "
            "mix of the two."
        )
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more .jsonl manifests and/or directories containing "
            ".jsonl manifests."
        ),
    )
    p.add_argument("-o", "--output", required=True, help="Output cache .pt path.")
    p.add_argument(
        "--image-root",
        default=None,
        help="Optional root directory to resolve relative image paths against.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "When an input is a directory, also search subdirectories for "
            ".jsonl files. Default is non-recursive (top level only)."
        ),
    )
    p.add_argument("--encoder", default="conch_ViT-B-16")
    p.add_argument("--pretrained", default="hf_hub:MahmoodLab/conch")
    p.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. Defaults to HF_TOKEN env var.",
    )
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto by default).")
    p.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty cache instead of failing when no images are valid.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    manifest_paths = expand_manifest_inputs(args.inputs, recursive=args.recursive)
    print(f"Found {len(manifest_paths)} manifest file(s):", flush=True)
    for mp in manifest_paths:
        print(f"  - {mp}", flush=True)

    image_root = Path(args.image_root).resolve() if args.image_root else None

    print(f"Reading {len(manifest_paths)} manifest(s)...", flush=True)
    good, bad = collect_records(manifest_paths, image_root)
    print(f"  valid records: {len(good)}", flush=True)
    if bad:
        print(f"  skipped: {len(bad)} (first 5 reasons below)", flush=True)
        for mp, reason in bad[:5]:
            print(f"    [{mp.name}] {reason}", flush=True)

    if not good and not args.allow_empty:
        raise SystemExit("No valid image records to embed.")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    cache = build_cache(
        good,
        encoder=args.encoder,
        pretrained=args.pretrained,
        hf_token=hf_token,
        device=args.device,
        sources=[str(p) for p in manifest_paths],
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    shape = tuple(cache["embeddings"].shape)
    print(f"Saved cache to: {out} | embeddings shape: {shape}")


if __name__ == "__main__":
    main()
