"""
embed.py  (TITAN version)

Embed whole-slide images (.ndpi, .svs, ...) with MahmoodLab TITAN and save a
cache in exactly the same format the CONCH pipeline used, so train.py,
eval.py and run.sh work unchanged.

Per slide:
  1. tissue detection + non-overlapping 512 px grid at 20x      (wsi_tiling.py)
  2. CONCH v1.5 patch features, 768-d                          (titan.return_conch())
  3. TITAN slide embedding, 768-d                              (titan.encode_slide_from_patch_features)

Inputs
------
Any mix of:
  * .jsonl manifests (or folders containing .jsonl files), one record per line:
        {"image": "slides/A123.ndpi", "caption": "...", "disease": "...", "id": "A123"}
    "image" may also be a list of slides (or a ";"-separated string) for a case
    with several slides; their TITAN embeddings are averaged into one vector.
    "caption" is required for training caches and optional for inference.
  * .json / .csv manifests with the same fields
  * a slide file, a folder of slides, or a glob ("data/*.ndpi")

Output (.pt)
------------
    {
        "embeddings":  FloatTensor [N, 768],
        "meta": {
            "image_paths": [str, ...],     # ";"-joined for multi-slide cases
            "captions":    [str, ...],
            "disease":     [str, ...],
            "id":          [str, ...],     # only when any record has an id
            "n_patches":   [int, ...],
        },
        "encoder": "TITAN (CONCH v1.5 patches, 512px @ 20x)",
        "sources": [str, ...],
        "titan":   {...tiling / model config...},
    }

Per-slide patch features, coordinates and the slide embedding are also
written to --feat-dir/<slide>_<hash>.h5 (TRIDENT/CLAM compatible layout), so
re-running on the same slides is nearly free and the patch features can be
reused later (e.g. attention pooling / stage 2).

Usage
-----
    python embed.py manifest.jsonl -o opath/cache_titan.pt
    python embed.py /data/ndpi_dir -o cache.pt              # inference, no captions
    python embed.py slide.ndpi -o cache.pt --device cuda:1

Set HF_TOKEN (or pass --hf-token). Access to https://huggingface.co/MahmoodLab/TITAN
must be requested and granted first.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsi_tiling import SLIDE_EXTENSIONS, PatchGrid, PatchReader, build_patch_grid  # noqa: E402

ENCODER_NAME = "TITAN (CONCH v1.5 patches, 512px @ 20x)"
FEAT_VERSION = 1


# ---------------------------------------------------------------------------
# Input expansion
# ---------------------------------------------------------------------------

MANIFEST_SUFFIXES = {".jsonl", ".json", ".csv"}


def _is_slide(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SLIDE_EXTENSIONS


def expand_inputs(inputs: Iterable[str], recursive: bool) -> Tuple[List[Path], List[Path]]:
    """Split CLI inputs into (manifest files, bare slide files)."""
    manifests: List[Path] = []
    slides: List[Path] = []
    seen: set = set()

    def add(lst: List[Path], p: Path) -> None:
        if p not in seen:
            seen.add(p)
            lst.append(p)

    for raw in inputs:
        expanded = os.path.expandvars(os.path.expanduser(raw))
        if any(ch in expanded for ch in "*?[") and not Path(expanded).exists():
            matches = sorted(glob.glob(expanded, recursive=True))
            if not matches:
                raise SystemExit(f"Glob matched nothing: {raw}")
            m2, s2 = expand_inputs(matches, recursive)
            for p in m2:
                add(manifests, p)
            for p in s2:
                add(slides, p)
            continue

        p = Path(expanded).resolve()
        if not p.exists():
            raise SystemExit(f"Input not found: {raw}")
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            found_manifests = sorted(q for q in p.glob(pattern) if q.is_file() and q.suffix.lower() == ".jsonl")
            if found_manifests:
                for q in found_manifests:
                    add(manifests, q.resolve())
            else:
                found_slides = sorted(q for q in p.glob(pattern) if _is_slide(q))
                if not found_slides:
                    raise SystemExit(f"No .jsonl manifests or slide files found in directory: {p}")
                for q in found_slides:
                    add(slides, q.resolve())
        elif p.suffix.lower() in MANIFEST_SUFFIXES:
            add(manifests, p)
        elif _is_slide(p):
            add(slides, p)
        else:
            raise SystemExit(f"Unsupported input (expected manifest, slide, folder or glob): {p}")
    return manifests, slides


def _iter_rows(path: Path) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("records") or data.get("images") or data.get("data") or [data]
        yield from data
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)


def _resolve(image: str, manifest_dir: Path, image_root: Optional[Path]) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(image.strip())))
    if p.is_absolute():
        return p
    if image_root is not None:
        c = (image_root / p).resolve()
        if c.exists():
            return c
    c = (manifest_dir / p).resolve()
    if c.exists() or image_root is None:
        return c
    return (image_root / p).resolve()


def _split_images(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [s for s in str(value).split(";") if s.strip()]


def collect_records(
    manifests: List[Path],
    slides: List[Path],
    image_root: Optional[Path],
    image_key: str = "image",
    id_key: str = "id",
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    good: List[Dict[str, Any]] = []
    bad: List[Tuple[str, str]] = []

    for mpath in manifests:
        mdir = mpath.parent.resolve()
        for rec in _iter_rows(mpath):
            images = _split_images(rec.get(image_key, ""))
            if not images:
                bad.append((mpath.name, f"missing '{image_key}' field in {rec}"))
                continue
            paths = [_resolve(im, mdir, image_root) for im in images]
            missing = [str(p) for p in paths if not p.exists()]
            if missing:
                bad.append((mpath.name, f"slide not found: {missing[0]}"))
                continue
            rid = rec.get(id_key)
            good.append({
                "slides": [str(p) for p in paths],
                "caption": str(rec.get("caption", "") or ""),
                "disease": str(rec.get("disease", "") or ""),
                "id": None if rid is None else str(rid),
            })

    for s in slides:
        good.append({"slides": [str(s)], "caption": "", "disease": "", "id": s.stem})
    return good, bad


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _MockTitan(torch.nn.Module):
    """Deterministic stand-in used only when OPATH_TITAN_MOCK=1 (CPU smoke tests)."""

    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.patch_proj = torch.nn.Linear(3 * 16 * 16, 768)
        self.slide_proj = torch.nn.Linear(768, 768)
        with torch.no_grad():
            self.patch_proj.weight.copy_(torch.randn(768, 768, generator=g) * 0.02)
            self.slide_proj.weight.copy_(torch.randn(768, 768, generator=g) * 0.02)

    def encode_slide_from_patch_features(self, features, coords, patch_size_lv0):
        assert features.dim() == 3 and coords.dim() == 3, "expect [1, N, D] and [1, N, 2]"
        assert features.shape[1] == coords.shape[1]
        return self.slide_proj(features.float().mean(1))


def _mock_conch(mock: _MockTitan):
    class _Conch(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = mock

        def forward(self, x):
            x = torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(1)
            return self.m.patch_proj(x)

    def transform(img):
        arr = np.asarray(img.resize((64, 64)), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    return _Conch(), transform


def load_titan(model_id: str, hf_token: Optional[str], device: str):
    if os.environ.get("OPATH_TITAN_MOCK") == "1":
        print("WARNING: OPATH_TITAN_MOCK=1 -> using a random mock TITAN (testing only).", flush=True)
        titan = _MockTitan()
        conch, transform = _mock_conch(titan)
        return titan.to(device).eval(), conch.to(device).eval(), transform

    from transformers import AutoModel

    print(f"Loading TITAN: {model_id}", flush=True)
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token
    try:
        titan = AutoModel.from_pretrained(model_id, **kwargs)
    except TypeError:  # very old transformers used use_auth_token
        kwargs.pop("token", None)
        kwargs["use_auth_token"] = hf_token
        titan = AutoModel.from_pretrained(model_id, **kwargs)
    except OSError as exc:
        raise SystemExit(
            f"Could not load {model_id}: {exc}\n"
            "Check that (1) you requested and were granted access at https://huggingface.co/MahmoodLab/TITAN, "
            "(2) HF_TOKEN is set to a token of that account, (3) this machine can reach huggingface.co "
            "(or pass --titan-model /local/path/to/TITAN)."
        ) from exc
    conch, transform = titan.return_conch()
    return titan.to(device).eval(), conch.to(device).eval(), transform


def _conch_forward(conch, x):
    """CONCH v1.5 from titan.return_conch() is called directly (as in TITAN, CLAM, TRIDENT)."""
    out = conch(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() != 2 or out.shape[-1] != 768:
        raise RuntimeError(f"unexpected CONCH v1.5 output shape {tuple(out.shape)} (expected [B, 768])")
    return out


# ---------------------------------------------------------------------------
# Per-slide extraction
# ---------------------------------------------------------------------------

class _PatchDataset(torch.utils.data.Dataset):
    def __init__(self, grid: PatchGrid, transform, roi_mpp: float):
        self.reader = PatchReader(grid, roi_mpp=roi_mpp)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, i: int):
        return self.transform(self.reader.read(i))


def _feat_path(feat_dir: Path, slide: str) -> Path:
    h = hashlib.sha1(str(Path(slide).resolve()).encode()).hexdigest()[:8]
    return feat_dir / f"{Path(slide).stem}_{h}.h5"


def _tiling_signature(args) -> Dict[str, Any]:
    return {
        "feat_version": FEAT_VERSION,
        "patch_px": args.patch_px,
        "target_mag": args.target_mag,
        "min_tissue": args.min_tissue,
        "titan_model": args.titan_model if os.environ.get("OPATH_TITAN_MOCK") != "1" else "mock",
    }


def _load_feats(path: Path, sig: Dict[str, Any]):
    import h5py

    try:
        with h5py.File(path, "r") as f:
            stored = json.loads(f.attrs.get("opath_signature", "{}"))
            if stored != sig or "slide_embedding" not in f:
                return None
            return {
                "slide_embedding": torch.from_numpy(f["slide_embedding"][:]).float(),
                "n_patches": int(f["features"].shape[0]),
            }
    except Exception:
        return None


def _save_feats(path: Path, feats: np.ndarray, grid: PatchGrid, slide_emb: np.ndarray, sig: Dict[str, Any]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("features", data=feats.astype(np.float16), compression="gzip", compression_opts=4)
        c = f.create_dataset("coords", data=grid.coords.astype(np.int64))
        for k, v in grid.attrs().items():
            c.attrs[k] = v
        f.create_dataset("slide_embedding", data=slide_emb.astype(np.float32))
        f.attrs["slide_path"] = grid.path
        f.attrs["opath_signature"] = json.dumps(sig, sort_keys=True)
        f.attrs["encoder"] = ENCODER_NAME
    os.replace(tmp, path)


@torch.inference_mode()
def embed_slide(slide: str, titan, conch, transform, args, device: str) -> Tuple[torch.Tensor, int]:
    sig = _tiling_signature(args)
    fpath = _feat_path(Path(args.feat_dir), slide) if args.feat_dir else None
    if fpath is not None and fpath.exists() and not args.overwrite:
        cached = _load_feats(fpath, sig)
        if cached is not None:
            return cached["slide_embedding"], cached["n_patches"]

    grid = build_patch_grid(
        slide,
        patch_px=args.patch_px,
        target_mag=args.target_mag,
        min_tissue=args.min_tissue,
        default_mpp=args.default_mpp,
        roi_mpp=args.roi_mpp,
    )
    n = int(grid.coords.shape[0])
    if n == 0:
        raise RuntimeError("no tissue patches found (try --min-tissue 0.1 or check the slide)")
    if args.max_patches and n > args.max_patches:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(n, size=args.max_patches, replace=False))
        grid.coords = grid.coords[keep]
        n = args.max_patches

    ds = _PatchDataset(grid, transform, args.roi_mpp)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    use_amp = device.startswith("cuda")
    feats: List[torch.Tensor] = []
    for batch in tqdm(loader, desc=f"  patches {Path(slide).name} ({n})", leave=False):
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = _conch_forward(conch, batch)
        feats.append(out.float().cpu())
    features = torch.cat(feats, 0)  # [N, 768]

    coords = torch.from_numpy(grid.coords).long()
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        slide_emb = titan.encode_slide_from_patch_features(
            features.unsqueeze(0).to(device), coords.unsqueeze(0).to(device), int(grid.patch_size_lv0)
        )
    slide_emb = slide_emb.float().reshape(-1).cpu()

    if fpath is not None:
        _save_feats(fpath, features.numpy(), grid, slide_emb.numpy(), sig)
    return slide_emb, n


def build_cache(records: List[Dict[str, Any]], args, sources: List[str]) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    device = args.device or default_device()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    titan, conch, transform = load_titan(args.titan_model, hf_token, device)

    embeddings: List[torch.Tensor] = []
    meta: Dict[str, List[Any]] = {"image_paths": [], "captions": [], "disease": [], "n_patches": []}
    has_id = any(r.get("id") is not None for r in records)
    if has_id:
        meta["id"] = []
    failed: List[Tuple[str, str]] = []

    t0 = time.time()
    for rec in tqdm(records, desc="TITAN embedding"):
        slide_embs: List[torch.Tensor] = []
        n_total = 0
        err = None
        for s in rec["slides"]:
            try:
                e, n = embed_slide(s, titan, conch, transform, args, device)
                slide_embs.append(e)
                n_total += n
            except Exception as exc:  # keep going; report at the end
                err = f"{Path(s).name}: {exc}"
                if args.strict:
                    raise
        if not slide_embs:
            failed.append((rec.get("id") or rec["slides"][0], err or "no embedding"))
            continue
        if err:
            print(f"  WARNING partial case {rec.get('id')}: {err}", flush=True)
        embeddings.append(torch.stack(slide_embs).mean(0))
        meta["image_paths"].append(";".join(rec["slides"]))
        meta["captions"].append(rec["caption"])
        meta["disease"].append(rec["disease"])
        meta["n_patches"].append(n_total)
        if has_id:
            meta["id"].append(rec.get("id") or "")

    stacked = torch.stack(embeddings).float() if embeddings else torch.empty((0, 0))
    print(f"Embedded {len(embeddings)} record(s) in {time.time() - t0:.0f}s", flush=True)
    cache = {
        "embeddings": stacked,
        "meta": meta,
        "encoder": ENCODER_NAME,
        "sources": sources,
        "titan": {
            "model": args.titan_model,
            "patch_px": args.patch_px,
            "target_mag": args.target_mag,
            "min_tissue": args.min_tissue,
            "max_patches": args.max_patches,
            "multi_slide": "mean of per-slide TITAN embeddings",
        },
    }
    return cache, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a TITAN slide-embedding cache from manifests, slide files, folders or globs."
    )
    p.add_argument("inputs", nargs="+", help=".jsonl/.json/.csv manifests, slide files (.ndpi ...), folders or globs.")
    p.add_argument("-o", "--output", required=True, help="Output cache .pt path.")
    p.add_argument("--image-root", default=None, help="Root to resolve relative 'image' paths against.")
    p.add_argument("--recursive", action="store_true", help="Search folders recursively.")
    p.add_argument("--image-key", default=os.environ.get("IMAGE_KEY", "image"), help="Manifest field holding the slide path(s).")
    p.add_argument("--id-key", default=os.environ.get("ID_KEY", "id"), help="Manifest field holding the case id.")
    p.add_argument("--feat-dir", default=os.environ.get("TITAN_FEAT_DIR"),
                   help="Where per-slide .h5 patch features are stored/reused. "
                        "Default: $TITAN_FEAT_DIR or <output dir>/titan_feats. Use 'none' to disable.")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if a matching .h5 exists.")
    p.add_argument("--titan-model", default=os.environ.get("TITAN_MODEL", "MahmoodLab/TITAN"),
                   help="HF id or local folder of TITAN.")
    p.add_argument("--hf-token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env var.")
    p.add_argument("--device", default=None, help="cuda / cuda:1 / mps / cpu (auto by default).")
    p.add_argument("--batch-size", type=int, default=64, help="Patches per CONCH v1.5 forward pass.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers reading patches.")
    p.add_argument("--patch-px", type=int, default=512, help="Patch size in pixels at target magnification (TITAN: 512).")
    p.add_argument("--target-mag", type=int, default=20, help="Target magnification (TITAN: 20).")
    p.add_argument("--min-tissue", type=float, default=0.25, help="Minimum tissue fraction to keep a patch.")
    p.add_argument("--max-patches", type=int, default=0, help="Randomly subsample to at most this many patches (0 = all).")
    p.add_argument("--default-mpp", type=float, default=None, help="Fallback um/px when a slide has no resolution metadata.")
    p.add_argument("--roi-mpp", type=float, default=0.5, help="Assumed um/px for plain .png/.jpg images (0.5 = 20x).")
    p.add_argument("--strict", action="store_true", help="Stop on the first slide that fails.")
    p.add_argument("--allow-empty", action="store_true", help="Write an empty cache instead of failing.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    manifests, slides = expand_inputs(args.inputs, recursive=args.recursive)
    print(f"Found {len(manifests)} manifest(s) and {len(slides)} bare slide file(s).", flush=True)
    for mp in manifests:
        print(f"  - {mp}", flush=True)

    image_root = Path(args.image_root).resolve() if args.image_root else None
    good, bad = collect_records(manifests, slides, image_root, args.image_key, args.id_key)
    print(f"  valid records: {len(good)}", flush=True)
    if bad:
        print(f"  skipped: {len(bad)} (first 5 reasons below)", flush=True)
        for src, reason in bad[:5]:
            print(f"    [{src}] {reason}", flush=True)
    n_nocap = sum(1 for r in good if not r["caption"])
    if n_nocap:
        print(f"  note: {n_nocap} record(s) have no caption (fine for inference, not for training).", flush=True)
    if not good and not args.allow_empty:
        raise SystemExit("No valid slide records to embed.")

    out = Path(args.output)
    if args.feat_dir is None:
        args.feat_dir = str(out.resolve().parent / "titan_feats")
    elif args.feat_dir.lower() == "none":
        args.feat_dir = None
    if args.feat_dir:
        print(f"Patch features dir: {args.feat_dir}", flush=True)

    cache, failed = build_cache(good, args, sources=[str(p) for p in manifests + slides])
    if failed:
        print(f"  FAILED: {len(failed)} record(s) (first 10 below)", flush=True)
        for rid, reason in failed[:10]:
            print(f"    [{rid}] {reason}", flush=True)
    if cache["embeddings"].numel() == 0 and not args.allow_empty:
        raise SystemExit("All records failed; nothing to save.")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print(f"Saved cache to: {out} | embeddings shape: {tuple(cache['embeddings'].shape)}")


if __name__ == "__main__":
    main()
