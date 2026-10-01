"""
info.py

Small helper used by run.sh preflight checks.

    python info.py --field D   ../checkpoints/proj_abcde.pt   # -> 768
    python info.py --field llm ../checkpoints/proj_abcde.pt   # -> Qwen/Qwen2.5-7B-Instruct
    python info.py --cache-dim cache.pt                       # -> 768
    python info.py --latest ../checkpoints                    # newest TITAN projector path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _load(path: str):
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu")


def latest_titan_projector(ckpt_dir: Path) -> str:
    """Newest proj_<id>.pt whose model card says it was trained on a TITAN cache."""
    best = None
    for card_path in ckpt_dir.glob("proj_card_*.json"):
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "TITAN" not in str(card.get("encoder", "")):
            continue
        run_id = card.get("run_id") or card_path.stem.replace("proj_card_", "")
        ckpt = ckpt_dir / f"proj_{run_id}.pt"
        if not ckpt.exists():
            continue
        key = str(card.get("created_at", ""))
        if best is None or key > best[0]:
            best = (key, ckpt)
    if best is None:
        return ""
    return str(best[1])


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--field", help="Key to print from a projector checkpoint (D, H, V, llm, encoder).")
    g.add_argument("--cache-dim", action="store_true", help="Print embedding dim of a cache .pt.")
    g.add_argument("--latest", action="store_true", help="Print newest TITAN projector in a checkpoint dir.")
    p.add_argument("path")
    a = p.parse_args()

    if a.latest:
        out = latest_titan_projector(Path(a.path))
        if not out:
            sys.exit(1)
        print(out)
    elif a.cache_dim:
        emb = _load(a.path)["embeddings"]
        print(int(emb.shape[1]) if emb.dim() == 2 and emb.numel() else 0)
    else:
        val = _load(a.path).get(a.field, "")
        print(val)


if __name__ == "__main__":
    main()
