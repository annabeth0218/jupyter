"""
caption_io.py

Utilities to clean / normalize a pathology image manifest (.jsonl) so that
the training script only has to look at three fields:

    - image    (path to the image file)
    - caption  (a single self-contained training caption)
    - disease  (used as the diagnostic target; optional)

Selectable cleaning steps
-------------------------
Each cleaning operation is registered as a named step in CLEANING_STEPS.
A user can choose which steps run, in what order, either programmatically:

    from caption_io import clean_manifest
    clean_manifest(
        ["raw.jsonl"],
        "cleaned.jsonl",
        steps=["newlines", "compose", "drop_extra"],   # skip md_parens
    )

or on the CLI:

    python caption_io.py raw.jsonl -o cleaned.jsonl --steps newlines compose
    python caption_io.py raw.jsonl -o cleaned.jsonl --no-md-parens

Available steps (in their default order):

    newlines    Replace \\n / \\r\\n / \\r with spaces; collapse whitespace.
    md_parens   Remove parenthesised groups whose contents contain
                uppercase "MD".  Lowercase "md" is preserved.
    compose     Build a single training caption by merging title / caption /
                subcls / cls.
    drop_extra  Keep only `image`, `caption`, `disease` in the output row.

Passing no step flags runs all of them (full clean, matches the previous
default behavior).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Low-level text cleaners
# ---------------------------------------------------------------------------

# Match the innermost () that contain uppercase "MD" anywhere inside.
_PAREN_WITH_MD = re.compile(r"\([^()]*MD[^()]*\)")


def replace_newlines(text: str) -> str:
    """Replace any kind of newline with a single space and collapse runs."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def remove_md_parens(text: str) -> str:
    """Strip parenthesised segments whose contents contain uppercase 'MD'."""
    if not text:
        return ""
    cur = text
    for _ in range(10):  # iterate to handle nested / repeated groups
        nxt = _PAREN_WITH_MD.sub("", cur)
        if nxt == cur:
            break
        cur = nxt
    cur = re.sub(r"\s+([,.;:])", r"\1", cur)
    cur = re.sub(r"\s+", " ", cur)
    return cur.strip()


# ---------------------------------------------------------------------------
# Caption composition
# ---------------------------------------------------------------------------

def _low_first(text: str) -> str:
    """Lowercase only the first character (for slotting into a sentence)."""
    return text[:1].lower() + text[1:] if text else ""


def compose_caption(
    *,
    title: str = "",
    caption: str = "",
    disease: str = "",
    subcls: str = "",
    cls: str = "",
) -> str:
    """Compose a training caption from the manifest's text fields.

    The caller is responsible for any prior text cleaning (newlines /
    MD-parens) -- this function only does the merge.
    """
    parts: List[str] = []

    if title and (not disease or disease.lower() not in title.lower()):
        parts.append(f"{title} is shown.")

    if caption:
        parts.append(caption)

    if subcls or cls:
        if subcls and cls:
            parts.append(
                f"Classified as {_low_first(subcls)} within the broader "
                f"category of {_low_first(cls)}."
            )
        elif subcls:
            parts.append(f"Classified as {_low_first(subcls)}.")
        else:
            parts.append(f"Classified as {_low_first(cls)}.")

    merged = " ".join(parts)
    return re.sub(r"\s+", " ", merged).strip()


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------
#
# A "step" is a function that takes a record dict and returns a record dict.
# Steps run in the user-selected order.

# Fields kept in the output of the `drop_extra` step.
_KEEP_FIELDS = ("image", "caption", "disease")


def step_newlines(record: Dict[str, Any]) -> Dict[str, Any]:
    """Apply replace_newlines() to every string field."""
    out = dict(record)
    for k, v in record.items():
        if isinstance(v, str):
            out[k] = replace_newlines(v)
    return out


def step_md_parens(record: Dict[str, Any]) -> Dict[str, Any]:
    """Apply remove_md_parens() to every string field."""
    out = dict(record)
    for k, v in record.items():
        if isinstance(v, str):
            out[k] = remove_md_parens(v)
    return out


def step_compose(record: Dict[str, Any]) -> Dict[str, Any]:
    """Replace `caption` with the merged training caption."""
    out = dict(record)
    out["caption"] = compose_caption(
        title=record.get("title", "") or "",
        caption=record.get("caption", "") or record.get("captions", "") or "",
        disease=record.get("disease", "") or "",
        subcls=record.get("subcls", "") or "",
        cls=record.get("cls", "") or "",
    )
    return out


def step_drop_extra(record: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only image / caption / disease."""
    return {k: record.get(k, "") for k in _KEEP_FIELDS}


# Step name -> function. Order in this dict is also the default run order.
CLEANING_STEPS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "newlines": step_newlines,
    "md_parens": step_md_parens,
    "compose": step_compose,
    "drop_extra": step_drop_extra,
}

DEFAULT_STEPS: List[str] = list(CLEANING_STEPS.keys())


def resolve_steps(steps: Optional[Sequence[str]]) -> List[Callable]:
    """Turn a list of step names into the actual functions to run."""
    names = DEFAULT_STEPS if steps is None else list(steps)
    funcs: List[Callable] = []
    for name in names:
        if name not in CLEANING_STEPS:
            raise ValueError(
                f"Unknown cleaning step: {name!r}. "
                f"Available: {sorted(CLEANING_STEPS)}"
            )
        funcs.append(CLEANING_STEPS[name])
    return funcs


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def iter_manifest(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a .jsonl file, one per non-empty line."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {p}:{line_no}: {exc}") from exc


def clean_record(
    record: Dict[str, Any],
    steps: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run the selected cleaning steps over one record."""
    if not record.get("image"):
        raise ValueError(f"Record is missing required 'image' field: {record}")

    cur = record
    for fn in resolve_steps(steps):
        cur = fn(cur)
    return cur


def clean_manifest(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    steps: Optional[Sequence[str]] = None,
) -> int:
    """Read .jsonl manifests, run the chosen steps, write a single .jsonl.

    Returns the number of records written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    funcs = resolve_steps(steps)  # validate names up-front

    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for path in input_paths:
            for record in iter_manifest(path):
                if not record.get("image"):
                    raise ValueError(
                        f"Record is missing required 'image' field: {record}"
                    )
                cur = record
                for fn in funcs:
                    cur = fn(cur)
                out.write(json.dumps(cur, ensure_ascii=False) + "\n")
                count += 1
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Clean a pathology .jsonl manifest. By default runs every step "
            "(newlines, md_parens, compose, drop_extra). Use --steps to pick "
            "an explicit ordered subset, or the --no-<step> flags to drop "
            "individual default steps."
        )
    )
    p.add_argument("inputs", nargs="+", help="One or more .jsonl manifests.")
    p.add_argument("-o", "--output", required=True, help="Output .jsonl path.")
    p.add_argument(
        "--steps",
        nargs="+",
        choices=list(CLEANING_STEPS),
        default=None,
        help=(
            "Explicit ordered list of cleaning steps to apply. "
            "If given, overrides the per-step --no-* flags."
        ),
    )
    # Convenience opt-out flags for users who want the default minus one step.
    p.add_argument("--no-newlines", action="store_true", help="Skip the newlines step.")
    p.add_argument("--no-md-parens", action="store_true", help="Skip the md_parens step.")
    p.add_argument("--no-compose", action="store_true", help="Skip the compose step.")
    p.add_argument(
        "--no-drop-extra",
        action="store_true",
        help="Keep all original fields (don't reduce to image/caption/disease).",
    )
    return p.parse_args()


def _resolve_cli_steps(args: argparse.Namespace) -> List[str]:
    if args.steps is not None:
        return list(args.steps)
    skip = set()
    if args.no_newlines:
        skip.add("newlines")
    if args.no_md_parens:
        skip.add("md_parens")
    if args.no_compose:
        skip.add("compose")
    if args.no_drop_extra:
        skip.add("drop_extra")
    return [s for s in DEFAULT_STEPS if s not in skip]


def main() -> None:
    args = _parse_args()
    steps = _resolve_cli_steps(args)
    n = clean_manifest(args.inputs, args.output, steps=steps)
    print(f"Wrote {n} cleaned records to {args.output}")
    print(f"Steps applied (in order): {steps}")


if __name__ == "__main__":
    main()
