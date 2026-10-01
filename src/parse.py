#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys

# supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}

# duplicate-style suffix such as " (2)", "(3)"
DUP_NUM_RE = re.compile(r"\s*\(\d+\)")


def parse_folder_name(folder_name):
    """
    Parses folder name like:
    S123-45678_disease_optional
    Returns: (id, disease)
    """
    parts = folder_name.split("_")

    case_id = parts[0] if parts else None
    disease = parts[1] if len(parts) > 1 else None

    return case_id, disease


def make_caption(file, renamed, skipped):
    """
    Caption "The image shows {name}." where name is the filename without
    extension, (n) numbering and underscores. Returns None (and records the
    file in `skipped`) if the filename contains "(arrow)".
    Files whose numbering was removed are recorded in `renamed`.
    """
    stem = os.path.splitext(file)[0]
    if "(arrow)" in stem.lower():
        skipped.append(file)
        return None

    name = DUP_NUM_RE.sub("", stem)
    if name != stem:
        renamed.append(file)
    name = " ".join(name.replace("_", " ").split())
    return f"The image shows {name}."


def print_list(title, items):
    if items:
        print(f"{title}:\n" + "\n".join(f"  {i}" for i in items))


def parse_case(case_dir, wsi, f_out, renamed, skipped):
    """Parse one case folder (recursively); returns number of entries written."""
    case_dir = os.path.abspath(case_dir)
    case_id, disease = parse_folder_name(os.path.basename(case_dir))

    if not case_id or not case_id.startswith("S"):
        print(f"Skipping '{case_dir}': folder name does not match expected pattern", file=sys.stderr)
        return 0

    count = 0
    for subdir, dirs, files in os.walk(case_dir):
        dirs.sort()
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() not in IMG_EXTS:
                continue
            if ("wsi" in file.lower()) != wsi:
                continue

            caption = make_caption(file, renamed, skipped)
            if caption is None:
                continue

            entry = {
                "image": os.path.join(subdir, file),
                "id": case_id,
                "disease": disease,
                "caption": caption,
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(
        description="Generate a JSONL manifest of images from case folders."
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("-case", action="store_true",
                      help="root dir is a single case folder")
    mode.add_argument("-batch", action="store_true",
                      help="root dir is a parent folder containing one folder per case")
    ap.add_argument("-root", required=True, help="root directory")
    ap.add_argument("-output", default=None,
                    help="output file name (relative names go in root dir; "
                         "default: <case folder>_manifest.jsonl or manifest.jsonl)")
    ap.add_argument("-wsi", action="store_true", default=False,
                    help="parse only images with 'wsi' in the filename (case-insensitive); "
                         "by default those images are skipped")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        ap.error(f"root directory not found: {root}")

    if args.output:
        output = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    elif args.case:
        output = os.path.join(root, f"{os.path.basename(root)}_manifest.jsonl")
    else:
        output = os.path.join(root, "manifest.jsonl")

    total = 0
    renamed, skipped = [], []
    with open(output, "w", encoding="utf-8") as f_out:
        if args.case:
            total = parse_case(root, args.wsi, f_out, renamed, skipped)
        else:
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                if os.path.isdir(path):
                    total += parse_case(path, args.wsi, f_out, renamed, skipped)

    print_list("numbering removed", renamed)
    print_list("skipped (arrow)", skipped)

    print(f"Manifest saved to {output} ({total} images)")


if __name__ == "__main__":
    main()
