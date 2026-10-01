"""
tile_preview.py

Quality check for tissue detection and tiling before running TITAN on a
whole cohort. Needs only openslide-python, numpy and pillow (no GPU, no torch).

For each slide it writes <out_dir>/<slide>_tiles.png (thumbnail, blue = tissue
mask, green boxes = patches that will be embedded) and prints a summary line.

    python tile_preview.py /data/slides/A123.ndpi -o qc/
    python tile_preview.py "/data/slides/*.ndpi" -o qc/ --min-tissue 0.1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsi_tiling import SLIDE_EXTENSIONS, build_patch_grid, preview_image  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Preview tissue mask and TITAN patch grid.")
    p.add_argument("inputs", nargs="+", help="Slide files, folders or globs.")
    p.add_argument("-o", "--out-dir", default="tile_qc")
    p.add_argument("--patch-px", type=int, default=512)
    p.add_argument("--target-mag", type=int, default=20)
    p.add_argument("--min-tissue", type=float, default=0.25)
    p.add_argument("--default-mpp", type=float, default=None)
    p.add_argument("--roi-mpp", type=float, default=0.5)
    p.add_argument("--save-patches", type=int, default=0, help="Also save this many example patches per slide.")
    a = p.parse_args()

    slides = []
    for raw in a.inputs:
        raw = os.path.expanduser(raw)
        path = Path(raw)
        if path.is_dir():
            slides += sorted(q for q in path.iterdir() if q.suffix.lower() in SLIDE_EXTENSIONS)
        elif path.exists():
            slides.append(path)
        else:
            slides += [Path(m) for m in sorted(glob.glob(raw))]
    if not slides:
        raise SystemExit("No slides found.")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in slides:
        try:
            grid = build_patch_grid(
                str(s), patch_px=a.patch_px, target_mag=a.target_mag, min_tissue=a.min_tissue,
                default_mpp=a.default_mpp, roi_mpp=a.roi_mpp, keep_mask=True,
            )
        except Exception as exc:
            print(f"{s.name}: ERROR {exc}")
            continue
        img = preview_image(grid, roi_mpp=a.roi_mpp)
        img.save(out / f"{s.stem}_tiles.png")
        W, H = grid.dimensions
        print(
            f"{s.name}: {W}x{H} px, mpp={grid.mpp:.4f} ({grid.mpp_source}) -> {grid.level0_mag}x, "
            f"patch_size_lv0={grid.patch_size_lv0}, read level {grid.read_level} @ {grid.read_size}px, "
            f"{len(grid.coords)} patches"
        )
        if a.save_patches:
            from wsi_tiling import PatchReader

            reader = PatchReader(grid, roi_mpp=a.roi_mpp)
            step = max(1, len(grid.coords) // a.save_patches)
            for k, i in enumerate(range(0, len(grid.coords), step)):
                if k >= a.save_patches:
                    break
                reader.read(i).save(out / f"{s.stem}_patch{k:02d}.png")
            reader.close()
    print(f"Previews written to {out}/")


if __name__ == "__main__":
    main()
