"""
wsi_tiling.py

Whole-slide image (WSI) reading, tissue detection and patch-grid generation
for the TITAN pipeline. No torch dependency, so it can also be used by
tile_preview.py for quick QC on a laptop.

TITAN expects CONCH v1.5 features from 512 x 512 px patches at 20x
(~0.5 um/px), with coordinates stored in the level-0 reference frame and
`patch_size_lv0` = distance between adjacent patches at level 0
(1024 for a 40x slide, 512 for a 20x slide).

Supported inputs
----------------
* Any format OpenSlide reads: .ndpi, .svs, .mrxs, .scn, .vms, .vmu, .bif,
  pyramidal .tif/.tiff.
* Plain raster images (.png, .jpg, non-pyramidal .tif, ...) through a PIL
  fallback. Their magnification is unknown, so `roi_mpp` (default 0.5,
  i.e. treated as 20x) is assumed.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None  # large ROI images are expected

WSI_EXTENSIONS = {".ndpi", ".svs", ".mrxs", ".scn", ".vms", ".vmu", ".bif", ".tif", ".tiff", ".svslide"}
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jfif", ".tif", ".tiff"}
SLIDE_EXTENSIONS = WSI_EXTENSIONS | RASTER_EXTENSIONS


# ---------------------------------------------------------------------------
# Magnification helpers
# ---------------------------------------------------------------------------

def mpp_to_mag(mpp: float) -> int:
    """Snap microns-per-pixel to a nominal objective magnification.

    Same thresholds as MahmoodLab TRIDENT, so patch_size_lv0 matches what
    TITAN saw during pretraining (e.g. .ndpi at 0.2265 um/px -> 40x -> 1024).
    """
    if mpp < 0.16:
        return 80
    if mpp < 0.2:
        return 60
    if mpp < 0.3:
        return 40
    if mpp < 0.6:
        return 20
    if mpp < 1.2:
        return 10
    if mpp < 2.4:
        return 5
    raise ValueError(f"mpp={mpp} is implausibly coarse for a WSI (expected 20x or 40x).")


# ---------------------------------------------------------------------------
# Slide readers
# ---------------------------------------------------------------------------

class _PILSlide:
    """Minimal OpenSlide-like wrapper around a single-resolution raster image."""

    def __init__(self, path: str, mpp: float):
        with Image.open(path) as im:
            im.load()
            self._img = _to_rgb(im)
        self.dimensions = self._img.size
        self.level_count = 1
        self.level_dimensions = (self.dimensions,)
        self.level_downsamples = (1.0,)
        self.properties = {"openslide.mpp-x": str(mpp), "opath.backend": "PIL"}

    def get_best_level_for_downsample(self, downsample: float) -> int:  # noqa: ARG002
        return 0

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> Image.Image:  # noqa: ARG002
        x, y = location
        w, h = size
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        crop = self._img.crop((x, y, min(x + w, self.dimensions[0]), min(y + h, self.dimensions[1])))
        canvas.paste(crop, (0, 0))
        return canvas

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        thumb = self._img.copy()
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        return thumb

    def close(self) -> None:
        self._img = None


def _to_rgb(im: Image.Image) -> Image.Image:
    """RGBA/LA/P -> RGB on a white background (OpenSlide returns transparent
    pixels outside scanned regions, which must become white, not black)."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return im.convert("RGB")


def open_slide(path: str | os.PathLike, roi_mpp: float = 0.5):
    """Open a slide with OpenSlide, falling back to PIL for plain images."""
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in WSI_EXTENSIONS:
        try:
            import openslide  # type: ignore
        except ImportError as exc:
            if suffix in RASTER_EXTENSIONS:
                return _PILSlide(path, roi_mpp)
            raise ImportError(
                "Reading .ndpi/.svs requires OpenSlide: pip install openslide-python openslide-bin"
            ) from exc
        try:
            return openslide.OpenSlide(path)
        except Exception:
            if suffix in RASTER_EXTENSIONS:  # e.g. a flat .tif OpenSlide rejects
                return _PILSlide(path, roi_mpp)
            raise
    if suffix in RASTER_EXTENSIONS:
        return _PILSlide(path, roi_mpp)
    raise ValueError(f"Unsupported slide/image extension: {path}")


def slide_mpp(slide, default_mpp: Optional[float] = None) -> Tuple[float, str]:
    """Return (mpp, how) for level 0."""
    props = slide.properties
    for key in ("openslide.mpp-x", "openslide.mpp-y", "aperio.MPP"):
        val = props.get(key)
        if val:
            try:
                mpp = float(val)
                if mpp > 0:
                    return mpp, key
            except ValueError:
                pass
    # TIFF resolution tags (unit: centimeter or inch)
    res = props.get("tiff.XResolution")
    unit = (props.get("tiff.ResolutionUnit") or "").lower()
    if res:
        try:
            r = float(res)
            if r > 0 and unit in ("centimeter", "cm"):
                return 1e4 / r, "tiff.XResolution(cm)"
            if r > 0 and unit == "inch":
                mpp = 25400.0 / r
                if mpp < 5:  # ignore 72/96 dpi placeholders
                    return mpp, "tiff.XResolution(inch)"
        except ValueError:
            pass
    obj = props.get("openslide.objective-power") or props.get("hamamatsu.SourceLens")
    if obj:
        try:
            return 10.0 / float(obj), "objective-power"
        except ValueError:
            pass
    if default_mpp is not None:
        return float(default_mpp), "default"
    raise ValueError(
        "Could not determine slide resolution (mpp). Pass --default-mpp 0.25 (40x) or 0.5 (20x)."
    )


# ---------------------------------------------------------------------------
# Tissue detection
# ---------------------------------------------------------------------------

def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centers)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 / total - m0) ** 2 / (w0 * w1 / total)
    between = np.nan_to_num(between)
    return float(centers[int(np.argmax(between))])


def tissue_mask(thumb: Image.Image, *, sat_min: int = 15, val_min: int = 30, val_max: int = 250) -> np.ndarray:
    """Boolean tissue mask from an RGB thumbnail.

    Saturation-based (H&E tissue is coloured, glass is grey/white), with an Otsu
    threshold floored at `sat_min`, dark artefacts removed (`val_min`), a median
    filter against dust, and a small closing to fill gaps between nuclei.
    """
    hsv = np.asarray(thumb.convert("HSV"))
    s = hsv[..., 1]
    v = hsv[..., 2]
    thr = max(_otsu(s), float(sat_min))
    mask = (s > thr) & (v > val_min) & (v < val_max + 5)
    m = Image.fromarray((mask * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MedianFilter(5))
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))  # closing
    return np.asarray(m) > 127


# ---------------------------------------------------------------------------
# Patch grid
# ---------------------------------------------------------------------------

@dataclass
class PatchGrid:
    path: str
    coords: np.ndarray            # [N, 2] int64, top-left (x, y) at level 0
    patch_size_lv0: int           # TITAN's patch_size_lv0
    patch_px: int                 # output patch size (512)
    read_level: int               # pyramid level used for reading
    read_size: int                # region size read at read_level
    mpp: float
    mpp_source: str
    level0_mag: int
    target_mag: int
    dimensions: Tuple[int, int]
    backend: str
    thumb_downsample: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    thumb: Optional[Image.Image] = field(default=None, repr=False)

    def attrs(self) -> dict:
        return {
            "patch_size_level0": int(self.patch_size_lv0),
            "patch_size": int(self.patch_px),
            "level0_magnification": int(self.level0_mag),
            "target_magnification": int(self.target_mag),
            "mpp": float(self.mpp),
            "mpp_source": self.mpp_source,
            "width": int(self.dimensions[0]),
            "height": int(self.dimensions[1]),
            "backend": self.backend,
        }


def build_patch_grid(
    path: str,
    *,
    patch_px: int = 512,
    target_mag: int = 20,
    min_tissue: float = 0.25,
    default_mpp: Optional[float] = None,
    roi_mpp: float = 0.5,
    thumb_px_per_patch: int = 16,
    sat_min: int = 15,
    keep_all_if_empty: bool = True,
    keep_mask: bool = False,
) -> PatchGrid:
    """Tissue-filtered, non-overlapping grid of patches in level-0 coords."""
    slide = open_slide(path, roi_mpp=roi_mpp)
    try:
        backend = "PIL" if isinstance(slide, _PILSlide) else "openslide"
        mpp, mpp_src = slide_mpp(slide, default_mpp=default_mpp if backend == "openslide" else roi_mpp)
        mag0 = mpp_to_mag(mpp)
        if mag0 < target_mag:
            print(f"  WARNING {Path(path).name}: level-0 is {mag0}x (< {target_mag}x); patches will be upsampled.")
        patch_lv0 = max(1, int(round(patch_px * mag0 / target_mag)))
        W, H = slide.dimensions

        # Thumbnail sized so that one patch ~= thumb_px_per_patch pixels.
        ds = max(1.0, patch_lv0 / float(thumb_px_per_patch))
        tw, th = max(1, int(W / ds)), max(1, int(H / ds))
        cap = 8192
        if max(tw, th) > cap:
            f = cap / max(tw, th)
            tw, th = max(1, int(tw * f)), max(1, int(th * f))
        thumb = _to_rgb(slide.get_thumbnail((tw, th)))
        tw, th = thumb.size
        sx, sy = W / tw, H / th
        mask = tissue_mask(thumb, sat_min=sat_min)

        # Integral image for fast per-cell tissue fraction.
        ii = np.pad(mask.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

        nx, ny = math.ceil(W / patch_lv0), math.ceil(H / patch_lv0)
        xs = np.arange(nx) * patch_lv0
        ys = np.arange(ny) * patch_lv0
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        x0 = np.clip((gx / sx).astype(int), 0, tw)
        y0 = np.clip((gy / sy).astype(int), 0, th)
        x1 = np.clip(np.ceil((gx + patch_lv0) / sx).astype(int), 0, tw)
        y1 = np.clip(np.ceil((gy + patch_lv0) / sy).astype(int), 0, th)
        area = np.maximum((x1 - x0) * (y1 - y0), 1)
        tissue = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
        frac = tissue / area
        keep = frac >= min_tissue
        if not keep.any() and keep_all_if_empty and backend == "PIL":
            keep = np.ones_like(keep)  # small ROI photo: use every tile
        coords = np.stack([gx[keep], gy[keep]], axis=1).astype(np.int64)

        want_ds = patch_lv0 / float(patch_px)
        level = int(slide.get_best_level_for_downsample(want_ds + 1e-6))
        level_ds = float(slide.level_downsamples[level])
        read_size = max(1, int(round(patch_lv0 / level_ds)))

        return PatchGrid(
            path=str(path),
            coords=coords,
            patch_size_lv0=patch_lv0,
            patch_px=patch_px,
            read_level=level,
            read_size=read_size,
            mpp=mpp,
            mpp_source=mpp_src,
            level0_mag=mag0,
            target_mag=target_mag,
            dimensions=(W, H),
            backend=backend,
            thumb_downsample=(sx + sy) / 2,
            mask=mask if keep_mask else None,
            thumb=thumb if keep_mask else None,
        )
    finally:
        slide.close()


class PatchReader:
    """Reads patches for a PatchGrid; opens the slide lazily (safe with DataLoader workers)."""

    def __init__(self, grid: PatchGrid, roi_mpp: float = 0.5):
        self.grid = grid
        self.roi_mpp = roi_mpp
        self._slide = None

    def __len__(self) -> int:
        return int(self.grid.coords.shape[0])

    def read(self, i: int) -> Image.Image:
        if self._slide is None:
            self._slide = open_slide(self.grid.path, roi_mpp=self.roi_mpp)
        x, y = (int(v) for v in self.grid.coords[i])
        g = self.grid
        region = self._slide.read_region((x, y), g.read_level, (g.read_size, g.read_size))
        region = _to_rgb(region)
        if region.size != (g.patch_px, g.patch_px):
            region = region.resize((g.patch_px, g.patch_px), Image.Resampling.BILINEAR)
        return region

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
            self._slide = None


def preview_image(grid: PatchGrid, max_side: int = 2000, roi_mpp: float = 0.5) -> Image.Image:
    """Slide thumbnail with the tissue mask tinted blue and kept patches outlined green (QC)."""
    from PIL import ImageDraw

    if grid.mask is None:
        raise ValueError("build_patch_grid(..., keep_mask=True) is required for preview")
    W, H = grid.dimensions
    f = min(1.0, max_side / float(max(W, H)))
    size = (max(1, int(W * f)), max(1, int(H * f)))
    slide = open_slide(grid.path, roi_mpp=roi_mpp)
    try:
        base = _to_rgb(slide.get_thumbnail(size))
    finally:
        slide.close()
    mask_img = Image.fromarray((grid.mask * 70).astype(np.uint8)).resize(base.size, Image.Resampling.NEAREST)
    base.paste(Image.new("RGB", base.size, (0, 160, 255)), (0, 0), mask_img)
    draw = ImageDraw.Draw(base)
    sx, sy = W / base.size[0], H / base.size[1]
    for x, y in grid.coords:
        draw.rectangle(
            [x / sx, y / sy, (x + grid.patch_size_lv0) / sx - 1, (y + grid.patch_size_lv0) / sy - 1],
            outline=(0, 170, 0),
        )
    return base
