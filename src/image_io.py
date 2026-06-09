"""
Image input utilities for pathology interpretation workflows.

The functions here normalize many input styles into RGB PIL images:
single files, folders, globs, JSON/JSONL/CSV manifests, URLs, common raster
formats, TIFF pyramids, and DICOM when optional dependencies are installed.
"""

from __future__ import annotations

import base64
import csv
import glob
import io
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError


DEFAULT_IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".gif",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".j2k",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".svs",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
    ".dcm",
    ".dicom",
}


@dataclass
class ImageRecord:
    image: str
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    base_dir: Optional[Path] = None

    @property
    def resolved_image(self) -> str:
        if is_url(self.image) or self.image.startswith("data:"):
            return self.image
        path = Path(os.path.expandvars(os.path.expanduser(self.image)))
        if path.is_absolute() or self.base_dir is None:
            return str(path)
        return str((self.base_dir / path).resolve())


def is_url(value: str) -> bool:
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"}


def looks_like_image(path: Path, extensions: Sequence[str] = tuple(DEFAULT_IMAGE_EXTENSIONS)) -> bool:
    return path.suffix.lower() in set(extensions)


def iter_records(
    source: str | os.PathLike[str],
    image_key: str = "image",
    id_key: Optional[str] = None,
    recursive: bool = True,
    extensions: Sequence[str] = tuple(DEFAULT_IMAGE_EXTENSIONS),
) -> Iterator[ImageRecord]:
    """Yield image records from a file, folder, glob, URL, or manifest."""

    src = str(source)
    if is_url(src) or src.startswith("data:"):
        yield ImageRecord(image=src, id=src)
        return

    expanded = os.path.expandvars(os.path.expanduser(src))
    matches = sorted(glob.glob(expanded, recursive=recursive))
    if matches and not Path(expanded).exists():
        for match in matches:
            yield from iter_records(match, image_key=image_key, id_key=id_key, recursive=recursive, extensions=extensions)
        return

    path = Path(expanded)
    if path.is_dir():
        pattern = "**/*" if recursive else "*"
        for item in sorted(path.glob(pattern)):
            if item.is_file() and looks_like_image(item, extensions):
                yield ImageRecord(image=str(item), id=item.stem)
        return

    if not path.exists():
        raise FileNotFoundError(f"Input source not found: {source}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_manifest(path, image_key=image_key, id_key=id_key)
    elif suffix == ".json":
        yield from _iter_json_manifest(path, image_key=image_key, id_key=id_key)
    elif suffix == ".csv":
        yield from _iter_csv_manifest(path, image_key=image_key, id_key=id_key)
    elif looks_like_image(path, extensions):
        yield ImageRecord(image=str(path), id=path.stem)
    else:
        raise ValueError(
            f"Unsupported input source: {source}. Use an image file, directory, glob, URL, CSV, JSON, or JSONL manifest."
        )


def _record_from_mapping(row: Dict[str, Any], base_dir: Path, image_key: str, id_key: Optional[str]) -> ImageRecord:
    if image_key not in row or not row[image_key]:
        raise ValueError(f"Manifest row is missing required image field '{image_key}': {row}")
    metadata = dict(row)
    image = str(metadata.pop(image_key))
    record_id = str(metadata.get(id_key, "")) if id_key else str(metadata.get("id") or metadata.get("number") or "")
    return ImageRecord(image=image, id=record_id, metadata=metadata, base_dir=base_dir)


def _iter_jsonl_manifest(path: Path, image_key: str, id_key: Optional[str]) -> Iterator[ImageRecord]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            yield _record_from_mapping(row, path.parent, image_key, id_key)


def _iter_json_manifest(path: Path, image_key: str, id_key: Optional[str]) -> Iterator[ImageRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records") or data.get("images") or data.get("data") or [data]
    if not isinstance(data, list):
        raise ValueError(f"JSON manifest must be a list or contain records/images/data: {path}")
    for row in data:
        if not isinstance(row, dict):
            raise ValueError(f"JSON manifest rows must be objects: {path}")
        yield _record_from_mapping(row, path.parent, image_key, id_key)


def _iter_csv_manifest(path: Path, image_key: str, id_key: Optional[str]) -> Iterator[ImageRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if image_key not in (reader.fieldnames or []):
            raise ValueError(f"CSV manifest must contain an '{image_key}' column: {path}")
        for row in reader:
            yield _record_from_mapping(row, path.parent, image_key, id_key)


def load_image(
    image_ref: str | os.PathLike[str],
    *,
    max_pixels: int = 16_000_000,
    prefer_openslide: bool = True,
    timeout: int = 30,
) -> Image.Image:
    """Load an image-like input and return an RGB PIL image."""

    ref = str(image_ref)
    if ref.startswith("data:"):
        raw = _read_data_uri(ref)
        source_name = "data-uri"
    elif is_url(ref):
        raw = _read_url(ref, timeout=timeout)
        source_name = ref
    else:
        path = Path(os.path.expandvars(os.path.expanduser(ref)))
        source_name = str(path)
        if _is_dicom_path(path):
            return _load_dicom(path, max_pixels=max_pixels)
        if prefer_openslide and path.suffix.lower() in {".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"}:
            opened = _try_load_openslide(path, max_pixels=max_pixels)
            if opened is not None:
                return opened
        raw = path.read_bytes()

    try:
        with Image.open(io.BytesIO(raw)) as im:
            return normalize_pil_image(im, max_pixels=max_pixels)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not identify image format for {source_name}") from exc


def normalize_pil_image(image: Image.Image, *, max_pixels: int = 16_000_000) -> Image.Image:
    """Normalize orientation, frame/page, mode, alpha, and very large dimensions."""

    if getattr(image, "is_animated", False):
        image.seek(0)
    else:
        try:
            image = next(ImageSequence.Iterator(image))
        except Exception:
            pass

    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        image = _composite_alpha(image)
    elif image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    if image.mode == "L":
        image = image.convert("RGB")

    image = image.copy()
    image.thumbnail(_max_size_for_pixels(image.size, max_pixels), Image.Resampling.LANCZOS)
    return image.convert("RGB")


def _max_size_for_pixels(size: Tuple[int, int], max_pixels: int) -> Tuple[int, int]:
    width, height = size
    if width * height <= max_pixels:
        return size
    scale = (max_pixels / float(width * height)) ** 0.5
    return max(1, int(width * scale)), max(1, int(height * scale))


def _composite_alpha(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _read_url(url: str, *, timeout: int) -> bytes:
    request = Request(url, headers={"User-Agent": "OPathLM-image-loader/1.0"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read()
    guessed_type, _ = mimetypes.guess_type(url)
    if "html" in content_type and not (guessed_type or "").startswith("image/"):
        raise ValueError(f"URL returned HTML rather than an image: {url}")
    return raw


def _read_data_uri(uri: str) -> bytes:
    try:
        header, data = uri.split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid data URI") from exc
    if ";base64" in header:
        return base64.b64decode(data)
    return data.encode("utf-8")


def _is_dicom_path(path: Path) -> bool:
    return path.suffix.lower() in {".dcm", ".dicom"}


def _load_dicom(path: Path, *, max_pixels: int) -> Image.Image:
    try:
        import numpy as np
        import pydicom
    except ImportError as exc:
        raise ImportError("DICOM input requires optional packages: pip install pydicom numpy") from exc

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype("float32")
    arr -= float(arr.min())
    denom = float(arr.max()) or 1.0
    arr = (arr / denom * 255).clip(0, 255).astype("uint8")
    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    else:
        im = Image.fromarray(arr)
    return normalize_pil_image(im, max_pixels=max_pixels)


def _try_load_openslide(path: Path, *, max_pixels: int) -> Optional[Image.Image]:
    try:
        import openslide
    except ImportError:
        return None

    slide = openslide.OpenSlide(str(path))
    try:
        level = len(slide.level_dimensions) - 1
        for candidate, dims in enumerate(slide.level_dimensions):
            if dims[0] * dims[1] <= max_pixels:
                level = candidate
                break
        region = slide.read_region((0, 0), level, slide.level_dimensions[level])
        return normalize_pil_image(region, max_pixels=max_pixels)
    finally:
        slide.close()

