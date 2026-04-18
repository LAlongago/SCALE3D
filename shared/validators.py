from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


PLY_TO_STRUCT = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


@dataclass(frozen=True)
class ImageSetValidation:
    image_count: int
    width: int
    height: int
    total_bytes: int


@dataclass(frozen=True)
class PointCloudValidation:
    path: Path
    format_name: str
    vertex_count: int
    has_normals: bool


def validate_image_paths(paths: list[Path], max_total_mb: int = 512) -> ImageSetValidation:
    if not paths:
        raise ValueError("No image files were provided.")
    total_bytes = sum(path.stat().st_size for path in paths)
    if total_bytes > max_total_mb * 1024 * 1024:
        raise ValueError(f"Image set exceeds {max_total_mb} MB.")
    width = None
    height = None
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                current_width, current_height = image.size
        except Exception as exc:  # pragma: no cover - PIL exception shapes vary
            raise ValueError(f"Invalid image file: {path.name}") from exc
        if width is None:
            width, height = current_width, current_height
        elif width != current_width or height != current_height:
            raise ValueError("All uploaded images must share the same width and height.")
    return ImageSetValidation(
        image_count=len(paths),
        width=width or 0,
        height=height or 0,
        total_bytes=total_bytes,
    )


def validate_pointcloud_path(path: Path) -> PointCloudValidation:
    if path.suffix.lower() != ".ply":
        raise ValueError("Only PLY point-cloud files are supported in V1.")
    with path.open("rb") as handle:
        first = handle.readline().decode("utf-8", errors="ignore").strip()
        if first != "ply":
            raise ValueError("File is not a valid PLY point cloud.")
        format_name = None
        vertex_count = None
        properties: list[str] = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header.")
            decoded = line.decode("utf-8", errors="ignore").strip()
            if decoded.startswith("format "):
                format_name = decoded.split()[1]
            elif decoded.startswith("element "):
                tokens = decoded.split()
                in_vertex = len(tokens) >= 3 and tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
            elif decoded.startswith("property ") and in_vertex:
                tokens = decoded.split()
                if len(tokens) >= 3:
                    properties.append(tokens[-1])
            elif decoded == "end_header":
                break
    if format_name not in {"ascii", "binary_little_endian"}:
        raise ValueError("Only ASCII and binary_little_endian PLY files are supported.")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError("PLY file contains no vertices.")
    required = {"x", "y", "z"}
    if not required.issubset(properties):
        raise ValueError("PLY file must contain x/y/z vertex properties.")
    return PointCloudValidation(
        path=path,
        format_name=format_name,
        vertex_count=vertex_count,
        has_normals={"nx", "ny", "nz"}.issubset(properties),
    )
