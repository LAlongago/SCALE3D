from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.validators import PLY_TO_STRUCT, validate_pointcloud_path


PLY_TO_NUMPY = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class PointCloudPayload:
    coord: np.ndarray
    normal: np.ndarray
    bbox_min: list[float]
    bbox_max: list[float]
    vertex_count: int


def _parse_ply_header(path: Path) -> tuple[str, int, list[tuple[str, str]], int]:
    format_name = None
    vertex_count = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected EOF while parsing PLY header.")
            decoded = line.decode("utf-8", errors="ignore").strip()
            if decoded.startswith("format "):
                format_name = decoded.split()[1]
            elif decoded.startswith("element "):
                tokens = decoded.split()
                in_vertex = len(tokens) >= 3 and tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
                    properties = []
            elif decoded.startswith("property ") and in_vertex:
                tokens = decoded.split()
                if len(tokens) != 3:
                    raise ValueError(f"Unsupported property declaration: {decoded}")
                properties.append((tokens[2], tokens[1]))
            elif decoded == "end_header":
                return format_name or "", vertex_count or 0, properties, handle.tell()


def load_pointcloud_payload(path: Path) -> PointCloudPayload:
    validation = validate_pointcloud_path(path)
    format_name, vertex_count, properties, data_offset = _parse_ply_header(path)
    property_names = [name for name, _ in properties]
    has_normals = {"nx", "ny", "nz"}.issubset(property_names)

    if format_name == "ascii":
        coord = np.zeros((vertex_count, 3), dtype=np.float32)
        normal = np.zeros((vertex_count, 3), dtype=np.float32)
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            header_done = False
            data_idx = 0
            for line in handle:
                if not header_done:
                    header_done = line.strip() == "end_header"
                    continue
                parts = line.split()
                if not parts:
                    continue
                lookup = {
                    property_names[i]: float(parts[i]) for i in range(min(len(parts), len(property_names)))
                }
                coord[data_idx] = [lookup["x"], lookup["y"], lookup["z"]]
                if has_normals:
                    normal[data_idx] = [lookup["nx"], lookup["ny"], lookup["nz"]]
                data_idx += 1
    else:
        dtype = np.dtype([(name, PLY_TO_NUMPY[prop_type]) for name, prop_type in properties])
        with path.open("rb") as handle:
            handle.seek(data_offset)
            data = np.fromfile(handle, dtype=dtype, count=vertex_count)
        coord = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32, copy=False)
        if has_normals:
            normal = np.column_stack([data["nx"], data["ny"], data["nz"]]).astype(
                np.float32,
                copy=False,
            )
        else:
            normal = np.zeros((vertex_count, 3), dtype=np.float32)

    return PointCloudPayload(
        coord=coord,
        normal=normal,
        bbox_min=coord.min(axis=0).astype(float).tolist(),
        bbox_max=coord.max(axis=0).astype(float).tolist(),
        vertex_count=validation.vertex_count,
    )


def prepare_pointcept_dataset(sample_name: str, pointcloud_path: Path, dataset_root: Path) -> Path:
    payload = load_pointcloud_payload(pointcloud_path)
    sample_dir = dataset_root / "infer" / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "coord.npy", payload.coord)
    np.save(sample_dir / "normal.npy", payload.normal)
    return sample_dir
