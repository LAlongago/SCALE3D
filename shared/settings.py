from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from shared.paths import build_project_paths


@dataclass(frozen=True)
class Settings:
    api_token: str | None
    max_upload_mb: int
    runtime_root: Path
    use_celery: bool
    celery_broker_url: str
    celery_result_backend: str
    server_url: str
    colmap_command: str | None
    dgs_command: str | None
    pointcept_python: Path
    pc_skeletor_python: Path
    local_reconstruction_workers: int
    local_segmentation_workers: int
    local_geometry_workers: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    runtime_root = Path(
        os.environ.get("PIS_RUNTIME_ROOT", build_project_paths().runtime_root)
    ).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    return Settings(
        api_token=os.environ.get("PIS_API_TOKEN"),
        max_upload_mb=int(os.environ.get("PIS_MAX_UPLOAD_MB", "512")),
        runtime_root=runtime_root,
        use_celery=os.environ.get("PIS_USE_CELERY", "0") == "1",
        celery_broker_url=os.environ.get(
            "PIS_CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"
        ),
        celery_result_backend=os.environ.get(
            "PIS_CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/1"
        ),
        server_url=os.environ.get("PIS_SERVER_URL", "http://127.0.0.1:8000"),
        colmap_command=os.environ.get("PIS_COLMAP_COMMAND"),
        dgs_command=os.environ.get("PIS_3DGS_COMMAND"),
        pointcept_python=Path(
            os.environ.get(
                "PIS_POINTCEPT_PYTHON",
                "/home/qwer/miniconda3/envs/pointcept/bin/python",
            )
        ).resolve(),
        pc_skeletor_python=Path(
            os.environ.get(
                "PIS_PC_SKELETOR_PYTHON",
                "/home/qwer/miniconda3/envs/pc-skeletor/bin/python",
            )
        ).resolve(),
        local_reconstruction_workers=int(
            os.environ.get("PIS_LOCAL_RECONSTRUCTION_WORKERS", "1")
        ),
        local_segmentation_workers=int(
            os.environ.get("PIS_LOCAL_SEGMENTATION_WORKERS", "1")
        ),
        local_geometry_workers=int(os.environ.get("PIS_LOCAL_GEOMETRY_WORKERS", "2")),
    )
