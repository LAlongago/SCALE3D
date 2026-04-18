from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    ut_root: Path
    runtime_root: Path
    jobs_root: Path


def build_project_paths(runtime_root: Path | None = None) -> ProjectPaths:
    project_root = Path(__file__).resolve().parents[1]
    ut_root = project_root.parent
    actual_runtime = runtime_root or (project_root / ".runtime")
    jobs_root = actual_runtime / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        project_root=project_root,
        ut_root=ut_root,
        runtime_root=actual_runtime,
        jobs_root=jobs_root,
    )
