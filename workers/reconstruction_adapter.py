from __future__ import annotations

import os
import subprocess
from pathlib import Path

from shared.settings import get_settings


class ReconstructionConfigurationError(RuntimeError):
    pass


def _run_template(command_template: str, image_dir: Path, output_dir: Path) -> None:
    command = command_template.format(image_dir=str(image_dir), output_dir=str(output_dir))
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"External reconstruction command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )


def run_image_reconstruction(image_dir: Path, output_dir: Path) -> Path:
    settings = get_settings()
    if not settings.colmap_command or not settings.dgs_command:
        raise ReconstructionConfigurationError(
            "Image reconstruction is configured but PIS_COLMAP_COMMAND or PIS_3DGS_COMMAND is missing."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_template(settings.colmap_command, image_dir=image_dir, output_dir=output_dir)
    _run_template(settings.dgs_command, image_dir=image_dir, output_dir=output_dir)

    candidates = sorted(output_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(
            f"3DGS reconstruction finished but no point_cloud/iteration_*/point_cloud.ply was found under {output_dir}"
        )
    return candidates[-1]
