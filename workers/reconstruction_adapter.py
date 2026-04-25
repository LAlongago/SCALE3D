from __future__ import annotations

import os
import logging
import shlex
import subprocess
from pathlib import Path

from shared.settings import get_settings

logger = logging.getLogger("scale3d.reconstruction")


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


def _run_3dgs_denoise(pointcloud_path: Path, output_dir: Path) -> Path:
    settings = get_settings()
    denoise_script = Path(__file__).resolve().parents[2] / "UTtools" / "denoise_3dgs_point_cloud.py"
    if not denoise_script.exists():
        raise FileNotFoundError(f"3DGS denoise script was not found: {denoise_script}")

    denoised_dir = output_dir / "denoised"
    denoised_dir.mkdir(parents=True, exist_ok=True)
    denoised_path = denoised_dir / "point_cloud_denoised.ply"
    command = [
        str(settings.denoise_3dgs_python),
        str(denoise_script),
        "--input",
        str(pointcloud_path),
        "--output",
        str(denoised_path),
    ]
    if settings.denoise_3dgs_args:
        command.extend(shlex.split(settings.denoise_3dgs_args))

    logger.info("Executing 3DGS point cloud denoise command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(denoise_script.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        logger.info("3DGS denoise stdout:\n%s", completed.stdout.rstrip())
    if completed.stderr:
        logger.info("3DGS denoise stderr:\n%s", completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"3DGS point cloud denoise failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    if not denoised_path.exists():
        raise FileNotFoundError(f"3DGS denoise finished but output was not found: {denoised_path}")
    return denoised_path


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
    reconstructed_pointcloud = candidates[-1]
    if not settings.enable_3dgs_denoise:
        return reconstructed_pointcloud
    return _run_3dgs_denoise(reconstructed_pointcloud, output_dir)
