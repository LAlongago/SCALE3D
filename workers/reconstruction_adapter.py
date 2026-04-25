from __future__ import annotations

import os
import logging
import shlex
import shutil
import subprocess
from pathlib import Path

from shared.settings import get_settings

logger = logging.getLogger("scale3d.reconstruction")


class ReconstructionConfigurationError(RuntimeError):
    pass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def _copy_images_to_batch_scene(image_dir: Path, scene_dir: Path) -> None:
    input_dir = scene_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        shutil.copy2(image_path, input_dir / image_path.name)
        copied += 1
    if copied == 0:
        raise FileNotFoundError(f"No image files were found for reconstruction under {image_dir}")


def _run_default_batch_reconstruction(image_dir: Path, output_dir: Path) -> Path:
    settings = get_settings()
    ut_root = Path(__file__).resolve().parents[2]
    batch_script = ut_root / "gaussian-splatting" / "batch_reconstruct.py"
    if not batch_script.exists():
        raise ReconstructionConfigurationError(
            "Image reconstruction needs either PIS_COLMAP_COMMAND/PIS_3DGS_COMMAND "
            f"or gaussian-splatting/batch_reconstruct.py, but the batch script was not found: {batch_script}"
        )

    scene_name = "scene"
    dataset_root = output_dir / "batch_dataset"
    sfm_output_root = output_dir / "batch_sfm"
    model_output_root = output_dir / "batch_model"
    scene_dir = dataset_root / scene_name
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    _copy_images_to_batch_scene(image_dir, scene_dir)

    command = [
        str(settings.dgs_python),
        str(batch_script),
        "--dataset-root",
        str(dataset_root),
        "--sfm-output-root",
        str(sfm_output_root),
        "--model-output-root",
        str(model_output_root),
        "--scene",
        scene_name,
        "--overwrite",
        "--stop-on-error",
    ]
    if settings.dgs_batch_args:
        command.extend(shlex.split(settings.dgs_batch_args))

    logger.info("Executing default 3DGS batch reconstruction command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(batch_script.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        logger.info("3DGS batch reconstruction stdout:\n%s", completed.stdout.rstrip())
    if completed.stderr:
        logger.info("3DGS batch reconstruction stderr:\n%s", completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"3DGS batch reconstruction failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )

    model_scene = model_output_root / scene_name
    candidates = sorted(model_scene.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(
            f"3DGS batch reconstruction finished but no point_cloud/iteration_*/point_cloud.ply "
            f"was found under {model_scene}"
        )
    return candidates[-1]


def run_image_reconstruction(image_dir: Path, output_dir: Path) -> Path:
    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    if settings.colmap_command and settings.dgs_command:
        _run_template(settings.colmap_command, image_dir=image_dir, output_dir=output_dir)
        _run_template(settings.dgs_command, image_dir=image_dir, output_dir=output_dir)

        candidates = sorted(output_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
        if not candidates:
            raise FileNotFoundError(
                f"3DGS reconstruction finished but no point_cloud/iteration_*/point_cloud.ply was found under {output_dir}"
            )
        reconstructed_pointcloud = candidates[-1]
    else:
        reconstructed_pointcloud = _run_default_batch_reconstruction(image_dir, output_dir)
    if not settings.enable_3dgs_denoise:
        return reconstructed_pointcloud
    return _run_3dgs_denoise(reconstructed_pointcloud, output_dir)
