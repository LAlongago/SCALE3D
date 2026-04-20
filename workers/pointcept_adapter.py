from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from shared.palette import PALETTE_36
from shared.product_models import get_product_model
from shared.settings import get_settings

logger = logging.getLogger("scale3d.pointcept")


def _run_command(command: list[str], workdir: Path) -> None:
    logger.info("Executing external Pointcept command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        logger.info("Pointcept stdout:\n%s", completed.stdout.rstrip())
    if completed.stderr:
        logger.info("Pointcept stderr:\n%s", completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )


def run_pointcept_inference(
    ut_root: Path,
    product_model_id: str,
    dataset_root: Path,
    sample_name: str,
    output_dir: Path,
) -> dict[str, Path]:
    settings = get_settings()
    product_model = get_product_model(product_model_id)
    pointcept_root = ut_root / "Pointcept"
    runner_script = Path(__file__).resolve().with_name("pointcept_external_runner.py")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Dispatching Pointcept inference to external environment: python=%s sample=%s",
        settings.pointcept_python,
        sample_name,
    )

    part_names_json = output_dir / "part_names.json"
    palette_json = output_dir / "palette.json"
    result_json = output_dir / "pointcept_result_paths.json"
    part_names_json.write_text(
        json.dumps(product_model.part_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    palette_json.write_text(json.dumps(PALETTE_36, indent=2), encoding="utf-8")

    _run_command(
        [
            str(settings.pointcept_python),
            str(runner_script),
            "--ut-root",
            str(ut_root.resolve()),
            "--pointcept-root",
            str(pointcept_root.resolve()),
            "--config-file",
            str((ut_root / product_model.pointcept_model_config).resolve()),
            "--weight-file",
            str((ut_root / product_model.pointcept_weight_path).resolve()),
            "--dataset-root",
            str(dataset_root.resolve()),
            "--sample-name",
            sample_name,
            "--output-dir",
            str(output_dir.resolve()),
            "--num-classes",
            str(product_model.num_parts),
            "--part-names-json",
            str(part_names_json.resolve()),
            "--palette-json",
            str(palette_json.resolve()),
            "--result-json",
            str(result_json.resolve()),
        ],
        ut_root,
    )

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    return {key: Path(value) for key, value in payload.items()}
