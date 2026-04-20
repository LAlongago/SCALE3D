from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from shared.settings import get_settings

logger = logging.getLogger("scale3d.geometry")


def _run_command(command: list[str], workdir: Path) -> None:
    logger.info("Executing external geometry command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        logger.info("Geometry stdout:\n%s", completed.stdout.rstrip())
    if completed.stderr:
        logger.info("Geometry stderr:\n%s", completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )


def run_skeleton_and_length(
    ut_root: Path,
    coord_npy: Path,
    pred_npy: Path,
    output_dir: Path,
) -> dict[str, Path]:
    settings = get_settings()
    skeleton_script = ut_root / "pc-skeletor" / "tools" / "skeletonize_pointcept_instances.py"
    length_script = ut_root / "pc-skeletor" / "tools" / "compute_skeleton_curve_lengths.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            str(settings.pc_skeletor_python),
            str(skeleton_script),
            "--coord-npy",
            str(coord_npy),
            "--pred-npy",
            str(pred_npy),
            "--output-dir",
            str(output_dir),
            "--group-mode",
            "label",
            "--method",
            "lbc",
            "--down-sample",
            "0.01",
            "--min-points",
            "32",
        ],
        ut_root,
    )
    _run_command(
        [
            str(settings.pc_skeletor_python),
            str(length_script),
            "--skeleton-root",
            str(output_dir),
        ],
        ut_root,
    )
    return {
        "summary_json": output_dir / "summary.json",
        "curve_length_json": output_dir / "curve_length_summary.json",
        "curve_length_csv": output_dir / "curve_length_summary.csv",
    }


def parse_curve_length_summary(curve_length_summary_path: Path) -> dict[int, dict]:
    payload = json.loads(curve_length_summary_path.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    result: dict[int, dict] = {}
    for group in groups:
        label = group.get("label")
        if label is None:
            continue
        result[int(label)] = group
    return result
