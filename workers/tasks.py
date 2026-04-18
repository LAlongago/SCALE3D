from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

from server.reporting import build_inspection_summary, render_report_bundle
from server.repository import FileJobRepository
from shared.enums import ArtifactKind, InputType, JobStage, JobStatus
from shared.palette import PALETTE_36
from shared.paths import build_project_paths
from shared.product_models import get_product_model
from shared.schemas import (
    JobResultPayload,
    LengthPartResult,
    ReportsResult,
    SegmentationPartResult,
    SegmentationSummary,
    VisualizationResult,
)
from shared.settings import get_settings
from workers.celery_app import celery_app
from workers.geometry_adapter import parse_curve_length_summary, run_skeleton_and_length
from workers.pointcept_adapter import run_pointcept_inference
from workers.pointcloud_io import load_pointcloud_payload, prepare_pointcept_dataset
from workers.reconstruction_adapter import run_image_reconstruction


class LocalQueueDispatcher:
    def __init__(self) -> None:
        settings = get_settings()
        self.executors = {
            "reconstruction_gpu": ThreadPoolExecutor(
                max_workers=settings.local_reconstruction_workers,
                thread_name_prefix="pis-reconstruction",
            ),
            "segmentation_gpu": ThreadPoolExecutor(
                max_workers=settings.local_segmentation_workers,
                thread_name_prefix="pis-segmentation",
            ),
            "geometry_cpu": ThreadPoolExecutor(
                max_workers=settings.local_geometry_workers,
                thread_name_prefix="pis-geometry",
            ),
        }

    def submit_pipeline(self, job_id: str) -> None:
        Thread(target=run_job_pipeline, args=(job_id,), daemon=True).start()

    def run_stage(self, queue_name: str, fn, *args, **kwargs):
        future = self.executors[queue_name].submit(fn, *args, **kwargs)
        return future.result()


local_dispatcher = LocalQueueDispatcher()


def _copy_into_artifacts(repo: FileJobRepository, job_id: str, source: Path, name: str, kind: ArtifactKind) -> Path:
    artifacts_dir = repo.artifacts_dir(job_id)
    destination = artifacts_dir / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    repo.attach_artifact(job_id, name, destination, kind)
    return destination


def _build_segmentation_summary(product_model, segmentation_rows: list[SegmentationPartResult]) -> SegmentationSummary:
    detected_parts = sum(item.point_count > 0 for item in segmentation_rows)
    missing = [item.part_id for item in segmentation_rows if item.point_count == 0]
    low_confidence = [
        item.part_id
        for item in segmentation_rows
        if item.confidence is not None
        and item.point_count > 0
        and item.confidence < product_model.thresholds.confidence_warning_threshold
    ]
    notes = []
    if missing:
        notes.append(f"Missing expected parts: {', '.join(str(item) for item in missing)}")
    if low_confidence:
        notes.append(
            "Low-confidence parts: " + ", ".join(str(item) for item in low_confidence)
        )
    return SegmentationSummary(
        is_complete=not missing,
        detected_parts=detected_parts,
        expected_parts=len(product_model.expected_parts),
        missing_part_ids=missing,
        low_confidence_part_ids=low_confidence,
        notes=notes,
    )


def _build_length_rows(product_model, curve_length_map: dict[int, dict]) -> list[LengthPartResult]:
    rows = []
    for part_id in range(product_model.num_parts):
        raw = curve_length_map.get(part_id)
        rows.append(
            LengthPartResult(
                part_id=part_id,
                part_name=product_model.part_names[str(part_id)],
                length=None if raw is None else raw.get("curve_length_sum"),
                unit=product_model.length_unit,
                source_skeleton_ply=None if raw is None else raw.get("source_skeleton_ply"),
                status="ok" if raw and raw.get("status") == "ok" else "missing_or_failed",
            )
        )
    return rows


def _run_pointcloud_validation_stage(pointcloud_path: Path) -> dict:
    payload = load_pointcloud_payload(pointcloud_path)
    return {
        "pointcloud_path": str(pointcloud_path.resolve()),
        "vertex_count": payload.vertex_count,
        "bbox_min": payload.bbox_min,
        "bbox_max": payload.bbox_max,
    }


def run_job_pipeline(job_id: str) -> None:
    repo = FileJobRepository()
    paths = build_project_paths(get_settings().runtime_root)
    record = repo.get(job_id)
    product_model = get_product_model(record.product_model_id)
    job_dir = repo.job_dir(job_id)
    workspace_dir = repo.workspace_dir(job_id)
    uploads_dir = repo.uploads_dir(job_id)

    try:
        repo.update_stage(job_id, JobStatus.QUEUED, JobStage.UPLOAD, 5, "Job queued for execution.")

        if record.input_type == InputType.IMAGE_SET:
            repo.update_stage(
                job_id,
                JobStatus.RUNNING,
                JobStage.IMAGE_RECONSTRUCTION,
                15,
                "Running COLMAP and 3DGS reconstruction.",
                queue_name="reconstruction_gpu",
            )
            reconstruction_output = workspace_dir / "reconstruction"
            pointcloud_path = local_dispatcher.run_stage(
                "reconstruction_gpu",
                run_image_reconstruction,
                uploads_dir,
                reconstruction_output,
            )
            copied_pointcloud = _copy_into_artifacts(
                repo,
                job_id,
                pointcloud_path,
                "validated_point_cloud.ply",
                ArtifactKind.POINT_CLOUD,
            )
        else:
            original_pointcloud = next(iter(sorted(uploads_dir.glob("*.ply"))))
            copied_pointcloud = _copy_into_artifacts(
                repo,
                job_id,
                original_pointcloud,
                "validated_point_cloud.ply",
                ArtifactKind.POINT_CLOUD,
            )

        repo.update_stage(
            job_id,
            JobStatus.RUNNING,
            JobStage.POINTCLOUD_VALIDATION,
            30,
            "Validating uploaded point cloud and extracting metadata.",
            queue_name="geometry_cpu",
        )
        pointcloud_validation = local_dispatcher.run_stage(
            "geometry_cpu",
            _run_pointcloud_validation_stage,
            copied_pointcloud,
        )
        validation_json = workspace_dir / "pointcloud_validation.json"
        validation_json.write_text(
            json.dumps(pointcloud_validation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _copy_into_artifacts(repo, job_id, validation_json, "pointcloud_validation.json", ArtifactKind.INTERNAL)

        repo.update_stage(
            job_id,
            JobStatus.RUNNING,
            JobStage.PART_SEGMENTATION,
            50,
            "Preparing Pointcept inference sample and running segmentation.",
            queue_name="segmentation_gpu",
        )
        dataset_root = workspace_dir / "pointcept_dataset"
        sample_name = job_id
        sample_dir = prepare_pointcept_dataset(sample_name, copied_pointcloud, dataset_root)
        segmentation_output = local_dispatcher.run_stage(
            "segmentation_gpu",
            run_pointcept_inference,
            paths.ut_root,
            record.product_model_id,
            dataset_root,
            sample_name,
            workspace_dir / "segmentation",
        )

        pred_artifact = _copy_into_artifacts(
            repo,
            job_id,
            segmentation_output["pred_npy"],
            "prediction.npy",
            ArtifactKind.SEGMENTATION,
        )
        confidence_artifact = _copy_into_artifacts(
            repo,
            job_id,
            segmentation_output["confidence_npy"],
            "point_confidence.npy",
            ArtifactKind.SEGMENTATION,
        )
        segmentation_ply_artifact = _copy_into_artifacts(
            repo,
            job_id,
            segmentation_output["segmentation_ply"],
            "segmentation_pred.ply",
            ArtifactKind.SEGMENTATION,
        )

        repo.update_stage(
            job_id,
            JobStatus.RUNNING,
            JobStage.SEGMENTATION_REPORT,
            65,
            "Aggregating part completeness and confidence report.",
            queue_name="geometry_cpu",
        )
        raw_segmentation_rows = json.loads(
            segmentation_output["summary_json"].read_text(encoding="utf-8")
        )
        segmentation_rows = [
            SegmentationPartResult.model_validate(item) for item in raw_segmentation_rows
        ]
        segmentation_summary = _build_segmentation_summary(product_model, segmentation_rows)
        segmentation_summary_path = workspace_dir / "segmentation_summary.json"
        segmentation_summary_path.write_text(
            segmentation_summary.model_dump_json(indent=2),
            encoding="utf-8",
        )
        _copy_into_artifacts(
            repo,
            job_id,
            segmentation_summary_path,
            "segmentation_summary.json",
            ArtifactKind.SEGMENTATION,
        )

        repo.update_stage(
            job_id,
            JobStatus.RUNNING,
            JobStage.SKELETONIZATION_AND_LENGTH,
            80,
            "Running pc-skeletor skeletonization and curve length analysis.",
            queue_name="geometry_cpu",
        )
        geometry_outputs = local_dispatcher.run_stage(
            "geometry_cpu",
            run_skeleton_and_length,
            paths.ut_root,
            segmentation_output["coord_npy"],
            segmentation_output["pred_npy"],
            workspace_dir / "geometry",
        )
        skeleton_summary_artifact = _copy_into_artifacts(
            repo,
            job_id,
            geometry_outputs["summary_json"],
            "skeleton_summary.json",
            ArtifactKind.GEOMETRY,
        )
        curve_json_artifact = _copy_into_artifacts(
            repo,
            job_id,
            geometry_outputs["curve_length_json"],
            "curve_length_summary.json",
            ArtifactKind.GEOMETRY,
        )
        _copy_into_artifacts(
            repo,
            job_id,
            geometry_outputs["curve_length_csv"],
            "curve_length_summary.csv",
            ArtifactKind.GEOMETRY,
        )

        curve_length_map = parse_curve_length_summary(geometry_outputs["curve_length_json"])
        length_rows = _build_length_rows(product_model, curve_length_map)

        repo.update_stage(
            job_id,
            JobStatus.RUNNING,
            JobStage.REPORT_GENERATION,
            95,
            "Rendering inspection report bundle.",
            queue_name="geometry_cpu",
        )
        warnings = list(segmentation_summary.notes)
        inspection_summary = build_inspection_summary(product_model, warnings)
        result = JobResultPayload(
            segmentation=segmentation_rows,
            lengths=length_rows,
            reports=ReportsResult(
                segmentation_summary=segmentation_summary,
                inspection_summary=inspection_summary,
            ),
            visualization=VisualizationResult(
                segmentation_ply="artifacts/segmentation_pred.ply",
                point_confidence_npy="artifacts/point_confidence.npy",
                pred_npy="artifacts/prediction.npy",
                coord_npy=str(segmentation_output["coord_npy"].resolve()),
                palette=PALETTE_36,
                skeleton_summary_json="artifacts/skeleton_summary.json",
                curve_length_summary_json="artifacts/curve_length_summary.json",
            ),
            raw_outputs={
                "pointcloud_validation": pointcloud_validation,
            },
        )
        reports_dir = workspace_dir / "reports"
        pdf_path, json_path = render_report_bundle(reports_dir, product_model, result)
        pdf_artifact = _copy_into_artifacts(
            repo,
            job_id,
            pdf_path,
            "inspection_report.pdf",
            ArtifactKind.REPORT,
        )
        json_artifact = _copy_into_artifacts(
            repo,
            job_id,
            json_path,
            "inspection_report.json",
            ArtifactKind.REPORT,
        )
        result.reports.pdf_path = "artifacts/inspection_report.pdf"
        result.reports.json_path = "artifacts/inspection_report.json"
        repo.set_result(job_id, result)
        repo.mark_succeeded(job_id)
    except Exception as exc:
        repo.mark_failed(job_id, repo.get(job_id).current_stage, str(exc))
        raise


@celery_app.task(name="workers.tasks.run_job_task")
def run_job_task(job_id: str) -> str:
    run_job_pipeline(job_id)
    return job_id
