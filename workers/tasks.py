from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

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

logger = logging.getLogger("scale3d.worker")


@dataclass(frozen=True)
class PipelineStep:
    stage: JobStage
    progress: int
    queue_name: str
    message: str
    handler: Callable[[str], "PipelineStep | None"]


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
        repo = FileJobRepository()
        record = repo.get(job_id)
        repo.update_stage(job_id, JobStatus.QUEUED, JobStage.UPLOAD, 5, "任务已进入处理队列。")
        logger.info("Job %s accepted into staged local pipeline.", job_id)
        if record.input_type == InputType.IMAGE_SET:
            self._submit_step(job_id, IMAGE_RECONSTRUCTION_STEP)
        else:
            self._submit_step(job_id, POINTCLOUD_VALIDATION_STEP)

    def run_stage(self, queue_name: str, fn, *args, **kwargs):
        future = self.executors[queue_name].submit(fn, *args, **kwargs)
        return future.result()

    def _submit_step(self, job_id: str, step: PipelineStep) -> None:
        repo = FileJobRepository()
        repo.update_stage(
            job_id,
            JobStatus.RUNNING,
            step.stage,
            step.progress,
            step.message,
            queue_name=step.queue_name,
        )
        logger.info("Job %s submitted to %s for %s.", job_id, step.queue_name, step.stage.value)
        future = self.executors[step.queue_name].submit(step.handler, job_id)
        future.add_done_callback(lambda completed: self._on_step_done(job_id, step, completed))

    def _on_step_done(self, job_id: str, step: PipelineStep, future: Future) -> None:
        try:
            next_step = future.result()
        except Exception as exc:
            _mark_job_failed(job_id, step.stage, exc)
            return
        if next_step is None:
            logger.info("Job %s completed staged local pipeline.", job_id)
            return
        self._submit_step(job_id, next_step)

    def shutdown(self) -> None:
        for executor in self.executors.values():
            executor.shutdown(wait=False, cancel_futures=True)


local_dispatcher = LocalQueueDispatcher()


def _copy_into_artifacts(repo: FileJobRepository, job_id: str, source: Path, name: str, kind: ArtifactKind) -> Path:
    artifacts_dir = repo.artifacts_dir(job_id)
    destination = artifacts_dir / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    repo.attach_artifact(job_id, name, destination, kind)
    return destination


def _copy_skeleton_artifacts(repo: FileJobRepository, job_id: str, skeleton_summary_path: Path) -> list[Path]:
    payload = json.loads(skeleton_summary_path.read_text(encoding="utf-8"))
    copied: list[Path] = []
    full = payload.get("full")
    if isinstance(full, dict) and full.get("status") == "ok" and full.get("skeleton_ply"):
        source = Path(full["skeleton_ply"])
        if source.exists():
            copied.append(
                _copy_into_artifacts(repo, job_id, source, "skeleton_full.ply", ArtifactKind.GEOMETRY)
            )

    for group in payload.get("groups", []):
        if not isinstance(group, dict):
            continue
        if group.get("status") != "ok" or not group.get("skeleton_ply"):
            continue
        label = group.get("label", group.get("name"))
        if label is None:
            continue
        source = Path(group["skeleton_ply"])
        if source.exists():
            copied.append(
                _copy_into_artifacts(repo, job_id, source, f"skeleton_part_{label}.ply", ArtifactKind.GEOMETRY)
            )
    return copied


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
        notes.append(f"缺失预期部件：{', '.join(str(item) for item in missing)}")
    if low_confidence:
        notes.append(
            "低置信度部件：" + ", ".join(str(item) for item in low_confidence)
        )
    return SegmentationSummary(
        is_complete=not missing,
        detected_parts=detected_parts,
        expected_parts=len(product_model.expected_parts),
        missing_part_ids=missing,
        low_confidence_part_ids=low_confidence,
        notes=notes,
    )


def _raw_curve_length(raw: dict | None) -> float | None:
    if raw is None:
        return None
    value = raw.get("curve_length_sum")
    if value is None:
        return None
    return float(value)


def _length_scale(product_model, curve_length_map: dict[int, dict]) -> tuple[float | None, int | None]:
    reference_length = getattr(product_model, "reference_part_real_length", None)
    if reference_length is None:
        return None, None
    for part_id in sorted(curve_length_map):
        raw = curve_length_map[part_id]
        raw_length = _raw_curve_length(raw)
        if raw.get("status") == "ok" and raw_length is not None and raw_length > 0:
            return float(reference_length) / raw_length, part_id
    return None, None


def _build_length_rows(product_model, curve_length_map: dict[int, dict]) -> list[LengthPartResult]:
    rows = []
    scale_factor, reference_part_id = _length_scale(product_model, curve_length_map)
    for part_id in range(product_model.num_parts):
        raw = curve_length_map.get(part_id)
        raw_length = _raw_curve_length(raw)
        length = raw_length if getattr(product_model, "reference_part_real_length", None) is None else None
        if raw_length is not None and scale_factor is not None:
            length = raw_length * scale_factor
        rows.append(
            LengthPartResult(
                part_id=part_id,
                part_name=product_model.part_names[str(part_id)],
                length=length,
                unit=product_model.length_unit,
                raw_length=raw_length,
                scale_factor=scale_factor,
                reference_part_id=reference_part_id,
                source_skeleton_ply=None if raw is None else raw.get("source_skeleton_ply"),
                status="ok" if raw and raw.get("status") == "ok" else "missing_or_failed",
            )
        )
    return rows


def _length_calibration_payload(length_rows: list[LengthPartResult]) -> dict:
    reference_entry = next((item for item in length_rows if item.reference_part_id is not None), None)
    return {
        "reference_part_id": None if reference_entry is None else reference_entry.reference_part_id,
        "reference_real_length": None if reference_entry is None else reference_entry.length,
        "unit": None if reference_entry is None else reference_entry.unit,
        "scale_factor": None if reference_entry is None else reference_entry.scale_factor,
        "raw_reference_length": None if reference_entry is None else reference_entry.raw_length,
        "rule": "以计算成功且编号最小的部件作为参考部件，并将该部件真实长度设为 66.4 cm。",
    }


def _job_report_metadata(record, completed_at: datetime) -> dict:
    source_paths = record.client_meta.get("source_paths") if isinstance(record.client_meta, dict) else None
    if not source_paths:
        source_paths = record.uploads
    return {
        "job_id": record.job_id,
        "input_type": record.input_type.value,
        "source_paths": source_paths,
        "uploads": record.uploads,
        "created_at": record.created_at.isoformat(),
        "completed_at": completed_at.isoformat(),
    }


def _run_pointcloud_validation_stage(pointcloud_path: Path) -> dict:
    payload = load_pointcloud_payload(pointcloud_path)
    return {
        "pointcloud_path": str(pointcloud_path.resolve()),
        "vertex_count": payload.vertex_count,
        "bbox_min": payload.bbox_min,
        "bbox_max": payload.bbox_max,
    }


def _job_context(job_id: str) -> tuple[FileJobRepository, object, Path, Path, Path]:
    repo = FileJobRepository()
    record = repo.get(job_id)
    product_model = get_product_model(record.product_model_id)
    workspace_dir = repo.workspace_dir(job_id)
    uploads_dir = repo.uploads_dir(job_id)
    return repo, product_model, repo.job_dir(job_id), workspace_dir, uploads_dir


def _validated_pointcloud_path(repo: FileJobRepository, job_id: str) -> Path:
    return repo.artifacts_dir(job_id) / "validated_point_cloud.ply"


def _segmentation_outputs(workspace_dir: Path) -> dict[str, Path]:
    result_json = workspace_dir / "segmentation" / "pointcept_result_paths.json"
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    return {key: Path(value) for key, value in payload.items()}


def _stage_image_reconstruction(job_id: str) -> PipelineStep:
    repo, _product_model, _job_dir, workspace_dir, uploads_dir = _job_context(job_id)
    logger.info("Job %s running image reconstruction.", job_id)
    reconstruction_output = workspace_dir / "reconstruction"
    pointcloud_path = run_image_reconstruction(uploads_dir, reconstruction_output)
    _copy_into_artifacts(
        repo,
        job_id,
        pointcloud_path,
        "validated_point_cloud.ply",
        ArtifactKind.POINT_CLOUD,
    )
    return POINTCLOUD_VALIDATION_STEP


def _stage_pointcloud_validation(job_id: str) -> PipelineStep:
    repo, _product_model, _job_dir, workspace_dir, uploads_dir = _job_context(job_id)
    record = repo.get(job_id)
    copied_pointcloud = _validated_pointcloud_path(repo, job_id)
    if record.input_type == InputType.POINT_CLOUD:
        logger.info("Job %s using uploaded point cloud directly.", job_id)
        original_pointcloud = next(iter(sorted(uploads_dir.glob("*.ply"))))
        copied_pointcloud = _copy_into_artifacts(
            repo,
            job_id,
            original_pointcloud,
            "validated_point_cloud.ply",
            ArtifactKind.POINT_CLOUD,
        )
    logger.info("Job %s validating point cloud.", job_id)
    pointcloud_validation = _run_pointcloud_validation_stage(copied_pointcloud)
    validation_json = workspace_dir / "pointcloud_validation.json"
    validation_json.write_text(
        json.dumps(pointcloud_validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _copy_into_artifacts(repo, job_id, validation_json, "pointcloud_validation.json", ArtifactKind.INTERNAL)
    return PART_SEGMENTATION_STEP


def _stage_part_segmentation(job_id: str) -> PipelineStep:
    repo, _product_model, _job_dir, workspace_dir, _uploads_dir = _job_context(job_id)
    record = repo.get(job_id)
    paths = build_project_paths(get_settings().runtime_root)
    logger.info("Job %s running Pointcept segmentation.", job_id)
    dataset_root = workspace_dir / "pointcept_dataset"
    sample_name = job_id
    copied_pointcloud = _validated_pointcloud_path(repo, job_id)
    prepare_pointcept_dataset(sample_name, copied_pointcloud, dataset_root)
    segmentation_output = run_pointcept_inference(
        paths.ut_root,
        record.product_model_id,
        dataset_root,
        sample_name,
        workspace_dir / "segmentation",
    )
    _copy_into_artifacts(
        repo,
        job_id,
        segmentation_output["pred_npy"],
        "prediction.npy",
        ArtifactKind.SEGMENTATION,
    )
    _copy_into_artifacts(
        repo,
        job_id,
        segmentation_output["confidence_npy"],
        "point_confidence.npy",
        ArtifactKind.SEGMENTATION,
    )
    _copy_into_artifacts(
        repo,
        job_id,
        segmentation_output["segmentation_ply"],
        "segmentation_pred.ply",
        ArtifactKind.SEGMENTATION,
    )
    return SEGMENTATION_REPORT_STEP


def _stage_segmentation_report(job_id: str) -> PipelineStep:
    repo, product_model, _job_dir, workspace_dir, _uploads_dir = _job_context(job_id)
    logger.info("Job %s building segmentation report.", job_id)
    segmentation_output = _segmentation_outputs(workspace_dir)
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
    return SKELETONIZATION_AND_LENGTH_STEP


def _stage_skeletonization_and_length(job_id: str) -> PipelineStep:
    repo, _product_model, _job_dir, workspace_dir, _uploads_dir = _job_context(job_id)
    paths = build_project_paths(get_settings().runtime_root)
    segmentation_output = _segmentation_outputs(workspace_dir)
    logger.info("Job %s running skeletonization and curve length analysis.", job_id)
    geometry_outputs = run_skeleton_and_length(
        paths.ut_root,
        segmentation_output["coord_npy"],
        segmentation_output["pred_npy"],
        workspace_dir / "geometry",
    )
    _copy_into_artifacts(
        repo,
        job_id,
        geometry_outputs["summary_json"],
        "skeleton_summary.json",
        ArtifactKind.GEOMETRY,
    )
    _copy_into_artifacts(
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
    _copy_skeleton_artifacts(repo, job_id, geometry_outputs["summary_json"])
    return REPORT_GENERATION_STEP


def _stage_report_generation(job_id: str) -> None:
    repo, product_model, _job_dir, workspace_dir, _uploads_dir = _job_context(job_id)
    record = repo.get(job_id)
    logger.info("Job %s generating inspection report.", job_id)
    segmentation_output = _segmentation_outputs(workspace_dir)
    raw_segmentation_rows = json.loads(
        segmentation_output["summary_json"].read_text(encoding="utf-8")
    )
    segmentation_rows = [
        SegmentationPartResult.model_validate(item) for item in raw_segmentation_rows
    ]
    segmentation_summary = SegmentationSummary.model_validate(
        json.loads((workspace_dir / "segmentation_summary.json").read_text(encoding="utf-8"))
    )
    curve_length_map = parse_curve_length_summary(workspace_dir / "geometry" / "curve_length_summary.json")
    length_rows = _build_length_rows(product_model, curve_length_map)
    pointcloud_validation = json.loads(
        (workspace_dir / "pointcloud_validation.json").read_text(encoding="utf-8")
    )
    warnings = list(segmentation_summary.notes)
    inspection_summary = build_inspection_summary(product_model, warnings)
    completed_at = datetime.utcnow()
    job_metadata = _job_report_metadata(record, completed_at)
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
            "length_calibration": _length_calibration_payload(length_rows),
            "job_metadata": job_metadata,
        },
    )
    reports_dir = workspace_dir / "reports"
    pdf_path, json_path = render_report_bundle(reports_dir, product_model, result, job_metadata)
    _copy_into_artifacts(
        repo,
        job_id,
        pdf_path,
        "inspection_report.pdf",
        ArtifactKind.REPORT,
    )
    _copy_into_artifacts(
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
    if get_settings().cleanup_workspace_on_success:
        repo.cleanup_job_temp_resources(job_id)
        logger.info("Job %s temporary resources cleaned after success.", job_id)
    logger.info("Job %s completed successfully.", job_id)
    return None


def _mark_job_failed(job_id: str, stage: JobStage, exc: BaseException) -> None:
    repo = FileJobRepository()
    repo.mark_failed(job_id, stage, str(exc))
    if get_settings().cleanup_workspace_on_failure:
        repo.cleanup_job_temp_resources(job_id)
        logger.info("Job %s temporary resources cleaned after failure.", job_id)
    logger.exception("Job %s failed in %s: %s", job_id, stage.value, exc)


IMAGE_RECONSTRUCTION_STEP = PipelineStep(
    stage=JobStage.IMAGE_RECONSTRUCTION,
    progress=15,
    queue_name="reconstruction_gpu",
    message="正在执行 COLMAP 与 3DGS 重建。",
    handler=_stage_image_reconstruction,
)
POINTCLOUD_VALIDATION_STEP = PipelineStep(
    stage=JobStage.POINTCLOUD_VALIDATION,
    progress=30,
    queue_name="geometry_cpu",
    message="正在校验点云并提取元数据。",
    handler=_stage_pointcloud_validation,
)
PART_SEGMENTATION_STEP = PipelineStep(
    stage=JobStage.PART_SEGMENTATION,
    progress=50,
    queue_name="segmentation_gpu",
    message="正在准备 Pointcept 推理样本并执行部件分割。",
    handler=_stage_part_segmentation,
)
SEGMENTATION_REPORT_STEP = PipelineStep(
    stage=JobStage.SEGMENTATION_REPORT,
    progress=65,
    queue_name="geometry_cpu",
    message="正在汇总部件完整性与置信度结果。",
    handler=_stage_segmentation_report,
)
SKELETONIZATION_AND_LENGTH_STEP = PipelineStep(
    stage=JobStage.SKELETONIZATION_AND_LENGTH,
    progress=80,
    queue_name="geometry_cpu",
    message="正在执行骨架化与曲线长度计算。",
    handler=_stage_skeletonization_and_length,
)
REPORT_GENERATION_STEP = PipelineStep(
    stage=JobStage.REPORT_GENERATION,
    progress=95,
    queue_name="geometry_cpu",
    message="正在生成检测报告。",
    handler=_stage_report_generation,
)


def run_job_pipeline(job_id: str) -> None:
    repo = FileJobRepository()
    settings = get_settings()
    paths = build_project_paths(settings.runtime_root)
    record = repo.get(job_id)
    product_model = get_product_model(record.product_model_id)
    job_dir = repo.job_dir(job_id)
    workspace_dir = repo.workspace_dir(job_id)
    uploads_dir = repo.uploads_dir(job_id)

    try:
        logger.info("Job %s accepted into pipeline.", job_id)
        repo.update_stage(job_id, JobStatus.QUEUED, JobStage.UPLOAD, 5, "任务已进入处理队列。")
        logger.info("Job %s queued.", job_id)

        if record.input_type == InputType.IMAGE_SET:
            logger.info("Job %s entering image reconstruction.", job_id)
            repo.update_stage(
                job_id,
                JobStatus.RUNNING,
                JobStage.IMAGE_RECONSTRUCTION,
                15,
                "正在执行 COLMAP 与 3DGS 重建。",
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
            logger.info("Job %s using uploaded point cloud directly.", job_id)
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
            "正在校验点云并提取元数据。",
            queue_name="geometry_cpu",
        )
        logger.info("Job %s validating point cloud.", job_id)
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
            "正在准备 Pointcept 推理样本并执行部件分割。",
            queue_name="segmentation_gpu",
        )
        logger.info("Job %s running Pointcept segmentation.", job_id)
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
            "正在汇总部件完整性与置信度结果。",
            queue_name="geometry_cpu",
        )
        logger.info("Job %s building segmentation report.", job_id)
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
            "正在执行骨架化与曲线长度计算。",
            queue_name="geometry_cpu",
        )
        logger.info("Job %s running skeletonization and curve length analysis.", job_id)
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
        _copy_skeleton_artifacts(repo, job_id, geometry_outputs["summary_json"])

        curve_length_map = parse_curve_length_summary(geometry_outputs["curve_length_json"])
        length_rows = _build_length_rows(product_model, curve_length_map)

        repo.update_stage(
            job_id,
            JobStatus.RUNNING,
            JobStage.REPORT_GENERATION,
            95,
            "正在生成检测报告。",
            queue_name="geometry_cpu",
        )
        logger.info("Job %s generating inspection report.", job_id)
        warnings = list(segmentation_summary.notes)
        inspection_summary = build_inspection_summary(product_model, warnings)
        completed_at = datetime.utcnow()
        job_metadata = _job_report_metadata(record, completed_at)
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
                "length_calibration": _length_calibration_payload(length_rows),
                "job_metadata": job_metadata,
            },
        )
        reports_dir = workspace_dir / "reports"
        pdf_path, json_path = render_report_bundle(reports_dir, product_model, result, job_metadata)
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
        if settings.cleanup_workspace_on_success:
            repo.cleanup_job_temp_resources(job_id)
            logger.info("Job %s temporary resources cleaned after success.", job_id)
        logger.info("Job %s completed successfully.", job_id)
    except Exception as exc:
        repo.mark_failed(job_id, repo.get(job_id).current_stage, str(exc))
        if settings.cleanup_workspace_on_failure:
            repo.cleanup_job_temp_resources(job_id)
            logger.info("Job %s temporary resources cleaned after failure.", job_id)
        logger.exception("Job %s failed: %s", job_id, exc)
        raise


@celery_app.task(name="workers.tasks.run_job_task")
def run_job_task(job_id: str) -> str:
    run_job_pipeline(job_id)
    return job_id
