from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

from shared.enums import ArtifactKind, InputType, JobStage, JobStatus
from shared.paths import build_project_paths
from shared.schemas import JobArtifact, JobRecord, JobResultPayload, StageState
from shared.settings import get_settings


JOB_ID_PREFIXES = {
    InputType.IMAGE_SET: "Images",
    InputType.POINT_CLOUD: "Pointcloud",
}


class FileJobRepository:
    def __init__(self, runtime_root: Path | None = None) -> None:
        settings = get_settings()
        actual_runtime = runtime_root or settings.runtime_root
        self.paths = build_project_paths(actual_runtime)

    def jobs_root(self) -> Path:
        return self.paths.jobs_root

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root() / job_id

    def uploads_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def workspace_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifacts_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def result_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "result.json"

    def cleanup_job_temp_resources(self, job_id: str) -> None:
        for directory in (self.uploads_dir(job_id), self.workspace_dir(job_id)):
            shutil.rmtree(directory, ignore_errors=True)

    def delete_job(self, job_id: str) -> None:
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def cleanup_expired_jobs(self, retention_hours: int) -> list[str]:
        if retention_hours <= 0:
            return []
        now = datetime.utcnow()
        deleted: list[str] = []
        for record in self.list_jobs():
            if record.status not in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
                continue
            age_seconds = (now - record.updated_at).total_seconds()
            if age_seconds >= retention_hours * 3600:
                self.delete_job(record.job_id)
                deleted.append(record.job_id)
        return deleted

    def create_job(
        self,
        product_model_id: str,
        input_type: InputType,
        uploads: Iterable[str],
        client_meta: dict,
    ) -> JobRecord:
        job_id = self._generate_job_id(input_type)
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        record = JobRecord(
            job_id=job_id,
            product_model_id=product_model_id,
            input_type=input_type,
            status=JobStatus.UPLOADED,
            current_stage=JobStage.UPLOAD,
            current_progress=0,
            current_message="文件已接收，等待分发。",
            uploads=list(uploads),
            client_meta=client_meta,
            stage_history=[
                StageState(
                    stage=JobStage.UPLOAD,
                    progress=0,
                    message="Files uploaded and awaiting dispatch.",
                )
            ],
        )
        self.save(record)
        return record

    def _generate_job_id(self, input_type: InputType) -> str:
        prefix = JOB_ID_PREFIXES.get(input_type, "Job")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_id = f"{prefix}_{timestamp}"
        job_id = base_id
        suffix = 1
        while self.job_dir(job_id).exists():
            suffix += 1
            job_id = f"{base_id}_{suffix:02d}"
        return job_id

    def save(self, record: JobRecord) -> None:
        record.updated_at = datetime.utcnow()
        self.record_path(record.job_id).write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if record.result is not None:
            self.result_path(record.job_id).write_text(
                record.result.model_dump_json(indent=2),
                encoding="utf-8",
            )

    def get(self, job_id: str) -> JobRecord:
        path = self.record_path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Job '{job_id}' does not exist.")
        return JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list_jobs(self) -> list[JobRecord]:
        records = []
        for path in sorted(self.jobs_root().glob("*/job.json")):
            records.append(
                JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
            )
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def update_stage(
        self,
        job_id: str,
        status: JobStatus,
        stage: JobStage,
        progress: int,
        message: str,
        queue_name: str | None = None,
    ) -> JobRecord:
        record = self.get(job_id)
        record.status = status
        record.current_stage = stage
        record.current_progress = progress
        record.current_message = message
        record.queue_name = queue_name
        record.stage_history.append(
            StageState(
                stage=stage,
                progress=progress,
                queue_name=queue_name,
                message=message,
            )
        )
        self.save(record)
        return record

    def attach_artifact(
        self,
        job_id: str,
        name: str,
        path: Path,
        kind: ArtifactKind,
        content_type: str | None = None,
    ) -> JobArtifact:
        record = self.get(job_id)
        try:
            relative_path = str(path.resolve().relative_to(self.job_dir(job_id).resolve()))
        except ValueError:
            relative_path = str(path.resolve())
        artifact = JobArtifact(
            name=name,
            kind=kind,
            relative_path=relative_path,
            content_type=content_type,
            size_bytes=path.stat().st_size if path.exists() else None,
        )
        record.artifacts = [item for item in record.artifacts if item.name != name]
        record.artifacts.append(artifact)
        self.save(record)
        return artifact

    def set_result(self, job_id: str, result: JobResultPayload) -> JobRecord:
        record = self.get(job_id)
        record.result = result
        self.save(record)
        return record

    def mark_failed(self, job_id: str, stage: JobStage, message: str) -> JobRecord:
        record = self.get(job_id)
        record.status = JobStatus.FAILED
        record.current_stage = stage
        record.error = message
        record.current_progress = 100
        record.current_message = message
        record.stage_history.append(
            StageState(stage=stage, progress=100, message=message, queue_name=record.queue_name)
        )
        self.save(record)
        return record

    def mark_succeeded(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        record.status = JobStatus.SUCCEEDED
        record.current_progress = 100
        if record.stage_history:
            record.current_message = record.stage_history[-1].message
        self.save(record)
        return record
