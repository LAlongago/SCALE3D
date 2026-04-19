from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

from shared.enums import ArtifactKind, InputType, JobStage, JobStatus
from shared.paths import build_project_paths
from shared.schemas import JobArtifact, JobRecord, JobResultPayload, StageState
from shared.settings import get_settings


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

    def create_job(
        self,
        product_model_id: str,
        input_type: InputType,
        uploads: Iterable[str],
        client_meta: dict,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
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
