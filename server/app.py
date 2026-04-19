from __future__ import annotations

import json
import logging
import mimetypes
import shutil
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from server.dispatcher import dispatch_job
from server.repository import FileJobRepository
from shared.enums import InputType
from shared.product_models import get_product_model, load_product_models
from shared.schemas import CreateJobResponse, JobStatusResponse
from shared.settings import get_settings
from shared.validators import validate_image_paths, validate_pointcloud_path

app = FastAPI(title="UT Product Inspection System")
logger = logging.getLogger("scale3d.server")


def _check_token(authorization: str | None) -> None:
    token = get_settings().api_token
    if token and authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/api/v1/product-models")
def list_product_models(authorization: Annotated[str | None, Header()] = None):
    _check_token(authorization)
    return [model.model_dump() for model in load_product_models().values()]


@app.get("/api/v1/jobs")
def list_jobs(authorization: Annotated[str | None, Header()] = None):
    _check_token(authorization)
    repo = FileJobRepository()
    return [record.model_dump() for record in repo.list_jobs()]


@app.post("/api/v1/jobs", response_model=CreateJobResponse)
async def create_job(
    product_model_id: Annotated[str, Form(...)],
    input_type: Annotated[str, Form(...)],
    files: Annotated[list[UploadFile], File(...)],
    client_meta: Annotated[str | None, Form()] = None,
    authorization: Annotated[str | None, Header()] = None,
):
    _check_token(authorization)
    logger.info("Received create_job request: product_model_id=%s input_type=%s file_count=%s", product_model_id, input_type, len(files))
    try:
        normalized_input_type = InputType(input_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported input_type: {input_type}") from exc
    try:
        get_product_model(product_model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    repo = FileJobRepository()
    record = repo.create_job(
        product_model_id=product_model_id,
        input_type=normalized_input_type,
        uploads=[file.filename for file in files],
        client_meta=json.loads(client_meta) if client_meta else {},
    )
    uploads_dir = repo.uploads_dir(record.job_id)
    stored_paths = []
    for file in files:
        destination = uploads_dir / (file.filename or "upload.bin")
        logger.info("Receiving upload for job=%s file=%s", record.job_id, destination.name)
        with destination.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        stored_paths.append(destination)

    try:
        if normalized_input_type == InputType.IMAGE_SET:
            validate_image_paths(stored_paths, max_total_mb=get_settings().max_upload_mb)
        else:
            if len(stored_paths) != 1:
                raise ValueError("Point-cloud mode requires exactly one PLY file.")
            validate_pointcloud_path(stored_paths[0])
    except ValueError as exc:
        shutil.rmtree(repo.job_dir(record.job_id), ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dispatch_job(record.job_id)
    logger.info("Job dispatched: job_id=%s queue=%s", record.job_id, record.queue_name)
    return CreateJobResponse(
        job_id=record.job_id,
        status=record.status,
        current_stage=record.current_stage,
        current_progress=record.current_progress,
        current_message=record.current_message,
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, authorization: Annotated[str | None, Header()] = None):
    _check_token(authorization)
    repo = FileJobRepository()
    try:
        record = repo.get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        current_stage=record.current_stage,
        current_progress=record.current_progress,
        current_message=record.current_message,
        queue_name=record.queue_name,
        updated_at=record.updated_at,
        error=record.error,
        stage_history=record.stage_history,
    )


@app.get("/api/v1/jobs/{job_id}/result")
def get_result(job_id: str, authorization: Annotated[str | None, Header()] = None):
    _check_token(authorization)
    repo = FileJobRepository()
    try:
        record = repo.get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if record.result is None:
        raise HTTPException(status_code=409, detail="Result is not ready yet.")
    return record.result.model_dump()


@app.get("/api/v1/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str, authorization: Annotated[str | None, Header()] = None):
    _check_token(authorization)
    repo = FileJobRepository()
    try:
        record = repo.get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [artifact.model_dump() for artifact in record.artifacts]


@app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_name}")
def download_artifact(
    job_id: str,
    artifact_name: str,
    authorization: Annotated[str | None, Header()] = None,
):
    _check_token(authorization)
    repo = FileJobRepository()
    record = repo.get(job_id)
    artifact = next((item for item in record.artifacts if item.name == artifact_name), None)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_name}' was not found.")
    artifact_path = repo.job_dir(job_id) / artifact.relative_path
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact path missing: {artifact.relative_path}")
    media_type = artifact.content_type or mimetypes.guess_type(artifact_path.name)[0] or "application/octet-stream"
    return FileResponse(str(artifact_path), media_type=media_type, filename=artifact_path.name)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
