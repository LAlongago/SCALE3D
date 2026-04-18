from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from shared.enums import ArtifactKind, InputType, JobStage, JobStatus


class ThresholdRuleSet(BaseModel):
    min_part_points: int = 32
    confidence_warning_threshold: float = 0.50
    length_warning_rules: dict[str, dict[str, float]] = Field(default_factory=dict)


class ProductModelDefinition(BaseModel):
    product_model_id: str
    display_name: str
    num_parts: int
    part_names: dict[str, str]
    expected_parts: list[int]
    length_unit: str
    report_template_id: str
    pointcept_model_config: str
    pointcept_weight_path: str
    thresholds: ThresholdRuleSet


class JobArtifact(BaseModel):
    name: str
    kind: ArtifactKind
    relative_path: str
    content_type: str | None = None
    size_bytes: int | None = None


class StageState(BaseModel):
    stage: JobStage
    progress: int = 0
    queue_name: str | None = None
    message: str = ""
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SegmentationPartResult(BaseModel):
    part_id: int
    part_name: str
    point_count: int
    confidence: float | None = None
    status: str


class LengthPartResult(BaseModel):
    part_id: int
    part_name: str
    length: float | None = None
    unit: str
    source_skeleton_ply: str | None = None
    status: str


class SegmentationSummary(BaseModel):
    is_complete: bool
    detected_parts: int
    expected_parts: int
    missing_part_ids: list[int] = Field(default_factory=list)
    low_confidence_part_ids: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class InspectionSummary(BaseModel):
    product_model_id: str
    length_unit: str
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class VisualizationResult(BaseModel):
    segmentation_ply: str | None = None
    point_confidence_npy: str | None = None
    pred_npy: str | None = None
    coord_npy: str | None = None
    palette: list[list[int]] = Field(default_factory=list)
    skeleton_summary_json: str | None = None
    curve_length_summary_json: str | None = None


class ReportsResult(BaseModel):
    segmentation_summary: SegmentationSummary
    inspection_summary: InspectionSummary
    pdf_path: str | None = None
    json_path: str | None = None


class JobResultPayload(BaseModel):
    segmentation: list[SegmentationPartResult]
    lengths: list[LengthPartResult]
    reports: ReportsResult
    visualization: VisualizationResult
    raw_outputs: dict[str, Any] = Field(default_factory=dict)


class JobRecord(BaseModel):
    job_id: str
    product_model_id: str
    input_type: InputType
    status: JobStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    current_stage: JobStage = JobStage.UPLOAD
    current_progress: int = 0
    queue_name: str | None = None
    uploads: list[str] = Field(default_factory=list)
    client_meta: dict[str, Any] = Field(default_factory=dict)
    stage_history: list[StageState] = Field(default_factory=list)
    artifacts: list[JobArtifact] = Field(default_factory=list)
    result: JobResultPayload | None = None
    error: str | None = None


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    current_stage: JobStage
    current_progress: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    current_stage: JobStage
    current_progress: int
    queue_name: str | None = None
    updated_at: datetime
    error: str | None = None
