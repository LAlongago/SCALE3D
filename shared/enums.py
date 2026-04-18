from __future__ import annotations

from enum import Enum


class InputType(str, Enum):
    IMAGE_SET = "image_set"
    POINT_CLOUD = "point_cloud"


class JobStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(str, Enum):
    UPLOAD = "upload"
    IMAGE_RECONSTRUCTION = "image_reconstruction"
    POINTCLOUD_VALIDATION = "pointcloud_validation"
    PART_SEGMENTATION = "part_segmentation"
    SEGMENTATION_REPORT = "segmentation_report"
    SKELETONIZATION_AND_LENGTH = "skeletonization_and_length"
    REPORT_GENERATION = "report_generation"


class ArtifactKind(str, Enum):
    UPLOAD = "upload"
    POINT_CLOUD = "point_cloud"
    SEGMENTATION = "segmentation"
    GEOMETRY = "geometry"
    REPORT = "report"
    INTERNAL = "internal"
