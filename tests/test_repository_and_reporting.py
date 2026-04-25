from pathlib import Path

from server.reporting import build_inspection_summary, render_report_bundle
from server.repository import FileJobRepository
from shared.enums import InputType
from shared.product_models import get_product_model
from shared.schemas import (
    JobResultPayload,
    LengthPartResult,
    ReportsResult,
    SegmentationPartResult,
    SegmentationSummary,
    VisualizationResult,
)


def test_repository_create_and_get(tmp_path: Path):
    repo = FileJobRepository(tmp_path)
    record = repo.create_job("modela-36parts", InputType.POINT_CLOUD, ["a.ply"], {})
    loaded = repo.get(record.job_id)
    assert loaded.job_id == record.job_id
    assert loaded.uploads == ["a.ply"]


def test_render_report_bundle(tmp_path: Path):
    product_model = get_product_model("modela-36parts")
    summary = SegmentationSummary(
        is_complete=False,
        detected_parts=1,
        expected_parts=36,
        missing_part_ids=[1],
        notes=["缺失预期部件：1"],
    )
    inspection_summary = build_inspection_summary(product_model, ["warn"])
    result = JobResultPayload(
        segmentation=[
            SegmentationPartResult(
                part_id=0,
                part_name="part_00",
                point_count=42,
                confidence=0.9,
                status="detected",
            )
        ],
        lengths=[
            LengthPartResult(
                part_id=0,
                part_name="part_00",
                length=1.23,
                unit="cm",
                source_skeleton_ply="x.ply",
                status="ok",
            )
        ],
        reports=ReportsResult(
            segmentation_summary=summary,
            inspection_summary=inspection_summary,
        ),
        visualization=VisualizationResult(),
    )
    pdf_path, json_path = render_report_bundle(tmp_path, product_model, result)
    assert pdf_path.exists()
    assert json_path.exists()
