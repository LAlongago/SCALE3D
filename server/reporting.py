from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from shared.schemas import InspectionSummary, JobResultPayload, ProductModelDefinition

from datetime import datetime, date

def json_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def render_report_bundle(
    output_dir: Path,
    product_model: ProductModelDefinition,
    result: JobResultPayload,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "inspection_report.json"
    pdf_path = output_dir / "inspection_report.pdf"

    payload = {
        "product_model_id": product_model.product_model_id,
        "display_name": product_model.display_name,
        "segmentation": [item.model_dump() for item in result.segmentation],
        "lengths": [item.model_dump() for item in result.lengths],
        "reports": result.reports.model_dump(),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_serializer), encoding="utf-8")

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("UT Product Inspection Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Product Model: {product_model.display_name}", styles["Heading2"]),
        Paragraph(
            f"Length Unit: {result.reports.inspection_summary.length_unit}",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("Segmentation Summary", styles["Heading2"]),
        Paragraph(
            f"Detected parts: {result.reports.segmentation_summary.detected_parts} / "
            f"{result.reports.segmentation_summary.expected_parts}",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ]

    if result.reports.inspection_summary.warnings:
        story.append(Paragraph("Warnings", styles["Heading3"]))
        for warning in result.reports.inspection_summary.warnings:
            story.append(Paragraph(warning, styles["BodyText"]))
        story.append(Spacer(1, 10))

    table_rows = [["Part", "Points", "Confidence", "Length", "Status"]]
    lengths_by_part = {item.part_id: item for item in result.lengths}
    for part in result.segmentation:
        length_entry = lengths_by_part.get(part.part_id)
        table_rows.append(
            [
                part.part_name,
                str(part.point_count),
                "-" if part.confidence is None else f"{part.confidence:.3f}",
                "-"
                if length_entry is None or length_entry.length is None
                else f"{length_entry.length:.4f}",
                part.status,
            ]
        )

    table = Table(table_rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return pdf_path, json_path


def build_inspection_summary(
    product_model: ProductModelDefinition,
    warnings: list[str],
) -> InspectionSummary:
    return InspectionSummary(
        product_model_id=product_model.product_model_id,
        length_unit=product_model.length_unit,
        warnings=warnings,
    )
