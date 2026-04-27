from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from shared.schemas import InspectionSummary, JobResultPayload, ProductModelDefinition

from datetime import datetime, date

def json_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _register_chinese_font() -> str:
    font_name = "STSong-Light"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    return font_name


def _status_text(status: str) -> str:
    return {
        "detected": "已检测",
        "missing": "缺失",
        "ok": "正常",
        "missing_or_failed": "缺失或失败",
    }.get(status, status)


def _format_length(length: float | None, unit: str) -> str:
    if length is None:
        return "-"
    return f"{length:.2f} {unit}"


def render_report_bundle(
    output_dir: Path,
    product_model: ProductModelDefinition,
    result: JobResultPayload,
    job_metadata: dict | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "inspection_report.json"
    pdf_path = output_dir / "inspection_report.pdf"

    payload = {
        "任务信息": job_metadata or {},
        "产品型号ID": product_model.product_model_id,
        "产品型号名称": product_model.display_name,
        "长度单位": result.reports.inspection_summary.length_unit,
        "分割结果": [item.model_dump() for item in result.segmentation],
        "长度结果": [_report_length_item(item) for item in result.lengths],
        "报告摘要": result.reports.model_dump(),
        "原始输出": _report_raw_outputs(result.raw_outputs),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_serializer), encoding="utf-8")

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    chinese_font = _register_chinese_font()
    styles = getSampleStyleSheet()
    for style_name in ("Title", "Heading2", "Heading3", "BodyText"):
        styles[style_name].fontName = chinese_font
    story = [
        Paragraph("SCALE3D 产品检测报告", styles["Title"]),
        Spacer(1, 12),
        Paragraph("任务信息", styles["Heading2"]),
        Paragraph(f"数据来源：{_format_source_paths(job_metadata)}", styles["BodyText"]),
        Paragraph(f"任务创建时间：{_metadata_value(job_metadata, 'created_at')}", styles["BodyText"]),
        Paragraph(f"任务完成时间：{_metadata_value(job_metadata, 'completed_at')}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(f"产品型号：{product_model.display_name}", styles["Heading2"]),
        Paragraph(
            f"长度单位：{result.reports.inspection_summary.length_unit}",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("分割摘要", styles["Heading2"]),
        Paragraph(
            f"检测到的部件：{result.reports.segmentation_summary.detected_parts} / "
            f"{result.reports.segmentation_summary.expected_parts}",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ]

    if result.reports.inspection_summary.warnings:
        story.append(Paragraph("提示信息", styles["Heading3"]))
        for warning in result.reports.inspection_summary.warnings:
            story.append(Paragraph(warning, styles["BodyText"]))
        story.append(Spacer(1, 10))

    table_rows = [["部件", "点数", "置信度", "真实长度", "原始长度", "状态"]]
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
                else _format_length(length_entry.length, length_entry.unit),
                "-"
                if length_entry is None or length_entry.raw_length is None
                else f"{length_entry.raw_length:.4f}",
                _status_text(part.status),
            ]
        )

    table = Table(table_rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), chinese_font),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return pdf_path, json_path


def _report_length_item(item) -> dict:
    payload = item.model_dump()
    payload.pop("scale_factor", None)
    payload.pop("reference_part_id", None)
    return payload


def _report_raw_outputs(raw_outputs: dict) -> dict:
    payload = dict(raw_outputs)
    payload.pop("length_calibration", None)
    return payload


def _metadata_value(job_metadata: dict | None, key: str) -> str:
    if not job_metadata:
        return "-"
    value = job_metadata.get(key)
    return "-" if value in (None, "") else str(value)


def _format_source_paths(job_metadata: dict | None) -> str:
    if not job_metadata:
        return "-"
    source_paths = job_metadata.get("source_paths") or job_metadata.get("uploads") or []
    if isinstance(source_paths, str):
        return source_paths
    if not source_paths:
        return "-"
    return "<br/>".join(str(item) for item in source_paths)


def build_inspection_summary(
    product_model: ProductModelDefinition,
    warnings: list[str],
) -> InspectionSummary:
    return InspectionSummary(
        product_model_id=product_model.product_model_id,
        length_unit=product_model.length_unit,
        warnings=warnings,
    )
