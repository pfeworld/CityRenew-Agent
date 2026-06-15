"""报告生成与报告质量门禁接口（第7阶段）。

POST /api/reports/{project_id}/generate      生成结构化报告内容 + 跑质量门禁 + 落盘
GET  /api/reports/{project_id}/latest        读取最近一次生成的报告内容（含门禁字段）
GET  /api/reports/{project_id}/quality       对最近报告执行质量门禁
POST /api/reports/{project_id}/export-docx   将最近报告导出为 docx
GET  /api/reports/{project_id}/download      下载最近导出的 docx

红线：默认仅 train/val（include_test 恒 false）；不触碰 test；不调外部 API；不使用大模型；
报告数字均来自前序确定性计算并带 evidence_id；不返回 raw_json / 原始明细 / 企业名 / 小区名 /
地址明细；生成文件落在 gitignore 的 outputs 目录，不入库。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.report import (
    Phase75GateResponse,
    ReportContentResponse,
    ReportExportResponse,
    ReportQualityResponse,
)
from app.services import (
    phase75_gate_service,
    project_service,
    report_content_service,
    report_export_service,
    report_quality_service,
)

router = APIRouter(prefix="/api/reports", tags=["reports"])


def _get_project_or_404(db: Session, project_id: int):
    project = project_service.get_project(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"项目 {project_id} 不存在")
    return project


def _content_to_response(content: dict[str, Any], quality: dict[str, Any]) -> ReportContentResponse:
    return ReportContentResponse(
        report_id=content["report_id"],
        project_id=content["project_id"],
        project_name=content.get("project_name"),
        project_type=content.get("project_type"),
        generated_at=content["generated_at"],
        sections=content.get("sections", []),
        notes=content.get("notes", []),
        sections_count=content.get("sections_count", 0),
        required_sections_count=content.get("required_sections_count", 9),
        report_completeness=quality["report_completeness"],
        data_consistency=quality["data_consistency"],
        evidence_coverage=quality["evidence_coverage"],
        allowed_splits=content.get("allowed_splits", []),
        used_test=content.get("used_test", False),
        evidence_ids_count=content.get("evidence_ids_count", 0),
        leakage_check=quality["leakage_check"],
        quality_status=quality["overall_status"],
        can_enter_next_stage=quality["can_enter_next_stage"],
    )


@router.post("/{project_id}/generate", response_model=ReportContentResponse)
def generate_report(project_id: int, db: Session = Depends(get_db)) -> ReportContentResponse:
    project = _get_project_or_404(db, project_id)
    content = report_content_service.build_report_content(db, project, include_test=False)
    quality = report_quality_service.check_report_quality(db, content)
    return _content_to_response(content, quality)


@router.get("/{project_id}/latest", response_model=ReportContentResponse)
def latest_report(project_id: int, db: Session = Depends(get_db)) -> ReportContentResponse:
    _get_project_or_404(db, project_id)
    content = report_content_service.load_latest(project_id)
    if content is None:
        raise HTTPException(status_code=404, detail="该项目暂无已生成报告，请先调用 generate。")
    quality = report_quality_service.check_report_quality(db, content)
    return _content_to_response(content, quality)


@router.get("/{project_id}/quality", response_model=ReportQualityResponse)
def report_quality(project_id: int, db: Session = Depends(get_db)) -> ReportQualityResponse:
    _get_project_or_404(db, project_id)
    content = report_content_service.load_latest(project_id)
    if content is None:
        raise HTTPException(status_code=404, detail="该项目暂无已生成报告，请先调用 generate。")
    return ReportQualityResponse(**report_quality_service.check_report_quality(db, content))


@router.get("/{project_id}/phase75-gate", response_model=Phase75GateResponse)
def phase75_gate(project_id: int, db: Session = Depends(get_db)) -> Phase75GateResponse:
    """第7.5阶段独立质量门禁 + 反作弊校验（独立读 latest.json + DB 真值 + mutation tests）。"""
    project = _get_project_or_404(db, project_id)
    return Phase75GateResponse(**phase75_gate_service.run_phase75_gate(db, project))


@router.post("/{project_id}/export-docx", response_model=ReportExportResponse)
def export_report_docx(project_id: int, db: Session = Depends(get_db)) -> ReportExportResponse:
    _get_project_or_404(db, project_id)
    content = report_content_service.load_latest(project_id)
    if content is None:
        raise HTTPException(status_code=404, detail="该项目暂无已生成报告，请先调用 generate。")
    return ReportExportResponse(**report_export_service.export_docx(content))


@router.api_route("/{project_id}/download", methods=["GET", "HEAD"])
def download_report(project_id: int, db: Session = Depends(get_db)) -> FileResponse:
    _get_project_or_404(db, project_id)
    path = report_export_service.latest_docx_path(project_id)
    if path is None:
        raise HTTPException(status_code=404, detail="该项目暂无导出 docx，请先调用 export-docx。")
    return FileResponse(
        str(path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"cityrenew_report_p{project_id}.docx",
    )
