"""第12G：正式报告生成 / 导出 / 质量接口（面向产品前台）。

POST /api/report/generate                  生成对齐模板的 9 章报告（+四张量化表）
GET  /api/report/{report_id}/quality       报告质量评估结论
GET  /api/report/latest?project_id=         读取最近一次报告（结构化，供预览）
GET  /api/report/{report_id}/download-docx  下载 Word
GET  /api/report/{report_id}/download-pdf   导出 PDF

红线：报告数字来自本地确定性分析；仅 train/val，不触碰 test；不调外部 API；
文件落在已 gitignore 的 outputs 目录，不入库、不外发。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.services import model_inference_service as mi
from app.services import project_service
from app.services import report_builder_service as rb
from app.services import report_quality_v2_service as q
from app.services import report_word_service as rw

router = APIRouter(prefix="/api/report", tags=["report"])


class GenerateReportRequest(BaseModel):
    project_id: int
    case_style: str | None = Field(default=None, description="可选案例风格，如『按照鲁商1992案例风格』")
    requirement: str | None = Field(default=None, description="可选项目需求补充（不影响事实数字）")


def _project_or_404(db: Session, project_id: int):
    p = project_service.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"项目 {project_id} 不存在")
    return p


@router.post("/generate")
def generate(payload: GenerateReportRequest, db: Session = Depends(get_db)) -> dict:
    project = _project_or_404(db, payload.project_id)
    # fail-closed：先跑真实分析与自训练模型，无有效结果不出报告
    bundle = mi.run_inference(db, project)
    if bundle.get("status") != "ok" or not bundle.get("model_run_id") or not bundle.get("analysis_result"):
        raise HTTPException(status_code=422,
                            detail=bundle.get("message") or "项目分析数据不足，无法生成报告。")
    content = rb.build_report(db, project, case_style_key=payload.case_style)
    try:
        rendered = rw.build_and_convert(content, bundle)
    except rw.ReportGateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    quality = q.evaluate(content, pdf_ok=bool(rendered.get("pdf_size")),
                         docx_ok=bool(rendered.get("docx_size")))
    return {
        "report_id": content["report_id"],
        "project_id": content["project_id"],
        "project_name": content["project_name"],
        "project_type": bundle.get("renewal_type") or content["project_type"],
        "generated_at": content["generated_at"],
        "title": content["title"],
        "chapters": content["chapters"],
        "tables_index": content["tables_index"],
        "chapters_count": content["chapters_count"],
        "required_chapters": content["required_chapters"],
        "required_tables": content["required_tables"],
        "model_run_id": bundle.get("model_run_id"),
        "render": {
            "docx_size": rendered.get("docx_size"),
            "pdf_size": rendered.get("pdf_size"),
            "pdf_from_word": rendered.get("pdf_from_word"),
            "template_used": rendered.get("template_used"),
            "fill_stats": rendered.get("fill_stats"),
        },
        "quality": {
            "overall_status": quality["overall_status"],
            "passed": quality["passed"],
            "scores": quality["scores"],
            "metrics_status": quality["metrics_status"],
        },
        "download_docx": f"/api/report/{content['report_id']}/download-docx",
        "download_pdf": f"/api/report/{content['report_id']}/download-pdf",
    }


@router.get("/latest")
def latest(project_id: int = Query(default=1), db: Session = Depends(get_db)) -> dict:
    _project_or_404(db, project_id)
    content = rb.load_latest(project_id)
    if content is None:
        raise HTTPException(status_code=404, detail="该项目暂无已生成报告，请先调用 generate。")
    return {
        "report_id": content["report_id"],
        "project_id": content["project_id"],
        "project_name": content["project_name"],
        "project_type": content["project_type"],
        "generated_at": content["generated_at"],
        "title": content["title"],
        "chapters": content["chapters"],
        "tables_index": content["tables_index"],
        "chapters_count": content["chapters_count"],
        "download_docx": f"/api/report/{content['report_id']}/download-docx",
        "download_pdf": f"/api/report/{content['report_id']}/download-pdf",
    }


def _plain_text(content: dict) -> str:
    """把报告内容拼成纯文本正文（供前台复制；含封面/目录/9章，不含内部字段）。

    若已存在「就地填空」后从 Word 抽取的渲染文本，则直接采用，保证预览与正文一致。
    """
    if content.get("rendered_text"):
        return str(content["rendered_text"]).strip()
    name = content.get("project_name") or "城市更新项目"
    gen = content.get("generated_at", "")
    lines: list[str] = [
        "城市更新项目前策大数据分析报告",
        f"【{name}】前策大数据分析报告（含城市更新逻辑）",
        "生成模型：CityRenew Agent 城市更新前期策划智能体",
        "数据来源：黑客松比赛提供专用数据库；POI兴趣点补充自高德开放数据（GCJ02，覆盖上海全市）",
        "数据支撑：房价、人口、产业等多源数据来自比赛专用数据库，POI兴趣点覆盖全市（比赛数据+高德开放POI），融入城市更新前策分析逻辑",
        "",
        "报告目录",
    ]
    for ch in content.get("chapters", []):
        lines.append(f"{ch.get('no','')}. {ch.get('title','')}".strip())
    lines.append("")

    def emit(node):
        for p in node.get("paragraphs", []):
            lines.append(str(p))
        for b in node.get("bullets", []):
            lines.append(f"· {b}")
        for tb in node.get("tables", []):
            lines.append(tb.get("title", ""))
            for row in tb.get("rows", []):
                lines.append(
                    f"{row.get('label','')}：核心 {row.get('core','')} / "
                    f"近邻 {row.get('nearby','')} / 辐射 {row.get('radiation','')}")

    for ch in content.get("chapters", []):
        lines.append(f"{ch.get('no', '')}. {ch.get('title', '')}".strip())
        emit(ch)
        for sec in ch.get("sections", []):
            lines.append(sec.get("title", ""))
            emit(sec)
        lines.append("")
    return "\n".join(lines).strip()


@router.get("/{report_id}/content")
def content_view(report_id: str) -> dict:
    """读取报告正文内容（供前台预览 / 复制正文，不暴露任何内部字段）。"""
    content = rb.load_by_report_id(report_id)
    if content is None:
        raise HTTPException(status_code=404, detail="未找到该报告，请重新生成。")
    return {
        "report_id": content["report_id"],
        "title": content.get("title"),
        "project_name": content.get("project_name"),
        "generated_at": content.get("generated_at"),
        "cover": {
            "main_title": "城市更新项目前策大数据分析报告",
            "subtitle": f"【{content.get('project_name') or '城市更新项目'}】前策大数据分析报告（含城市更新逻辑）",
            "generator": "CityRenew Agent 城市更新前期策划智能体",
            "data_source": "黑客松比赛提供专用数据库；POI兴趣点补充自高德开放数据（GCJ02，覆盖上海全市）",
            "data_support": "房价、人口、产业等多源数据来自比赛专用数据库，POI兴趣点覆盖全市（比赛数据+高德开放POI），融入城市更新前策分析逻辑",
        },
        "directory": content.get("directory", [f"{c.get('no')}. {c.get('title')}" for c in content.get("chapters", [])]),
        "chapters": [
            {"no": c.get("no"), "title": c.get("title"),
             "paragraphs": c.get("paragraphs", []), "bullets": c.get("bullets", []),
             "tables": c.get("tables", []), "sections": c.get("sections", [])}
            for c in content.get("chapters", [])
        ],
        # 与最终 Word/PDF 完全一致的「就地填空」渲染块（供前台预览）
        "blocks": content.get("rendered_blocks", []),
        "plain_text": _plain_text(content),
    }


@router.get("/{report_id}/quality")
def quality(report_id: str) -> dict:
    content = rb.load_by_report_id(report_id)
    if content is None:
        raise HTTPException(status_code=404, detail="未找到该报告，请重新生成。")
    docx_ok = rw.file_for_report(report_id, "docx") is not None
    pdf_ok = rw.file_for_report(report_id, "pdf") is not None
    return q.evaluate(content, pdf_ok=pdf_ok, docx_ok=docx_ok)


@router.api_route("/{report_id}/download-docx", methods=["GET", "HEAD"])
def download_docx(report_id: str) -> FileResponse:
    path = rw.file_for_report(report_id, "docx")
    if path is None:
        raise HTTPException(status_code=404, detail="Word 文件不存在，请先生成报告。")
    return FileResponse(
        str(path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="城市更新前期策划报告.docx",
    )


@router.api_route("/{report_id}/download-pdf", methods=["GET", "HEAD"])
def download_pdf(report_id: str) -> FileResponse:
    path = rw.file_for_report(report_id, "pdf")
    if path is None:
        raise HTTPException(status_code=404, detail="PDF 文件不存在，请先生成报告。")
    return FileResponse(str(path), media_type="application/pdf", filename="城市更新前期策划报告.pdf")
