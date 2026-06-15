"""第12G：案例回归测试服务（RegressionTestService，内部能力）。

基于两个正式案例做输入输出回归测试，验证智能体是否真正"跑通"：
- 测试1 华建案例：能识别项目类型、生成对齐模板的报告，并在第6章引用相关华建案例；
- 测试2 鲁商1992案例：按其风格生成策略时能命中历史建筑活化/商业更新/外挂连廊/
  空间复合/场所记忆等更新方向；
- 测试3 风格迁移：按"鲁商1992风格"生成时，输出确实参考该案例逻辑；
- 测试4 完整报告：能按模板生成 9 章 Word 报告；
- 测试5 PDF 导出成功；
- 测试6 文本洁净：无星号 / Markdown / raw key / 后台字段。

红线：仅内部使用（不暴露前台）；仅 train/val；不调外部 API；不使用大模型生成事实数字。
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.models import Project
from app.services import case_learning_service as cases
from app.services import model_inference_service as mi
from app.services import project_service
from app.services import report_builder_service as rb
from app.services import report_quality_v2_service as q
from app.services import report_word_service as rw

logger = logging.getLogger("cityrenew.regression")

LUSHANG_DIRECTIONS = ["历史", "商业", "外挂连廊", "空间复合", "场所记忆", "首层", "活化", "存量"]


def _t(name: str, passed: bool, detail: str, evidence: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, "evidence": evidence}


def run_case_regression(db: Session, project_id: int = 1) -> dict[str, Any]:
    project: Project | None = project_service.get_project(db, project_id)
    tests: list[dict[str, Any]] = []

    corpus = cases.case_corpus_status()
    tests.append(_t(
        "案例语料加载", corpus["huajian_case_count"] > 0 and corpus["lushang_present"],
        f"华建案例 {corpus['huajian_case_count']} 例、{len(corpus['huajian_categories'])} 类；鲁商1992 画像{'已' if corpus['lushang_present'] else '未'}就绪。",
        {"huajian_count": corpus["huajian_case_count"], "categories": corpus["huajian_categories"]},
    ))

    if project is None:
        tests.append(_t("项目可用", False, f"项目 {project_id} 不存在，后续测试跳过。"))
        return _summary(tests)

    # 标准报告（华建相关方向）
    content = rb.build_report(db, project, case_style_key=None)
    ptype = content.get("project_type")
    ch6 = next((c for c in content["chapters"] if c["no"] == "6"), {})
    ch6_body = "\n".join(ch6.get("bullets", []) + ch6.get("paragraphs", []))
    hj_names = [n for n in (content.get("source_facts", {}).get("case_ref_names") or [])]
    hj_cited = any(n.split("（")[0][:6] in ch6_body for n in hj_names)
    tests.append(_t(
        "测试1·华建案例输入输出", bool(ptype) and content["chapters_count"] == 9 and hj_cited,
        f"识别类型「{ptype}」，生成 {content['chapters_count']} 章报告，第6章引用案例 {len(hj_names)} 个。",
        {"project_type": ptype, "case_refs": hj_names},
    ))

    # 鲁商风格报告
    style_content = rb.build_report(db, project, case_style_key="按照鲁商1992案例风格")
    ls_body = "\n".join(
        ch6_text(style_content, "6") + ch6_text(style_content, "8")
    )
    ls_hits = [k for k in LUSHANG_DIRECTIONS if k in ls_body]
    tests.append(_t(
        "测试2·鲁商1992策略方向", len(ls_hits) >= 3,
        f"按鲁商1992风格生成时命中更新方向：{('、'.join(ls_hits)) or '无'}。",
        {"hits": ls_hits},
    ))

    style_ref = cases.style_reference("按照鲁商1992案例风格生成策略")
    tests.append(_t(
        "测试3·风格迁移生效", bool(style_ref) and "按「鲁商1992" in ls_body,
        "「按鲁商1992风格」请求被识别，且输出引用该案例逻辑。" if style_ref else "未能加载鲁商风格参考。",
        {"keywords": (style_ref or {}).get("keywords")},
    ))

    # 完整报告 + 导出（基于模板复制填充 + Word 转 PDF，绝不使用 ReportLab）
    bundle = mi.run_inference(db, project)
    rendered = None
    if bundle.get("status") == "ok":
        try:
            rendered = rw.build_and_convert(style_content, bundle)
        except rw.ReportGateError as exc:
            logger.warning("回归报告生成失败：%s", exc)
    docx_ok = bool(rendered and rendered.get("docx_size"))
    pdf_ok = bool(rendered and rendered.get("pdf_size"))
    quality = q.evaluate(style_content, pdf_ok=pdf_ok, docx_ok=docx_ok)
    tests.append(_t(
        "测试4·完整报告生成", style_content["chapters_count"] == 9 and docx_ok,
        f"基于模板复制填充生成 9 章 Word 报告（{(rendered or {}).get('docx_size')} 字节）。",
        {"docx": (rendered or {}).get("docx_path"), "template": (rendered or {}).get("template_used")},
    ))
    tests.append(_t(
        "测试5·PDF 由 Word 转换", pdf_ok and bool(rendered and rendered.get("pdf_from_word")),
        f"PDF 由生成后的 Word 转换得到（{(rendered or {}).get('pdf_size')} 字节）。",
        {"pdf": (rendered or {}).get("pdf_path"), "pdf_from_word": (rendered or {}).get("pdf_from_word")},
    ))
    tests.append(_t(
        "测试6·文本洁净", not quality["markdown_hits"] and not quality["forbidden_field_hits"],
        "无星号 / Markdown / raw key / 后台字段。" if not quality["markdown_hits"] and not quality["forbidden_field_hits"]
        else f"检出：{quality['markdown_hits']} {quality['forbidden_field_hits']}",
        {"markdown": quality["markdown_hits"], "fields": quality["forbidden_field_hits"]},
    ))
    tests.append(_t(
        "报告质量门禁", quality["passed"],
        f"质量评估结论：{quality['overall_status']}。",
        {"scores": quality["scores"]},
    ))

    return _summary(tests, quality)


def ch6_text(content: dict, no: str) -> list[str]:
    ch = next((c for c in content.get("chapters", []) if c.get("no") == no), {})
    return ch.get("bullets", []) + ch.get("paragraphs", [])


def _summary(tests: list[dict], quality: dict | None = None) -> dict[str, Any]:
    passed = sum(1 for t in tests if t["passed"])
    total = len(tests)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "all_passed": passed == total,
        "tests": tests,
        "quality": quality,
        "notes": [
            "回归测试为内部能力，仅 train/val，不触碰最终评估留出数据，不调用外部 API。",
        ],
    }
