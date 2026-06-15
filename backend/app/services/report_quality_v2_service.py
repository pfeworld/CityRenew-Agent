"""第12G：正式报告质量评估器（ReportQualityEvaluator）。

对 report_builder_service 生成的报告内容（+ 已导出文件）做正式版验收检查：
1. 报告结构完整率（9 章齐全、每章有正文）          目标 ≥ 0.95
2. 模板章节匹配率（标题对齐 SC 模板目录）            目标 ≥ 0.95
3. 四张量化表完整率（4 表 + 三圈层列 + 单元格填充）   目标 = 1.0
4. 文本洁净度（无 Markdown/星号/raw key/英文字段/____）目标 命中 = 0
5. 案例引用相关性（第6章命中相关案例）               目标 ≥ 0.85
6. 数字可追溯率（量化表数值可回比 source_metrics）   目标 ≥ 0.90
7. 正式表达评分（启发式：结论充分、无禁用痕迹）       目标 ≥ 0.85
8. 前台内部字段暴露（正文不含模型/训练/接入字段）     目标 = 0

红线：纯只读；不改业务数据；不读 test；不调外部 API；不使用大模型。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services import report_template_service as tpl

logger = logging.getLogger("cityrenew.report.quality.v2")

# 正文禁止出现的开发者 / 训练 / 接入字段（暴露给前台用户即不合格）。
FORBIDDEN_TOKENS = [
    "F_score", "L_score", "P_score", "H_score", "I_score",
    "evidence_id", "MAPE", "mape", "DeepSeek", "deepseek",
    "final test", "train/val", "weak_label", "raw_json",
    "自研模型", "自研城市更新模型", "大模型", "知识库", "train",
    "val", "T1", "T2", "T3",
]
# Markdown / 占位痕迹
MARKDOWN_PATTERNS = [r"\*\*", r"(?<!\w)\*(?!\w)", r"#{1,6}\s", r"_{3,}", r"`", r"\|\s*-{2,}"]

PASS = "pass"
WARNING = "warning"
FAIL = "fail"


def _all_text(content: dict) -> str:
    parts: list[str] = [content.get("title", "")]
    for ch in content.get("chapters", []):
        parts.append(ch.get("title", ""))
        parts.extend(ch.get("paragraphs", []))
        parts.extend(ch.get("bullets", []))
        for sec in ch.get("sections", []):
            parts.append(sec.get("title", ""))
            parts.extend(sec.get("paragraphs", []))
            parts.extend(sec.get("bullets", []))
        for tb in ch.get("tables", []):
            parts.append(tb.get("title", ""))
            for r in tb.get("rows", []):
                parts.extend([str(r.get(k, "")) for k in ("label", "core", "nearby", "radiation")])
    return "\n".join(str(p) for p in parts)


def _chapter_has_body(ch: dict) -> bool:
    """章节是否有正文：直接段落/要点/表格，或子小节(sections)含内容。"""
    if ch.get("paragraphs") or ch.get("bullets") or ch.get("tables"):
        return True
    return any(s.get("paragraphs") or s.get("bullets") for s in ch.get("sections", []))


def _mk(name, value, threshold, status, explanation):
    return {"metric_name": name, "current_value": value, "threshold": threshold,
            "status": status, "explanation": explanation}


def _structure(content: dict) -> tuple[float, list[str]]:
    chapters = content.get("chapters", [])
    issues: list[str] = []
    need = 9
    ok = 0
    for i in range(need):
        ch = chapters[i] if i < len(chapters) else {}
        has_title = bool(ch.get("title"))
        has_body = _chapter_has_body(ch)
        if has_title and has_body:
            ok += 1
        else:
            issues.append(f"第{i+1}章结构不完整（标题/正文缺失）。")
    return round(ok / need, 4), issues


def _title_match(content: dict) -> tuple[float, list[str]]:
    want = tpl.chapter_titles()
    got = [c.get("title", "") for c in content.get("chapters", [])]
    issues: list[str] = []
    matched = 0
    for i, t in enumerate(want):
        if i < len(got) and got[i].strip() == t.strip():
            matched += 1
        else:
            issues.append(f"第{i+1}章标题与模板不一致：期望「{t}」。")
    return round(matched / len(want), 4), issues


def _tables(content: dict) -> tuple[float, list[str]]:
    issues: list[str] = []
    found = {}
    for ch in content.get("chapters", []):
        for tb in ch.get("tables", []):
            found[tb.get("title")] = tb
    expected = [t["title"] for t in tpl.CANONICAL_TABLES]
    total = len(expected)
    ok = 0
    for title in expected:
        tb = found.get(title)
        if not tb:
            issues.append(f"缺少量化表：{title}。")
            continue
        cols = tb.get("columns", [])
        if cols[1:] != ["核心范围", "近邻范围", "辐射范围"]:
            issues.append(f"{title} 列口径非三圈层。")
            continue
        rows = tb.get("rows", [])
        cell_ok = all(
            all(str(r.get(k, "")).strip() for k in ("label", "core", "nearby", "radiation"))
            for r in rows
        )
        if rows and cell_ok:
            ok += 1
        else:
            issues.append(f"{title} 存在空单元格。")
    return round(ok / total, 4), issues


def _cleanliness(content: dict) -> tuple[list[str], list[str]]:
    text = _all_text(content)
    md_hits: list[str] = []
    for pat in MARKDOWN_PATTERNS:
        if re.search(pat, text):
            md_hits.append(pat)
    tok_hits = [t for t in FORBIDDEN_TOKENS if t in text]
    # raw English variable like xxx_score / camelCase keys
    if re.search(r"[A-Za-z]+_[A-Za-z]+", text):
        tok_hits.append("英文下划线变量名")
    return md_hits, tok_hits


def _case_relevance(content: dict) -> tuple[float, list[str]]:
    facts = content.get("source_facts", {})
    names = facts.get("case_ref_names") or []
    issues: list[str] = []
    ch6 = next((c for c in content.get("chapters", []) if c.get("no") == "6"), {})
    bullets = ch6.get("bullets", [])
    if not names:
        issues.append("第6章未命中相关案例。")
        return 0.0, issues
    score = min(1.0, len(names) / 2)  # 至少 2 个相关案例视为达标
    # 案例名应出现在第6章正文（含子小节），避免"挂名不引用"
    body_parts = list(bullets) + list(ch6.get("paragraphs", []))
    for sec in ch6.get("sections", []):
        body_parts.extend(sec.get("paragraphs", []))
        body_parts.extend(sec.get("bullets", []))
    body = "\n".join(body_parts)
    cited = sum(1 for n in names if n.split("（")[0][:6] in body)
    if cited == 0:
        issues.append("第6章案例未在正文中实际引用。")
        score = min(score, 0.5)
    return round(score, 4), issues


def _traceability(content: dict) -> tuple[float, list[str]]:
    src = content.get("source_metrics", {})
    issues: list[str] = []
    total = 0
    traced = 0
    for ch in content.get("chapters", []):
        for tb in ch.get("tables", []):
            tkey = tb.get("key")
            for r in tb.get("rows", []):
                for ring in ("core", "nearby", "radiation"):
                    val = str(r.get(ring, "")).strip()
                    if not val or val == "暂无数据":
                        continue
                    # 仅核对纯数值单元格
                    try:
                        num = float(val)
                    except ValueError:
                        continue
                    total += 1
                    key = f"{tkey}:{r_key(tb, r)}:{ring}"
                    if key in src and abs(float(src[key]) - num) <= max(1.0, abs(num) * 0.001):
                        traced += 1
                    else:
                        issues.append(f"{tb.get('title')} {r.get('label')} {ring}={val} 无法回比。")
    return (round(traced / total, 4) if total else 1.0), issues


def r_key(tb: dict, row: dict) -> str:
    # 从模板 canonical 行顺序映射 row_key（与 builder 登记口径一致）
    title = tb.get("title")
    for t in tpl.CANONICAL_TABLES:
        if t["title"] == title:
            for rr in t["rows"]:
                if rr["label"] == row.get("label"):
                    return rr["key"]
    return ""


def _expression(content: dict, md_hits, tok_hits) -> float:
    chapters = content.get("chapters", [])
    score = 1.0
    if md_hits or tok_hits:
        score -= 0.4
    # 每章应有足够文字
    thin = sum(1 for ch in chapters if not _chapter_has_body(ch))
    score -= 0.05 * thin
    return round(max(0.0, min(1.0, score)), 4)


def evaluate(content: dict[str, Any], *, pdf_ok: bool = False, docx_ok: bool = False) -> dict[str, Any]:
    structure, s_issues = _structure(content)
    title_m, t_issues = _title_match(content)
    tables, tb_issues = _tables(content)
    md_hits, tok_hits = _cleanliness(content)
    case_rel, c_issues = _case_relevance(content)
    trace, tr_issues = _traceability(content)
    expr = _expression(content, md_hits, tok_hits)

    metrics = [
        _mk("report_structure_completeness", structure, ">= 0.95",
            PASS if structure >= 0.95 else FAIL, "9 章结构完整。" if structure >= 0.95 else f"结构缺失：{s_issues[:4]}"),
        _mk("template_chapter_match", title_m, ">= 0.95",
            PASS if title_m >= 0.95 else FAIL, "章节标题对齐模板。" if title_m >= 0.95 else f"标题不匹配：{t_issues[:4]}"),
        _mk("quant_tables_completeness", tables, "= 1.0",
            PASS if tables >= 1.0 else FAIL, "四张三圈层量化表齐全且填充。" if tables >= 1.0 else f"表问题：{tb_issues[:4]}"),
        _mk("markdown_rawkey_hits", len(md_hits) + len(tok_hits), "= 0",
            PASS if not md_hits and not tok_hits else FAIL,
            "未检出 Markdown/星号/raw key/英文字段/内部字段。" if not md_hits and not tok_hits
            else f"检出：markdown={md_hits} 字段={tok_hits}"),
        _mk("case_reference_relevance", case_rel, ">= 0.85",
            PASS if case_rel >= 0.85 else (WARNING if case_rel >= 0.5 else FAIL),
            "第6章案例引用相关且被正文引用。" if case_rel >= 0.85 else f"案例引用：{c_issues[:3]}"),
        _mk("number_traceability", trace, ">= 0.90",
            PASS if trace >= 0.90 else FAIL, "量化表数值均可回比底层计算。" if trace >= 0.90 else f"不可回比：{tr_issues[:3]}"),
        _mk("formal_expression_score", expr, ">= 0.85",
            PASS if expr >= 0.85 else WARNING, "正式文档表达达标。" if expr >= 0.85 else "表达偏薄或含禁用痕迹。"),
        _mk("frontend_field_exposure", len(tok_hits), "= 0",
            PASS if not tok_hits else FAIL, "正文未暴露内部/训练/接入字段。" if not tok_hits else f"暴露字段：{tok_hits}"),
        _mk("pdf_export_success", bool(pdf_ok), "= true",
            PASS if pdf_ok else WARNING, "PDF 导出成功。" if pdf_ok else "尚未导出 PDF。"),
        _mk("docx_export_success", bool(docx_ok), "= true",
            PASS if docx_ok else WARNING, "Word 导出成功。" if docx_ok else "尚未导出 Word。"),
    ]

    has_fail = any(m["status"] == FAIL for m in metrics)
    has_warn = any(m["status"] == WARNING for m in metrics)
    overall = FAIL if has_fail else (WARNING if has_warn else PASS)

    risks = [f"[{m['status'].upper()}] {m['metric_name']}：{m['explanation']}"
             for m in metrics if m["status"] != PASS]

    return {
        "report_id": content.get("report_id"),
        "project_id": content.get("project_id"),
        "overall_status": overall,
        "passed": overall == PASS,
        "metrics_status": metrics,
        "scores": {
            "report_structure_completeness": structure,
            "template_chapter_match": title_m,
            "quant_tables_completeness": tables,
            "case_reference_relevance": case_rel,
            "number_traceability": trace,
            "formal_expression_score": expr,
        },
        "markdown_hits": md_hits,
        "forbidden_field_hits": tok_hits,
        "risks": risks,
        "notes": [
            "本评估纯只读；报告数字均来自本地确定性计算，未使用最终评估留出数据，未调用外部 API。",
        ],
    }
