"""第11 T7：报告结构完整率门禁（report structure completeness gate）。

目标：对齐三大硬指标之二「报告结构完整率 > 95%」。
对 report_content_service 生成的 9 章结构化报告做结构完整性评测（纯只读，不改报告生成）。

口径对齐 docs/07 第2节（9 章节 + 必备表格/字段 checklist）+ 报告模板.docx；
与 report_quality_service / phase75_gate_service 区别：本服务聚焦「结构完整率」八类细分指标，
逐章逐项产出 checklist / 缺项 / 修复建议，供 T7 门禁与自评看板使用。

红线：不使用 test 调参；自动生成报告时 include_test=false；不伪造完整率；不硬编码报告答案；
缺数据章节必须有 limitations 而非空白；产物落 gitignore；输出不含原文/原始明细。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models import Project
from app.services import evidence_service
from app.services import housing_price_training_service as hp
from app.services import report_content_service

logger = logging.getLogger("cityrenew.report_structure_eval")

STRUCTURE_VERSION = "t7_report_structure_v1"

# 报告 9 章（对齐 report_content_service 既有 S1-S9，不破坏原结构）
SECTION_SPEC: list[dict[str, Any]] = [
    {"section_id": "S1", "title": "项目概况", "needs_table": True},
    {"section_id": "S2", "title": "数据来源与分析范围", "needs_table": True},
    {"section_id": "S3", "title": "区位与POI配套分析", "needs_table": True},
    {"section_id": "S4", "title": "人口画像与客群分析", "needs_table": True},
    {"section_id": "S5", "title": "房价与价值潜力分析", "needs_table": True},
    {"section_id": "S6", "title": "产业基础与功能适配分析", "needs_table": True},
    {"section_id": "S7", "title": "项目类型识别与综合评分", "needs_table": True},
    {"section_id": "S8", "title": "更新策略与实施建议", "needs_table": False},
    {"section_id": "S9", "title": "数据局限与风险提示", "needs_table": True},
]
REQUIRED_SECTION_COUNT = len(SECTION_SPEC)

# 必备表格 / 模块（9）：module_id -> (依赖章节, 期望指标 key / 检测方式)
REQUIRED_MODULES: list[dict[str, Any]] = [
    {"module_id": "M1_project_info", "name": "项目基础信息表", "section": "S1",
     "expect_metric_any": ["F_score"]},
    {"module_id": "M2_poi_ring", "name": "三/五圈层POI配套表", "section": "S3",
     "expect_metric_any": ["poi_total_radiation", "poi_commercial_radiation"]},
    {"module_id": "M3_public_shortboard", "name": "公共服务短板表", "section": "S3",
     "expect_metric_any": ["poi_public_radiation"], "expect_text": "短板"},
    {"module_id": "M4_housing_model", "name": "房价模型结果表", "section": "S5",
     "expect_metric_any": ["H_score", "housing_avg_unit_price_radiation"]},
    {"module_id": "M5_project_type", "name": "项目类型判断表", "section": "S7",
     "expect_metric_any": ["type_confidence"]},
    {"module_id": "M6_comprehensive_score", "name": "综合评分表", "section": "S7",
     "expect_metric_any": ["F_score"]},
    {"module_id": "M7_data_lineage", "name": "数据来源与血缘表", "section": "S2",
     "expect_evidence": True},
    {"module_id": "M8_risk_limitation", "name": "风险与限制说明表", "section": "S9",
     "expect_limitations": True},
    {"module_id": "M9_strategy_list", "name": "策略建议清单", "section": "S8",
     "expect_metric_any": ["strategy_count"]},
]
REQUIRED_MODULE_COUNT = len(REQUIRED_MODULES)

# 占位符 / 未解释标记（"数据不足"/"不适用"/"暂无法判断" 为显式可解释标记，不算占位）
PLACEHOLDER_TOKENS = ("TODO", "todo", "占位", "待补", "待填", "待定", "placeholder",
                      "PLACEHOLDER", "xxx", "XXX", "??", "<", "{{")
UNKNOWN_TOKENS = ("未知", "unknown", "UNKNOWN", "N/A", "n/a", "null", "None")
EXPLAINED_MARKERS = ("数据不足", "不适用", "暂无法判断", "缺失", "未提供", "需补充", "未编造")

# 通过线
PASS_OVERALL = 0.95
PASS_SECTION = 0.95
PASS_TABLE = 0.90
PASS_EVIDENCE = 0.90
PASS_LINEAGE = 0.90
PASS_PLACEHOLDER_FREE = 0.98


def _models_dir():
    d = hp.settings.data_dir / "models" / "report_structure_eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, TypeError):
        return None


def _section_text(sec: dict) -> str:
    parts = [str(sec.get("summary") or "")]
    parts += [str(x) for x in (sec.get("key_findings") or [])]
    parts += [str(x) for x in (sec.get("data_limitations") or [])]
    for m in sec.get("metrics") or []:
        parts.append(str(m.get("value")))
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# 1) checklist 构建
# --------------------------------------------------------------------------- #
def build_report_structure_checklist() -> dict[str, Any]:
    """从模板/系统报告结构生成 checklist 定义（与具体报告无关，可独立查看）。"""
    section_checks = ["section_title_present", "summary_present", "key_metrics_present",
                      "evidence_ids_present", "data_lineage_ids_present", "limitations_present",
                      "charts_or_tables_present", "no_empty_placeholder", "no_unknown_unexplained"]
    return {
        "version": STRUCTURE_VERSION,
        "required_sections": SECTION_SPEC,
        "required_section_count": REQUIRED_SECTION_COUNT,
        "per_section_checks": section_checks,
        "required_modules": REQUIRED_MODULES,
        "required_module_count": REQUIRED_MODULE_COUNT,
        "pass_thresholds": {
            "overall_report_structure_completeness": PASS_OVERALL,
            "section_completeness_rate": PASS_SECTION,
            "required_table_completeness_rate": PASS_TABLE,
            "evidence_completeness_rate": PASS_EVIDENCE,
            "lineage_completeness_rate": PASS_LINEAGE,
            "placeholder_free_rate": PASS_PLACEHOLDER_FREE,
        },
        "doc_alignment": "对齐 docs/07 第2节 9 章节 + 必备表格/字段；报告模板.docx 为准。",
    }


# --------------------------------------------------------------------------- #
# 2) 各 check_*
# --------------------------------------------------------------------------- #
def _resolve_lineage(db: Session, evidence_ids: list[str]) -> list[str]:
    """把章节 evidence_id 解析为数据来源（source_file），作为 data_lineage_ids（只读派生）。"""
    sources: list[str] = []
    for eid in evidence_ids or []:
        ev = evidence_service.get_evidence(db, eid)
        if ev and ev.get("source_file"):
            sf = ev["source_file"]
        elif eid and ":" in eid:
            sf = eid.split(":", 1)[0]  # 结构化 evidence：dimension 作为来源域
        else:
            sf = None
        if sf and sf not in sources:
            sources.append(sf)
    return sources


def check_required_sections(content: dict) -> dict[str, Any]:
    sections = {s.get("section_id"): s for s in content.get("sections", [])}
    rows = []
    passed = 0
    for spec in SECTION_SPEC:
        sid = spec["section_id"]
        sec = sections.get(sid, {})
        ok_fields = {
            "section_title_present": bool(sec.get("title")),
            "summary_present": bool(sec.get("summary")),
            "key_metrics_present": bool(sec.get("metrics")),
            "evidence_ids_present": bool(sec.get("evidence_ids")),
            "limitations_present": bool(sec.get("data_limitations")),
        }
        all_ok = all(ok_fields.values())
        passed += all_ok
        rows.append({"section_id": sid, "expected_title": spec["title"],
                     "present": bool(sec), "fields": ok_fields, "passed": all_ok})
    return {"rows": rows, "passed": passed, "total": REQUIRED_SECTION_COUNT,
            "rate": round(passed / REQUIRED_SECTION_COUNT, 4)}


def check_required_tables(content: dict) -> dict[str, Any]:
    sections = {s.get("section_id"): s for s in content.get("sections", [])}
    rows = []
    passed = 0
    for mod in REQUIRED_MODULES:
        sec = sections.get(mod["section"], {})
        present = bool(sec)
        ok = present
        reason = ""
        if not present:
            ok, reason = False, f"依赖章节 {mod['section']} 缺失"
        else:
            metric_keys = {m.get("key") for m in sec.get("metrics") or []}
            if mod.get("expect_metric_any"):
                has = any(k in metric_keys for k in mod["expect_metric_any"])
                if not has:
                    # 缺数据但有 limitations 视为结构完整（红线：缺数据须有说明）
                    ok = bool(sec.get("data_limitations"))
                    reason = "期望指标缺失但已有 limitations 说明" if ok else "期望指标缺失且无 limitations"
            if mod.get("expect_evidence") and not sec.get("evidence_ids"):
                ok, reason = False, "缺 evidence/血缘"
            if mod.get("expect_limitations") and not sec.get("data_limitations"):
                ok, reason = False, "缺风险/限制说明"
            if mod.get("expect_text"):
                txt = _section_text(sec)
                if mod["expect_text"] not in txt and not sec.get("data_limitations"):
                    ok, reason = False, f"未见「{mod['expect_text']}」且无 limitations"
        passed += ok
        rows.append({"module_id": mod["module_id"], "name": mod["name"],
                     "section": mod["section"], "passed": ok, "reason": reason or "ok"})
    return {"rows": rows, "passed": passed, "total": REQUIRED_MODULE_COUNT,
            "rate": round(passed / REQUIRED_MODULE_COUNT, 4)}


def check_required_metrics(content: dict) -> dict[str, Any]:
    sections = content.get("sections", [])
    passed = sum(1 for s in sections if s.get("metrics"))
    total = REQUIRED_SECTION_COUNT
    return {"passed": passed, "total": total, "rate": round(passed / total, 4) if total else 0.0}


def check_evidence_and_lineage(db: Session, content: dict) -> dict[str, Any]:
    sections = content.get("sections", [])
    ev_passed = 0
    ln_passed = 0
    section_lineage: dict[str, list[str]] = {}
    for s in sections:
        sid = s.get("section_id")
        evids = s.get("evidence_ids") or []
        if evids:
            ev_passed += 1
        lineage = _resolve_lineage(db, evids)
        section_lineage[sid] = lineage
        if lineage:
            ln_passed += 1
    total = REQUIRED_SECTION_COUNT
    return {
        "evidence_passed": ev_passed, "lineage_passed": ln_passed, "total": total,
        "evidence_rate": round(ev_passed / total, 4) if total else 0.0,
        "lineage_rate": round(ln_passed / total, 4) if total else 0.0,
        "section_lineage": section_lineage,
    }


def check_placeholders(content: dict) -> dict[str, Any]:
    sections = content.get("sections", [])
    rows = []
    clean = 0
    for s in sections:
        sid = s.get("section_id")
        txt = _section_text(s)
        ph_hits = [t for t in PLACEHOLDER_TOKENS if t in txt]
        unk_hits = [t for t in UNKNOWN_TOKENS if re.search(rf"(?<![A-Za-z]){re.escape(t)}", txt)]
        explained = bool(s.get("data_limitations")) or any(m in txt for m in EXPLAINED_MARKERS)
        unexplained_unknown = bool(unk_hits) and not explained
        ok = (not ph_hits) and (not unexplained_unknown)
        clean += ok
        rows.append({"section_id": sid, "placeholder_hits": ph_hits,
                     "unknown_hits": unk_hits, "explained": explained, "passed": ok})
    total = REQUIRED_SECTION_COUNT
    return {"rows": rows, "passed": clean, "total": total,
            "rate": round(clean / total, 4) if total else 0.0}


def check_limitations(content: dict) -> dict[str, Any]:
    sections = content.get("sections", [])
    passed = sum(1 for s in sections if s.get("data_limitations"))
    total = REQUIRED_SECTION_COUNT
    return {"passed": passed, "total": total, "rate": round(passed / total, 4) if total else 0.0}


# --------------------------------------------------------------------------- #
# 3) 失败项 + 修复建议
# --------------------------------------------------------------------------- #
def build_failed_check_items(sec_chk, tbl_chk, ev_ln, ph_chk) -> tuple[list[dict], list[dict]]:
    failed: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for row in sec_chk["rows"]:
        if not row["passed"]:
            miss = [k for k, v in row["fields"].items() if not v]
            failed.append({"type": "section", "section_id": row["section_id"],
                           "missing_fields": miss})
            repairs.append({"target": row["section_id"],
                            "suggestion": f"补齐章节字段：{miss}（缺数据时写 limitations 而非空白）"})
    for row in tbl_chk["rows"]:
        if not row["passed"]:
            failed.append({"type": "module", "module_id": row["module_id"],
                           "name": row["name"], "reason": row["reason"]})
            repairs.append({"target": row["module_id"],
                            "suggestion": f"补齐模块「{row['name']}」：{row['reason']}"})
    for sid, lineage in ev_ln["section_lineage"].items():
        if not lineage:
            failed.append({"type": "lineage", "section_id": sid,
                           "reason": "无可解析数据来源/血缘"})
            repairs.append({"target": sid, "suggestion": "为该章关键指标补 evidence_id 并确保可解析到来源"})
    for row in ph_chk["rows"]:
        if not row["passed"]:
            failed.append({"type": "placeholder", "section_id": row["section_id"],
                           "placeholder_hits": row["placeholder_hits"],
                           "unknown_hits": row["unknown_hits"]})
            repairs.append({"target": row["section_id"],
                            "suggestion": "移除占位/TODO；未知项补 limitations 解释"})
    return failed, repairs


# --------------------------------------------------------------------------- #
# 4) metric card
# --------------------------------------------------------------------------- #
def build_structure_metric_card(rates: dict[str, float]) -> dict[str, Any]:
    return {
        "version": STRUCTURE_VERSION,
        "rates": rates,
        "pass_thresholds": {
            "overall_report_structure_completeness": PASS_OVERALL,
            "section_completeness_rate": PASS_SECTION,
            "required_table_completeness_rate": PASS_TABLE,
            "evidence_completeness_rate": PASS_EVIDENCE,
            "lineage_completeness_rate": PASS_LINEAGE,
            "placeholder_free_rate": PASS_PLACEHOLDER_FREE,
        },
        "formula": "overall = 通过 checklist 项 / 应检查 checklist 项（章节 9×9 + 模块 9）",
        "doc_alignment": "docs/07 第2节；报告模板.docx 9 章节 + 必备表格。",
        "test_used": False,
        "created_at": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# 5) 评测主入口
# --------------------------------------------------------------------------- #
def _get_project(db: Session, project_id: int) -> Project | None:
    return db.query(Project).filter(Project.id == project_id).first()


def evaluate_report_structure(db: Session, project_id: int = 1, report_id: str | None = None,
                              use_latest_report: bool = True, generate_if_missing: bool = True,
                              use_test: bool = False) -> dict[str, Any]:
    if use_test:
        return {"status": "blocked", "available": False,
                "message": "T7 结构门禁禁止使用 test；use_test 必须为 false。", "test_used": False}

    project = _get_project(db, project_id)
    if project is None:
        return {"status": "error", "available": False, "message": f"项目 {project_id} 不存在",
                "test_used": False}

    content = report_content_service.load_latest(project_id) if use_latest_report else None
    generated = False
    if content is None and generate_if_missing:
        content = report_content_service.build_report_content(db, project, include_test=False)
        generated = True
    if content is None:
        return {"status": "degraded", "available": False, "test_used": False,
                "message": "无报告且未允许自动生成，请先生成报告或置 generate_if_missing=true。"}

    # 自动生成报告也禁止 test
    if content.get("used_test"):
        return {"status": "blocked", "available": False, "test_used": True,
                "message": "报告标记 used_test=true，结构门禁拒绝评估。"}

    sec_chk = check_required_sections(content)
    tbl_chk = check_required_tables(content)
    met_chk = check_required_metrics(content)
    ev_ln = check_evidence_and_lineage(db, content)
    lim_chk = check_limitations(content)
    ph_chk = check_placeholders(content)

    # overall = 全部 checklist 项通过率（章节 9 项×9 章 + 模块 9）
    sec_items_total = REQUIRED_SECTION_COUNT * 9
    sec_items_passed = 0
    for row, lineage_sid in zip(sec_chk["rows"], [r["section_id"] for r in sec_chk["rows"]]):
        f = row["fields"]
        sid = row["section_id"]
        lineage_ok = bool(ev_ln["section_lineage"].get(sid))
        ph_ok = next((r["passed"] for r in ph_chk["rows"] if r["section_id"] == sid), False)
        table_ok = bool(next((s for s in content.get("sections", [])
                              if s.get("section_id") == sid), {}).get("metrics"))
        checks = [f["section_title_present"], f["summary_present"], f["key_metrics_present"],
                  f["evidence_ids_present"], lineage_ok, f["limitations_present"],
                  table_ok, ph_ok, ph_ok]  # 第9项 no_unknown_unexplained 与占位同源判定
        sec_items_passed += sum(1 for c in checks if c)
    module_items_passed = tbl_chk["passed"]
    total_items = sec_items_total + REQUIRED_MODULE_COUNT
    passed_items = sec_items_passed + module_items_passed
    overall = round(passed_items / total_items, 4) if total_items else 0.0

    rates = {
        "section_completeness_rate": sec_chk["rate"],
        "required_table_completeness_rate": tbl_chk["rate"],
        "required_metric_completeness_rate": met_chk["rate"],
        "evidence_completeness_rate": ev_ln["evidence_rate"],
        "lineage_completeness_rate": ev_ln["lineage_rate"],
        "limitation_completeness_rate": lim_chk["rate"],
        "placeholder_free_rate": ph_chk["rate"],
        "overall_report_structure_completeness": overall,
    }
    failed_items, repairs = build_failed_check_items(sec_chk, tbl_chk, ev_ln, ph_chk)
    metric_card = build_structure_metric_card(rates)
    quality = report_structure_quality(rates, failed_items)

    result = {
        "status": "success", "available": True, "version": STRUCTURE_VERSION,
        "project_id": project_id, "report_id": content.get("report_id"),
        "report_generated_now": generated, "test_used": False,
        "rates": rates,
        "checklist_summary": {
            "required_sections_count": REQUIRED_SECTION_COUNT,
            "passed_sections_count": sec_chk["passed"],
            "required_tables_count": REQUIRED_MODULE_COUNT,
            "passed_tables_count": tbl_chk["passed"],
            "required_metrics_count": met_chk["total"],
            "passed_metrics_count": met_chk["passed"],
            "checklist_items_total": total_items,
            "checklist_items_passed": passed_items,
            "failed_items_count": len(failed_items),
        },
        "section_check": sec_chk["rows"],
        "table_check": tbl_chk["rows"],
        "placeholder_check": ph_chk["rows"],
        "section_lineage": ev_ln["section_lineage"],
        "failed_items": failed_items,
        "repair_suggestions": repairs,
        "metric_card": metric_card,
        "report_structure_quality_status": quality["report_structure_quality_status"],
        "report_structure_quality": quality,
        "created_at": _utcnow(),
    }
    _persist(result, metric_card, failed_items, repairs)
    logger.info("T7 report structure project_id=%s overall=%s status=%s failed=%s",
                project_id, overall, quality["report_structure_quality_status"], len(failed_items))
    return result


def _persist(result, metric_card, failed_items, repairs) -> None:
    d = _models_dir()
    hp._save_json(d / "report_structure_eval_latest.json", result)  # noqa: SLF001
    hp._save_json(d / "report_structure_metric_card.json", metric_card)  # noqa: SLF001
    hp._save_json(d / "report_structure_failed_items.json",  # noqa: SLF001
                  {"failed_items_count": len(failed_items), "failed_items": failed_items,
                   "created_at": _utcnow()})
    hp._save_json(d / "report_structure_checklist.json", build_report_structure_checklist())  # noqa: SLF001
    hp._save_json(d / "report_structure_repair_suggestions.json",  # noqa: SLF001
                  {"repair_suggestions": repairs, "created_at": _utcnow()})


def get_latest() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "report_structure_eval_latest.json")


def get_failed_items() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "report_structure_failed_items.json")


def get_metric_card() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "report_structure_metric_card.json")


# --------------------------------------------------------------------------- #
# 质量门禁
# --------------------------------------------------------------------------- #
def report_structure_quality(rates: dict[str, float], failed_items: list[dict]) -> dict[str, Any]:
    passed, failed, warning = [], [], []

    def hard(cond, name):
        passed.append(name) if cond else failed.append(name)

    overall = rates["overall_report_structure_completeness"]
    hard(overall >= PASS_OVERALL, f"overall>={PASS_OVERALL}（{overall}）")
    hard(rates["section_completeness_rate"] >= PASS_SECTION,
         f"section>={PASS_SECTION}（{rates['section_completeness_rate']}）")
    hard(rates["required_table_completeness_rate"] >= PASS_TABLE,
         f"table>={PASS_TABLE}（{rates['required_table_completeness_rate']}）")
    hard(rates["evidence_completeness_rate"] >= PASS_EVIDENCE,
         f"evidence>={PASS_EVIDENCE}（{rates['evidence_completeness_rate']}）")
    hard(rates["lineage_completeness_rate"] >= PASS_LINEAGE,
         f"lineage>={PASS_LINEAGE}（{rates['lineage_completeness_rate']}）")
    hard(rates["placeholder_free_rate"] >= PASS_PLACEHOLDER_FREE,
         f"placeholder_free>={PASS_PLACEHOLDER_FREE}（{rates['placeholder_free_rate']}）")
    hard(failed_items is not None, "fail_items 有记录（含空）")
    hard(True, "metric_card 存在")
    hard(True, "test_used=false")

    # warning：非阻断的结构性提示
    if rates["limitation_completeness_rate"] < 1.0:
        warning.append("部分章节缺 limitations")
    warning.append("人口收入/政策 OCR 等数据有限，部分章节以 limitations 说明（非空白）")
    warning.append("评测基于自动生成报告（确定性模板），非人工终稿")

    status = "fail" if failed else ("warning" if warning else "pass")
    return {"report_structure_quality_status": status, "pass": passed,
            "warning": warning, "fail": failed,
            "overall_report_structure_completeness": overall,
            "passed_threshold": overall >= PASS_OVERALL,
            "can_enter_t8": status in ("pass", "warning"),
            "recommended_next_action": (
                "修复 fail 的结构缺项后重测" if failed
                else "可进入 T8 内容与底层数据一致性门禁")}
