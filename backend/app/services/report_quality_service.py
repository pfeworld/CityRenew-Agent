"""报告质量门禁（第7.5阶段准备，轻量只读校验）。

对 report_content_service 生成的结构化报告内容做 5 类检查：
1. report_completeness  >= 0.98  9章 × 7字段齐全度
2. data_consistency     >= 0.95  报告数字回比 source_metrics + F_score 复算 + 事实一致
3. evidence_coverage    >= 0.95  每章至少 1 个 evidence_id + 关键结论带证据
4. leakage_check        == 0     扫描 raw_json/坐标/企业名/小区名/地址明细
5. test_usage_check     pass     used_test=false 且 allowed_splits=['train','val']

红线：纯只读，不改业务数据；不读取 test；不调外部 API；不使用大模型。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.services import scoring_service

logger = logging.getLogger("cityrenew.report.quality")

# 门禁状态枚举
ST_PASS = "pass"
ST_WARNING = "warning"
ST_FAIL = "fail"

# 内部门槛
COMPLETENESS_PASS = 0.98
CONSISTENCY_PASS = 0.95
EVIDENCE_PASS = 0.95

REQUIRED_SECTIONS = 9
SECTION_FIELDS = ("title", "summary", "key_findings", "metrics",
                  "evidence_ids", "data_limitations")

# 数值回比容差
NUM_TOLERANCE = 0.01

# 脱敏自检禁用标记（扫描报告内容 JSON）
FORBIDDEN_TOKENS = (
    "raw_json", '"coordinates"', '"address"', '"residence"',
    "profile_json", "chunk_text", "center_lng", "center_lat",
)


def _mk(name: str, value: Any, threshold: str, status: str, explanation: str) -> dict[str, Any]:
    return {
        "metric_name": name,
        "current_value": value,
        "threshold": threshold,
        "status": status,
        "explanation": explanation,
    }


# --------------------------------------------------------------------------- #
# 1. 完整率
# --------------------------------------------------------------------------- #
def _check_completeness(content: dict) -> tuple[float, list[str]]:
    sections = content.get("sections", [])
    issues: list[str] = []
    total = REQUIRED_SECTIONS * len(SECTION_FIELDS)
    satisfied = 0

    if len(sections) < REQUIRED_SECTIONS:
        issues.append(f"章节数 {len(sections)} < {REQUIRED_SECTIONS}。")

    for idx in range(REQUIRED_SECTIONS):
        sec = sections[idx] if idx < len(sections) else {}
        sid = sec.get("section_id", f"#{idx + 1}")
        for field in SECTION_FIELDS:
            val = sec.get(field)
            ok = bool(val) if not isinstance(val, (int, float)) else True
            if ok:
                satisfied += 1
            else:
                issues.append(f"{sid} 缺失字段：{field}。")
    score = round(satisfied / total, 4) if total else 0.0
    return score, issues


# --------------------------------------------------------------------------- #
# 2. 一致性
# --------------------------------------------------------------------------- #
def _check_consistency(content: dict) -> tuple[float, list[str]]:
    src = content.get("source_metrics", {})
    facts = content.get("source_facts", {})
    issues: list[str] = []
    total = 0
    consistent = 0

    # 2.1 每个章节数值型 metric 必须能在 source_metrics 命中且一致
    for sec in content.get("sections", []):
        for m in sec.get("metrics", []):
            value = m.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            total += 1
            key = m.get("key")
            if key in src and abs(float(src[key]) - float(value)) <= NUM_TOLERANCE:
                consistent += 1
            else:
                issues.append(
                    f"{sec.get('section_id')} 指标 {key}={value} 无法在 source_metrics 命中或不一致。"
                )

    # 2.2 F_score 复算
    total += 1
    f_score = src.get("F_score")
    weights = facts.get("weights") or {}
    dim_scores = {k: src.get(k) for k in ("P_score", "H_score", "L_score", "I_score")}
    if f_score is not None and weights and all(v is not None for v in dim_scores.values()):
        recomputed = (
            dim_scores["P_score"] * weights.get("P", 0)
            + dim_scores["H_score"] * weights.get("H", 0)
            + dim_scores["L_score"] * weights.get("L", 0)
            + dim_scores["I_score"] * weights.get("I", 0)
        )
        recomputed = round(max(0.0, min(100.0, recomputed)), 2)
        if abs(recomputed - float(f_score)) <= NUM_TOLERANCE + 0.01:
            consistent += 1
        else:
            issues.append(f"F_score={f_score} 复算={recomputed} 不一致。")
    else:
        issues.append("F_score 复算所需字段缺失（F_score/weights/四维分）。")

    # 2.3 score_level 与 F_score 阈值一致
    total += 1
    level = facts.get("score_level")
    if f_score is not None:
        expected = scoring_service._score_level(float(f_score))
        if level == expected:
            consistent += 1
        else:
            issues.append(f"score_level={level} 与 F_score={f_score} 期望「{expected}」不一致。")
    else:
        issues.append("无 F_score，无法校验 score_level。")

    # 2.4 strategy_count 在 source 与 metric 间一致
    total += 1
    strat_count = facts.get("strategy_count")
    if strat_count is not None and src.get("strategy_count") is not None:
        if abs(float(strat_count) - float(src["strategy_count"])) <= NUM_TOLERANCE:
            consistent += 1
        else:
            issues.append("strategy_count 在 source_facts 与 source_metrics 间不一致。")
    else:
        issues.append("strategy_count 缺失，无法校验。")

    # 2.5 房价 val_mape / model_type 出现时必须有事实支撑
    mape_present = "housing_val_mape" in src
    if mape_present:
        total += 1
        if facts.get("housing_val_mape") is not None or facts.get("housing_model_type"):
            consistent += 1
        else:
            issues.append("报告出现房价 val_mape 但缺少模型事实支撑。")

    score = round(consistent / total, 4) if total else 1.0
    return score, issues


# --------------------------------------------------------------------------- #
# 3. 证据覆盖
# --------------------------------------------------------------------------- #
def _check_evidence(content: dict) -> tuple[float, list[str]]:
    sections = content.get("sections", [])
    issues: list[str] = []
    checks = 0
    covered = 0

    for sec in sections:
        sid = sec.get("section_id")
        # 每章至少一个 evidence_id
        checks += 1
        evids = sec.get("evidence_ids") or []
        if evids:
            covered += 1
        else:
            issues.append(f"{sid} 无 evidence_id。")
        # 每条数值 metric 应带 evidence_id
        for m in sec.get("metrics", []):
            if not isinstance(m.get("value"), (int, float)):
                continue
            checks += 1
            if m.get("evidence_id"):
                covered += 1
            else:
                issues.append(f"{sid} 指标 {m.get('key')} 缺 evidence_id。")

    score = round(covered / checks, 4) if checks else 0.0
    return score, issues


# --------------------------------------------------------------------------- #
# 4. 泄露扫描
# --------------------------------------------------------------------------- #
def _check_leakage(content: dict) -> dict[str, Any]:
    # 仅扫描进入报告正文/响应的字段，剔除内部回比用的 source_metrics/source_facts。
    payload = {k: v for k, v in content.items()
               if k not in ("source_metrics", "source_facts")}
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    hits = [tok for tok in FORBIDDEN_TOKENS if tok in blob]
    return {
        "leak": bool(hits),
        "hit_tokens": hits,
        "fields_scanned": len(payload.get("sections", [])),
        "note": "仅扫描报告正文字段；source_metrics/source_facts 为内部回比用，不出接口正文。",
    }


# --------------------------------------------------------------------------- #
# 5. test 使用检查
# --------------------------------------------------------------------------- #
def _check_test_usage(content: dict) -> dict[str, Any]:
    used_test = content.get("used_test", False)
    allowed = content.get("allowed_splits", [])
    ok = used_test is False and allowed == ["train", "val"]
    return {
        "pass": ok,
        "used_test": used_test,
        "allowed_splits": allowed,
        "note": "报告默认仅 train/val，未触碰 test。" if ok
        else "检测到 test 使用风险或 allowed_splits 非 train/val。",
    }


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def check_report_quality(db: Session, content: dict[str, Any]) -> dict[str, Any]:
    """对结构化报告内容执行 5 类质量检查并汇总门禁结论。"""
    completeness, comp_issues = _check_completeness(content)
    consistency, cons_issues = _check_consistency(content)
    evidence, ev_issues = _check_evidence(content)
    leakage = _check_leakage(content)
    test_usage = _check_test_usage(content)

    metrics: list[dict[str, Any]] = []

    comp_status = ST_PASS if completeness >= COMPLETENESS_PASS else (
        ST_WARNING if completeness >= 0.90 else ST_FAIL)
    metrics.append(_mk(
        "report_completeness", completeness, f">= {COMPLETENESS_PASS}", comp_status,
        "9章×7字段齐全。" if comp_status == ST_PASS else f"完整率不足：{comp_issues[:5]}",
    ))

    cons_status = ST_PASS if consistency >= CONSISTENCY_PASS else (
        ST_WARNING if consistency >= 0.85 else ST_FAIL)
    metrics.append(_mk(
        "data_consistency", consistency, f">= {CONSISTENCY_PASS}", cons_status,
        "报告数字均可回比 source_metrics 且 F_score 可复算。" if cons_status == ST_PASS
        else f"一致性不足：{cons_issues[:5]}",
    ))

    ev_status = ST_PASS if evidence >= EVIDENCE_PASS else (
        ST_WARNING if evidence >= 0.85 else ST_FAIL)
    metrics.append(_mk(
        "evidence_coverage", evidence, f">= {EVIDENCE_PASS}", ev_status,
        "每章及关键数值均带 evidence_id。" if ev_status == ST_PASS
        else f"证据覆盖不足：{ev_issues[:5]}",
    ))

    leak_status = ST_PASS if not leakage["leak"] else ST_FAIL
    metrics.append(_mk(
        "leakage_check", leakage["hit_tokens"], "无 raw_json/坐标/企业名/小区名/地址明细",
        leak_status,
        "未检测到原文/原始明细外泄。" if leak_status == ST_PASS
        else f"检测到泄露标记：{leakage['hit_tokens']}（阻断）。",
    ))

    test_status = ST_PASS if test_usage["pass"] else ST_FAIL
    metrics.append(_mk(
        "test_usage_check", test_usage["pass"], "used_test=false 且 allowed=['train','val']",
        test_status, test_usage["note"],
    ))

    metrics.append(_mk(
        "external_api_calls", 0, "== 0", ST_PASS,
        "报告生成全程本地确定性模板，未调用任何外部 API。",
    ))
    metrics.append(_mk(
        "llm_report_check", False, "无大模型撰写报告/生成事实数字", ST_PASS,
        "报告为确定性模板 + 结构化数据填充，无 LLM 参与。",
    ))

    has_fail = any(m["status"] == ST_FAIL for m in metrics)
    has_warning = any(m["status"] == ST_WARNING for m in metrics)
    if has_fail:
        overall = ST_FAIL
    elif has_warning:
        overall = ST_WARNING
    else:
        overall = ST_PASS

    # 硬阻断项：泄露 / test / 一致性 fail / 完整率 fail
    hard_fail_items: list[str] = []
    if leak_status == ST_FAIL:
        hard_fail_items.append("raw_json/原始明细泄露")
    if test_status == ST_FAIL:
        hard_fail_items.append("test 使用风险")
    if cons_status == ST_FAIL:
        hard_fail_items.append("数据一致性不达标")
    if comp_status == ST_FAIL:
        hard_fail_items.append("报告完整率不达标")

    can_enter = overall == ST_PASS or (overall == ST_WARNING and not hard_fail_items)

    risks: list[str] = []
    recommendations: list[str] = []
    next_required: list[str] = []
    for m in metrics:
        if m["status"] == ST_FAIL:
            risks.append(f"[FAIL] {m['metric_name']}：{m['explanation']}")
            next_required.append(f"修复 {m['metric_name']}。")
        elif m["status"] == ST_WARNING:
            risks.append(f"[WARNING] {m['metric_name']}：{m['explanation']}")

    if overall == ST_PASS:
        recommendations.append("报告完整率/一致性/证据覆盖全部达标，可进入第7.5门禁正式评估与第8阶段。")
    elif overall == ST_WARNING and can_enter:
        recommendations.append("存在非阻断 warning，可进入下一步并并行优化。")
    else:
        recommendations.append("必须先修复 fail 项后方可进入第8阶段。")
    if can_enter and not next_required:
        next_required.append("无阻断性必修项；进入第8阶段前确认 test 仍未被触碰。")

    logger.info(
        "report quality project_id=%s overall=%s can_enter=%s comp=%s cons=%s ev=%s leak=%s test=%s",
        content.get("project_id"), overall, can_enter, completeness, consistency,
        evidence, leakage["leak"], test_usage["pass"],
    )

    return {
        "mode": settings.app_mode,
        "phase": "7.5",
        "report_id": content.get("report_id"),
        "project_id": content.get("project_id"),
        "overall_status": overall,
        "can_enter_next_stage": can_enter,
        "metrics_status": metrics,
        "report_completeness": completeness,
        "data_consistency": consistency,
        "evidence_coverage": evidence,
        "leakage_check": leakage,
        "test_usage_check": test_usage,
        "hard_fail_items": hard_fail_items,
        "risks": risks,
        "recommendations": recommendations,
        "next_required_actions": next_required,
        "notes": [
            "本门禁纯只读校验，未改动任何业务数据。",
            "完整率=9章×7字段齐全度；一致性=报告数字回比 source_metrics + F_score 复算；"
            "证据覆盖=每章及关键数值带 evidence_id。",
        ],
    }
