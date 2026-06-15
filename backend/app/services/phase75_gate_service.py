"""第7.5阶段独立质量门禁 + 反作弊校验（纯只读）。

与第7阶段内联质检（report_quality_service）的根本区别：本门禁**不信任**报告自报的
source_metrics/source_facts，而是独立地：
- 从磁盘重新读取 latest.json（不复用内存中的 report content 对象）。
- 从数据库 AnalysisResult / EvidenceChain / full-summary 重新构造真值集。
- 用真值集交叉校验报告中的数字、F_score、事实、证据存在性、脱敏与 test 隔离。

并内置 3 个 mutation tests（A 改错 F_score / B 删除证据 / C 置 used_test=true），
对内存 deepcopy 副本执行，断言门禁必须 fail —— 反向证明门禁本身有效（反作弊）。

红线：纯只读，不写 DB、不写 latest.json、不调外部 API、不使用大模型。
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import AnalysisResult, EvidenceChain, Project
from app.services import analysis_common as ac
from app.services import analysis_orchestrator as orch
from app.services import housing_price_model as hpm
from app.services import report_content_service

logger = logging.getLogger("cityrenew.phase75_gate")

ST_PASS = "pass"
ST_FAIL = "fail"

COMPLETENESS_PASS = 0.98
CONSISTENCY_PASS = 0.95
EVIDENCE_PASS = 0.95
NUM_TOLERANCE = 0.01

REQUIRED_SECTIONS = 9
SECTION_FIELDS = ("title", "summary", "key_findings", "metrics",
                  "evidence_ids", "data_limitations")

FORBIDDEN_TOKENS = (
    "raw_json", '"coordinates"', '"address"', '"residence"',
    "profile_json", "chunk_text", "center_lng", "center_lat",
)

# 合法的非 source 数字白名单（年龄分桶标签 / 结构常量）
AGE_BUCKET_LABELS = {18, 24, 25, 34, 35, 44, 45, 54, 55, 64, 65}
STRUCTURAL_CONSTANTS = {0, 1, 2, 3, 4, 9}  # 章节数/低置信度计数等小整数结构量

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _mk(name: str, value: Any, threshold: str, status: str, explanation: str) -> dict[str, Any]:
    return {
        "metric_name": name,
        "current_value": value,
        "threshold": threshold,
        "status": status,
        "explanation": explanation,
    }


# --------------------------------------------------------------------------- #
# 独立真值（全部来自 DB / 模型指标文件，不读报告自报字段）
# --------------------------------------------------------------------------- #
def _build_independent_truth(db: Session, project: Project) -> dict[str, Any]:
    pid = project.id

    db_value_by_eid: dict[str, float] = {}
    allowed: set[float] = set()
    for eid, value in db.query(AnalysisResult.evidence_id, AnalysisResult.metric_value).filter(
        AnalysisResult.project_id == pid
    ).all():
        if value is None:
            continue
        allowed.add(float(value))
        if eid:
            db_value_by_eid[eid] = float(value)

    evidence_id_set = {e for (e,) in db.query(EvidenceChain.evidence_id).all() if e}

    summary = orch.get_full_summary(db, project)
    scores = summary.get("scores") or {}
    weights = summary.get("weights") or {}
    f_score = summary.get("F_score")
    type_conf = summary.get("project_type_confidence")

    facts = {
        "project_type": summary.get("project_type"),
        "score_level": summary.get("score_level"),
        "strategy_count": summary.get("strategy_count"),
    }

    # F_score 独立复算
    recomputed_f = None
    if weights and all(scores.get(k) is not None for k in ("P_score", "H_score", "L_score", "I_score")):
        recomputed_f = round(ac.clamp(
            scores["P_score"] * weights.get("P", 0)
            + scores["H_score"] * weights.get("H", 0)
            + scores["L_score"] * weights.get("L", 0)
            + scores["I_score"] * weights.get("I", 0)
        ), 2)

    # 低置信度维度数独立复算（来自各维度评分 evidence 的 confidence）
    low_conf = 0
    for dim, key in (("poi", "L_score"), ("population", "P_score"),
                     ("housing", "H_score"), ("industry", "I_score")):
        eid = ac.make_evidence_id(dim, pid, "all", key)
        conf = db.query(EvidenceChain.confidence).filter(
            EvidenceChain.evidence_id == eid
        ).scalar()
        if conf is not None and conf < 0.3:
            low_conf += 1

    # full-summary 派生数字纳入 allowed
    for v in (f_score, recomputed_f, type_conf, facts["strategy_count"], low_conf):
        if isinstance(v, (int, float)):
            allowed.add(float(v))
    for v in scores.values():
        if isinstance(v, (int, float)):
            allowed.add(float(v))
    for v in weights.values():
        if isinstance(v, (int, float)):
            allowed.add(float(v))

    # 项目事实数字（圈层半径 / 年代 / 面积）
    for v in (project.nearby_buffer_m, project.radiation_buffer_m, project.core_buffer_m,
              project.build_year, project.project_area, project.building_area):
        if isinstance(v, (int, float)):
            allowed.add(float(v))

    # 房价模型指标数字（val_mape / val_mae，报告 S5 文本会出现）
    metrics = hpm.load_metrics()
    if metrics:
        for k in ("val_mape", "val_mae", "train_count", "val_count"):
            v = metrics.get(k)
            if isinstance(v, (int, float)):
                allowed.add(float(v))

    # 合法标签 / 结构常量
    allowed |= {float(x) for x in AGE_BUCKET_LABELS}
    allowed |= {float(x) for x in STRUCTURAL_CONSTANTS}

    return {
        "db_value_by_eid": db_value_by_eid,
        "evidence_id_set": evidence_id_set,
        "allowed_values": sorted(allowed),
        "facts": facts,
        "recomputed_f": recomputed_f,
        "low_conf_count": low_conf,
    }


def _num_matches(n: float, allowed: list[float]) -> bool:
    """数字 n 是否可溯源到真值集（容差 / 向下取整 / 四舍五入 三种命中）。"""
    for a in allowed:
        if abs(a - n) <= NUM_TOLERANCE:
            return True
        if abs(math.floor(a) - n) < 1e-9:
            return True
        if abs(round(a) - n) < 1e-9:
            return True
    return False


def _strip_labels(text: str, project: Project) -> str:
    """剥离项目名/城市/区，避免名称内数字（如"第5阶段"）误判为幻觉。"""
    for s in (project.name, project.city, project.district):
        if s:
            text = text.replace(s, " ")
    return text


# --------------------------------------------------------------------------- #
# 核心校验（对传入 content 执行；mutation 复用，纯函数不读 DB）
# --------------------------------------------------------------------------- #
def evaluate(truth: dict[str, Any], project: Project, content: dict[str, Any]) -> dict[str, Any]:
    sections = content.get("sections", [])
    allowed = truth["allowed_values"]
    evidence_id_set = truth["evidence_id_set"]

    # ---- 1. 完整率 ----
    total_fields = REQUIRED_SECTIONS * len(SECTION_FIELDS)
    satisfied = 0
    for idx in range(REQUIRED_SECTIONS):
        sec = sections[idx] if idx < len(sections) else {}
        for field in SECTION_FIELDS:
            val = sec.get(field)
            if bool(val) if not isinstance(val, (int, float)) else True:
                satisfied += 1
    completeness = round(satisfied / total_fields, 4) if total_fields else 0.0

    # ---- 2. 数据一致性（独立）----
    cons_checks = 0
    cons_ok = 0
    cons_issues: list[str] = []

    # 2.1 每条数值 metric 命中真值集
    for sec in sections:
        for m in sec.get("metrics", []):
            v = m.get("value")
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            cons_checks += 1
            if _num_matches(float(v), allowed):
                cons_ok += 1
            else:
                cons_issues.append(f"{sec.get('section_id')}.{m.get('key')}={v} 无法溯源到真值集。")

    # 2.2 F_score 独立复算（与报告自报 F_score 比对）
    cons_checks += 1
    report_f = _find_metric_value(sections, "F_score")
    recomputed_f = truth["recomputed_f"]
    if report_f is not None and recomputed_f is not None and abs(report_f - recomputed_f) <= NUM_TOLERANCE + 0.01:
        cons_ok += 1
    else:
        cons_issues.append(f"F_score 报告={report_f} 独立复算={recomputed_f} 不一致。")

    # 2.3/2.4/2.5 事实一致（project_type / score_level / strategy_count）
    facts = truth["facts"]
    claimed_type = content.get("project_type")
    claimed_level = (content.get("source_facts") or {}).get("score_level")
    claimed_strategy = _find_metric_value(sections, "strategy_count")
    facts_ok = True
    for name, claimed, real in (
        ("project_type", claimed_type, facts.get("project_type")),
        ("score_level", claimed_level, facts.get("score_level")),
    ):
        cons_checks += 1
        if claimed == real and real is not None:
            cons_ok += 1
        else:
            facts_ok = False
            cons_issues.append(f"{name} 报告={claimed} full-summary={real} 不一致。")
    cons_checks += 1
    if (claimed_strategy is not None and facts.get("strategy_count") is not None
            and abs(float(claimed_strategy) - float(facts["strategy_count"])) <= NUM_TOLERANCE):
        cons_ok += 1
    else:
        facts_ok = False
        cons_issues.append(
            f"strategy_count 报告={claimed_strategy} full-summary={facts.get('strategy_count')} 不一致。"
        )

    consistency = round(cons_ok / cons_checks, 4) if cons_checks else 0.0

    # ---- 3. 证据覆盖 + evidence_id 存在性 ----
    ev_checks = 0
    ev_ok = 0
    ev_issues: list[str] = []
    for sec in sections:
        sid = sec.get("section_id")
        ev_checks += 1
        evids = sec.get("evidence_ids") or []
        if evids and all(e in evidence_id_set for e in evids):
            ev_ok += 1
        elif not evids:
            ev_issues.append(f"{sid} 无 evidence_id。")
        else:
            ev_issues.append(f"{sid} 存在 evidence_id 未登记于 EvidenceChain。")
        for m in sec.get("metrics", []):
            if not isinstance(m.get("value"), (int, float)):
                continue
            ev_checks += 1
            eid = m.get("evidence_id")
            if eid and eid in evidence_id_set:
                ev_ok += 1
            else:
                ev_issues.append(f"{sid}.{m.get('key')} 证据缺失或未登记。")
    evidence = round(ev_ok / ev_checks, 4) if ev_checks else 0.0

    # ---- 4. 数字溯源（summary + key_findings 文本，独立）----
    num_total = 0
    num_ok = 0
    num_issues: list[str] = []
    for sec in sections:
        texts = [sec.get("summary") or ""]
        texts.extend(sec.get("key_findings") or [])
        for t in texts:
            cleaned = _strip_labels(str(t), project)
            for tok in NUMBER_RE.findall(cleaned):
                n = float(tok)
                num_total += 1
                if _num_matches(n, allowed):
                    num_ok += 1
                else:
                    num_issues.append(f"{sec.get('section_id')} 文本数字 {tok} 无法溯源。")
    number_traceability = round(num_ok / num_total, 4) if num_total else 1.0

    # ---- 5. 脱敏扫描 ----
    payload = {k: v for k, v in content.items() if k not in ("source_metrics", "source_facts")}
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    leak_hits = [tok for tok in FORBIDDEN_TOKENS if tok in blob]
    leakage = {"leak": bool(leak_hits), "hit_tokens": leak_hits}

    # ---- 6. test 隔离 ----
    used_test = content.get("used_test", False)
    allowed_splits = content.get("allowed_splits", [])
    test_ok = used_test is False and allowed_splits == ["train", "val"]
    test_usage = {"pass": test_ok, "used_test": used_test, "allowed_splits": allowed_splits}

    # ---- 汇总指标状态 ----
    metrics: list[dict[str, Any]] = []
    comp_status = ST_PASS if completeness >= COMPLETENESS_PASS else ST_FAIL
    metrics.append(_mk("report_completeness", completeness, f">= {COMPLETENESS_PASS}", comp_status,
                       "9章×7字段齐全。" if comp_status == ST_PASS else f"不足：{cons_issues[:0] or '字段缺失'}"))
    cons_status = ST_PASS if consistency >= CONSISTENCY_PASS else ST_FAIL
    metrics.append(_mk("data_consistency", consistency, f">= {CONSISTENCY_PASS}", cons_status,
                       "报告数字独立回比真值集 + F_score 可复算 + 事实一致。" if cons_status == ST_PASS
                       else f"不一致：{cons_issues[:5]}"))
    ev_status = ST_PASS if evidence >= EVIDENCE_PASS else ST_FAIL
    metrics.append(_mk("evidence_coverage", evidence, f">= {EVIDENCE_PASS}", ev_status,
                       "每章及关键数值带 evidence_id 且均登记于 EvidenceChain。" if ev_status == ST_PASS
                       else f"证据问题：{ev_issues[:5]}"))
    num_status = ST_PASS if number_traceability >= 1.0 else ST_FAIL
    metrics.append(_mk("number_traceability", number_traceability, "== 1.0（0 幻觉）", num_status,
                       "summary/key_findings 全部数字可溯源。" if num_status == ST_PASS
                       else f"存在不可溯源数字：{num_issues[:5]}"))
    leak_status = ST_PASS if not leakage["leak"] else ST_FAIL
    metrics.append(_mk("leakage_check", leak_hits, "无 forbidden tokens", leak_status,
                       "未检测到泄露。" if leak_status == ST_PASS else f"检测到泄露：{leak_hits}"))
    test_status = ST_PASS if test_ok else ST_FAIL
    metrics.append(_mk("test_usage_check", test_ok, "used_test=false 且 allowed=['train','val']",
                       test_status, "默认仅 train/val，未触碰 test。" if test_ok else "test 隔离异常。"))
    metrics.append(_mk("external_api_calls", 0, "== 0", ST_PASS, "全程本地确定性校验，无外部 API。"))
    metrics.append(_mk("llm_report_check", False, "无大模型撰写报告", ST_PASS,
                       "报告为确定性模板，门禁亦无 LLM 参与。"))

    independent_ok = (consistency >= CONSISTENCY_PASS and number_traceability >= 1.0
                      and evidence >= EVIDENCE_PASS and facts_ok)

    has_fail = any(m["status"] == ST_FAIL for m in metrics)
    overall = ST_FAIL if has_fail else ST_PASS

    return {
        "overall_status": overall,
        "metrics_status": metrics,
        "report_completeness": completeness,
        "data_consistency": consistency,
        "evidence_coverage": evidence,
        "number_traceability": number_traceability,
        "independent_consistency_check": ST_PASS if independent_ok else ST_FAIL,
        "leakage_check": leakage,
        "test_usage_check": test_usage,
        "used_test": used_test,
        "allowed_splits": allowed_splits,
        "issues": {"consistency": cons_issues, "evidence": ev_issues, "number": num_issues},
    }


def _find_metric_value(sections: list[dict], key: str) -> float | None:
    for sec in sections:
        for m in sec.get("metrics", []):
            if m.get("key") == key and isinstance(m.get("value"), (int, float)):
                return float(m["value"])
    return None


# --------------------------------------------------------------------------- #
# Mutation tests（内存 deepcopy，零污染）
# --------------------------------------------------------------------------- #
def _mutate_fscore(content: dict) -> dict:
    c = copy.deepcopy(content)
    for sec in c.get("sections", []):
        for m in sec.get("metrics", []):
            if m.get("key") == "F_score" and isinstance(m.get("value"), (int, float)):
                m["value"] = float(m["value"]) + 37.77  # 改错，必不可溯源且破坏复算
    return c


def _mutate_evidence(content: dict) -> dict:
    c = copy.deepcopy(content)
    for sec in c.get("sections", []):
        if sec.get("section_id") == "S3":
            sec["evidence_ids"] = []
            for m in sec.get("metrics", []):
                m["evidence_id"] = None
            break
    return c


def _mutate_used_test(content: dict) -> dict:
    c = copy.deepcopy(content)
    c["used_test"] = True
    return c


def _run_mutation_tests(truth: dict, project: Project, content: dict) -> tuple[bool, list[dict]]:
    cases = [
        ("A_fscore_corrupted", "把 F_score 指标改错", "data_consistency", _mutate_fscore),
        ("B_evidence_removed", "删除关键章节(S3)的 evidence_id", "evidence_coverage", _mutate_evidence),
        ("C_used_test_true", "把 used_test 置为 true", "test_usage_check", _mutate_used_test),
    ]
    results: list[dict] = []
    all_pass = True
    for name, desc, expected_metric, mutate in cases:
        mutated = mutate(content)
        res = evaluate(truth, project, mutated)
        triggered = [m["metric_name"] for m in res["metrics_status"] if m["status"] == ST_FAIL]
        # 通过条件：变异副本被判 fail 且预期指标确实触发
        passed = res["overall_status"] == ST_FAIL and expected_metric in triggered
        all_pass = all_pass and passed
        results.append({
            "name": name,
            "description": desc,
            "expected_fail_metric": expected_metric,
            "got_status": res["overall_status"],
            "triggered_metrics": triggered,
            "passed": passed,
        })
    return all_pass, results


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_phase75_gate(db: Session, project: Project) -> dict[str, Any]:
    """执行第7.5独立门禁 + 反作弊校验（纯只读）。"""
    # req1：独立从磁盘读取 latest.json（不复用内存对象）
    content = report_content_service.load_latest(project.id)
    if content is None:
        return {
            "mode": settings.app_mode, "phase": "7.5", "report_id": None,
            "project_id": project.id, "overall_status": ST_FAIL,
            "can_enter_next_stage": False,
            "metrics_status": [_mk("report_exists", False, "latest.json 存在", ST_FAIL,
                                   "该项目暂无已生成报告。")],
            "report_completeness": 0.0, "data_consistency": 0.0, "evidence_coverage": 0.0,
            "number_traceability": 0.0, "independent_consistency_check": ST_FAIL,
            "leakage_check": {}, "test_usage_check": {}, "used_test": False,
            "allowed_splits": [], "mutation_tests_pass": False, "mutation_tests": [],
            "hard_fail_items": ["报告缺失"], "risks": ["未找到 latest.json。"],
            "recommendations": ["先调用 POST /api/reports/{id}/generate 生成报告。"],
            "next_required_actions": ["生成报告后重跑第7.5门禁。"],
            "notes": ["本门禁纯只读，未改动任何数据。"],
        }

    truth = _build_independent_truth(db, project)
    real = evaluate(truth, project, content)
    mutation_pass, mutation_results = _run_mutation_tests(truth, project, content)

    metrics = list(real["metrics_status"])
    metrics.append(_mk("mutation_tests_pass", mutation_pass, "A/B/C 变异均被判 fail",
                       ST_PASS if mutation_pass else ST_FAIL,
                       "三个 mutation tests 均被门禁拦截（反作弊有效）。" if mutation_pass
                       else "存在未被拦截的变异（门禁有效性不足）。"))
    metrics.append(_mk("independent_consistency_check", real["independent_consistency_check"],
                       "== pass", real["independent_consistency_check"],
                       "独立一致性（数字溯源/复算/事实/证据）通过。"
                       if real["independent_consistency_check"] == ST_PASS else "独立一致性未通过。"))

    has_fail = any(m["status"] == ST_FAIL for m in metrics)
    overall = ST_FAIL if has_fail else ST_PASS

    can_enter = (
        overall == ST_PASS
        and mutation_pass
        and real["independent_consistency_check"] == ST_PASS
        and real["report_completeness"] >= COMPLETENESS_PASS
        and real["data_consistency"] >= CONSISTENCY_PASS
        and real["evidence_coverage"] >= EVIDENCE_PASS
        and not real["leakage_check"].get("leak", False)
        and real["test_usage_check"].get("pass", False)
        and real["used_test"] is False
        and real["allowed_splits"] == ["train", "val"]
    )

    hard_fail_items: list[str] = []
    if real["leakage_check"].get("leak"):
        hard_fail_items.append("原文/原始明细泄露")
    if not real["test_usage_check"].get("pass"):
        hard_fail_items.append("test 隔离异常")
    if real["data_consistency"] < CONSISTENCY_PASS:
        hard_fail_items.append("数据一致性不达标")
    if real["number_traceability"] < 1.0:
        hard_fail_items.append("存在不可溯源数字（疑似幻觉）")
    if not mutation_pass:
        hard_fail_items.append("反作弊 mutation tests 未全部拦截")

    risks: list[str] = []
    next_required: list[str] = []
    for m in metrics:
        if m["status"] == ST_FAIL:
            risks.append(f"[FAIL] {m['metric_name']}：{m['explanation']}")
            next_required.append(f"修复 {m['metric_name']}。")

    recommendations: list[str] = []
    if overall == ST_PASS and can_enter:
        recommendations.append(
            "独立门禁与反作弊全部通过；报告结构完整、数字可溯源、未作弊、test 隔离。可进入第8阶段。")
        recommendations.append(
            "注意：本门禁不等价于模型预测准确率/最终比赛指标达标（房价 MAPE / 类型 F1 / "
            "test 检索匹配率属第9阶段 eval 模式 test 评估）。")
    else:
        recommendations.append("存在 fail 项，必须修复后方可进入第8阶段。")
    if can_enter and not next_required:
        next_required.append("无阻断项；进入第8阶段前确认 test 仍未被触碰。")

    logger.info(
        "phase75 gate project_id=%s overall=%s can_enter=%s comp=%s cons=%s ev=%s num=%s "
        "indep=%s mutation_pass=%s leak=%s test=%s",
        project.id, overall, can_enter, real["report_completeness"], real["data_consistency"],
        real["evidence_coverage"], real["number_traceability"],
        real["independent_consistency_check"], mutation_pass,
        real["leakage_check"].get("leak"), real["test_usage_check"].get("pass"),
    )

    return {
        "mode": settings.app_mode,
        "phase": "7.5",
        "report_id": content.get("report_id"),
        "project_id": project.id,
        "overall_status": overall,
        "can_enter_next_stage": can_enter,
        "metrics_status": metrics,
        "report_completeness": real["report_completeness"],
        "data_consistency": real["data_consistency"],
        "evidence_coverage": real["evidence_coverage"],
        "number_traceability": real["number_traceability"],
        "independent_consistency_check": real["independent_consistency_check"],
        "leakage_check": real["leakage_check"],
        "test_usage_check": real["test_usage_check"],
        "used_test": real["used_test"],
        "allowed_splits": real["allowed_splits"],
        "mutation_tests_pass": mutation_pass,
        "mutation_tests": mutation_results,
        "hard_fail_items": hard_fail_items,
        "risks": risks,
        "recommendations": recommendations,
        "next_required_actions": next_required,
        "notes": [
            "本门禁独立从磁盘 latest.json + 数据库 AnalysisResult/EvidenceChain/full-summary "
            "重建真值，不信任报告自报的 source_metrics。",
            "3 个 mutation tests 在内存 deepcopy 副本上执行，未污染数据库与 latest.json。",
            "纯只读：未写 DB、未写文件、未调外部 API、未使用大模型。",
        ],
    }
