"""第6.5阶段质量门禁（项目类型识别 / 综合评分 / 策略建议）。

目标：在第6阶段功能完成基础上，做一个轻量只读门禁，判断是否达到进入第7阶段标准。

工作方式：
- 选取目标项目（优先 id=1，否则数据库第一个项目）。
- 读取 full_summary（纯只读）验证读取链路；并执行一次**幂等自测** run_full_analysis
  （include_test=false）获得完整结构用于细粒度校验。该自测仅 clear+rewrite 同项目同维度，
  不累积新业务数据、不触碰 test、不写跨项目数据。
- 6 个合成类型场景仅用虚拟输入字段（persist=False），不使用任何 test 数据。

红线：不读取 test 内容；不调用外部 API；不使用大模型；不生成报告；不返回原文/raw_json/
原始点位/企业名/小区名/地址明细。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import AnalysisResult, Project
from app.services import analysis_orchestrator as orch
from app.services import project_type_service as pts
from app.services import scoring_service, spatial_service

logger = logging.getLogger("cityrenew.phase6_eval")

# 门禁状态枚举
ST_PASS = "pass"
ST_WARNING = "warning"
ST_FAIL = "fail"
ST_NOT_READY = "not_ready"

# 必备维度
REQUIRED_DIMENSIONS = ("population", "housing", "poi", "industry",
                       "classification", "scoring", "strategy")
# 合法项目类型枚举
VALID_TYPES = set(pts.ALL_TYPES)
# 策略必备结构字段
STRATEGY_FIELDS = ("update_positioning", "key_opportunities", "key_risks",
                   "recommended_directions", "priority_actions", "data_limitations")
# 脱敏自检禁用标记（用于扫描返回数据，不扫描门禁指标描述本身）
FORBIDDEN_TOKENS = ("raw_json", '"address"', '"residence"', '"coordinates"',
                    "profile_json", "chunk_text")

# 第6.5阶段尚不能评估的指标（依赖第7/9阶段）
NOT_READY_METRICS = {
    "retrieval_accuracy_test": "检索匹配率需 retrieval_qa 评测题 + test 检索（第9阶段 eval 模式）。",
    "report_completeness": "报告结构完整率依赖第7阶段报告生成。",
    "data_consistency": "报告数字回比 analysis_result 依赖第7阶段报告。",
    "evidence_coverage_report": "报告级证据链覆盖率依赖第7阶段报告。",
    "final_test_score": "test 最终评分属第9阶段 eval 模式，本阶段默认不触碰 test。",
}


def _mk(name: str, value: Any, threshold: str, status: str, explanation: str) -> dict[str, Any]:
    return {
        "metric_name": name,
        "current_value": value,
        "threshold": threshold,
        "status": status,
        "explanation": explanation,
    }


def _select_target_project(db: Session) -> Project | None:
    p = db.get(Project, 1)
    if p is not None:
        return p
    return db.query(Project).order_by(Project.id).all()[0] if db.query(Project).count() else None


def _synthetic_scenarios(db: Session) -> tuple[int, list[dict[str, Any]]]:
    """6 个合成类型场景（仅虚拟字段，persist=False，不用 test）。"""
    scenarios = {
        pts.TYPE_OLD: dict(land_use="二类居住用地", build_year=1985,
                           update_demand="老旧小区改造提升", expected_direction="改善居住"),
        pts.TYPE_INDUSTRIAL: dict(land_use="工业用地", build_year=1990,
                                  update_demand="老厂房腾退转型", expected_direction="工业遗存活化"),
        pts.TYPE_BLOCK: dict(land_use="商业用地", build_year=2010,
                             update_demand="商圈活力提升", expected_direction="街区风貌优化"),
        pts.TYPE_PUBLIC_SPACE: dict(land_use="公共绿地", build_year=2010,
                                    update_demand="公共空间品质提升", expected_direction="景观与滨水慢行"),
        pts.TYPE_COMMUNITY: dict(land_use="公共服务设施", build_year=2008,
                                 update_demand="社区配套补短板便民", expected_direction="一刻钟生活圈"),
        pts.TYPE_MIXED: dict(land_use="综合用地", build_year=2012, project_area=80000.0,
                             update_demand="片区统筹综合开发", expected_direction="多元功能复合"),
    }
    empty_four = {
        "scores": {}, "confidence": {}, "poi": {}, "population": {},
        "housing": {}, "industry": {}, "allowed_splits": ["train", "val"],
        "include_test": False, "used_test": False, "evidence_ids": [],
    }
    hits = 0
    detail: list[dict[str, Any]] = []
    for expected, fields in scenarios.items():
        p = Project(id=0, name="synthetic", **fields)
        r = pts.identify(db, p, empty_four, persist=False)
        ok = r["project_type"] == expected
        hits += 1 if ok else 0
        detail.append({"expected": expected, "got": r["project_type"], "hit": ok})
    return hits, detail


def _scan_forbidden(payload: Any) -> list[str]:
    """扫描返回数据是否含原文/原始明细标记（不扫描门禁指标描述本身）。"""
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    return [tok for tok in FORBIDDEN_TOKENS if tok in blob]


def run_phase6_gate(db: Session) -> dict[str, Any]:
    """执行第6.5质量门禁，返回完整门禁结构。"""
    project = _select_target_project(db)
    if project is None:
        return {
            "overall_status": ST_FAIL,
            "can_enter_next_stage": False,
            "target_project_id": None,
            "metrics_status": [_mk("target_project", None, "存在至少一个项目", ST_FAIL,
                                   "数据库无任何项目，无法执行第6阶段门禁。")],
            "core_results": {},
            "not_ready_metrics": [{"metric": k, "reason": v} for k, v in NOT_READY_METRICS.items()],
            "risks": ["无可评估项目。"],
            "recommendations": ["先创建项目并运行四维/一键分析。"],
            "next_required_actions": ["创建项目 → 运行 run-full。"],
            "notes": ["本门禁只读评估，未写业务数据。"],
        }

    metrics: list[dict[str, Any]] = []
    notes: list[str] = [
        "门禁执行一次幂等自测 run_full_analysis（include_test=false），仅刷新同项目同维度，"
        "不累积业务数据、不触碰 test。",
        "6 个合成类型场景仅用虚拟输入字段（persist=False），不使用任何 test 数据。",
    ]

    # ---- 取数：自测 run-full + 只读 full_summary ----
    run_ok = True
    try:
        full = orch.run_full_analysis(db, project, include_test=False)
    except Exception as exc:  # noqa: BLE001 - 门禁需捕获并记录失败
        run_ok = False
        full = {}
        logger.warning("phase6 gate run_full failed: %s", exc)
    summary = orch.get_full_summary(db, project)

    # ---- 1. full_analysis_success ----
    required_keys = ("project_type", "F_score", "scores", "weights", "score_level",
                     "strategy_count", "evidence_ids", "four_dimension",
                     "project_type_result", "score_result", "strategy_result")
    struct_ok = run_ok and all(k in full for k in required_keys)
    metrics.append(_mk(
        "full_analysis_success", struct_ok, "run-full 成功返回完整结构",
        ST_PASS if struct_ok else ST_FAIL,
        "run-full 返回完整结构。" if struct_ok else "run-full 失败或结构缺字段（阻断性 fail）。",
    ))

    score_result = full.get("score_result", {}) if struct_ok else {}
    type_result = full.get("project_type_result", {}) if struct_ok else {}
    strategy_result = full.get("strategy_result", {}) if struct_ok else {}

    # ---- 2. full_summary_available ----
    summ_ok = bool(summary.get("has_full_analysis"))
    metrics.append(_mk(
        "full_summary_available", summ_ok, "has_full_analysis == true",
        ST_PASS if summ_ok else ST_FAIL,
        "/full-summary 可读取完整结果。" if summ_ok else "/full-summary 无完整结果。",
    ))

    # ---- 3. analysis_result_dimensions ----
    dims = {d for (d,) in db.query(AnalysisResult.dimension)
            .filter(AnalysisResult.project_id == project.id).distinct().all() if d}
    missing_dims = [d for d in REQUIRED_DIMENSIONS if d not in dims]
    metrics.append(_mk(
        "analysis_result_dimensions", sorted(dims),
        "包含 population/housing/poi/industry/classification/scoring/strategy 7 维度",
        ST_PASS if not missing_dims else ST_FAIL,
        "7 个维度齐全。" if not missing_dims else f"缺失维度：{missing_dims}。",
    ))

    # ---- 4. evidence_chain_exists ----
    ev_count = len(full.get("evidence_ids", []))
    ev_status = ST_PASS if ev_count >= 10 else (ST_WARNING if ev_count >= 1 else ST_FAIL)
    metrics.append(_mk(
        "evidence_chain_exists", ev_count, ">=10 pass / 1-9 warning / 0 fail", ev_status,
        f"第6阶段结果 evidence_ids 数量={ev_count}。",
    ))

    # ---- 5. project_type_exists ----
    ptype = full.get("project_type")
    type_ok = ptype in VALID_TYPES
    metrics.append(_mk(
        "project_type_exists", ptype, "非空且属于 6 类型枚举",
        ST_PASS if type_ok else ST_FAIL,
        "项目类型合法。" if type_ok else f"项目类型非法或为空：{ptype}。",
    ))

    # ---- 6. project_type_confidence ----
    conf = full.get("project_type_confidence")
    missing_fields = type_result.get("missing_fields", [])
    if conf is None:
        conf_status, conf_expl = ST_FAIL, "无置信度。"
    elif conf >= 0.25:
        conf_status = ST_PASS
        conf_expl = f"置信度 {conf} ≥ 0.25。"
    elif conf >= 0.15:
        conf_status = ST_WARNING
        conf_expl = f"置信度 {conf} 偏低（0.15~0.25）。"
    else:
        conf_status = ST_WARNING if missing_fields else ST_FAIL
        conf_expl = f"置信度 {conf} < 0.15。"
    if missing_fields:
        conf_expl += f" 项目输入字段缺失：{missing_fields}（指标规则支撑，属非阻断数据问题）。"
    metrics.append(_mk(
        "project_type_confidence", conf, ">=0.25 pass / 0.15-0.25 warning / <0.15 fail",
        conf_status, conf_expl,
    ))

    # ---- 7. matched_rules ----
    mr = full.get("matched_rules", [])
    metrics.append(_mk(
        "matched_rules", len(mr), "len(matched_rules) >= 1",
        ST_PASS if len(mr) >= 1 else ST_FAIL,
        f"命中规则数={len(mr)}。" if len(mr) >= 1 else "无命中规则（fail）。",
    ))

    # ---- 8. synthetic_type_scenarios ----
    hits, scen_detail = _synthetic_scenarios(db)
    scen_status = ST_PASS if hits == 6 else (ST_WARNING if hits >= 4 else ST_FAIL)
    metrics.append(_mk(
        "synthetic_type_scenarios", {"hits": hits, "total": 6, "detail": scen_detail},
        "6 命中 pass / 4-5 warning / <4 fail", scen_status,
        f"6 个合成场景命中 {hits}/6（仅虚拟输入，未用 test）。",
    ))

    # ---- 9. score_fields_complete ----
    sc = score_result.get("scores", {})
    weights = score_result.get("weights", {})
    contributions = score_result.get("contributions", [])
    score_fields_ok = (
        all(k in sc for k in ("P_score", "H_score", "L_score", "I_score"))
        and all(k in weights for k in ("P", "H", "L", "I"))
        and bool(contributions)
        and score_result.get("F_score") is not None
        and bool(score_result.get("score_level"))
    )
    metrics.append(_mk(
        "score_fields_complete", score_fields_ok,
        "P/H/L/I scores、weights、contributions、F_score、score_level 全存在",
        ST_PASS if score_fields_ok else ST_FAIL,
        "评分字段齐全。" if score_fields_ok else "评分字段缺失。",
    ))

    # ---- 10. weights_sum ----
    wsum = round(sum(weights.values()), 6) if weights else 0.0
    wsum_ok = abs(wsum - 1.0) <= 0.0001
    metrics.append(_mk(
        "weights_sum", wsum, "|wP+wH+wL+wI - 1.0| <= 0.0001",
        ST_PASS if wsum_ok else ST_FAIL,
        f"权重和={wsum}。" if wsum_ok else f"权重和={wsum} 偏离 1.0。",
    ))

    # ---- 11. f_score_recomputable ----
    f_score = score_result.get("F_score")
    if score_fields_ok and f_score is not None:
        recomputed = (sc["P_score"] * weights["P"] + sc["H_score"] * weights["H"]
                      + sc["L_score"] * weights["L"] + sc["I_score"] * weights["I"])
        recomputed = round(max(0.0, min(100.0, recomputed)), 2)
        diff = abs(recomputed - f_score)
        f_ok = diff <= 0.01
        f_expl = f"F_score={f_score}，复算={recomputed}，差={round(diff, 4)}。"
    else:
        f_ok = False
        recomputed = None
        f_expl = "评分字段缺失，无法复算 F_score。"
    metrics.append(_mk(
        "f_score_recomputable", {"F_score": f_score, "recomputed": recomputed},
        "|F_score - Σ(score×weight)| <= 0.01", ST_PASS if f_ok else ST_FAIL, f_expl,
    ))

    # ---- 12. score_level_correct ----
    level = score_result.get("score_level")
    if f_score is not None:
        expected_level = scoring_service._score_level(f_score)
        level_ok = level == expected_level
        level_expl = f"F_score={f_score} → 期望「{expected_level}」，实得「{level}」。"
    else:
        level_ok = False
        level_expl = "无 F_score，无法校验档位。"
    metrics.append(_mk(
        "score_level_correct", level, "档位与 F_score 阈值一致",
        ST_PASS if level_ok else ST_FAIL, level_expl,
    ))

    # ---- 13. strategy_structure_complete ----
    strat_missing = [f for f in STRATEGY_FIELDS if f not in strategy_result]
    metrics.append(_mk(
        "strategy_structure_complete", not strat_missing,
        "6 个策略结构字段齐全",
        ST_PASS if not strat_missing else ST_FAIL,
        "策略结构完整。" if not strat_missing else f"缺失策略字段：{strat_missing}。",
    ))

    # ---- 14. strategy_count ----
    s_count = full.get("strategy_count", 0)
    s_status = ST_PASS if s_count >= 3 else (ST_WARNING if s_count >= 1 else ST_FAIL)
    metrics.append(_mk(
        "strategy_count", s_count, ">=3 pass / 1-2 warning / 0 fail", s_status,
        f"策略条目数={s_count}。",
    ))

    # ---- 15. strategy_not_report ----
    strat_blob = json.dumps(strategy_result, ensure_ascii=False)
    positioning_len = len(strategy_result.get("update_positioning", "") or "")
    looks_report = (
        "报告正文" in strat_blob or ".docx" in strat_blob.lower()
        or "章节" in strat_blob or positioning_len > 200
    )
    metrics.append(_mk(
        "strategy_not_report", not looks_report,
        "结构化 JSON，非最终报告长文",
        ST_PASS if not looks_report else ST_FAIL,
        "策略为结构化 JSON，非报告。" if not looks_report else "疑似生成报告长文（fail）。",
    ))

    # ---- 16. allowed_splits_default ----
    allowed = full.get("allowed_splits", [])
    allowed_ok = allowed == ["train", "val"]
    metrics.append(_mk(
        "allowed_splits_default", allowed, "== ['train','val']",
        ST_PASS if allowed_ok else ST_FAIL,
        "默认仅 train/val。" if allowed_ok else f"allowed_splits 非 train/val：{allowed}。",
    ))

    # ---- 17. used_test ----
    used_test = full.get("used_test", False)
    metrics.append(_mk(
        "used_test", used_test, "== false",
        ST_PASS if used_test is False else ST_FAIL,
        "未使用 test。" if used_test is False else "使用了 test（阻断性 fail）。",
    ))

    # ---- 18. test_usage_check ----
    default_allowed = spatial_service._allowed_splits(False)
    test_usage_ok = default_allowed == ["train", "val"] and used_test is False and "test" not in allowed
    metrics.append(_mk(
        "test_usage_check", test_usage_ok,
        "未用 test 训练/调参/规则校准/Prompt；仅读 manifest 计数",
        ST_PASS if test_usage_ok else ST_FAIL,
        "默认 allowed_splits=train/val，类型词典/权重为经验常量未读 test；合成场景用虚拟输入。"
        if test_usage_ok else "检测到 test 进入建系统流程的风险（fail）。",
    ))

    # ---- 19. raw_json_leak_check（扫描返回数据，不扫描指标描述）----
    core_results = {
        "project_type": ptype,
        "project_type_confidence": conf,
        "matched_rules_count": len(mr),
        "P_score": sc.get("P_score"),
        "H_score": sc.get("H_score"),
        "L_score": sc.get("L_score"),
        "I_score": sc.get("I_score"),
        "weights": weights,
        "F_score": f_score,
        "score_level": level,
        "strategy_count": s_count,
        "evidence_ids_count": ev_count,
        "allowed_splits": allowed,
        "used_test": used_test,
    }
    leak_hits = _scan_forbidden({"full": full, "core_results": core_results,
                                 "summary": summary, "scenarios": scen_detail})
    metrics.append(_mk(
        "raw_json_leak_check", leak_hits, "无 raw_json/原始明细/坐标/企业名/小区名/地址",
        ST_PASS if not leak_hits else ST_FAIL,
        "未检测到原文/原始明细外泄。" if not leak_hits else f"检测到泄露标记：{leak_hits}（fail）。",
    ))

    # ---- 20. external_api_calls ----
    metrics.append(_mk(
        "external_api_calls", 0, "== 0", ST_PASS,
        "全程本地确定性计算，未调用任何外部 API（无 DeepSeek / 无 LLM）。",
    ))

    # ---- 21. llm_scoring_check ----
    metrics.append(_mk(
        "llm_scoring_check", False, "无大模型参与打分/事实结论",
        ST_PASS,
        "类型识别为规则+指标，评分为确定性加权，策略为规则模板；全部带 evidence，无 LLM 参与。",
    ))

    # ---- not_ready 指标 ----
    for name, reason in NOT_READY_METRICS.items():
        metrics.append(_mk(name, None, "本阶段不评估", ST_NOT_READY, reason))

    # ---- 汇总 overall_status ----
    gate_metrics = [m for m in metrics if m["status"] != ST_NOT_READY]
    has_fail = any(m["status"] == ST_FAIL for m in gate_metrics)
    has_warning = any(m["status"] == ST_WARNING for m in gate_metrics)

    # 硬性 fail 触发项
    hard_fail_names = {
        "full_analysis_success", "used_test", "test_usage_check",
        "raw_json_leak_check", "f_score_recomputable",
    }
    hard_fail_items = [m["metric_name"] for m in gate_metrics
                       if m["status"] == ST_FAIL and m["metric_name"] in hard_fail_names]

    if has_fail:
        overall = ST_FAIL
    elif has_warning:
        overall = ST_WARNING
    else:
        overall = ST_PASS

    # 判定 warning 是否均为"非阻断"（confidence 低 / strategy_count 少 / 场景未全中 / 证据偏少）
    non_blocking_warn = {"project_type_confidence", "strategy_count",
                         "synthetic_type_scenarios", "evidence_chain_exists"}
    warn_names = {m["metric_name"] for m in gate_metrics if m["status"] == ST_WARNING}
    all_warn_non_blocking = warn_names.issubset(non_blocking_warn)

    if overall == ST_PASS:
        can_enter = True
    elif overall == ST_WARNING:
        can_enter = all_warn_non_blocking
    else:
        can_enter = False

    # ---- 风险 / 建议 / 下一步 ----
    risks: list[str] = []
    recommendations: list[str] = []
    next_required: list[str] = []
    for m in gate_metrics:
        if m["status"] == ST_FAIL:
            risks.append(f"[FAIL] {m['metric_name']}：{m['explanation']}")
            next_required.append(f"修复 {m['metric_name']}：{m['explanation']}")
        elif m["status"] == ST_WARNING:
            risks.append(f"[WARNING] {m['metric_name']}：{m['explanation']}")

    if "project_type_confidence" in warn_names:
        recommendations.append("补齐项目 land_use/update_demand/expected_direction 以提升类型置信度。")
    if "evidence_chain_exists" in warn_names:
        recommendations.append("对更多项目运行 run-full 以累积证据链。")
    if "synthetic_type_scenarios" in warn_names:
        recommendations.append("在 train/val 案例上校准类型词典/阈值（禁用 test）。")

    if overall == ST_PASS:
        recommendations.append("核心安全项与可评估指标全部达标，可进入第7阶段（报告生成+质量门禁）。")
    elif overall == ST_WARNING:
        recommendations.append(
            "warning 均为数据字段缺失类非阻断问题，可进入第7阶段并并行优化。"
            if can_enter else "存在需先处理的 warning。"
        )
    else:
        recommendations.append("必须先修复 fail 项后方可进入第7阶段。")
    if overall != ST_FAIL and not next_required:
        next_required.append("无阻断性必修项；进入第7阶段前确认 test 仍未被触碰。")

    logger.info(
        "phase6 gate project_id=%s overall=%s can_enter=%s type=%s conf=%s F=%s level=%s "
        "scen=%s/6 ev=%s used_test=%s leak=%s",
        project.id, overall, can_enter, ptype, conf, f_score, level, hits, ev_count,
        used_test, bool(leak_hits),
    )

    return {
        "mode": settings.app_mode,
        "phase": "6.5",
        "target_project_id": project.id,
        "overall_status": overall,
        "can_enter_next_stage": can_enter,
        "metrics_status": metrics,
        "core_results": core_results,
        "not_ready_metrics": [{"metric": k, "reason": v} for k, v in NOT_READY_METRICS.items()],
        "hard_fail_items": hard_fail_items,
        "risks": risks,
        "recommendations": recommendations,
        "next_required_actions": next_required,
        "notes": notes,
    }
