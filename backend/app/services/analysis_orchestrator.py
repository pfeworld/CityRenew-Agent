"""分析编排（第5阶段四维 + 第6阶段类型识别/综合评分/策略/一键分析）。

第5阶段：统一运行 POI / 人口 / 房价 / 产业四维分析，汇总 P/H/L/I 维度分、核心指标、
合并证据、allowed_splits 与是否使用 test。
第6阶段：在四维结果之上做项目类型识别 → 综合评分 → 策略建议，并提供 run_full_analysis
一键流水线与 full_summary 汇总。

默认仅 train/val（include_test 默认 false）；单步接口缺四维结果时自动补跑四维（仍默认
不含 test）。本模块不调用外部 API、不使用大模型、不生成最终报告。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.models import AnalysisResult, EvidenceChain, Project
from app.services import (
    analysis_common as ac,
    housing_analysis_service,
    industry_analysis_service,
    poi_analysis_service,
    population_analysis_service,
    project_type_service,
    scoring_service,
    strategy_service,
)

logger = logging.getLogger("cityrenew.analysis.orchestrator")

# 维度 -> 维度分键
SCORE_KEY = {
    "poi": "L_score",
    "population": "P_score",
    "housing": "H_score",
    "industry": "I_score",
}


def run_four_dimension_analysis(
    db: Session, project: Project, include_test: bool = False
) -> dict[str, Any]:
    """顺序运行四维分析并汇总。"""
    poi = poi_analysis_service.analyze(db, project, include_test)
    population = population_analysis_service.analyze(db, project, include_test)
    housing = housing_analysis_service.analyze(db, project, include_test)
    industry = industry_analysis_service.analyze(db, project, include_test)

    results = {
        "poi": poi,
        "population": population,
        "housing": housing,
        "industry": industry,
    }

    scores = {
        "L_score": poi["score"],
        "P_score": population["score"],
        "H_score": housing["score"],
        "I_score": industry["score"],
    }
    confidence = {
        "L": poi["confidence"],
        "P": population["confidence"],
        "H": housing["confidence"],
        "I": industry["confidence"],
    }
    evidence_ids: list[str] = []
    notes: list[str] = []
    for dim in results.values():
        evidence_ids.extend(dim["evidence_ids"])

    allowed_splits = poi["allowed_splits"]
    notes.append("四维评分为阶段性维度分（可解释经验权重），非最终综合评分；综合评分在第6阶段。")
    if include_test:
        notes.append("本次分析按 include_test=true 纳入了 test 数据（非默认；仅用于评估场景）。")

    logger.info(
        "four-dimension analyze project_id=%s scores=%s include_test=%s",
        project.id, scores, include_test,
    )
    return {
        "project_id": project.id,
        "allowed_splits": allowed_splits,
        "include_test": include_test,
        "used_test": include_test,
        "scores": scores,
        "confidence": confidence,
        "poi": poi,
        "population": population,
        "housing": housing,
        "industry": industry,
        "evidence_ids": evidence_ids,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# 第6阶段：单步自动补跑 + 一键完整流水线
# --------------------------------------------------------------------------- #
def run_project_type(
    db: Session, project: Project, include_test: bool = False
) -> dict[str, Any]:
    """项目类型识别（自动补跑四维，默认仅 train/val）。"""
    four_dim = run_four_dimension_analysis(db, project, include_test)
    return project_type_service.identify(db, project, four_dim, persist=True)


def run_score(
    db: Session, project: Project, include_test: bool = False
) -> dict[str, Any]:
    """综合评分（自动补跑四维 + 类型识别，默认仅 train/val）。"""
    four_dim = run_four_dimension_analysis(db, project, include_test)
    type_result = project_type_service.identify(db, project, four_dim, persist=True)
    return scoring_service.score(
        db, project, four_dim, type_result["project_type"], persist=True
    )


def run_strategy(
    db: Session, project: Project, include_test: bool = False
) -> dict[str, Any]:
    """策略建议（自动补跑四维 + 类型识别 + 综合评分，默认仅 train/val）。"""
    four_dim = run_four_dimension_analysis(db, project, include_test)
    type_result = project_type_service.identify(db, project, four_dim, persist=True)
    score_result = scoring_service.score(
        db, project, four_dim, type_result["project_type"], persist=True
    )
    return strategy_service.build(
        db, project, four_dim, type_result["project_type"], score_result, persist=True
    )


def run_full_analysis(
    db: Session, project: Project, include_test: bool = False
) -> dict[str, Any]:
    """一键完整分析流水线（第6阶段）。

    顺序：四维分析 → 类型识别 → 综合评分 → 策略建议 → 落库 → 返回完整结构化结果。
    返回结构含第6.5 质量门禁可直接扫描的扁平字段（project_type/F_score/scores/weights 等）。
    """
    four_dim = run_four_dimension_analysis(db, project, include_test)
    type_result = project_type_service.identify(db, project, four_dim, persist=True)
    project_type = type_result["project_type"]

    # 自训练类型模型 + 用户文本先验（主链路优先调用训练模型；延迟导入避免循环依赖）
    renewal_type = project_type
    renewal_type_source = "rule_based"
    renewal_type_confidence = type_result.get("confidence")
    try:
        from app.services import model_inference_service as _mis
        rt = _mis._infer_renewal_type(db, project)  # noqa: SLF001
        if rt.get("label_cn"):
            renewal_type = rt["label_cn"]
            renewal_type_source = rt.get("model_source", "model")
            renewal_type_confidence = rt.get("confidence", renewal_type_confidence)
    except Exception as exc:  # noqa: BLE001
        logger.warning("模型类型推断失败，报告使用规则类型兜底：%s", type(exc).__name__)

    score_result = scoring_service.score(db, project, four_dim, project_type, persist=True)
    strategy_result = strategy_service.build(
        db, project, four_dim, project_type, score_result, persist=True
    )

    # 合并全链路证据（去重保序）
    all_evidence: list[str] = []
    for src in (four_dim["evidence_ids"], type_result["evidence_ids"],
                score_result["evidence_ids"], strategy_result["evidence_ids"]):
        for eid in src:
            if eid not in all_evidence:
                all_evidence.append(eid)

    allowed_splits = four_dim["allowed_splits"]
    used_test = four_dim["used_test"]
    notes = [
        "一键流水线：四维(确定性) → 类型识别(规则+指标) → 综合评分(加权) → 策略(规则模板)。",
        "全程未调用外部 API、未使用大模型打分/生成事实结论；仅返回脱敏指标与证据ID，不含底层原始明细。",
    ]
    if used_test:
        notes.append("本次按 include_test=true 纳入 test（非默认；仅评估场景）。")

    logger.info(
        "run_full project_id=%s type=%s conf=%.3f F_score=%s level=%s splits=%s used_test=%s ev=%d",
        project.id, project_type, type_result["confidence"], score_result["F_score"],
        score_result["score_level"], allowed_splits, used_test, len(all_evidence),
    )

    return {
        "project_id": project.id,
        # ---- 第6.5 门禁可扫描的扁平字段 ----
        "project_type": project_type,
        "renewal_type": renewal_type,
        "renewal_type_source": renewal_type_source,
        "renewal_type_confidence": renewal_type_confidence,
        "project_type_confidence": type_result["confidence"],
        "matched_rules": type_result["matched_rules"],
        "F_score": score_result["F_score"],
        "scores": score_result["scores"],
        "weights": score_result["weights"],
        "score_level": score_result["score_level"],
        "strategy_count": strategy_result["strategy_count"],
        "allowed_splits": allowed_splits,
        "include_test": include_test,
        "used_test": used_test,
        "evidence_ids": all_evidence,
        # ---- 各阶段完整结果 ----
        "four_dimension": four_dim,
        "project_type_result": type_result,
        "score_result": score_result,
        "strategy_result": strategy_result,
        "notes": notes,
    }


def get_full_summary(db: Session, project: Project) -> dict[str, Any]:
    """读取已落库的完整分析汇总（不重算；供第6.5 门禁与前端使用）。"""
    base = get_summary(db, project)

    def _latest(dimension: str, metric_key: str) -> AnalysisResult | None:
        return (
            db.query(AnalysisResult)
            .filter(
                AnalysisResult.project_id == project.id,
                AnalysisResult.dimension == dimension,
                AnalysisResult.metric_key == metric_key,
            )
            .order_by(AnalysisResult.id.desc())
            .first()
        )

    type_row = _latest("classification", "project_type")
    conf_row = _latest("classification", "type_confidence")
    f_row = _latest("scoring", "F_score")
    strat_row = _latest("strategy", "strategy_count")

    matched_rules: list[str] = []
    weights: dict[str, float] | None = None
    score_level: str | None = None
    if f_row and f_row.evidence_id:
        ev = (
            db.query(EvidenceChain)
            .filter(EvidenceChain.evidence_id == f_row.evidence_id)
            .first()
        )
        if ev and ev.metadata_json:
            meta = json.loads(ev.metadata_json)
            weights = meta.get("weights")
            score_level = meta.get("score_level")
    if type_row and type_row.evidence_id:
        ev = (
            db.query(EvidenceChain)
            .filter(EvidenceChain.evidence_id == type_row.evidence_id)
            .first()
        )
        if ev and ev.metadata_json:
            matched_rules = json.loads(ev.metadata_json).get("matched_rules", [])

    has_full = bool(type_row and f_row and strat_row)
    notes = list(base.get("notes", []))
    if not has_full:
        notes.append("尚未运行完整分析（run-full）；类型/评分/策略部分为空，请先运行 run-full。")

    return {
        "project_id": project.id,
        "project_type": type_row.metric_text if type_row else None,
        "project_type_confidence": conf_row.metric_value if conf_row else None,
        "matched_rules": matched_rules,
        "F_score": f_row.metric_value if f_row else None,
        "score_level": score_level,
        "weights": weights,
        "scores": base.get("scores", {}),
        "strategy_count": int(strat_row.metric_value) if strat_row and strat_row.metric_value is not None else None,
        "dimensions": base.get("dimensions", []),
        "total_metrics": base.get("total_metrics", 0),
        "total_evidence": base.get("total_evidence", 0),
        "has_full_analysis": has_full,
        "notes": notes,
    }


def get_summary(db: Session, project: Project) -> dict[str, Any]:
    """读取已落库的 AnalysisResult 汇总（不重算）。"""
    rows = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.project_id == project.id)
        .all()
    )
    by_dim: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    evidence_ids: set[str] = set()
    for r in rows:
        dim = r.dimension or "unknown"
        node = by_dim.setdefault(dim, {"metric_count": 0, "score": None, "evidence_ids": set()})
        node["metric_count"] += 1
        if r.evidence_id:
            node["evidence_ids"].add(r.evidence_id)
            evidence_ids.add(r.evidence_id)
        if r.metric_key == SCORE_KEY.get(dim):
            node["score"] = r.metric_value
            scores[SCORE_KEY[dim]] = r.metric_value

    dimensions = [
        {
            "dimension": dim,
            "score": node["score"],
            "metric_count": node["metric_count"],
            "evidence_count": len(node["evidence_ids"]),
        }
        for dim, node in sorted(by_dim.items())
    ]
    total_evidence = (
        db.query(EvidenceChain)
        .filter(EvidenceChain.evidence_id.in_(evidence_ids))
        .count()
        if evidence_ids
        else 0
    )
    notes = []
    if not rows:
        notes.append("该项目暂无四维分析结果，请先运行四维分析。")
    return {
        "project_id": project.id,
        "dimensions": dimensions,
        "scores": scores,
        "total_metrics": len(rows),
        "total_evidence": total_evidence,
        "notes": notes,
    }
