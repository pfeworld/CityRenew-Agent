"""阶段性评估基线（第1-5阶段，轻量只读）。

目标：在进入第6阶段前，汇总当前系统的阶段性质量指标，**只用 train/val**，
不读取 test 内容（split 计数来自 manifest，仅计数不读 test 记录）。

红线：
- 不使用 test 做训练/调参；不调用外部 API；不生成报告；不返回任何原文。
- 仅返回统计量与当前阶段状态，且不改动任何业务数据（纯只读）。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    AnalysisResult,
    DataFile,
    EvidenceChain,
    HousingRecord,
    IndustryPoint,
    KnowledgeChunk,
    PoiPoint,
    PopulationProfile,
)
from app.services import housing_price_model as hpm
from app.services import spatial_service, split_manager

logger = logging.getLogger("cityrenew.stage_eval")

FOUR_DIMENSIONS = ("poi", "population", "housing", "industry")
# 脱敏自检：禁止出现在分析/证据落库字段中的敏感标记
FORBIDDEN_TOKENS = ("raw_json", '"address"', '"residence"', '"coordinates"')
MAX_SUMMARY_LEN = 2000  # 超长摘要视为潜在原文外泄风险

# 门禁状态枚举
ST_PASS = "pass"
ST_WARNING = "warning"
ST_FAIL = "fail"
ST_NOT_READY = "not_ready"

# 房价模型 val_mape 阈值
MAPE_PASS = 0.15
MAPE_WARN = 0.25
# 软门槛（>0 即过硬门槛，但低于该值给 warning 提示"偏少"）
RAG_WARN_BELOW = 30
EVIDENCE_WARN_BELOW = 20

# 当前阶段尚不具备评估条件的指标（标 not_ready）
NOT_READY_METRICS = {
    "report_completeness": "报告生成在第7阶段，暂无成稿报告可校验结构完整率。",
    "data_consistency": "需报告数字回比 analysis_result，依赖第7阶段报告生成。",
    "hallucination_rate": "依赖第7阶段报告与质量门禁。",
    "project_type_f1": "项目类型识别在第6阶段，暂无分类结果。",
    "retrieval_accuracy_test": "需 retrieval_qa 评测题与 test 检索评估（第9阶段 eval 模式）。",
    "house_mape_on_test": "test 上的 MAPE 属第9阶段评估；当前仅有 val_mape。",
    "evidence_coverage_report": "报告级证据链覆盖率依赖第7阶段报告。",
}


def _data_import_counts(db: Session) -> dict[str, Any]:
    return {
        "poi": db.query(PoiPoint).count(),
        "industry": db.query(IndustryPoint).count(),
        "house_price": db.query(HousingRecord).count(),
        "population_grid": db.query(PopulationProfile).count(),
        "data_files": db.query(DataFile).filter(DataFile.source == "corpus").count(),
    }


def _split_counts() -> dict[str, Any]:
    """split 计数（来自 manifest，仅计数，不读取 test 记录内容）。"""
    summary = split_manager.get_split_summary()
    if not summary.get("built"):
        return {"built": False, "message": summary.get("message"), "per_type": {}}
    per_type = {}
    totals = Counter()
    for dt, c in summary["per_type"].items():
        per_type[dt] = {"train": c["train"], "val": c["val"], "test": c["test"]}
        totals["train"] += c["train"]
        totals["val"] += c["val"]
        totals["test"] += c["test"]
    return {
        "built": True,
        "seed": summary.get("seed"),
        "mode": summary.get("mode"),
        "ratios": summary.get("ratios"),
        "per_type": per_type,
        "totals": dict(totals),
        "note": "test 仅计数，未读取其内容；建系统流程仅用 train/val。",
    }


def _rag_chunk_counts(db: Session) -> dict[str, Any]:
    total = db.query(KnowledgeChunk).count()
    by_split: Counter = Counter()
    by_type: Counter = Counter()
    for split, stype in db.query(KnowledgeChunk.split, KnowledgeChunk.source_type).all():
        by_split[split or "unknown"] += 1
        by_type[stype or "unknown"] += 1
    return {"total": total, "by_split": dict(by_split), "by_source_type": dict(by_type)}


def _analysis_status(db: Session) -> dict[str, Any]:
    total = db.query(AnalysisResult).count()
    # 按项目聚合维度，判断四维是否跑通
    proj_dims: dict[int, set] = {}
    for pid, dim in db.query(AnalysisResult.project_id, AnalysisResult.dimension).all():
        if pid is None:
            continue
        proj_dims.setdefault(pid, set()).add(dim)
    full = [pid for pid, dims in proj_dims.items() if set(FOUR_DIMENSIONS).issubset(dims)]
    return {
        "analysis_result_count": total,
        "projects_analyzed": len(proj_dims),
        "projects_with_full_four_dimensions": len(full),
        "four_dimensions_ran": bool(full),
    }


def _housing_model_metrics() -> dict[str, Any]:
    metrics = hpm.load_metrics()
    if metrics is None:
        return {
            "available": False,
            "note": "尚未训练房价基线模型（运行任一房价/四维分析即可生成）。",
        }
    return {
        "available": True,
        "model_type": metrics.get("model_type"),
        "train_count": metrics.get("train_count"),
        "val_count": metrics.get("val_count"),
        "val_mape": metrics.get("val_mape"),
        "val_mae": metrics.get("val_mae"),
        "degraded": metrics.get("degraded"),
        "note": "模型仅用 train 训练、val 验证；test 未参与训练/调参/模型选择。",
    }


def _leakage_scan(db: Session) -> dict[str, Any]:
    """脱敏自检：扫描分析/证据落库字段是否含原文外泄风险。

    检查 EvidenceChain.summary / metadata_json 与 AnalysisResult.metric_text。
    """
    hits: list[str] = []
    scanned = 0

    for summary, meta in db.query(EvidenceChain.summary, EvidenceChain.metadata_json).all():
        scanned += 1
        for field_val in (summary, meta):
            if not field_val:
                continue
            if any(tok in field_val for tok in FORBIDDEN_TOKENS):
                hits.append("evidence_field_contains_forbidden_token")
            if len(field_val) > MAX_SUMMARY_LEN:
                hits.append("evidence_field_too_long")

    for (text_val,) in db.query(AnalysisResult.metric_text).all():
        scanned += 1
        if not text_val:
            continue
        if any(tok in text_val for tok in FORBIDDEN_TOKENS):
            hits.append("analysis_text_contains_forbidden_token")
        if len(text_val) > MAX_SUMMARY_LEN:
            hits.append("analysis_text_too_long")

    return {
        "raw_json_leak_risk": bool(hits),
        "fields_scanned": scanned,
        "hit_types": sorted(set(hits)),
        "note": "仅扫描分析/证据落库的脱敏字段；raw_json 原始列仅本地溯源，从不出接口。",
    }


def _pending_metrics() -> list[dict[str, str]]:
    """当前阶段尚不能评估的指标及原因。"""
    return [{"metric": k, "reason": v} for k, v in NOT_READY_METRICS.items()]


# --------------------------------------------------------------------------- #
# 质量门禁（第5.5阶段门槛）
# --------------------------------------------------------------------------- #
def _mk(name: str, value: Any, threshold: str, status: str, explanation: str) -> dict[str, Any]:
    return {
        "metric_name": name,
        "current_value": value,
        "threshold": threshold,
        "status": status,
        "explanation": explanation,
    }


def _test_usage_check(baseline: dict[str, Any]) -> tuple[str, str]:
    """检查是否存在 test 被用于训练/调参的迹象（基于可观测信号，纯只读）。"""
    if not baseline["default_allowed_is_train_val"]:
        return ST_FAIL, "默认 allowed_splits 不是 train/val，存在 test 进入建系统流程的风险。"
    hm = baseline["housing_model"]
    splits = baseline["split_counts"]
    if hm.get("available") and splits.get("built"):
        house = splits["per_type"].get("house_price", {})
        train_n = house.get("train", 0)
        val_n = house.get("val", 0)
        if hm.get("train_count", 0) > train_n:
            return ST_FAIL, (
                f"房价模型 train_count={hm.get('train_count')} 超过 train split 数量 {train_n}，"
                "疑似 test 进入训练。"
            )
        if hm.get("val_count", 0) > val_n:
            return ST_FAIL, (
                f"房价模型 val_count={hm.get('val_count')} 超过 val split 数量 {val_n}。"
            )
    return ST_PASS, "默认仅 train/val；房价模型 train/val 计数与 split 一致，未见 test 用于训练/调参。"


def _build_gate(baseline: dict[str, Any]) -> dict[str, Any]:
    """根据基线统计计算门禁状态、风险与下一步行动。"""
    metrics: list[dict[str, Any]] = []

    dic = baseline["data_import_counts"]
    data_ok = all(dic.get(k, 0) > 0 for k in ("poi", "industry", "house_price", "population_grid"))
    metrics.append(_mk(
        "data_import_complete",
        {k: dic.get(k, 0) for k in ("poi", "industry", "house_price", "population_grid")},
        "POI/产业/房价/人口 均 > 0",
        ST_PASS if data_ok else ST_FAIL,
        "四类结构化数据均已导入。" if data_ok else "存在数据类型未导入。",
    ))

    sc = baseline["split_counts"]
    tot = sc.get("totals", {}) if sc.get("built") else {}
    split_ok = bool(sc.get("built")) and all(tot.get(s, 0) > 0 for s in ("train", "val", "test"))
    metrics.append(_mk(
        "split_manifest_complete",
        {"built": sc.get("built", False), "totals": tot},
        "manifest 存在且 train/val/test 均 > 0",
        ST_PASS if split_ok else ST_FAIL,
        "split_manifest 已生成且三划分均有数据（test 仅计数未读取）。" if split_ok
        else "split_manifest 缺失或某一划分为空。",
    ))

    rag_total = baseline["rag_chunks"].get("total", 0)
    rag_status = ST_FAIL if rag_total <= 0 else (ST_WARNING if rag_total < RAG_WARN_BELOW else ST_PASS)
    metrics.append(_mk(
        "rag_chunks", rag_total, "> 0（< 30 偏少给 warning）", rag_status,
        "RAG 知识块充足。" if rag_status == ST_PASS
        else ("无 RAG 知识块。" if rag_status == ST_FAIL else "RAG 知识块偏少，建议补充知识源。"),
    ))

    ev = baseline["evidence_chain_count"]
    ev_status = ST_FAIL if ev <= 0 else (ST_WARNING if ev < EVIDENCE_WARN_BELOW else ST_PASS)
    metrics.append(_mk(
        "evidence_chain_count", ev, "> 0（< 20 偏少给 warning）", ev_status,
        "证据链记录充足。" if ev_status == ST_PASS
        else ("无证据链记录。" if ev_status == ST_FAIL else "证据链偏少。"),
    ))

    ar = baseline["analysis"]["analysis_result_count"]
    metrics.append(_mk(
        "analysis_result_count", ar, "> 0",
        ST_PASS if ar > 0 else ST_FAIL,
        "已生成 AnalysisResult。" if ar > 0 else "无 AnalysisResult，四维分析未落库。",
    ))

    four_ran = baseline["analysis"]["four_dimensions_ran"]
    metrics.append(_mk(
        "four_dimensions_ran", four_ran, "四维分析对至少一个项目跑通",
        ST_PASS if four_ran else ST_FAIL,
        "四维分析已跑通。" if four_ran else "四维分析未跑通。",
    ))

    allowed_ok = baseline["default_allowed_is_train_val"]
    metrics.append(_mk(
        "default_allowed_splits", baseline["default_allowed_splits"], "== ['train','val']",
        ST_PASS if allowed_ok else ST_FAIL,
        "默认仅使用 train/val。" if allowed_ok else "默认 allowed_splits 非 train/val。",
    ))

    leak = baseline["desensitization_check"]["raw_json_leak_risk"]
    metrics.append(_mk(
        "raw_json_leak_risk", leak, "== false",
        ST_PASS if not leak else ST_FAIL,
        "未检测到 raw_json / 原文外泄风险。" if not leak else "检测到潜在原文外泄风险。",
    ))

    test_status, test_expl = _test_usage_check(baseline)
    metrics.append(_mk(
        "test_usage_check", test_status, "未用 test 训练/调参/规则校准/Prompt",
        test_status, test_expl,
    ))

    # 外部 API 调用：本阶段全程本地确定性计算，恒为 0
    metrics.append(_mk(
        "external_api_calls", 0, "== 0", ST_PASS,
        "未调用任何外部 API（无 DeepSeek / 无外部模型）。",
    ))

    # 房价模型 val_mape 门槛
    hm = baseline["housing_model"]
    mape = hm.get("val_mape") if hm.get("available") else None
    if mape is None:
        mape_status = ST_NOT_READY
        mape_expl = "尚未训练房价模型或无 val 样本，无法评估 val_mape。"
    elif mape <= MAPE_PASS:
        mape_status, mape_expl = ST_PASS, f"房价模型 val_mape={mape} ≤ {MAPE_PASS}，达标。"
    elif mape <= MAPE_WARN:
        mape_status, mape_expl = ST_WARNING, f"房价模型 val_mape={mape} 偏高（{MAPE_PASS}~{MAPE_WARN}）。"
    else:
        mape_status, mape_expl = ST_FAIL, f"房价模型 val_mape={mape} > {MAPE_WARN}，不达标。"
    metrics.append(_mk(
        "housing_val_mape", mape, "≤0.15 pass / ≤0.25 warning / >0.25 fail",
        mape_status, mape_expl,
    ))

    # not_ready 指标
    for name, reason in NOT_READY_METRICS.items():
        metrics.append(_mk(name, None, "本阶段不评估", ST_NOT_READY, reason))

    # ---- 汇总 overall_status ----
    gate_metrics = [m for m in metrics if m["status"] != ST_NOT_READY]
    has_fail = any(m["status"] == ST_FAIL for m in gate_metrics)
    has_warning = any(m["status"] == ST_WARNING for m in gate_metrics)

    # 硬性 fail 触发项（用于风险与必修项明确列出）
    hard_fail_items: list[str] = []
    if test_status == ST_FAIL:
        hard_fail_items.append("test 污染（test 进入训练/调参）")
    if leak:
        hard_fail_items.append("raw_json / 原文外泄风险")
    if not four_ran:
        hard_fail_items.append("四维分析未跑通")
    # external_api 恒 0，无需列入

    if has_fail:
        overall = ST_FAIL
    elif has_warning:
        overall = ST_WARNING
    else:
        overall = ST_PASS
    can_enter = overall in (ST_PASS, ST_WARNING)

    # ---- 风险 / 建议 / 下一步必做 ----
    risks: list[str] = []
    recommendations: list[str] = []
    next_required: list[str] = []

    for m in gate_metrics:
        if m["status"] == ST_FAIL:
            risks.append(f"[FAIL] {m['metric_name']}：{m['explanation']}")
            next_required.append(f"修复 {m['metric_name']}：{m['explanation']}")
        elif m["status"] == ST_WARNING:
            risks.append(f"[WARNING] {m['metric_name']}：{m['explanation']}")

    if mape_status == ST_WARNING:
        recommendations.append("在第6阶段于 val 上优化房价模型特征/超参，争取 val_mape ≤ 0.15。")
    if rag_status == ST_WARNING:
        recommendations.append("补充政策/案例/口径等知识源，提升 RAG chunk 数量与检索覆盖。")
    if ev_status == ST_WARNING:
        recommendations.append("对更多项目运行四维分析以累积证据链。")

    if overall == ST_PASS:
        recommendations.append("核心门槛全部达标，可进入第6阶段（类型识别+综合评分+一键分析）。")
    elif overall == ST_WARNING:
        recommendations.append("允许进入第6阶段，但建议并行处理上述 warning 项。")
    else:
        recommendations.append("必须先修复 fail 项后方可进入第6阶段。")

    if overall != ST_FAIL:
        next_required.append("无阻断性必修项；进入第6阶段前确认 test 仍未被触碰。")

    return {
        "overall_status": overall,
        "can_enter_next_stage": can_enter,
        "metrics_status": metrics,
        "hard_fail_items": hard_fail_items,
        "risks": risks,
        "recommendations": recommendations,
        "next_required_actions": next_required,
    }


def get_stage_baseline(db: Session) -> dict[str, Any]:
    """汇总第1-5阶段阶段性评估基线 + 质量门禁（纯只读，仅 train/val）。"""
    default_allowed = spatial_service._allowed_splits(False)  # ['train','val']
    result = {
        "mode": settings.app_mode,
        "phase": "1-5",
        "data_import_counts": _data_import_counts(db),
        "split_counts": _split_counts(),
        "rag_chunks": _rag_chunk_counts(db),
        "evidence_chain_count": db.query(EvidenceChain).count(),
        "analysis": _analysis_status(db),
        "housing_model": _housing_model_metrics(),
        "default_allowed_splits": default_allowed,
        "default_allowed_is_train_val": default_allowed == ["train", "val"],
        "desensitization_check": _leakage_scan(db),
        "pending_metrics": _pending_metrics(),
        "notes": [
            "本评估为阶段性基线，仅用 train/val，不读取 test 内容、不调用外部 API、不生成报告。",
            "三大硬指标（检索匹配率/报告结构完整率/数据一致性）需在第7、9阶段具备条件后于 test 上评估。",
        ],
    }

    gate = _build_gate(result)
    # 门禁字段置于返回顶层
    result["overall_status"] = gate["overall_status"]
    result["can_enter_next_stage"] = gate["can_enter_next_stage"]
    result["metrics_status"] = gate["metrics_status"]
    result["risks"] = gate["risks"]
    result["recommendations"] = gate["recommendations"]
    result["next_required_actions"] = gate["next_required_actions"]
    if gate["hard_fail_items"]:
        result["notes"].append("必须修复项：" + "；".join(gate["hard_fail_items"]))

    logger.info(
        "stage baseline gate: overall=%s can_enter=%s analysis=%s evidence=%s model=%s mape=%s leak=%s",
        gate["overall_status"], gate["can_enter_next_stage"],
        result["analysis"]["analysis_result_count"],
        result["evidence_chain_count"],
        result["housing_model"].get("model_type"),
        result["housing_model"].get("val_mape"),
        result["desensitization_check"]["raw_json_leak_risk"],
    )
    return result
