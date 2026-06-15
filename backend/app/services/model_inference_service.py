"""ModelInferenceService（第一阶段·真链路核心）。

把"用户项目 → 本地空间分析 + 自训练模型"串成一条可追溯的真实推理链，产出结构化
analysis_result（对齐报告 9 章口径）、evidence_map、model_run_id、model_source。

数据与结论来源（红线）：
- 数字/指标：本地四维空间分析（坐标驱动，查共享数据表）+ 房价基线模型 + 综合评分；
- 更新类型：优先调用自训练分类模型 backend/data/models/project_type/model.pkl，
  模型不可用时规则兜底，并如实标注 model_source；
- 文字成文：由上层（DeepSeek 受约束）完成，本服务只产出结构化事实，不编造；
- 缺坐标 / 超覆盖范围 / 范围内无数据 → fail-closed，返回明确状态，绝不出假结果。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models import Project
from app.services import analysis_orchestrator as ao
from app.services import feature_engineering_service as fe
from app.services import project_type_training_service as ptt
from app.services.spatial_service import SpatialError

logger = logging.getLogger("cityrenew.model.inference")

# 自训练分类模型英文标签 → 中文业务展示（前台/报告只出中文）
TYPE_LABEL_CN = {
    # 7 类统一 taxonomy（与 project_type_training_service 对齐）
    "commercial_vitality_upgrade": "商业活力提升型",
    "community_facility_upgrade": "社区配套升级型",
    "old_area_stock_renewal": "老旧片区/存量地块更新型",
    "industrial_heritage_activation": "工业遗存活化型",
    "block_quality_improvement": "街区提升型",
    "public_space_optimization": "公共空间优化型",
    "comprehensive_function_plot": "综合功能地块型",
    "uncertain": "待明确更新类型",
    # 历史英文 id 兼容（旧 model.pkl / 旧记录回显，不影响新链路）
    "public_service_improvement": "社区配套升级型",
    "residential_living_quality": "老旧片区/存量地块更新型",
    "industry_upgrade": "工业遗存活化型",
    "TOD_transport_oriented": "综合功能地块型",
    "culture_tourism_activation": "街区提升型",
    "green_open_space_improvement": "公共空间优化型",
    "comprehensive_renewal": "综合功能地块型",
    "low_efficiency_land_redevelopment": "老旧片区/存量地块更新型",
}

# 状态码（内部用；前台话术由上层映射，不直接展示）
STATUS_OK = "ok"
STATUS_MISSING_LOCATION = "missing_location"
STATUS_OUT_OF_COVERAGE = "out_of_coverage"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_ERROR = "inference_error"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ring(dim: dict, name: str) -> dict:
    for r in (dim or {}).get("rings") or []:
        if r.get("ring") == name:
            return r
    return {}


def _ev(value: Any, source_type: str, source_ref: str, confidence: Any = None,
        note: str = "") -> dict[str, Any]:
    """构造一条 evidence_map 记录（结论→来源可追溯）。"""
    return {
        "value": value,
        "source_type": source_type,   # model_analysis / local_analysis / user_input / housing_model / missing
        "source_ref": source_ref,
        "confidence": confidence,
        "note": note,
    }


def _type_cn(label: str | None) -> str:
    if not label:
        return TYPE_LABEL_CN["uncertain"]
    return TYPE_LABEL_CN.get(label, label)


# --------------------------------------------------------------------------- #
# 自训练类型模型接入（模型优先 + 规则兜底）
# --------------------------------------------------------------------------- #
# 用户文本 → 类型先验（按特异性排序；命中即采纳并标注 rule_fallback）。
# 用户对自己项目的定性最权威，应高于纯 POI 模型；POI 模型仅在用户未明确描述时使用。
_TEXT_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("industrial_heritage_activation",
     ("工业遗存", "工业厂房", "老厂房", "旧厂房", "厂区", "工业园改造", "工业用地", "锅炉房", "车间")),
    ("old_area_stock_renewal",
     ("老旧片区", "存量地块", "存量空间", "旧城", "旧区改造", "棚户", "老旧仓库", "仓储建筑",
      "老旧建筑", "历史建筑", "历史风貌", "城中村")),
    ("public_space_optimization",
     ("公共空间", "慢行系统", "绿地", "公园", "滨水", "滨江", "口袋公园", "开放空间", "步行")),
    ("comprehensive_function_plot",
     ("综合开发", "产城融合", "混合功能", "功能复合", "tod", "站城一体", "综合体", "枢纽")),
    ("community_facility_upgrade",
     ("老旧社区", "老旧小区", "居住片区", "社区配套", "一刻钟生活圈", "适老化", "公共服务",
      "便民", "民生", "养老", "社区更新")),
    ("block_quality_improvement",
     ("街区提升", "街区更新", "沿街界面", "首层界面", "风貌提升", "街道更新", "立面")),
    ("commercial_vitality_upgrade",
     ("商业活力", "商圈", "商业街区", "消费场景", "业态", "夜间经济", "商业氛围", "零售")),
]


def _text_type_prior(project: Project) -> tuple[str, str] | None:
    """从用户输入文本识别明确的项目类型先验，返回 (label, 命中关键词) 或 None。"""
    text = " ".join(str(t) for t in (
        getattr(project, "name", None), getattr(project, "description", None),
        getattr(project, "update_demand", None), getattr(project, "expected_direction", None),
        getattr(project, "land_use", None)) if t).lower()
    if not text.strip():
        return None
    best: tuple[str, int, str] | None = None
    for label, kws in _TEXT_TYPE_KEYWORDS:
        hits = [kw for kw in kws if kw.lower() in text]
        if hits and (best is None or len(hits) > best[1]):
            best = (label, len(hits), "、".join(hits[:3]))
    return (best[0], best[2]) if best else None


def _infer_renewal_type(db: Session, project: Project) -> dict[str, Any]:
    """类型判定优先级：用户文本明确描述 > 自训练模型 > 规则兜底。均如实标注 model_source。"""
    # 先确保 T2 特征存在（模型推理依赖 POI 组成特征）
    try:
        fe.build_features(db, project)
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_features 失败（类型模型将尝试已有特征/规则兜底）：%s", type(exc).__name__)

    try:
        exp = ptt.explain_project_type_prediction(db, project.id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("类型模型推理异常：%s", type(exc).__name__)
        exp = {"available": False}

    model_label = exp.get("model_assisted_type")
    rule_label = exp.get("rule_based_type")

    # 用户文本明确描述类型时优先采纳（最权威），标注为 rule_fallback 并记录命中关键词
    text_prior = _text_type_prior(project)
    if text_prior:
        t_label, matched = text_prior
        return {
            "label": t_label,
            "label_cn": _type_cn(t_label),
            "confidence": 0.7,
            "model_source": "rule_fallback",
            "model_artifact": None,
            "rule_label_cn": _type_cn(model_label or rule_label),
            "reason_codes": [f"用户输入明确指向该更新类型（命中关键词：{matched}）"],
            "top_features": [],
            "note": (f"项目类型依据用户输入文本判定（关键词：{matched}）；"
                     f"POI 模型参考判型为「{_type_cn(model_label or rule_label)}」。"),
        }

    if exp.get("available") and model_label:
        return {
            "label": model_label,
            "label_cn": _type_cn(model_label),
            "confidence": exp.get("confidence"),
            "model_source": "model",
            "model_artifact": "project_type/model.pkl",
            "rule_label_cn": _type_cn(rule_label),
            "reason_codes": exp.get("reason_codes", []),
            "top_features": exp.get("top_contributing_features", []),
        }
    # 规则兜底
    return {
        "label": rule_label,
        "label_cn": _type_cn(rule_label),
        "confidence": exp.get("confidence"),
        "model_source": "rule_fallback",
        "model_artifact": None,
        "rule_label_cn": _type_cn(rule_label),
        "reason_codes": exp.get("reason_codes", []),
        "top_features": [],
        "note": exp.get("message", "自训练分类模型暂不可用，已使用规则兜底并如实标注。"),
    }


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_inference(db: Session, project: Project) -> dict[str, Any]:
    """对单个项目运行真实推理链，产出结构化 analysis_result。

    返回：
      {status, model_run_id, model_source, models_used, analysis_result, evidence_map,
       data_gaps, confidence_notes, message}
    fail-closed：缺坐标 / 超覆盖 / 范围内无数据 → status != ok，且不含编造结果。
    """
    run_id = f"mr_{project.id}_{_now()}_{uuid.uuid4().hex[:6]}"

    if project.center_lng is None or project.center_lat is None:
        return {"status": STATUS_MISSING_LOCATION, "model_run_id": run_id,
                "message": "项目缺少可分析的中心坐标。", "analysis_result": None}

    # 1) 本地四维空间分析 + 类型(规则) + 综合评分 + 策略（坐标驱动，真实计算）
    try:
        full = ao.run_full_analysis(db, project, include_test=False)
    except SpatialError as exc:
        return {"status": STATUS_OUT_OF_COVERAGE, "model_run_id": run_id,
                "message": str(exc), "analysis_result": None}
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_full_analysis 异常：%s", type(exc).__name__)
        return {"status": STATUS_ERROR, "model_run_id": run_id,
                "message": "分析执行失败。", "analysis_result": None}

    fd = full.get("four_dimension", {})
    poi_r = _ring(fd.get("poi"), "radiation")
    pop_r = _ring(fd.get("population"), "radiation")
    house_r = _ring(fd.get("housing"), "radiation")
    ind_r = _ring(fd.get("industry"), "radiation")

    # fail-closed：辐射圈四维全空 → 该坐标附近无语料覆盖，不出假结果
    coverage = {
        "poi_total": poi_r.get("total") or 0,
        "population": pop_r.get("residential") or 0,
        "housing_samples": house_r.get("sample_count") or 0,
        "industry_enterprises": ind_r.get("enterprise_count") or 0,
    }
    if sum(1 for v in coverage.values() if v and v > 0) == 0:
        return {"status": STATUS_INSUFFICIENT_DATA, "model_run_id": run_id,
                "coverage": coverage,
                "message": "项目坐标周边暂无可用分析数据，请补充资料或确认项目位置。",
                "analysis_result": None}

    # 2) 自训练类型模型（模型优先 + 规则兜底）
    type_info = _infer_renewal_type(db, project)

    # 3) 房价基线模型指标（真实模型/统计基线）
    housing_dim = fd.get("housing") or {}
    model_metrics = housing_dim.get("model_metrics") or {}
    baseline = housing_dim.get("baseline_interval") or {}

    # ---- evidence_map：关键结论 → 来源可追溯 ----
    evidence_map: dict[str, Any] = {
        "renewal_type": _ev(
            type_info["label_cn"],
            "model_analysis" if type_info["model_source"] == "model" else "rule_based",
            type_info.get("model_artifact") or "project_type_service(规则)",
            type_info.get("confidence"),
            "自训练分类模型优先；不可用时规则兜底，已如实标注。",
        ),
        "comprehensive_score": _ev(
            full.get("F_score"), "local_analysis", "scoring_service(加权综合评分)",
            None, full.get("score_level")),
        "poi_total_radiation": _ev(
            poi_r.get("total"), "local_analysis", "poi_analysis_service(坐标驱动圈层归集)"),
        "population_radiation": _ev(
            pop_r.get("residential"), "local_analysis", "population_analysis_service"),
        "housing_avg_price_radiation": _ev(
            house_r.get("avg_unit_price"), "housing_model",
            f"housing_price_model({model_metrics.get('model_type')})",
            None, f"基线区间 mid≈{baseline.get('mid')}"),
        "industry_enterprises_radiation": _ev(
            ind_r.get("enterprise_count"), "local_analysis", "industry_analysis_service"),
    }

    # ---- data_gaps：现有数据集未提供的指标，如实标注 ----
    data_gaps: list[str] = []
    if not project.boundary_geojson:
        data_gaps.append("项目红线（核心范围精确边界）待补充，当前以中心点构建圈层。")
    if house_r.get("avg_unit_price") is None:
        data_gaps.append("辐射范围房价样本不足，价格基线置信度受限。")
    if (pop_r.get("residential") or 0) == 0:
        data_gaps.append("范围内人口画像数据较少。")

    # ---- 结构化 analysis_result（对齐报告 9 章口径；数字均来自上面真实计算）----
    strat = full.get("strategy_result") or {}
    analysis_result = {
        "project_understanding": {
            "name": project.name,
            "address": project.address,
            "district": project.district,
            "center": {"lng": project.center_lng, "lat": project.center_lat},
            "rings_m": {"core": project.core_buffer_m or 0,
                        "nearby": project.nearby_buffer_m or 500,
                        "radiation": project.radiation_buffer_m or 1500},
            "land_use": project.land_use,
            "update_demand": project.update_demand,
            "expected_direction": project.expected_direction,
        },
        "renewal_type": type_info["label_cn"],
        "renewal_type_reason": "；".join(
            [r.get("rule", "") if isinstance(r, dict) else str(r)
             for r in type_info.get("reason_codes", [])][:4]) or None,
        "renewal_type_source": type_info["model_source"],
        "renewal_type_confidence": type_info.get("confidence"),
        "comprehensive_score": full.get("F_score"),
        "score_level": full.get("score_level"),
        "dimension_scores": full.get("scores"),
        "location_poi_analysis": fd.get("poi"),
        "population_analysis": fd.get("population"),
        "housing_space_analysis": fd.get("housing"),
        "industry_analysis": fd.get("industry"),
        "demand_potential_analysis": {
            "key_opportunities": strat.get("key_opportunities", []),
            "key_risks": strat.get("key_risks", []),
            "recommended_directions": strat.get("recommended_directions", []),
        },
        "core_recommendations": {
            "update_positioning": strat.get("update_positioning"),
            "priority_actions": strat.get("priority_actions", []),
            "strategy_count": full.get("strategy_count"),
        },
        "data_gaps": data_gaps,
        "evidence_ids": full.get("evidence_ids", []),
    }

    models_used = [
        {"name": "项目类型分类模型", "artifact": type_info.get("model_artifact"),
         "source": type_info["model_source"]},
        {"name": "房价基线模型", "artifact": "housing_baseline.pkl",
         "source": "model" if not model_metrics.get("degraded") else "statistical_baseline",
         "model_type": model_metrics.get("model_type"), "val_mape": model_metrics.get("val_mape")},
    ]

    confidence_notes: list[str] = []
    if type_info["model_source"] != "model":
        confidence_notes.append("更新类型由规则兜底得到（自训练模型暂不可用）。")
    if model_metrics.get("degraded"):
        confidence_notes.append("房价采用统计基线（样本不足以训练回归模型）。")

    logger.info(
        "inference ok project_id=%s run_id=%s type=%s(src=%s) F=%s cover=%s",
        project.id, run_id, type_info["label_cn"], type_info["model_source"],
        full.get("F_score"), coverage,
    )
    return {
        "status": STATUS_OK,
        "model_run_id": run_id,
        "model_source": type_info["model_source"],
        "models_used": models_used,
        "renewal_type": type_info["label_cn"],
        "coverage": coverage,
        "analysis_result": analysis_result,
        "evidence_map": evidence_map,
        "data_gaps": data_gaps,
        "confidence_notes": confidence_notes,
        "generated_at": _now(),
        "message": "",
    }
