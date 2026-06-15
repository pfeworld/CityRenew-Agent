"""第12E：智能体工具层（Agent Tools）。

把已有的自研模型 / 数据 / 评估能力封装为**只读**工具，供编排层调用。
所有工具：
- 仅读取已落库 / 已产出的结构化结果（进程内直接调用现有 service 的 GET 类函数）；
- 绝不触发训练 / 调参 / 评测 run / 导出等写操作；
- 任一工具失败均被捕获，返回 status=error 且不伪造结果，不影响其他工具继续运行。

返回结构统一为：
  {tool, name, status: 'ok'|'empty'|'error', message, data}
其中 data 为现有 service 已脱敏的统计/分类/评分/证据摘要（不含原文/原始明细）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.services import (
    analysis_orchestrator,
    data_lineage_service,
    feature_engineering_service,
    final_test_eval_service,
    housing_price_training_service,
    phase115_gate_service,
    project_service,
    project_type_training_service,
    score_calibration_service,
)

logger = logging.getLogger("cityrenew.agent.tools")


# --------------------------------------------------------------------------- #
# 工具元信息（供 capabilities 接口与前端工具卡展示）
# --------------------------------------------------------------------------- #
TOOL_META: list[dict[str, str]] = [
    {"tool": "get_project_features", "name": "圈层特征工具", "description": "项目五圈层 POI 空间特征向量与覆盖率", "source": "特征工程结果"},
    {"tool": "get_poi_summary", "name": "POI 圈层分析工具", "description": "核心/近邻/辐射圈层 POI 配套结构摘要", "source": "圈层 POI 统计"},
    {"tool": "get_feature_quality", "name": "特征质量工具", "description": "特征覆盖率与质量门禁状态", "source": "特征质量评估"},
    {"tool": "get_score_result", "name": "综合评分工具", "description": "十维评分与综合评分（可复算，非大模型打分）", "source": "评分校准结果"},
    {"tool": "get_project_type", "name": "更新类型识别工具", "description": "项目更新类型识别（规则辅助识别）与依据", "source": "类型识别结果"},
    {"tool": "get_housing_model_result", "name": "房价预测工具", "description": "房价预测模型结果与最终冻结评估摘要", "source": "房价模型 / 最终评估"},
    {"tool": "get_evidence_lineage", "name": "证据链追溯工具", "description": "证据来源与数据血缘流向（脱敏）", "source": "证据链 / 数据血缘"},
    {"tool": "get_risk_summary", "name": "可信边界工具", "description": "可信边界提示与补数据建议", "source": "风险摘要"},
    {"tool": "get_trust_summary", "name": "可信度校验工具", "description": "数据一致性、证据链与最终冻结评估状态", "source": "总门禁 / 指标卡"},
    {"tool": "get_full_summary", "name": "项目研判汇总工具", "description": "四维分析+类型+评分+策略的已落库汇总", "source": "分析汇总"},
]


def _ok(tool: str, name: str, data: Any, message: str = "") -> dict[str, Any]:
    if data is None or (isinstance(data, dict) and data.get("available") is False):
        return {"tool": tool, "name": name, "status": "empty", "message": message or "该数据暂不可用", "data": None}
    return {"tool": tool, "name": name, "status": "ok", "message": "", "data": data}


def _safe(tool: str, name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    """安全执行单个工具，捕获所有异常，绝不向上抛。"""
    try:
        data = fn()
        return _ok(tool, name, data)
    except Exception as exc:  # noqa: BLE001 - 工具级兜底
        logger.warning("Agent 工具 %s 执行失败：%s", tool, type(exc).__name__)
        return {
            "tool": tool,
            "name": name,
            "status": "error",
            "message": f"该工具暂不可用（{type(exc).__name__}）",
            "data": None,
        }


# --------------------------------------------------------------------------- #
# 各只读工具实现
# --------------------------------------------------------------------------- #
def get_project_features(db: Session, project_id: int) -> dict[str, Any]:
    return _safe("get_project_features", "圈层特征工具",
                 lambda: feature_engineering_service.get_latest(db, project_id))


def get_poi_summary(db: Session, project_id: int) -> dict[str, Any]:
    return _safe("get_poi_summary", "POI 圈层分析工具",
                 lambda: feature_engineering_service.build_poi_summary(db, project_id))


def get_feature_quality(db: Session, project_id: int) -> dict[str, Any]:
    return _safe("get_feature_quality", "特征质量工具",
                 lambda: feature_engineering_service.build_feature_quality(db, project_id))


def get_score_result(db: Session, project_id: int) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        explain = score_calibration_service.explain_project(db, project_id)
        latest = score_calibration_service.get_latest()
        return {"explain": explain, "latest_summary": _trim_latest(latest)}
    return _safe("get_score_result", "综合评分工具", _run)


def get_project_type(db: Session, project_id: int) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        explain = project_type_training_service.explain_project_type_prediction(db, project_id)
        latest = project_type_training_service.get_latest()
        return {"explain": explain, "latest_summary": _trim_latest(latest)}
    return _safe("get_project_type", "更新类型识别工具", _run)


def get_housing_model_result(db: Session, project_id: int) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        latest = housing_price_training_service.get_latest()
        final_test = final_test_eval_service.get_latest()
        return {
            "housing_model": _trim_latest(latest),
            "final_test": _trim_latest(final_test),
        }
    return _safe("get_housing_model_result", "房价预测工具", _run)


def get_evidence_lineage(db: Session, project_id: int) -> dict[str, Any]:
    # export=False：仅读取聚合，不写盘（保持工具只读）。
    return _safe("get_evidence_lineage", "证据链追溯工具",
                 lambda: data_lineage_service.build_lineage(db, export=False))


def get_risk_summary(db: Session, project_id: int) -> dict[str, Any]:
    def _run() -> dict[str, Any] | None:
        data = phase115_gate_service.get_risk_summary()
        if data is None:
            phase115_gate_service.build_phase115_gate_result(db, project_id)
            data = phase115_gate_service.get_risk_summary()
        return data
    return _safe("get_risk_summary", "可信边界工具", _run)


def get_trust_summary(db: Session, project_id: int) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        gate = phase115_gate_service.build_phase115_gate_result(db, project_id)
        metric_card = final_test_eval_service.get_metric_card()
        return {"gate": _trim_gate(gate), "final_test_metric_card": metric_card}
    return _safe("get_trust_summary", "可信度校验工具", _run)


def get_full_summary(db: Session, project_id: int) -> dict[str, Any]:
    def _run() -> dict[str, Any] | None:
        project = project_service.get_project(db, project_id)
        if project is None:
            return None
        return analysis_orchestrator.get_full_summary(db, project)
    return _safe("get_full_summary", "项目研判汇总工具", _run)


# --------------------------------------------------------------------------- #
# 摘要裁剪：控制传给大模型的体量，且只保留统计/状态类字段
# --------------------------------------------------------------------------- #
def _trim_latest(data: Any) -> Any:
    """裁剪 latest 类结果：去掉超长列表，仅保留概要键。"""
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 12:
            out[k] = {"count": len(v), "preview": v[:8]}
        else:
            out[k] = v
    return out


def _trim_gate(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    # 只保留门禁概要与三大硬指标，避免回灌过多内部字段。
    keep = {
        "overall_status", "summary", "hard_metrics", "three_hard_metrics",
        "final_test_status", "compliance", "data_safety", "warnings_count",
        "warning_count", "pass", "metrics",
    }
    return {k: v for k, v in data.items() if k in keep} or data


def get_project_info(db: Session, project_id: int) -> dict[str, Any]:
    """项目基础信息（名称/地址/坐标可用性），非工具卡，仅供编排上下文。"""
    def _run() -> dict[str, Any] | None:
        project = project_service.get_project(db, project_id)
        if project is None:
            return None
        return {
            "project_id": getattr(project, "id", project_id),
            "name": getattr(project, "name", None),
            "address": getattr(project, "address", None),
            "city": getattr(project, "city", None),
        }
    return _safe("get_project_info", "项目基础信息", _run)


# 工具注册表（编排层按任务类型选择）
TOOL_REGISTRY: dict[str, Callable[[Session, int], dict[str, Any]]] = {
    "get_project_features": get_project_features,
    "get_poi_summary": get_poi_summary,
    "get_feature_quality": get_feature_quality,
    "get_score_result": get_score_result,
    "get_project_type": get_project_type,
    "get_housing_model_result": get_housing_model_result,
    "get_evidence_lineage": get_evidence_lineage,
    "get_risk_summary": get_risk_summary,
    "get_trust_summary": get_trust_summary,
    "get_full_summary": get_full_summary,
}


def available_tool_count(db: Session, project_id: int = 1) -> tuple[int, int]:
    """统计可用工具数（status=ok）/ 总数，供 health 接口。失败计为不可用。"""
    total = len(TOOL_REGISTRY)
    ok = 0
    for name, fn in TOOL_REGISTRY.items():
        try:
            res = fn(db, project_id)
            if res.get("status") == "ok":
                ok += 1
        except Exception:  # noqa: BLE001
            continue
    return ok, total
