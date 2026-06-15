"""四维分析公共工具（第5阶段）。

职责（被 poi / population / housing / industry 四个分析服务复用）：
- 圈层常量与顺序。
- 评分工具：clamp、min-max 归一化、归一化香农熵（功能/产业多样性）。
- 落库工具：清理并重建某项目某维度的 AnalysisResult；写 EvidenceChain。
- evidence_id 规则：{dimension}:{source_hash8}:p{project_id}:{ring}#{metric}，
  稳定可复算、按项目/圈层/指标唯一（对齐 docs/06）。

红线：
- 本模块不读取任何 test 数据；split 控制在 spatial_service 完成。
- 不持有/输出任何语料原文；summary 仅写脱敏口径短语。
"""

from __future__ import annotations

import logging
import math
from typing import Any

from sqlalchemy.orm import Session

from app.models import AnalysisResult
from app.services import evidence_service

logger = logging.getLogger("cityrenew.analysis")

RING_CORE = "core"
RING_NEARBY = "nearby"
RING_RADIATION = "radiation"
RING_ORDER = (RING_CORE, RING_NEARBY, RING_RADIATION)

# 各维度对应的来源语料文件名（仅用于 evidence 溯源标注，不读原文）
# 第6阶段新增派生维度（classification / scoring / strategy）：其结论派生自
# 四维 analysis_result + 项目输入字段 + 规则，故来源标注为"派生计算"，不指向语料原文。
DIMENSION_SOURCE_FILE: dict[str, str] = {
    "poi": "POI兴趣点分布数据.json",
    "population": "区域人口总量.json",
    "housing": "房价历史交易数据.json",
    "industry": "产业布局数据.json",
    # ---- 第6阶段派生维度（决策层，非语料原文）----
    "classification": "derived:analysis_result(P/H/L/I)+project_input",
    "scoring": "derived:analysis_result(P/H/L/I)+type_weights",
    "strategy": "derived:analysis_result(P/H/L/I)+rules",
}


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def minmax_score(value: float | None, lo: float, hi: float) -> float:
    """把 value 线性映射到 0~100（超界裁剪）。lo>=hi 时返回 0。"""
    if value is None or hi <= lo:
        return 0.0
    return clamp((value - lo) / (hi - lo) * 100.0)


def normalized_entropy(counts: list[float]) -> float:
    """归一化香农熵（0~1）。用于功能混合度 / 产业多样性。

    单一类目或空集返回 0；类目越均衡越接近 1。
    """
    total = sum(c for c in counts if c and c > 0)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c and c > 0]
    if len(probs) <= 1:
        return 0.0
    ent = -sum(p * math.log(p) for p in probs)
    return round(ent / math.log(len(probs)), 4)


def safe_div(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def percentile(sorted_values: list[float], q: float) -> float | None:
    """线性插值分位数（q ∈ [0,1]），入参需已排序且非空。"""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return percentile(sorted(values), 0.5)


def make_evidence_id(dimension: str, project_id: int, ring: str, metric_key: str) -> str:
    """生成稳定且按项目/圈层/指标唯一的 evidence_id。"""
    source_file = DIMENSION_SOURCE_FILE.get(dimension, dimension)
    record_ref = f"p{project_id}:{ring}#{metric_key}"
    return evidence_service.make_evidence_id(dimension, source_file, record_ref)


def clear_dimension_results(db: Session, project_id: int, dimension: str) -> int:
    """删除某项目某维度的旧 AnalysisResult，保证重算幂等。"""
    deleted = (
        db.query(AnalysisResult)
        .filter(
            AnalysisResult.project_id == project_id,
            AnalysisResult.dimension == dimension,
        )
        .delete(synchronize_session=False)
    )
    return int(deleted or 0)


def record_metric(
    db: Session,
    *,
    project_id: int,
    dimension: str,
    ring: str | None,
    metric_key: str,
    value: float | None = None,
    text: str | None = None,
    unit: str | None = None,
    summary: str | None = None,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """写一条 AnalysisResult + 对应 EvidenceChain，返回 evidence_id。

    数字仅来自调用方的确定性统计；本函数不产生任何数字。
    """
    ring_key = ring or "all"
    evidence_id = make_evidence_id(dimension, project_id, ring_key, metric_key)
    source_file = DIMENSION_SOURCE_FILE.get(dimension, dimension)

    db.add(
        AnalysisResult(
            project_id=project_id,
            ring=ring,
            dimension=dimension,
            metric_key=metric_key,
            metric_value=value,
            metric_text=text,
            unit=unit,
            evidence_id=evidence_id,
        )
    )
    evidence_service.upsert_evidence(
        db,
        evidence_id=evidence_id,
        data_type=dimension,
        source_file=source_file,
        record_ref=f"p{project_id}:{ring_key}#{metric_key}",
        summary=summary,
        confidence=confidence,
        metadata=metadata,
    )
    return evidence_id
