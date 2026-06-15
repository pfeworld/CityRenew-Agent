"""综合评分（第6阶段，确定性加权，绝不使用大模型打分）。

F_score = P_score*wP + H_score*wH + L_score*wL + I_score*wI

权重按项目类型切换（经验值，可在 val 校准；禁用 test）。输出包含 P/H/L/I 原始分、
权重、各维加权贡献、F_score、score_level 与可解释说明。

红线：
- 分数仅来自第5阶段四维 analysis_result（确定性计算），本模块只做加权，不产生新数字。
- 不调用外部 API；不使用大模型；权重与档位阈值均为可解释常量。
- 维度无数据（置信度=0）时在 explanation 标注，不误导。
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.models import Project
from app.services import analysis_common as ac
from app.services import project_type_service as pts

logger = logging.getLogger("cityrenew.scorer")

DIMENSION = "scoring"

# --------------------------------------------------------------------------- #
# 按项目类型切换的四维权重（P=人口 / H=房价 / L=区位POI / I=产业）
# 经验值，可在 val 上校准（docs/08 第5节）；禁用 test。
# --------------------------------------------------------------------------- #
TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    pts.TYPE_OLD: {"P": 0.25, "H": 0.30, "L": 0.30, "I": 0.15},
    pts.TYPE_INDUSTRIAL: {"P": 0.15, "H": 0.25, "L": 0.25, "I": 0.35},
    pts.TYPE_BLOCK: {"P": 0.20, "H": 0.20, "L": 0.40, "I": 0.20},
    pts.TYPE_PUBLIC_SPACE: {"P": 0.35, "H": 0.10, "L": 0.40, "I": 0.15},
    pts.TYPE_COMMUNITY: {"P": 0.40, "H": 0.15, "L": 0.35, "I": 0.10},
    pts.TYPE_MIXED: {"P": 0.25, "H": 0.30, "L": 0.20, "I": 0.25},
}
# 兜底权重（未知类型时使用综合功能地块权重）
DEFAULT_WEIGHTS = TYPE_WEIGHTS[pts.TYPE_MIXED]

# 维度键 -> 四维 scores 中的键 / 中文名
DIM_DEF = [
    ("P", "P_score", "人口客群"),
    ("H", "H_score", "房价价值"),
    ("L", "L_score", "区位配套"),
    ("I", "I_score", "产业经济"),
]

# F_score 档位阈值（可解释常量）
LEVEL_HIGH = 80.0
LEVEL_MID_HIGH = 65.0
LEVEL_MID = 50.0
LEVEL_MID_LOW = 35.0


def get_weights(project_type: str | None) -> dict[str, float]:
    """返回项目类型对应的四维权重（未知类型用兜底）。"""
    return TYPE_WEIGHTS.get(project_type or "", DEFAULT_WEIGHTS)


def _score_level(f_score: float) -> str:
    if f_score >= LEVEL_HIGH:
        return "高"
    if f_score >= LEVEL_MID_HIGH:
        return "中高"
    if f_score >= LEVEL_MID:
        return "中"
    if f_score >= LEVEL_MID_LOW:
        return "中低"
    return "低"


def score(
    db: Session,
    project: Project,
    four_dim: dict[str, Any],
    project_type: str,
    persist: bool = True,
) -> dict[str, Any]:
    """计算综合评分 F_score（按项目类型权重加权四维分）。"""
    raw_scores = four_dim.get("scores") or {}
    confidence = four_dim.get("confidence") or {}
    weights = get_weights(project_type)

    contributions: list[dict[str, Any]] = []
    f_score = 0.0
    low_conf_dims: list[str] = []
    for key, score_key, label in DIM_DEF:
        s = float(raw_scores.get(score_key, 0.0) or 0.0)
        w = weights[key]
        contrib = round(s * w, 2)
        f_score += s * w
        conf_key = key  # confidence 字典键为 P/H/L/I
        c = confidence.get(conf_key)
        if c is not None and c < 0.3:
            low_conf_dims.append(f"{label}({score_key})")
        contributions.append({
            "dimension": key,
            "score_key": score_key,
            "label": label,
            "score": round(s, 2),
            "weight": w,
            "contribution": contrib,
            "confidence": c,
        })

    f_score = round(ac.clamp(f_score), 2)
    level = _score_level(f_score)

    # 可解释说明：主导维度（贡献最大）与拖累维度（贡献最小）
    sorted_contrib = sorted(contributions, key=lambda d: d["contribution"], reverse=True)
    lead = sorted_contrib[0]
    drag = sorted_contrib[-1]
    parts = [
        f"综合评分 F_score={f_score}（{level}）。",
        f"权重档（{project_type}）：P={weights['P']}, H={weights['H']}, L={weights['L']}, I={weights['I']}。",
        f"主导维度：{lead['label']}（贡献 {lead['contribution']}）；"
        f"相对拖累：{drag['label']}（贡献 {drag['contribution']}）。",
    ]
    notes = [
        "F_score 由第5阶段四维确定性分按项目类型权重加权得到，本模块不产生新数字、不使用大模型。",
    ]
    if low_conf_dims:
        notes.append(
            f"以下维度数据置信度偏低，其得分参考性有限：{'、'.join(low_conf_dims)}（未补零误导）。"
        )

    explanation = " ".join(parts)

    evidence_ids: list[str] = []
    # 引用四维各维度分的 evidence（溯源）
    for dim, key in (("poi", "L_score"), ("population", "P_score"),
                     ("housing", "H_score"), ("industry", "I_score")):
        if four_dim.get(dim):
            evidence_ids.append(ac.make_evidence_id(dim, project.id, "all", key))

    if persist:
        ac.clear_dimension_results(db, project.id, DIMENSION)
        f_evid = ac.record_metric(
            db, project_id=project.id, dimension=DIMENSION, ring=None,
            metric_key="F_score", value=f_score, unit="score",
            summary="综合评分 F_score（按项目类型权重加权四维分）",
            confidence=None,
            metadata={
                "project_type": project_type,
                "weights": weights,
                "score_level": level,
            },
        )
        evidence_ids.insert(0, f_evid)
        for c in contributions:
            evidence_ids.append(
                ac.record_metric(
                    db, project_id=project.id, dimension=DIMENSION, ring=None,
                    metric_key=f"contribution_{c['dimension']}", value=c["contribution"],
                    unit="score",
                    summary=f"{c['label']}维度加权贡献（score×weight）",
                    confidence=c["confidence"],
                    metadata={"score": c["score"], "weight": c["weight"]},
                )
            )
        db.commit()

    logger.info(
        "score project_id=%s type=%s F_score=%.2f level=%s weights=%s used_test=%s",
        project.id, project_type, f_score, level, weights, four_dim.get("used_test", False),
    )

    return {
        "project_id": project.id,
        "project_type": project_type,
        "scores": {sk: round(float(raw_scores.get(sk, 0.0) or 0.0), 2)
                   for _, sk, _ in DIM_DEF},
        "weights": weights,
        "contributions": contributions,
        "F_score": f_score,
        "score_level": level,
        "explanation": explanation,
        "allowed_splits": four_dim.get("allowed_splits", []),
        "include_test": four_dim.get("include_test", False),
        "used_test": four_dim.get("used_test", False),
        "evidence_ids": evidence_ids,
        "notes": notes,
    }
