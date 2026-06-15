"""产业经济分析（I 维度，第5阶段）。

目标：基于产业点位统计三圈层企业数量、密度、空间集聚，给出可解释的 I_score
与方向性产业适配建议。

数据现状（如实反映，不编造）：
- 训练语料产业点位 category_name 为单一类目（"公司企业;公司;公司"），
  **细分行业字段缺失，不可细分**；不得编造集成电路/软件信息/金融/文创等不存在的分类。
- 因此本维度以企业数量、产业密度、空间集聚、区位适配为主；产业多样性受限于单一类目（≈0）。
默认仅 train/val。

红线：仅输出统计量/分类/密度/建议/评分/置信度/notes/evidence_id；
不返回企业名称、地址、坐标、raw_json。
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from sqlalchemy.orm import Session

from app.models import Project
from app.services import analysis_common as ac
from app.services import spatial_service

logger = logging.getLogger("cityrenew.analysis.industry")

DIMENSION = "industry"
DENSITY_HI_PER_KM2 = 30.0  # 企业密度归一化经验上界
SCALE_HI = 120.0  # 辐射圈企业数量归一化经验上界


def _first_level(category_name: str | None) -> str:
    if not category_name:
        return "未分类"
    return category_name.split(";")[0].strip() or "未分类"


def _ring_stat(rows: list[Any], radius_m: int, ring: str) -> dict[str, Any]:
    cat = Counter(_first_level(r.category_name) for r in rows)
    area = spatial_service.ring_area_km2(radius_m)
    density = ac.safe_div(len(rows), area)
    diversity = ac.normalized_entropy(list(cat.values()))
    return {
        "ring": ring,
        "radius_m": radius_m,
        "enterprise_count": len(rows),
        "density_per_km2": round(density, 3) if density is not None else None,
        "diversity_index": diversity,
        "_cat": cat,
    }


def _suggestions(rad: dict[str, Any], single_category: bool) -> list[str]:
    out: list[str] = []
    density = rad.get("density_per_km2") or 0.0
    count = rad.get("enterprise_count", 0)
    if count == 0:
        out.append("辐射范围内暂无产业点位样本，建议结合区位与人口配套补充产业导入。")
        return out
    if density >= DENSITY_HI_PER_KM2 * 0.6:
        out.append("产业点位空间集聚度较高，适合存量提质、产业载体升级与配套完善。")
    else:
        out.append("产业点位密度偏低，适合培育产业功能、引导产城融合与就业岗位补充。")
    out.append("结合人口职住与区位配套优化产业空间布局。")
    if single_category:
        out.append("数据集产业为单一类目，细分行业字段缺失，不可细分；产业方向需结合外部产业规划进一步确认。")
    return out


def analyze(db: Session, project: Project, include_test: bool = False) -> dict[str, Any]:
    collected = spatial_service.collect_ring_records(db, project, "industry", include_test)
    radii = collected["radii"]
    records = collected["records"]

    ring_stats = {ring: _ring_stat(records[ring], radii[ring], ring) for ring in ac.RING_ORDER}
    rad = ring_stats[ac.RING_RADIATION]
    cat_dist: Counter = rad["_cat"]
    distinct_cats = [c for c in cat_dist if c != "未分类"]
    single_category = len(distinct_cats) <= 1
    dominant = cat_dist.most_common(1)[0][0] if cat_dist else None

    # ---- 评分 ----
    density_score = ac.minmax_score(rad["density_per_km2"], 0.0, DENSITY_HI_PER_KM2)
    scale_score = ac.minmax_score(float(rad["enterprise_count"]), 0.0, SCALE_HI)
    diversity_score = rad["diversity_index"] * 100.0
    score = ac.clamp(0.50 * density_score + 0.30 * scale_score + 0.20 * diversity_score)

    confidence = round(min(1.0, rad["enterprise_count"] / 50.0), 3)
    notes = [
        "圈层为累计圆；产业密度=企业数/圈层面积(km²)。",
        "数据集产业为单一类目，细分行业字段缺失，不可细分（未编造行业分类）。",
        "多样性指数受单一类目限制（≈0），故评分权重较低。",
    ]
    if rad["enterprise_count"] == 0:
        notes.append("辐射范围内无 train/val 产业点位，评分置信度低。")
    if collected["skipped_no_coord"]:
        notes.append(f"{collected['skipped_no_coord']} 条产业点位因坐标不可用被跳过。")

    suggestions = _suggestions(rad, single_category)

    # ---- 落库 ----
    ac.clear_dimension_results(db, project.id, DIMENSION)
    evidence_ids: list[str] = []
    for ring in ac.RING_ORDER:
        st = ring_stats[ring]
        evidence_ids.append(
            ac.record_metric(
                db, project_id=project.id, dimension=DIMENSION, ring=ring,
                metric_key="enterprise_count", value=float(st["enterprise_count"]), unit="家",
                summary=f"{ring} 圈层企业数量", confidence=confidence,
                metadata={"allowed_splits": collected["allowed_splits"]},
            )
        )
        if st["density_per_km2"] is not None:
            evidence_ids.append(
                ac.record_metric(
                    db, project_id=project.id, dimension=DIMENSION, ring=ring,
                    metric_key="density_per_km2", value=st["density_per_km2"], unit="家/km²",
                    summary=f"{ring} 圈层产业密度", confidence=confidence,
                )
            )
        evidence_ids.append(
            ac.record_metric(
                db, project_id=project.id, dimension=DIMENSION, ring=ring,
                metric_key="diversity_index", value=float(st["diversity_index"]), unit="ratio",
                summary=f"{ring} 圈层产业多样性（归一化香农熵）", confidence=confidence,
            )
        )
    if dominant:
        evidence_ids.append(
            ac.record_metric(
                db, project_id=project.id, dimension=DIMENSION, ring=ac.RING_RADIATION,
                metric_key="dominant_industry", text=dominant,
                summary="主导产业（一级类目，受数据单一类目限制）", confidence=confidence,
            )
        )
    evidence_ids.append(
        ac.record_metric(
            db, project_id=project.id, dimension=DIMENSION, ring=None,
            metric_key="I_score", value=float(round(score, 2)), unit="score",
            summary="产业经济 I 维度评分（密度/规模/多样性 可解释加权）",
            confidence=confidence,
        )
    )
    db.commit()

    rings_out = [
        {k: v for k, v in ring_stats[ring].items() if not k.startswith("_")}
        for ring in ac.RING_ORDER
    ]
    logger.info(
        "industry analyze project_id=%s I_score=%.1f count_rad=%s single_cat=%s",
        project.id, score, rad["enterprise_count"], single_category,
    )
    return {
        "project_id": project.id,
        "dimension": DIMENSION,
        "score": round(score, 2),
        "confidence": confidence,
        "allowed_splits": collected["allowed_splits"],
        "include_test": include_test,
        "used_test": include_test,
        "center_status": collected["center_status"],
        "evidence_ids": evidence_ids,
        "notes": notes,
        "rings": rings_out,
        "category_dist": {k: v for k, v in cat_dist.items() if k != "未分类"},
        "dominant_industry": dominant,
        "adaptation_suggestions": suggestions,
    }
