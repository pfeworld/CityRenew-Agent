"""人口画像分析（P 维度，第5阶段）。

目标：基于人口网格总量与画像字段，分析居住/工作人口、职住比与客群结构，
产出可解释的 P_score。

口径：
- 人口网格按**网格中心点**归集到圈层（累计圆），不做面积加权（与现有 spatial 一致）。
- 规模用总量表 residential/worker；结构占比由画像计数（residential_* / worker_*）
  汇总后求比，画像计数基数与总量可能不同（画像为抽样口径），在 notes 标注。
- **无收入字段**：income 一律标注"数据缺失/不适用"，绝不编造（红线）。
- 默认仅 train/val。

红线：仅输出统计量/结构占比/评分/置信度/notes/evidence_id；不返回 raw_json/网格坐标明细。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from sqlalchemy.orm import Session

from app.models import Project
from app.services import analysis_common as ac
from app.services import spatial_service

logger = logging.getLogger("cityrenew.analysis.population")

DIMENSION = "population"

AGE_FIELDS = [
    ("residential_age_18", "<18"),
    ("residential_age_18_to_24", "18-24"),
    ("residential_age_25_to_34", "25-34"),
    ("residential_age_35_to_44", "35-44"),
    ("residential_age_45_to_54", "45-54"),
    ("residential_age_55_to_64", "55-64"),
    ("residential_age_65", "65+"),
]
CONSUMPTION_FIELDS = [
    ("residential_consumption_low", "low"),
    ("residential_consumption_middle", "middle"),
    ("residential_consumption_high", "high"),
]
EDUCATION_FIELDS = [
    ("residential_education_middle", "middle"),
    ("residential_education_college", "college"),
    ("residential_education_university", "university"),
]
CAR_FIELDS = [
    ("residential_auto_none", "none"),
    ("residential_auto_auto", "auto"),
]

P_SCALE_HI = 60000.0  # 居住人口规模归一化经验上界（辐射圈累计）


def _sum_profile(rows: list[Any]) -> Counter:
    agg: Counter = Counter()
    for r in rows:
        if not r.profile_json:
            continue
        try:
            prof = json.loads(r.profile_json)
        except (TypeError, ValueError):
            continue
        for k, v in prof.items():
            if isinstance(v, (int, float)):
                agg[k] += v
    return agg


def _structure(agg: Counter, fields: list[tuple[str, str]]) -> dict[str, Any]:
    base = sum(agg.get(f, 0) for f, _ in fields)
    if base <= 0:
        return {"available": False, "base_count": 0, "ratios": {}}
    ratios = {label: round(agg.get(f, 0) / base, 4) for f, label in fields}
    return {"available": True, "base_count": int(base), "ratios": ratios}


def _ring_stat(rows: list[Any], radius_m: int, ring: str) -> dict[str, Any]:
    residential = sum(int(r.residential or 0) for r in rows)
    worker = sum(int(r.worker or 0) for r in rows)
    jhr = ac.safe_div(worker, residential)
    # 逐圈层聚合画像（年龄/消费），供报告表格按真实分析填充三圈层
    agg = _sum_profile(rows)
    return {
        "ring": ring,
        "radius_m": radius_m,
        "grid_count": len(rows),
        "residential": residential,
        "worker": worker,
        "job_housing_ratio": round(jhr, 4) if jhr is not None else None,
        "age_structure": _structure(agg, AGE_FIELDS),
        "consumption_structure": _structure(agg, CONSUMPTION_FIELDS),
    }


def _main_segment(age: dict, consumption: dict, car: dict) -> str | None:
    parts: list[str] = []
    if age.get("available"):
        top_age = max(age["ratios"].items(), key=lambda kv: kv[1])[0]
        parts.append(f"{top_age}岁为主")
    if consumption.get("available"):
        r = consumption["ratios"]
        if r.get("high", 0) + r.get("middle", 0) >= 0.6:
            parts.append("中高消费")
        elif r.get("low", 0) >= 0.5:
            parts.append("中低消费")
    if car.get("available") and car["ratios"].get("auto", 0) >= 0.4:
        parts.append("有车比例较高")
    return "、".join(parts) if parts else None


def analyze(db: Session, project: Project, include_test: bool = False) -> dict[str, Any]:
    collected = spatial_service.collect_ring_records(db, project, "population", include_test)
    radii = collected["radii"]
    records = collected["records"]

    ring_stats = {ring: _ring_stat(records[ring], radii[ring], ring) for ring in ac.RING_ORDER}
    radiation_rows = records[ac.RING_RADIATION]
    rad = ring_stats[ac.RING_RADIATION]

    agg = _sum_profile(radiation_rows)
    age = _structure(agg, AGE_FIELDS)
    consumption = _structure(agg, CONSUMPTION_FIELDS)
    education = _structure(agg, EDUCATION_FIELDS)
    car = _structure(agg, CAR_FIELDS)
    main_segment = _main_segment(age, consumption, car)

    # ---- 评分 ----
    scale_score = ac.minmax_score(rad["residential"], 0.0, P_SCALE_HI)
    jhr = rad["job_housing_ratio"]
    balance_score = 0.0 if jhr is None else ac.clamp(100.0 * (1 - min(1.0, abs(jhr - 1.0) / 1.5)))
    consumption_score = 0.0
    if consumption["available"]:
        cr = consumption["ratios"]
        consumption_score = ac.clamp((cr.get("high", 0) + 0.5 * cr.get("middle", 0)) * 100.0)
    vitality_score = 0.0
    if age["available"]:
        ar = age["ratios"]
        labor = ar.get("25-34", 0) + ar.get("35-44", 0) + ar.get("45-54", 0)
        vitality_score = ac.clamp(labor * 100.0)

    score = ac.clamp(
        0.40 * scale_score
        + 0.20 * balance_score
        + 0.20 * consumption_score
        + 0.20 * vitality_score
    )

    profile_available = age["available"] or consumption["available"]
    confidence = round(min(1.0, rad["grid_count"] / 20.0) * (1.0 if profile_available else 0.5), 3)

    notes = [
        "人口网格按中心点归集（未做面积加权）；规模用总量表，结构占比由画像计数汇总求比。",
        "画像计数为抽样口径，其基数可能与居住/工作人口总量不一致。",
        "收入字段：数据缺失/不适用（数据集未提供，未编造）。",
    ]
    if rad["grid_count"] == 0:
        notes.append("辐射范围内无 train/val 人口网格，评分置信度低。")
    if not profile_available and rad["grid_count"] > 0:
        notes.append("落圈网格画像字段为空，结构摘要不可用。")

    # ---- 落库 ----
    ac.clear_dimension_results(db, project.id, DIMENSION)
    evidence_ids: list[str] = []
    for ring in ac.RING_ORDER:
        st = ring_stats[ring]
        for key, unit in (("residential", "人"), ("worker", "人"), ("grid_count", "个")):
            evidence_ids.append(
                ac.record_metric(
                    db, project_id=project.id, dimension=DIMENSION, ring=ring,
                    metric_key=f"pop_{key}", value=float(st[key]), unit=unit,
                    summary=f"{ring} 圈层人口 {key}", confidence=confidence,
                    metadata={"allowed_splits": collected["allowed_splits"]},
                )
            )
        if st["job_housing_ratio"] is not None:
            evidence_ids.append(
                ac.record_metric(
                    db, project_id=project.id, dimension=DIMENSION, ring=ring,
                    metric_key="job_housing_ratio", value=float(st["job_housing_ratio"]),
                    unit="ratio", summary=f"{ring} 圈层职住比 worker/residential",
                    confidence=confidence,
                )
            )
    if main_segment:
        evidence_ids.append(
            ac.record_metric(
                db, project_id=project.id, dimension=DIMENSION, ring=ac.RING_RADIATION,
                metric_key="main_segment", text=main_segment,
                summary="主力客群判断（年龄/消费/有车规则推断）", confidence=confidence,
            )
        )
    evidence_ids.append(
        ac.record_metric(
            db, project_id=project.id, dimension=DIMENSION, ring=None,
            metric_key="P_score", value=float(round(score, 2)), unit="score",
            summary="人口潜力 P 维度评分（规模/职住/消费/活力 可解释加权）",
            confidence=confidence,
        )
    )
    db.commit()

    rings_out = [ring_stats[ring] for ring in ac.RING_ORDER]
    logger.info(
        "population analyze project_id=%s P_score=%.1f res_rad=%s grids=%s",
        project.id, score, rad["residential"], rad["grid_count"],
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
        "age_structure": age,
        "consumption_structure": consumption,
        "education_structure": education,
        "car_ownership": car,
        "income_structure": "数据缺失/不适用",
        "main_segment": main_segment,
    }
