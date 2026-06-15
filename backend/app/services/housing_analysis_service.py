"""房价价值分析（H 维度，第5阶段）。

目标：基于房价历史交易，统计项目三圈层价格水平、价格梯度，并结合房价基线模型
给出基线预测区间与可解释的 H_score。

异常值过滤：unit_price<=0 删除；area<=0 删除；year==0 视为缺失。
模型：housing_price_model（仅 train/val 训练验证，test 不参与）。
默认仅 train/val。

红线：仅输出统计量/分布/区间/模型指标/评分/置信度/notes/evidence_id；
不返回小区名、地址、坐标、raw_json。
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from sqlalchemy.orm import Session

from app.models import PoiPoint, Project
from app.services import analysis_common as ac
from app.services import housing_price_model as hpm
from app.services import housing_price_training_service as hpt
from app.services import spatial_service
from app.utils import geo_utils

logger = logging.getLogger("cityrenew.analysis.housing")

DIMENSION = "housing"
PRICE_LO = 20000.0  # H_score 价值水平归一化经验下界（元/㎡）
PRICE_HI = 120000.0  # 经验上界（元/㎡）
# 落圈成交样本不足该阈值时，启用「本区授权脱敏成交样本」真实基线（覆盖全市）
MIN_LOCAL_SAMPLES = 5


def _resolve_district(db: Session, project: Project) -> str | None:
    """识别项目所在行政区：优先项目字段，其次由周边高德 POI（全市覆盖）就近取众数。"""
    for txt in (getattr(project, "district", None), getattr(project, "address", None),
                getattr(project, "name", None)):
        d = hpt.normalize_district(txt)
        if d:
            return d
    lng, lat = project.center_lng, project.center_lat
    if lng is None or lat is None:
        return None
    dd = 0.03  # 约 3km 包围盒，先用 SQL 粗筛再算距离
    rows = (
        db.query(PoiPoint.district_name, PoiPoint.lng, PoiPoint.lat)
        .filter(PoiPoint.district_name.isnot(None),
                PoiPoint.lng.between(lng - dd, lng + dd),
                PoiPoint.lat.between(lat - dd, lat + dd))
        .all()
    )
    counter: Counter = Counter()
    for dn, x, y in rows:
        if x is None or y is None:
            continue
        if geo_utils.haversine_m(lng, lat, x, y) <= 1500:
            counter[hpt.normalize_district(dn) or dn] += 1
    return counter.most_common(1)[0][0] if counter else None


def _valid_unit_prices(rows: list[Any]) -> list[float]:
    out: list[float] = []
    for r in rows:
        if r.unit_price and r.unit_price > 0 and r.area and r.area > 0:
            out.append(float(r.unit_price))
    return out


def _ring_stat(rows: list[Any], radius_m: int, ring: str) -> dict[str, Any]:
    prices = _valid_unit_prices(rows)
    areas = [float(r.area) for r in rows if r.area and r.area > 0]
    years = [int(r.year) for r in rows if r.year and r.year > 0]
    room_dist = Counter(r.room_type for r in rows if r.room_type)
    year_summary = {
        "min_year": float(min(years)) if years else None,
        "max_year": float(max(years)) if years else None,
        "median_year": ac.median([float(y) for y in years]) if years else None,
        "missing_year": float(sum(1 for r in rows if not (r.year and r.year > 0))),
    }
    return {
        "ring": ring,
        "radius_m": radius_m,
        "sample_count": len(prices),
        "avg_unit_price": round(sum(prices) / len(prices), 2) if prices else None,
        "median_unit_price": round(ac.median(prices), 2) if prices else None,
        "avg_area": round(sum(areas) / len(areas), 2) if areas else None,
        "room_type_dist": dict(room_dist.most_common(8)),
        "year_summary": year_summary,
    }


def analyze(db: Session, project: Project, include_test: bool = False) -> dict[str, Any]:
    collected = spatial_service.collect_ring_records(db, project, "housing", include_test)
    radii = collected["radii"]
    records = collected["records"]

    ring_stats = {ring: _ring_stat(records[ring], radii[ring], ring) for ring in ac.RING_ORDER}
    rad = ring_stats[ac.RING_RADIATION]

    price_gradient = {ring: ring_stats[ring]["avg_unit_price"] for ring in ac.RING_ORDER}

    # ---- 房价基线模型（仅 train/val）----
    bundle = hpm.train_baseline(db, force_retrain=True)
    observed_median = (
        rad["median_unit_price"]
        if rad["median_unit_price"] is not None
        else ring_stats[ac.RING_NEARBY]["median_unit_price"]
    )
    interval = hpm.baseline_interval(
        bundle,
        lng=project.center_lng,
        lat=project.center_lat,
        area=project.building_area,
        year=project.build_year,
        observed_median=observed_median,
    )
    model_metrics = bundle.to_metrics()

    # ---- 外区无本地落圈成交样本：启用本区授权脱敏成交样本真实基线（覆盖全市）----
    district = _resolve_district(db, project)
    district_base = hpt.district_price_baseline().get(district) if district else None
    district_year_base = hpt.district_build_year_baseline().get(district) if district else None
    rent_reference = hpt.citywide_rent_baseline()
    used_district_baseline = False
    if (rad["sample_count"] or 0) < MIN_LOCAL_SAMPLES and district_base:
        # 本地落圈样本不足，徐汇训练的逐点模型对外区为外推、不可靠；
        # 改用本区真实成交样本中位/分位作为价格基线（真实、可回溯）。
        observed_median = district_base["median"]
        interval = {"low": district_base["p25"], "mid": district_base["median"],
                    "high": district_base["p75"]}
        used_district_baseline = True

    # ---- 评分 ----
    level_value = interval.get("mid") if interval.get("mid") is not None else observed_median
    level_score = ac.minmax_score(level_value, PRICE_LO, PRICE_HI)
    # 样本充足度
    sample_score = ac.clamp(min(1.0, rad["sample_count"] / 50.0) * 100.0)
    # 价格梯度合理性：核心高于辐射（向心溢价）得高分
    core_p = price_gradient.get(ac.RING_CORE)
    rad_p = price_gradient.get(ac.RING_RADIATION)
    if core_p and rad_p:
        gradient_score = ac.clamp(50.0 + (core_p - rad_p) / rad_p * 100.0)
    else:
        gradient_score = 50.0

    score = ac.clamp(0.40 * level_score + 0.30 * sample_score + 0.30 * gradient_score)

    confidence = round(min(1.0, rad["sample_count"] / 30.0), 3)
    notes = [
        "异常值过滤：unit_price<=0、area<=0 删除；year==0 视为缺失。",
        "圈层为累计圆；价格梯度为各圈层均价。",
        f"房价基线模型仅用 train/val（{model_metrics['model_type']}），test 不参与训练。",
    ]
    if bundle.degraded:
        notes.append(f"模型已降级：{bundle.note}")
    if used_district_baseline:
        notes.append(
            f"辐射范围内暂无本地落圈成交样本；价格基线采用本区（{district}）授权脱敏成交样本中位价"
            f"约 {interval['mid']:.0f} 元/㎡（样本量 {district_base['count']} 套，来源：科研授权脱敏房价样本，与正式房价模型同源）。"
        )
        confidence = round(min(0.85, max(confidence, district_base["count"] / 1000.0)), 3)
    elif rad["sample_count"] == 0:
        notes.append("辐射范围内暂无本地成交样本，价格基线置信度受限。")

    # ---- 落库 ----
    ac.clear_dimension_results(db, project.id, DIMENSION)
    evidence_ids: list[str] = []
    for ring in ac.RING_ORDER:
        st = ring_stats[ring]
        evidence_ids.append(
            ac.record_metric(
                db, project_id=project.id, dimension=DIMENSION, ring=ring,
                metric_key="housing_sample_count", value=float(st["sample_count"]), unit="套",
                summary=f"{ring} 圈层有效房价样本数", confidence=confidence,
                metadata={"allowed_splits": collected["allowed_splits"]},
            )
        )
        if st["avg_unit_price"] is not None:
            evidence_ids.append(
                ac.record_metric(
                    db, project_id=project.id, dimension=DIMENSION, ring=ring,
                    metric_key="avg_unit_price", value=st["avg_unit_price"], unit="元/㎡",
                    summary=f"{ring} 圈层平均单价", confidence=confidence,
                )
            )
            evidence_ids.append(
                ac.record_metric(
                    db, project_id=project.id, dimension=DIMENSION, ring=ring,
                    metric_key="median_unit_price", value=st["median_unit_price"], unit="元/㎡",
                    summary=f"{ring} 圈层单价中位数", confidence=confidence,
                )
            )
    baseline_source = (
        f"区级授权脱敏成交样本（{district}，n={district_base['count']}）"
        if used_district_baseline else model_metrics["model_type"]
    )
    for bound in ("low", "mid", "high"):
        if interval.get(bound) is not None:
            evidence_ids.append(
                ac.record_metric(
                    db, project_id=project.id, dimension=DIMENSION, ring=None,
                    metric_key=f"baseline_{bound}", value=interval[bound], unit="元/㎡",
                    summary=f"房价基线区间 {bound}", confidence=confidence,
                    metadata={"model_type": model_metrics["model_type"],
                              "baseline_source": baseline_source},
                )
            )
    evidence_ids.append(
        ac.record_metric(
            db, project_id=project.id, dimension=DIMENSION, ring=None,
            metric_key="H_score", value=float(round(score, 2)), unit="score",
            summary="房价价值 H 维度评分（价值水平/样本充足/梯度合理 可解释加权）",
            confidence=confidence,
            metadata={"model_metrics": model_metrics},
        )
    )
    db.commit()

    rings_out = [ring_stats[ring] for ring in ac.RING_ORDER]
    logger.info(
        "housing analyze project_id=%s H_score=%.1f samples_rad=%s model=%s mape=%s",
        project.id, score, rad["sample_count"], model_metrics["model_type"],
        model_metrics["val_mape"],
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
        "price_gradient": price_gradient,
        "baseline_interval": interval,
        "model_metrics": model_metrics,
    }
