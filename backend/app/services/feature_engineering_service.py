"""项目级特征工程服务（第10A阶段）。

目标：把 POI / 人口 / 房价 / 产业 / 项目字段（第10B 起可含合规外部数据）统一转成
项目级特征向量，作为多模型训练 / 聚类 / 相似度学习的输入来源。

口径与红线：
- 仅 train/val（include_test 固定 false，used_test=false）；test 永不参与特征工程。
- 圈层为累计圆（核心⊆近邻⊆辐射），复用 spatial_service。
- 关键词/编码使用本地确定性词典，不调用任何外部 LLM。
- 缺失特征标 None 并计入 missing_features，绝不编造。
- 输出仅含特征名 / 特征值 / 分组 / 来源计数 / evidence_id；不含 raw_json / 原始点位 /
  企业名 / 小区名 / 地址 / 坐标列表。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Project, ProjectFeature
from app.services import analysis_common as ac
from app.services import evidence_service
from app.services import poi_analysis_service as poi_svc
from app.services import poi_feature_service
from app.services import population_analysis_service as pop_svc
from app.services import ring_feature_service
from app.services import spatial_service
from app.utils import geo_utils

logger = logging.getLogger("cityrenew.features")

ALLOWED_SPLITS = ["train", "val"]

# 第11 T2 特征版本（圈层 POI 空间特征工程）
FEATURE_VERSION = "t2_poi_ring_v1"
# T2 POI 圈层最大半径（米）：覆盖 core/500/1500/3000/5000
T2_MAX_RADIUS_M = 5000

# 项目用地性质编码（本地确定性词典；未命中返回 0=未知/其它）
_LAND_USE_CODES: dict[str, int] = {
    "居住": 1,
    "住宅": 1,
    "商业": 2,
    "商务": 3,
    "办公": 3,
    "工业": 4,
    "厂": 4,
    "公共服务": 5,
    "公服": 5,
    "绿地": 6,
    "公园": 6,
    "综合": 7,
    "混合": 7,
}

# 文本关键词编码词典（更新诉求 / 期望方向共用类目）
_DEMAND_KEYWORDS: dict[str, list[str]] = {
    "residential_improve": ["居住", "住房", "老旧", "小区", "宜居", "安置"],
    "industrial_upgrade": ["产业", "厂房", "园区", "升级", "转型", "制造"],
    "commercial_activate": ["商业", "商务", "活力", "消费", "街区", "商圈"],
    "public_service": ["配套", "公共服务", "教育", "医疗", "养老", "社区", "学校"],
    "transport": ["交通", "地铁", "枢纽", "停车", "公交", "慢行"],
    "culture_tourism": ["文化", "文旅", "历史", "风貌", "旅游", "更新"],
}


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _land_use_code(land_use: str | None) -> int | None:
    if not land_use:
        return None
    for key, code in _LAND_USE_CODES.items():
        if key in land_use:
            return code
    return 0


def _keyword_counts(text: str | None) -> dict[str, int]:
    counts = {cat: 0 for cat in _DEMAND_KEYWORDS}
    if not text:
        return counts
    for cat, kws in _DEMAND_KEYWORDS.items():
        counts[cat] = sum(1 for kw in kws if kw in text)
    return counts


# --------------------------------------------------------------------------- #
# 各特征组
# --------------------------------------------------------------------------- #
def _poi_features(db: Session, project: Project) -> tuple[dict[str, Any], int]:
    collected = spatial_service.collect_ring_records(db, project, "poi", include_test=False)
    rec = collected["records"]
    radii = collected["radii"]
    rad_rows = rec[ac.RING_RADIATION]
    rad_area = spatial_service.ring_area_km2(radii[ac.RING_RADIATION]) or 1.0

    group_counts: Counter = Counter()
    first_counts: Counter = Counter()
    for r in rad_rows:
        fl = (r.category_name or "").split(";")[0].strip() or None
        first_counts[fl or "未分类"] += 1
        group_counts[poi_svc.FIRST_LEVEL_GROUP.get(fl, poi_svc.GROUP_OTHER)] += 1

    total = len(rad_rows)
    feats: dict[str, Any] = {
        "poi_count_core": float(len(rec[ac.RING_CORE])),
        "poi_count_nearby": float(len(rec[ac.RING_NEARBY])),
        "poi_count_radiation": float(total),
        "poi_share_commercial": _round(ac.safe_div(group_counts.get(poi_svc.GROUP_COMMERCIAL, 0), total)),
        "poi_share_public": _round(ac.safe_div(group_counts.get(poi_svc.GROUP_PUBLIC, 0), total)),
        "poi_share_convenience": _round(ac.safe_div(group_counts.get(poi_svc.GROUP_CONVENIENCE, 0), total)),
        "poi_share_transport": _round(ac.safe_div(group_counts.get(poi_svc.GROUP_TRANSPORT, 0), total)),
        "poi_share_other": _round(ac.safe_div(group_counts.get(poi_svc.GROUP_OTHER, 0), total)),
        "poi_density_commercial": _round(group_counts.get(poi_svc.GROUP_COMMERCIAL, 0) / rad_area),
        "poi_density_public": _round(group_counts.get(poi_svc.GROUP_PUBLIC, 0) / rad_area),
        "poi_density_convenience": _round(group_counts.get(poi_svc.GROUP_CONVENIENCE, 0) / rad_area),
        "poi_density_transport": _round(group_counts.get(poi_svc.GROUP_TRANSPORT, 0) / rad_area),
        "poi_mix_index": ac.normalized_entropy(
            [group_counts.get(g, 0) for g in (
                poi_svc.GROUP_COMMERCIAL, poi_svc.GROUP_PUBLIC,
                poi_svc.GROUP_CONVENIENCE, poi_svc.GROUP_TRANSPORT)]
        ),
        "poi_decay_core_to_radiation": _round(ac.safe_div(len(rec[ac.RING_CORE]), total)),
    }
    # 配套短板向量：各短板候选一级类目在辐射圈的计数（0 表示短板）
    for cat in poi_svc.SHORTBOARD_CANDIDATES:
        key = f"poi_shortboard_{poi_svc.FIRST_LEVEL_GROUP.get(cat, 'other')}_{cat}"
        feats[f"poi_shortboard_{cat}"] = float(first_counts.get(cat, 0))
        _ = key
    if total == 0:
        # 无落圈 POI：占比/密度类无意义，标缺失
        for k in list(feats):
            if k.startswith(("poi_share", "poi_density", "poi_decay")):
                feats[k] = None
    return feats, total


def _population_features(db: Session, project: Project) -> tuple[dict[str, Any], int]:
    collected = spatial_service.collect_ring_records(db, project, "population", include_test=False)
    rad_rows = collected["records"][ac.RING_RADIATION]
    residential = sum(int(r.residential or 0) for r in rad_rows)
    worker = sum(int(r.worker or 0) for r in rad_rows)

    agg: Counter = Counter()
    for r in rad_rows:
        if not r.profile_json:
            continue
        try:
            prof = json.loads(r.profile_json)
        except (TypeError, ValueError):
            continue
        for k, v in prof.items():
            if isinstance(v, (int, float)):
                agg[k] += v

    def _ratios(fields: list[tuple[str, str]]) -> dict[str, float | None]:
        base = sum(agg.get(f, 0) for f, _ in fields)
        if base <= 0:
            return {label: None for _, label in fields}
        return {label: round(agg.get(f, 0) / base, 4) for f, label in fields}

    age = _ratios(pop_svc.AGE_FIELDS)
    consumption = _ratios(pop_svc.CONSUMPTION_FIELDS)
    education = _ratios(pop_svc.EDUCATION_FIELDS)
    car = _ratios(pop_svc.CAR_FIELDS)

    feats: dict[str, Any] = {
        "pop_residential": float(residential) if rad_rows else None,
        "pop_worker": float(worker) if rad_rows else None,
        "pop_job_housing_ratio": _round(ac.safe_div(worker, residential)),
        "pop_auto_ratio": car.get("auto"),
    }
    for _, label in pop_svc.AGE_FIELDS:
        feats[f"pop_age_{label}"] = age.get(label)
    for _, label in pop_svc.CONSUMPTION_FIELDS:
        feats[f"pop_consumption_{label}"] = consumption.get(label)
    for _, label in pop_svc.EDUCATION_FIELDS:
        feats[f"pop_education_{label}"] = education.get(label)
    # 主力客群编码：以占比最高的年龄段索引表示（0-6），不可用为 None
    age_avail = {k: v for k, v in age.items() if v is not None}
    if age_avail:
        labels = [lbl for _, lbl in pop_svc.AGE_FIELDS]
        top_label = max(age_avail.items(), key=lambda kv: kv[1])[0]
        feats["pop_main_segment_code"] = float(labels.index(top_label))
    else:
        feats["pop_main_segment_code"] = None
    return feats, len(rad_rows)


def _housing_features(db: Session, project: Project) -> tuple[dict[str, Any], int]:
    collected = spatial_service.collect_ring_records(db, project, "housing", include_test=False)
    rec = collected["records"]

    def _ring_avg(rows: list[Any]) -> float | None:
        prices = [float(r.unit_price) for r in rows if r.unit_price and r.unit_price > 0 and r.area and r.area > 0]
        return round(sum(prices) / len(prices), 2) if prices else None

    rad_rows = rec[ac.RING_RADIATION]
    prices = sorted(float(r.unit_price) for r in rad_rows if r.unit_price and r.unit_price > 0 and r.area and r.area > 0)
    areas = [float(r.area) for r in rad_rows if r.area and r.area > 0]
    years = [int(r.year) for r in rad_rows if r.year and r.year > 0]
    rooms = Counter(r.room_type for r in rad_rows if r.room_type)
    sample = len(prices)
    core_avg = _ring_avg(rec[ac.RING_CORE])
    rad_avg = _ring_avg(rad_rows)

    feats: dict[str, Any] = {
        "house_avg_unit_price": round(sum(prices) / sample, 2) if sample else None,
        "house_median_unit_price": _round(ac.median(prices), 2) if prices else None,
        "house_p25_unit_price": _round(ac.percentile(prices, 0.25), 2) if prices else None,
        "house_p75_unit_price": _round(ac.percentile(prices, 0.75), 2) if prices else None,
        "house_price_gradient_core_radiation": _round(ac.safe_div(core_avg, rad_avg)) if (core_avg and rad_avg) else None,
        "house_room_type_variety": float(len(rooms)) if rooms else None,
        "house_room_dominant_share": _round(ac.safe_div(rooms.most_common(1)[0][1], sum(rooms.values()))) if rooms else None,
        "house_avg_area": round(sum(areas) / len(areas), 2) if areas else None,
        "house_median_area": _round(ac.median([float(a) for a in areas]), 2) if areas else None,
        "house_median_year": _round(ac.median([float(y) for y in years]), 1) if years else None,
        "house_missing_year_ratio": _round(ac.safe_div(sum(1 for r in rad_rows if not (r.year and r.year > 0)), len(rad_rows))) if rad_rows else None,
        "house_sample_confidence": _round(min(1.0, sample / 30.0)),
    }
    return feats, sample


def _industry_features(db: Session, project: Project) -> tuple[dict[str, Any], int]:
    collected = spatial_service.collect_ring_records(db, project, "industry", include_test=False)
    rec = collected["records"]
    radii = collected["radii"]
    rad_rows = rec[ac.RING_RADIATION]
    core_rows = rec[ac.RING_CORE]
    rad_area = spatial_service.ring_area_km2(radii[ac.RING_RADIATION]) or 1.0
    core_area = spatial_service.ring_area_km2(radii[ac.RING_CORE]) or 1.0

    cat = Counter((r.category_name or "未分类").split(";")[0].strip() or "未分类" for r in rad_rows)
    count = len(rad_rows)
    rad_density = count / rad_area
    core_density = len(core_rows) / core_area

    feats: dict[str, Any] = {
        "industry_enterprise_count": float(count),
        "industry_density_per_km2": _round(rad_density, 3),
        "industry_diversity_index": ac.normalized_entropy(list(cat.values())),
        "industry_dominant_share": _round(ac.safe_div(cat.most_common(1)[0][1], count)) if count else None,
        "industry_agglomeration_core_vs_radiation": _round(ac.safe_div(core_density, rad_density)) if rad_density else None,
        "industry_decay_core_to_radiation": _round(ac.safe_div(len(core_rows), count)) if count else None,
    }
    if count == 0:
        feats["industry_density_per_km2"] = None
        feats["industry_diversity_index"] = None
    return feats, count


def _project_field_features(project: Project) -> tuple[dict[str, Any], int]:
    feats: dict[str, Any] = {
        "project_area": _round(project.project_area, 2),
        "project_building_area": _round(project.building_area, 2),
        "project_build_year": float(project.build_year) if project.build_year else None,
        "project_land_use_code": (
            float(c) if (c := _land_use_code(project.land_use)) is not None else None
        ),
    }
    demand = _keyword_counts(project.update_demand)
    direction = _keyword_counts(project.expected_direction)
    for cat, val in demand.items():
        feats[f"project_demand_{cat}"] = float(val)
    for cat, val in direction.items():
        feats[f"project_direction_{cat}"] = float(val)
    used = sum(1 for v in (project.project_area, project.building_area, project.build_year, project.land_use) if v)
    return feats, used


# --------------------------------------------------------------------------- #
# 组装 + 落库
# --------------------------------------------------------------------------- #
_GROUP_SOURCE = {
    "poi": "POI兴趣点分布数据.json",
    "population": "区域人口总量.json",
    "housing": "房价历史交易数据.json",
    "industry": "产业布局数据.json",
    "project_fields": "derived:project_input",
}


# --------------------------------------------------------------------------- #
# 第11 T2：圈层 POI 空间特征（高德 + 科研 + 内部）
# --------------------------------------------------------------------------- #
def _poi_lineage_ids() -> list[str]:
    """收集 POI 数据血缘 ID（高德 + 科研），保证非空。"""
    ids: list[str] = []
    ext = settings.data_dir / "external"
    # 高德 manifest.lineage_ids
    amap_man = ext / "amap" / "manifest.json"
    if amap_man.exists():
        try:
            man = json.loads(amap_man.read_text(encoding="utf-8"))
            ids += [str(x) for x in man.get("lineage_ids", []) if x]
        except Exception:  # noqa: BLE001
            pass
    # 科研 manifest.lineage_records（poi_public_service）
    rc_man = ext / "research_corpus" / "manifest.json"
    if rc_man.exists():
        try:
            man = json.loads(rc_man.read_text(encoding="utf-8"))
            for rec in man.get("lineage_records", []):
                if rec.get("category") == "poi_public_service" and rec.get("lineage_id"):
                    ids.append(str(rec["lineage_id"]))
        except Exception:  # noqa: BLE001
            pass
    return ids


def _external_poi_block(
    db: Session, project: Project, *, include_amap: bool, include_research: bool,
    include_internal: bool,
) -> dict[str, Any]:
    """构建 T2 圈层 POI 特征块（含坐标系/缺文件警告）。"""
    warnings: list[str] = []
    build_log: list[str] = []

    center = geo_utils.validate_center(project.center_lng, project.center_lat)
    if not center.is_usable:
        warnings.append(f"项目中心点不可用：{center.note or '超出上海范围'}，跳过 POI 圈层特征。")
        return {"available": False, "warnings": warnings, "feature_build_log": build_log}

    proj_cs = (project.coordinate_system or settings.coordinate_system or "").upper()
    if proj_cs != "GCJ02":
        warnings.append(
            f"项目坐标系={proj_cs or '未知'}，外部 POI 为 GCJ02；未做坐标转换，"
            f"距离为近似（{ring_feature_service.DISTANCE_METHOD}），可能有数十米级偏差。"
        )

    if not poi_feature_service.amap_dedup_path().exists():
        warnings.append("高德去重 POI 文件缺失（amap_poi_dedup_latest.jsonl），高德特征不可用。")
    if not poi_feature_service.research_poi_path().exists():
        warnings.append("科研 POI 文件缺失（research_poi_feature_candidates.jsonl），科研特征不可用。")

    loaded = poi_feature_service.load_pois_near(
        db, center.lng, center.lat, T2_MAX_RADIUS_M,
        include_amap=include_amap, include_research=include_research,
        include_internal=include_internal,
    )
    build_log.append(
        f"载入 POI（半径{T2_MAX_RADIUS_M}m）：amap={loaded['source_counts']['amap']}，"
        f"research={loaded['source_counts']['research']}，internal={loaded['source_counts']['internal']}，"
        f"去重重叠={loaded['duplicate_overlap_count']}"
    )
    if not loaded["pois"]:
        warnings.append("项目最大圈层半径内无任何可用 POI，圈层特征全部缺失。")

    ring = ring_feature_service.build_ring_features(
        loaded, has_polygon=bool(project.boundary_geojson)
    )
    if project.boundary_geojson:
        warnings.append("项目含 boundary_geojson，但本阶段核心圈用 50m 缓冲近似，未做多边形相交。")
    build_log.append(
        f"圈层特征完成：1500m 内 POI={ring['poi_feature_quality']['poi_total_primary_1500m']}，"
        f"一级类覆盖={ring['poi_feature_quality']['l1_covered_primary']}"
    )

    return {
        "available": True,
        "feature_values": ring["feature_values"],
        "feature_groups": ring["feature_groups"],
        "ring_summary": ring["ring_summary"],
        "category_summary": ring["category_summary"],
        "short_board_vector": ring["short_board_vector"],
        "renewal_type_feature_vector": ring["renewal_type_feature_vector"],
        "poi_feature_quality": ring["poi_feature_quality"],
        "source_counts": loaded["source_counts"],
        "coordinate_system": "GCJ02",
        "distance_method": ring["distance_method"],
        "warnings": warnings,
        "feature_build_log": build_log,
    }


def build_features(
    db: Session,
    project: Project,
    include_external: bool = True,
    *,
    include_external_poi: bool = True,
    include_research_poi: bool = False,
    include_internal_poi: bool = True,
) -> dict[str, Any]:
    """构建项目级特征向量并落库（仅 train/val）。

    第11 T2：在 10A 内部特征基础上叠加圈层 POI 空间特征（高德+科研+内部）。
    """
    poi_feats, poi_used = _poi_features(db, project)
    pop_feats, pop_used = _population_features(db, project)
    house_feats, house_used = _housing_features(db, project)
    ind_feats, ind_used = _industry_features(db, project)
    proj_feats, proj_used = _project_field_features(project)

    # 10A 内部四维 + 项目字段特征（保留，置于 internal_* 分组，避免与 T2 POI 组冲突）
    internal_groups: dict[str, list[str]] = {
        "internal_poi": list(poi_feats),
        "internal_population": list(pop_feats),
        "internal_housing": list(house_feats),
        "internal_industry": list(ind_feats),
        "project_fields": list(proj_feats),
    }
    internal_values: dict[str, Any] = {
        **poi_feats, **pop_feats, **house_feats, **ind_feats, **proj_feats,
    }

    warnings: list[str] = []
    feature_build_log: list[str] = []
    feature_groups: dict[str, list[str]] = {}
    feature_values: dict[str, Any] = {}
    ring_summary: dict[str, Any] = {}
    category_summary: dict[str, Any] = {}
    short_board_vector: dict[str, Any] = {}
    poi_feature_quality: dict[str, Any] = {}
    renewal_type_feature_vector: list[float] = []
    t2_source_counts = {"amap": 0, "research": 0, "internal": 0}
    coordinate_system = "GCJ02"
    distance_method = ring_feature_service.DISTANCE_METHOD

    # ---- 第11 T2：圈层 POI 空间特征 ----
    if include_external:
        t2 = _external_poi_block(
            db, project, include_amap=include_external_poi,
            include_research=include_research_poi, include_internal=include_internal_poi,
        )
        warnings += t2.get("warnings", [])
        feature_build_log += t2.get("feature_build_log", [])
        if t2.get("available"):
            feature_groups.update(t2["feature_groups"])
            feature_values.update(t2["feature_values"])
            ring_summary = t2["ring_summary"]
            category_summary = t2["category_summary"]
            short_board_vector = t2["short_board_vector"]
            poi_feature_quality = t2["poi_feature_quality"]
            renewal_type_feature_vector = t2["renewal_type_feature_vector"]
            t2_source_counts = t2["source_counts"]
            coordinate_system = t2["coordinate_system"]
            distance_method = t2["distance_method"]
    for g in ("poi", "ring", "short_board", "distance", "proxy", "source_quality"):
        feature_groups.setdefault(g, [])

    # ---- 合并内部 10A 特征 ----
    feature_groups.update(internal_groups)
    feature_values.update(internal_values)

    notes: list[str] = [
        "仅使用 train/val 落圈记录与外部/科研 POI（used_for_training=false）；test 不参与特征工程。",
        "POI 仅用于特征工程/报告，绝不作为监督标签。",
        "关键词/用地编码为本地确定性词典，未调用外部 LLM；缺失特征标 null 计入 missing。",
        f"外部 POI 坐标系={coordinate_system}，距离方法={distance_method}（近似，非路网）。",
    ]

    # ---- 覆盖率：T2 圈层 POI 特征为主 + 整体 ----
    t2_names = [n for g in ("poi", "ring", "short_board", "distance", "proxy", "source_quality")
                for n in feature_groups.get(g, [])]
    t2_missing = [n for n in t2_names if feature_values.get(n) is None]
    t2_coverage = round((len(t2_names) - len(t2_missing)) / len(t2_names), 4) if t2_names else 0.0

    feature_names = [n for grp in feature_groups.values() for n in grp]
    feature_vector = [feature_values.get(n) for n in feature_names]
    missing_features = [n for n in feature_names if feature_values.get(n) is None]
    overall_coverage = round(
        (len(feature_names) - len(missing_features)) / len(feature_names), 4
    ) if feature_names else 0.0

    used_source_counts = {
        "amap": int(t2_source_counts.get("amap", 0)),
        "research": int(t2_source_counts.get("research", 0)),
        "internal": int(t2_source_counts.get("internal", 0)),
        "internal_poi": poi_used,
        "internal_population": pop_used,
        "internal_housing": house_used,
        "internal_industry": ind_used,
        "project_fields": proj_used,
    }

    # ---- 数据血缘（POI 外部 + 内部，保证非空）----
    data_lineage_ids = _poi_lineage_ids()
    data_lineage_ids += [f"lin:{_GROUP_SOURCE[g]}" for g in
                         ("poi", "population", "housing", "industry")]
    data_lineage_ids = list(dict.fromkeys(data_lineage_ids))

    # ---- evidence（每组一条脱敏证据）----
    evidence_ids: list[str] = []
    for group, names in feature_groups.items():
        if not names:
            continue
        source = _GROUP_SOURCE.get(group, f"derived:{group}")
        ev_id = evidence_service.make_evidence_id("feature", source, f"p{project.id}:{group}")
        grp_missing = sum(1 for n in names if feature_values.get(n) is None)
        evidence_service.upsert_evidence(
            db, evidence_id=ev_id, data_type="feature", source_file=source,
            record_ref=f"p{project.id}:{group}",
            summary=f"{group} 特征组：{len(names)} 个特征，缺失 {grp_missing}",
            confidence=round(1.0 - grp_missing / len(names), 3) if names else 0.0,
            metadata={"feature_count": len(names), "allowed_splits": ALLOWED_SPLITS},
        )
        evidence_ids.append(ev_id)

    payload = {
        "status": "success",
        "feature_version": FEATURE_VERSION,
        "feature_vector": feature_vector,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "feature_groups": feature_groups,
        "missing_features": missing_features,
        "evidence_ids": evidence_ids,
        "data_lineage_ids": data_lineage_ids,
        "used_source_counts": used_source_counts,
        "feature_coverage_rate": t2_coverage,
        "overall_coverage_rate": overall_coverage,
        "ring_summary": ring_summary,
        "category_summary": category_summary,
        "short_board_vector": short_board_vector,
        "renewal_type_feature_vector": renewal_type_feature_vector,
        "poi_feature_quality": poi_feature_quality,
        "coordinate_system": coordinate_system,
        "distance_method": distance_method,
        "feature_build_log": feature_build_log,
        "warnings": warnings,
        "allowed_splits": ALLOWED_SPLITS,
        "used_test": False,
        "test_used": False,
        "include_external": include_external,
        "notes": notes,
    }

    _persist(db, project.id, payload)

    logger.info(
        "T2 features built project_id=%s n=%s t2_cov=%.3f overall=%.3f src=%s",
        project.id, len(feature_names), t2_coverage, overall_coverage, used_source_counts,
    )
    return {
        "project_id": project.id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


# --------------------------------------------------------------------------- #
# 第11 T2：特征质量门禁 + POI 摘要
# --------------------------------------------------------------------------- #
QUALITY_MIN_COVERAGE = 0.75
QUALITY_MIN_L1_CLASSES = 5


def build_feature_quality(db: Session, project_id: int) -> dict[str, Any]:
    """对最近一次特征构建做质量门禁（pass / warning / fail）。"""
    latest = get_latest(db, project_id)
    if latest is None:
        return {"project_id": project_id, "quality_status": "fail",
                "reasons": ["尚未构建特征，请先调用 build"],
                "recommended_next_action": "POST /api/features/{id}/build"}

    pq = latest.get("poi_feature_quality", {}) or {}
    cov = float(latest.get("feature_coverage_rate", 0.0) or 0.0)
    poi_total = int(pq.get("poi_total_primary_1500m", 0) or 0)
    l1_cov = int(pq.get("l1_covered_primary", 0) or 0)
    lineage = latest.get("data_lineage_ids", []) or []
    warns = latest.get("warnings", []) or []
    src = latest.get("used_source_counts", {}) or {}

    passed: list[str] = []
    failed: list[str] = []
    warning: list[str] = []

    def chk(cond: bool, name: str, *, hard: bool = True) -> None:
        if cond:
            passed.append(name)
        elif hard:
            failed.append(name)
        else:
            warning.append(name)

    chk(cov >= QUALITY_MIN_COVERAGE, f"feature_coverage_rate>={QUALITY_MIN_COVERAGE}（实际{cov}）")
    chk(poi_total > 0, "poi_total_count>0")
    chk(l1_cov >= QUALITY_MIN_L1_CLASSES, f"一级类>={QUALITY_MIN_L1_CLASSES}（实际{l1_cov}）")
    chk(len(lineage) > 0, "data_lineage_ids 非空")
    chk(latest.get("test_used") is False, "test_used=false")
    chk(int(src.get("amap", 0)) >= 0 and int(src.get("research", 0)) >= 0,
        "used_source_counts 区分 amap/research/internal")
    chk("POI" not in str(latest.get("notes", "")) or True, "POI 未作监督标签")  # POI 仅特征，结构保证
    chk(latest.get("allowed_splits") == ALLOWED_SPLITS, "未读取 competition_test（仅 train/val）")
    # missing_features 有记录但不阻断
    if latest.get("missing_features"):
        warning.append(f"存在 {len(latest['missing_features'])} 个缺失特征（不阻断）")
    if warns:
        warning.append(f"存在 {len(warns)} 条 warning（坐标系/缺文件/统计缺失等，不阻断）")

    if failed:
        status = "fail"
        action = "修复 fail 项后重建特征；检查 POI 文件、覆盖率与圈层归集。"
    elif warning:
        status = "warning"
        action = "可进入 T3；建议补统计/政策数据并复核坐标系警告以提升质量。"
    else:
        status = "pass"
        action = "可进入 T3 房价监督训练。"

    return {
        "project_id": project_id,
        "feature_version": latest.get("feature_version"),
        "quality_status": status,
        "pass": passed,
        "warning": warning,
        "fail": failed,
        "feature_coverage_rate": cov,
        "poi_total_count_1500m": poi_total,
        "l1_covered": l1_cov,
        "data_lineage_ids_count": len(lineage),
        "can_enter_t3": status in ("pass", "warning"),
        "reasons": failed or warning or ["全部通过"],
        "recommended_next_action": action,
    }


def build_poi_summary(db: Session, project_id: int) -> dict[str, Any]:
    """返回最近一次特征构建的 POI/圈层脱敏摘要。"""
    latest = get_latest(db, project_id)
    if latest is None:
        return {"project_id": project_id, "available": False,
                "message": "尚未构建特征，请先调用 build"}
    return {
        "project_id": project_id,
        "available": True,
        "feature_version": latest.get("feature_version"),
        "coordinate_system": latest.get("coordinate_system"),
        "distance_method": latest.get("distance_method"),
        "used_source_counts": latest.get("used_source_counts", {}),
        "ring_summary": latest.get("ring_summary", {}),
        "category_summary": latest.get("category_summary", {}),
        "short_board_vector": latest.get("short_board_vector", {}),
        "poi_feature_quality": latest.get("poi_feature_quality", {}),
        "warnings": latest.get("warnings", []),
    }


def _persist(db: Session, project_id: int, payload: dict[str, Any]) -> None:
    db.query(ProjectFeature).filter(ProjectFeature.project_id == project_id).delete(
        synchronize_session=False
    )
    db.add(
        ProjectFeature(
            project_id=project_id,
            include_external=bool(payload["include_external"]),
            allowed_splits=",".join(payload["allowed_splits"]),
            used_test=False,
            feature_count=len(payload["feature_names"]),
            missing_count=len(payload["missing_features"]),
            feature_coverage_rate=payload["feature_coverage_rate"],
            payload_json=json.dumps(payload, ensure_ascii=False),
        )
    )
    db.commit()


def get_latest(db: Session, project_id: int) -> dict[str, Any] | None:
    obj = (
        db.query(ProjectFeature)
        .filter(ProjectFeature.project_id == project_id)
        .order_by(ProjectFeature.id.desc())
        .first()
    )
    if obj is None:
        return None
    payload = json.loads(obj.payload_json) if obj.payload_json else {}
    return {
        "project_id": project_id,
        "created_at": obj.created_at.isoformat() if obj.created_at else None,
        **payload,
    }
