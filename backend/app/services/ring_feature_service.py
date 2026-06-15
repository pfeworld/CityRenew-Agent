"""第11 T2：圈层空间特征计算服务。

输入 poi_feature_service.load_pois_near 的归集结果（每条 {dist_m, l1, l2, source}），
按 core / 500 / 1500 / 3000 / 5000 圈层计算稳定可复算的空间特征：
基础数量(A) / 功能混合度代理(B) / 圈层衰减(C) / 配套短板(D) / 最近距离(E) / 类型辅助代理(F)。

红线：
- 代理指标字段名一律带 proxy 或 score，绝不伪装为真实人口画像。
- 缺数据输出 None（计入 missing），不编造；分母为 0 安全处理。
- 距离为 Haversine 近似（distance_method=haversine_approx_gcj02），不声称路网精准。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from app.services.poi_feature_service import (
    L1_COMMERCIAL,
    L1_CULTURE_SPORTS,
    L1_GOVERNMENT,
    L1_GREEN_SPACE,
    L1_INDUSTRY_OFFICE,
    L1_PUBLIC_SERVICE,
    L1_RESIDENTIAL,
    L1_TRANSPORT,
    L1_UNKNOWN,
    L1_URBAN_RENEWAL,
    L1_CLASSES,
)

# 圈层定义（米）；core 默认 50（仅中心点时），主分析圈 1500
RING_DEFS: tuple[tuple[str, int], ...] = (
    ("core", 50),
    ("ring_500m", 500),
    ("ring_1500m", 1500),
    ("ring_3000m", 3000),
    ("ring_5000m", 5000),
)
MANDATORY_RINGS = ("core", "ring_500m", "ring_1500m")
PRIMARY_RING_M = 1500

# 配套短板期望阈值（1500m 内"较好供给"的参考计数；写入常量，不依赖 test）
GAP_EXPECTED: dict[str, float] = {
    "education": 15, "medical": 12, "elderly_care": 5, "transit": 20,
    "parking": 25, "culture_sports": 10, "green_space": 6,
    "community_service": 10, "industry_support": 15, "commercial_supply": 60,
}

DISTANCE_METHOD = "haversine_approx_gcj02"


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _round(v: float | None, n: int = 4) -> float | None:
    return None if v is None else round(float(v), n)


def _area_km2(radius_m: int) -> float:
    return math.pi * (radius_m / 1000.0) ** 2


def _shannon_norm(counts: list[int]) -> float | None:
    total = sum(counts)
    if total <= 0:
        return None
    k = sum(1 for c in counts if c > 0)
    if k <= 1:
        return 0.0
    ent = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
    return round(ent / math.log(k), 4)


def _within(pois: list[dict[str, Any]], radius_m: int) -> list[dict[str, Any]]:
    return [p for p in pois if p["dist_m"] <= radius_m]


def _l1_counter(rows: list[dict[str, Any]]) -> Counter:
    return Counter(p["l1"] for p in rows)


def _l2_counter(rows: list[dict[str, Any]], l1: str) -> Counter:
    return Counter(p["l2"] for p in rows if p["l1"] == l1 and p["l2"])


def build_ring_features(
    loaded: dict[str, Any],
    *,
    has_polygon: bool = False,
) -> dict[str, Any]:
    """计算全部圈层空间特征，返回特征值/分组/摘要/短板/质量。"""
    pois: list[dict[str, Any]] = loaded["pois"]
    nearest: dict[str, float] = loaded.get("nearest", {})

    rings_rows = {name: _within(pois, r) for name, r in RING_DEFS}
    primary = rings_rows["ring_1500m"]
    r500 = rings_rows["ring_500m"]

    feat: dict[str, Any] = {}
    groups: dict[str, list[str]] = {
        "poi": [], "ring": [], "short_board": [], "distance": [], "proxy": [],
        "source_quality": [],
    }

    def put(group: str, name: str, value: Any) -> None:
        feat[name] = value
        groups[group].append(name)

    # ---- A 基础数量（每圈层）----
    for name, radius in RING_DEFS:
        rows = rings_rows[name]
        lc = _l1_counter(rows)
        put("poi", f"poi_total_count_{name}", float(len(rows)))
        put("poi", f"category_l1_count_{name}", float(len([c for c in lc.values() if c > 0])))
        put("poi", f"poi_density_per_km2_{name}", _round(len(rows) / _area_km2(radius), 3))

    # 主分析圈（1500m）一级类计数 / 占比 / 二级类
    lc_primary = _l1_counter(primary)
    total_primary = len(primary)
    l2_total = 0
    for l1 in L1_CLASSES:
        cnt = lc_primary.get(l1, 0)
        put("poi", f"category_l1_count_{l1}", float(cnt))
        put("poi", f"category_l1_ratio_{l1}",
            _round(_safe_div(cnt, total_primary)) if total_primary else None)
        l2c = _l2_counter(primary, l1)
        l2_total += len([1 for v in l2c.values() if v > 0])
    put("poi", "category_l2_count_total", float(l2_total))
    unknown_ratio = _safe_div(lc_primary.get(L1_UNKNOWN, 0), total_primary) if total_primary else None
    put("poi", "unknown_category_ratio", _round(unknown_ratio))

    # ---- source_quality ----
    sc = loaded.get("source_counts", {})
    put("source_quality", "source_count_amap", float(sc.get("amap", 0)))
    put("source_quality", "source_count_research", float(sc.get("research", 0)))
    put("source_quality", "source_count_internal", float(sc.get("internal", 0)))
    put("source_quality", "duplicate_overlap_count", float(loaded.get("duplicate_overlap_count", 0)))

    # ---- B 功能混合度代理（1500m）----
    l1_list_for_entropy = [lc_primary.get(l1, 0) for l1 in L1_CLASSES if l1 != L1_UNKNOWN]
    all_l2 = Counter(p["l2"] for p in primary if p["l2"])
    put("proxy", "shannon_entropy_l1", _shannon_norm(l1_list_for_entropy))
    put("proxy", "shannon_entropy_l2", _shannon_norm(list(all_l2.values())))
    put("proxy", "function_mix_score", _shannon_norm(l1_list_for_entropy))

    n_comm = lc_primary.get(L1_COMMERCIAL, 0)
    n_pub = lc_primary.get(L1_PUBLIC_SERVICE, 0)
    put("proxy", "commercial_public_balance",
        _round(_safe_div(n_comm, n_comm + n_pub)) if (n_comm + n_pub) else None)
    pub_l2 = _l2_counter(primary, L1_PUBLIC_SERVICE)
    put("proxy", "public_service_diversity",
        _round(len([1 for v in pub_l2.values() if v > 0]) / 4.0))

    def _l2cnt(l1: str, l2: str) -> int:
        return sum(1 for p in primary if p["l1"] == l1 and p["l2"] == l2)

    catering = _l2cnt(L1_COMMERCIAL, "catering")
    entertainment = _l2cnt(L1_COMMERCIAL, "entertainment")
    hotel = _l2cnt(L1_COMMERCIAL, "hotel")
    shopping = _l2cnt(L1_COMMERCIAL, "shopping") + _l2cnt(L1_COMMERCIAL, "supermarket")
    education = _l2cnt(L1_PUBLIC_SERVICE, "education")
    supermarket = _l2cnt(L1_COMMERCIAL, "supermarket")
    community = _l2cnt(L1_PUBLIC_SERVICE, "community_service")
    elderly = _l2cnt(L1_PUBLIC_SERVICE, "elderly_care")
    medical = _l2cnt(L1_PUBLIC_SERVICE, "medical")
    green = lc_primary.get(L1_GREEN_SPACE, 0)

    def _proxy_share(num: int) -> float | None:
        return _round(_safe_div(num, total_primary)) if total_primary else None

    put("proxy", "night_life_intensity_proxy", _proxy_share(catering + entertainment + hotel))
    put("proxy", "family_life_intensity_proxy", _proxy_share(education + supermarket + community + green))
    put("proxy", "youth_consumption_proxy", _proxy_share(entertainment + catering + shopping))
    put("proxy", "elderly_service_proxy", _proxy_share(elderly + medical + community))

    # ---- C 圈层衰减（500→1500 密度比）----
    def _density(rows: list[dict[str, Any]], radius: int, pred=None) -> float | None:
        n = len([p for p in rows if (pred is None or pred(p))])
        return n / _area_km2(radius)

    def _decay(pred=None) -> float | None:
        d500 = _density(r500, 500, pred)
        d1500 = _density(primary, 1500, pred)
        return _round(_safe_div(d500, d1500))

    put("ring", "total_poi_decay_500_to_1500", _decay())
    put("ring", "public_service_decay_500_to_1500", _decay(lambda p: p["l1"] == L1_PUBLIC_SERVICE))
    put("ring", "commercial_decay_500_to_1500", _decay(lambda p: p["l1"] == L1_COMMERCIAL))
    put("ring", "transport_decay_500_to_1500", _decay(lambda p: p["l1"] == L1_TRANSPORT))
    put("ring", "industry_decay_500_to_1500", _decay(lambda p: p["l1"] == L1_INDUSTRY_OFFICE))
    put("ring", "urban_renewal_decay_500_to_1500", _decay(lambda p: p["l1"] == L1_URBAN_RENEWAL))

    # ---- D 配套短板向量（1500m，规则阈值）----
    def _gap(actual: int, key: str) -> float | None:
        if total_primary == 0:
            return None  # 无任何 POI → 缺失，不编造
        exp = GAP_EXPECTED[key]
        return _round(max(0.0, min(1.0, 1.0 - actual / exp)))

    transit = (_l2cnt(L1_TRANSPORT, "metro") + _l2cnt(L1_TRANSPORT, "bus"))
    parking = _l2cnt(L1_TRANSPORT, "parking")
    culture_sports = lc_primary.get(L1_CULTURE_SPORTS, 0)
    industry = lc_primary.get(L1_INDUSTRY_OFFICE, 0)
    commercial = lc_primary.get(L1_COMMERCIAL, 0)

    short_board = {
        "education_gap_score": _gap(education, "education"),
        "medical_gap_score": _gap(medical, "medical"),
        "elderly_care_gap_score": _gap(elderly, "elderly_care"),
        "transit_gap_score": _gap(transit, "transit"),
        "parking_gap_score": _gap(parking, "parking"),
        "culture_sports_gap_score": _gap(culture_sports, "culture_sports"),
        "green_space_gap_score": _gap(green, "green_space"),
        "community_service_gap_score": _gap(community, "community_service"),
        "industry_support_gap_score": _gap(industry, "industry_support"),
        "commercial_supply_gap_score": _gap(commercial, "commercial_supply"),
    }
    for k, v in short_board.items():
        put("short_board", k, v)

    # ---- E 最近距离（米，Haversine 近似）----
    dist_map = {
        "nearest_metro_distance": "metro",
        "nearest_bus_distance": "bus",
        "nearest_hospital_distance": "hospital",
        "nearest_school_distance": "school",
        "nearest_elderly_care_distance": "elderly_care",
        "nearest_park_distance": "park",
        "nearest_mall_distance": "mall",
        "nearest_industrial_park_distance": "industrial_park",
        "nearest_government_service_distance": "government_service",
    }
    for fname, tgt in dist_map.items():
        d = nearest.get(tgt)
        put("distance", fname, _round(d, 1) if d is not None else None)

    # ---- F 项目类型辅助代理（仅特征，不分类）----
    renewal = lc_primary.get(L1_URBAN_RENEWAL, 0)
    metro_cnt = _l2cnt(L1_TRANSPORT, "metro")
    put("proxy", "community_support_score", _proxy_share(community + education + medical + elderly))
    put("proxy", "TOD_potential_score",
        _round(min(1.0, (metro_cnt + transit / 5.0) / 10.0)))
    put("proxy", "industry_upgrade_score", _proxy_share(industry + renewal))
    put("proxy", "commercial_vitality_score", _proxy_share(commercial))
    psg = short_board["education_gap_score"]
    medg = short_board["medical_gap_score"]
    if psg is not None and medg is not None:
        put("proxy", "public_service_shortage_score", _round((psg + medg) / 2.0))
    else:
        put("proxy", "public_service_shortage_score", None)
    put("proxy", "residential_support_score", _proxy_share(supermarket + community + green + medical))
    put("proxy", "culture_tourism_potential_score", _proxy_share(culture_sports + green))

    renewal_type_feature_vector = [
        float(sum(1 for p in primary if p["l1"] == L1_URBAN_RENEWAL and p["l2"] == l2))
        for l2 in ("old_factory", "old_residential", "urban_village",
                   "renovation_project", "TOD", "waterfront", "low_efficiency_land")
    ]

    # ---- 摘要 ----
    ring_summary = {
        name: {
            "radius_m": radius,
            "poi_total": len(rings_rows[name]),
            "l1_covered": len([c for c in _l1_counter(rings_rows[name]).values() if c > 0]),
        }
        for name, radius in RING_DEFS
    }
    category_summary = {
        "primary_ring_m": PRIMARY_RING_M,
        "l1_counts": {l1: lc_primary.get(l1, 0) for l1 in L1_CLASSES},
        "l2_top": dict(all_l2.most_common(15)),
    }

    # ---- POI 特征质量 ----
    l1_covered_primary = len([c for c in lc_primary.values() if c > 0])
    poi_feature_quality = {
        "poi_total_primary_1500m": total_primary,
        "l1_covered_primary": l1_covered_primary,
        "unknown_category_ratio": _round(unknown_ratio),
        "duplicate_overlap_count": loaded.get("duplicate_overlap_count", 0),
        "source_counts": sc,
        "distance_method": DISTANCE_METHOD,
        "has_polygon": has_polygon,
    }

    return {
        "feature_values": feat,
        "feature_groups": groups,
        "ring_summary": ring_summary,
        "category_summary": category_summary,
        "short_board_vector": short_board,
        "renewal_type_feature_vector": renewal_type_feature_vector,
        "poi_feature_quality": poi_feature_quality,
        "distance_method": DISTANCE_METHOD,
    }
