"""坐标解析与合法性校验工具（第2阶段）。

对齐 docs/11 D1 决策：
- 一律按 ``coordinates = [lng, lat]``（经度在前、纬度在后）解析。
- 上海范围合法性校验：lng ∈ [120, 123]，lat ∈ [30, 32]。
- 若顺序明显反了（即 [lat, lng]）则自动纠正并标记 ``corrected``。
- 人口为网格：coordinates = [[lng,lat],[lng,lat]] 两角点。

本阶段只做解析与校验，**不计算距离**（距离留第4阶段米制投影/Haversine）。
本模块不打印、不记录任何坐标原值到日志，仅返回结构化结果供上层聚合统计。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# 上海经纬度合法范围（D1）
SHANGHAI_LNG_MIN, SHANGHAI_LNG_MAX = 120.0, 123.0
SHANGHAI_LAT_MIN, SHANGHAI_LAT_MAX = 30.0, 32.0

# 坐标状态枚举
STATUS_OK = "ok"
STATUS_CORRECTED = "corrected"
STATUS_INVALID = "invalid"
STATUS_MISSING = "missing"


def in_shanghai(lng: float, lat: float) -> bool:
    """判断经纬度是否落在上海合法范围内。"""
    return (
        SHANGHAI_LNG_MIN <= lng <= SHANGHAI_LNG_MAX
        and SHANGHAI_LAT_MIN <= lat <= SHANGHAI_LAT_MAX
    )


@dataclass
class PointParseResult:
    """单点坐标解析结果。"""

    lng: float | None
    lat: float | None
    status: str
    note: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.status in (STATUS_OK, STATUS_CORRECTED) and self.lng is not None


def _to_float_pair(coordinates) -> tuple[float, float] | None:
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
        return None
    try:
        return float(coordinates[0]), float(coordinates[1])
    except (TypeError, ValueError):
        return None


def parse_point(coordinates) -> PointParseResult:
    """解析单点 [lng, lat]，必要时自动纠正 lat/lng 顺序。

    返回的 status：ok / corrected / invalid / missing。
    """
    pair = _to_float_pair(coordinates)
    if pair is None:
        return PointParseResult(None, None, STATUS_MISSING, "coordinates 非法或缺失")

    a, b = pair  # 默认 a=lng, b=lat（D1）
    if in_shanghai(a, b):
        return PointParseResult(a, b, STATUS_OK)
    # 顺序可能反了：尝试 [lat, lng]
    if in_shanghai(b, a):
        return PointParseResult(b, a, STATUS_CORRECTED, "已纠正 lat/lng 顺序")
    return PointParseResult(a, b, STATUS_INVALID, "超出上海合法范围")


@dataclass
class BboxParseResult:
    """人口网格边界框解析结果。"""

    center_lng: float | None
    center_lat: float | None
    grid_key: str | None
    bbox_geojson: str | None
    status: str
    note: str | None = None


def grid_key_from_points(points: list[tuple[float, float]]) -> str:
    """由两角点生成稳定 grid_key（四舍五入 + 排序，消除顺序差异）。

    已验证：人口总量与人口画像两文件用该 key 可 1:1 对齐。
    """
    norm = tuple(sorted((round(p[0], 4), round(p[1], 4)) for p in points))
    digest = hashlib.sha1(repr(norm).encode("utf-8")).hexdigest()[:12]
    return f"grid_{digest}"


def _bbox_geojson(min_lng: float, min_lat: float, max_lng: float, max_lat: float) -> str:
    """构造矩形 GeoJSON Polygon 字符串（不含任何业务原文）。"""
    import json

    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lng, min_lat],
                [max_lng, min_lat],
                [max_lng, max_lat],
                [min_lng, max_lat],
                [min_lng, min_lat],
            ]
        ],
    }
    return json.dumps(polygon, ensure_ascii=False)


def parse_bbox(coordinates) -> BboxParseResult:
    """解析人口网格 [[lng,lat],[lng,lat]] 两角点。

    返回中心点、grid_key、bbox GeoJSON 与状态。
    """
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
        return BboxParseResult(None, None, None, None, STATUS_MISSING, "网格 coordinates 非法")

    p0 = parse_point(coordinates[0])
    p1 = parse_point(coordinates[1])
    if not (p0.is_usable and p1.is_usable):
        # 任一角点不可用：保留 grid_key（基于原值）以便排查，但状态置 invalid/missing
        status = STATUS_INVALID if (p0.status != STATUS_MISSING and p1.status != STATUS_MISSING) else STATUS_MISSING
        return BboxParseResult(None, None, None, None, status, "网格角点坐标不可用")

    pts = [(p0.lng, p0.lat), (p1.lng, p1.lat)]
    grid_key = grid_key_from_points(pts)
    lngs = [pts[0][0], pts[1][0]]
    lats = [pts[0][1], pts[1][1]]
    center_lng = round(sum(lngs) / 2, 6)
    center_lat = round(sum(lats) / 2, 6)
    geojson = _bbox_geojson(min(lngs), min(lats), max(lngs), max(lats))
    status = STATUS_CORRECTED if STATUS_CORRECTED in (p0.status, p1.status) else STATUS_OK
    note = "网格角点顺序已纠正" if status == STATUS_CORRECTED else None
    return BboxParseResult(center_lng, center_lat, grid_key, geojson, status, note)


# 地球平均半径（米），WGS84 IUGG 推荐均值，用于 Haversine 距离
EARTH_RADIUS_M = 6371008.8


def haversine_m(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """两点间 Haversine 地理距离（米）。

    第4阶段圈层归集专用：**严禁直接用经纬度差值当距离**（D1）。
    入参顺序统一为 [lng, lat]，与 D1 解析口径一致。
    """
    import math

    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_M * c


def validate_center(lng, lat) -> PointParseResult:
    """校验项目中心点（手动输入 [lng, lat]）。

    复用 parse_point 的合法性校验与 lat/lng 自动纠偏，返回可用状态。
    """
    return parse_point([lng, lat])


def cell_key(lng: float, lat: float, size_deg: float = 0.002) -> str:
    """坐标粗网格 cell key，用于 POI/产业空间整组切分防近邻泄露。

    采用 floor 分箱：同一 cell 内（默认 ~0.002°≈200m）的点整组进入同一 split，
    既保证近邻不跨 split，又能获得足够多的空间组以接近目标切分比例。
    """
    import math

    cx = math.floor(lng / size_deg)
    cy = math.floor(lat / size_deg)
    return f"cell_{cx}_{cy}"
