"""空间圈层分析服务（第4阶段）。

职责：
- 校验项目中心点（D1：[lng, lat]、上海范围、自动纠偏）。
- 无红线时基于中心点生成 core / nearby(500) / radiation(1500) 三圈层。
- 用 **Haversine 米制距离** 把 POI / 房价 / 产业点归集到圈层（严禁经纬度差值）。
- 人口网格先用网格中心点归集（后续可升级面积加权）。
- 默认仅用 train / val（排除 test）；include_test=true 才纳入 test。
- 仅输出统计数量与按 split 分组摘要，不返回 raw_json / 原始坐标 / 原始明细。

圈层口径（D4）：
- core：核心缓冲半径 = core_buffer_m；为 0 且无红线时兜底 default_core_buffer_m(150)。
- nearby：nearby_buffer_m（默认 500）。
- radiation：radiation_buffer_m（默认 1500）。
红线相关：boundary_geojson 本阶段仅保存与返回，不做多边形相交计算（预留）。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    HousingRecord,
    IndustryPoint,
    PoiPoint,
    PopulationProfile,
    Project,
)
from app.utils import geo_utils

logger = logging.getLogger("cityrenew.spatial")

RING_CORE = "core"
RING_NEARBY = "nearby"
RING_RADIATION = "radiation"
RING_ORDER = (RING_CORE, RING_NEARBY, RING_RADIATION)

# data_type -> (Model, 坐标类型, RingCounts 字段名)
SPATIAL_SOURCES: dict[str, tuple[Any, str, str]] = {
    "poi": (PoiPoint, "point", "poi_count"),
    "housing": (HousingRecord, "point", "housing_count"),
    "industry": (IndustryPoint, "point", "industry_count"),
    "population": (PopulationProfile, "grid", "population_grid_count"),
}

USABLE_COORD_STATUS = {None, geo_utils.STATUS_OK, geo_utils.STATUS_CORRECTED}


class SpatialError(ValueError):
    """中心点缺失或不合法等空间分析前置错误。"""


def _resolve_center(project: Project) -> tuple[float, float, str]:
    """解析并校验项目中心点，返回 (lng, lat, status)。"""
    if project.center_lng is None or project.center_lat is None:
        raise SpatialError("项目缺少中心点（center_lng/center_lat），无法生成圈层")
    result = geo_utils.validate_center(project.center_lng, project.center_lat)
    if not result.is_usable:
        raise SpatialError(f"项目中心点不合法：{result.note or '超出上海合法范围'}")
    return result.lng, result.lat, result.status


def _ring_radii(project: Project) -> tuple[int, int, int]:
    """计算 core / nearby / radiation 三圈层外缘半径（米）。"""
    core_r = project.core_buffer_m or 0
    if core_r <= 0:
        # core_buffer_m=0：无论是否有红线，本阶段均用兜底半径（红线多边形暂不处理）
        core_r = settings.default_core_buffer_m
    nearby_r = project.nearby_buffer_m or settings.nearby_buffer_m
    radiation_r = project.radiation_buffer_m or settings.radiation_buffer_m
    return core_r, nearby_r, radiation_r


def _allowed_splits(include_test: bool) -> list[str]:
    """默认仅 train/val；include_test=true 才追加 test（即使 demo 模式亦如此）。"""
    splits = ["train", "val"]
    if include_test:
        splits.append("test")
    return splits


def _usable_coord(row: Any, coord_type: str) -> tuple[float, float] | None:
    if coord_type == "grid":
        lng, lat = row.center_lng, row.center_lat
    else:
        lng, lat = row.lng, row.lat
    if lng is None or lat is None:
        return None
    if getattr(row, "coord_status", None) not in USABLE_COORD_STATUS:
        return None
    return float(lng), float(lat)


def _classify_ring(distance_m: float, core_r: int, nearby_r: int, radiation_r: int) -> str | None:
    """按距离归入互斥圈层带；超出 radiation 返回 None。"""
    if distance_m <= core_r:
        return RING_CORE
    if distance_m <= nearby_r:
        return RING_NEARBY
    if distance_m <= radiation_r:
        return RING_RADIATION
    return None


def _aggregate_one(
    db: Session,
    model: Any,
    coord_type: str,
    center_lng: float,
    center_lat: float,
    radii: tuple[int, int, int],
    allowed_splits: list[str],
) -> dict[str, Any]:
    """归集单个数据类型：返回互斥带计数 / 累计计数 / 按 split 分组 / 跳过数。"""
    core_r, nearby_r, radiation_r = radii

    exclusive = {r: 0 for r in RING_ORDER}
    cumulative = {r: 0 for r in RING_ORDER}
    by_split: dict[str, dict[str, int]] = {r: defaultdict(int) for r in RING_ORDER}
    skipped = 0

    rows = db.query(model).filter(model.split.in_(allowed_splits)).all()
    for row in rows:
        coord = _usable_coord(row, coord_type)
        if coord is None:
            skipped += 1
            continue
        lng, lat = coord
        dist = geo_utils.haversine_m(center_lng, center_lat, lng, lat)
        ring = _classify_ring(dist, core_r, nearby_r, radiation_r)
        if ring is None:
            continue
        exclusive[ring] += 1
        by_split[ring][row.split] += 1
        # 累计圆内计数（嵌套）
        if dist <= core_r:
            cumulative[RING_CORE] += 1
        if dist <= nearby_r:
            cumulative[RING_NEARBY] += 1
        if dist <= radiation_r:
            cumulative[RING_RADIATION] += 1

    return {
        "exclusive": exclusive,
        "cumulative": cumulative,
        "by_split": {r: dict(by_split[r]) for r in RING_ORDER},
        "skipped": skipped,
    }


def _build_core(
    db: Session, project: Project, include_test: bool
) -> dict[str, Any]:
    """共用归集逻辑，供 rings 与 spatial-summary 复用。"""
    center_lng, center_lat, center_status = _resolve_center(project)
    radii = _ring_radii(project)
    core_r, nearby_r, radiation_r = radii
    allowed_splits = _allowed_splits(include_test)

    per_type: dict[str, dict[str, Any]] = {}
    for data_type, (model, coord_type, _field) in SPATIAL_SOURCES.items():
        per_type[data_type] = _aggregate_one(
            db, model, coord_type, center_lng, center_lat, radii, allowed_splits
        )

    radius_by_ring = {RING_CORE: core_r, RING_NEARBY: nearby_r, RING_RADIATION: radiation_r}

    def _ring_counts(scope: str, ring: str) -> dict[str, Any]:
        counts = {
            "ring": ring,
            "radius_m": radius_by_ring[ring],
            "poi_count": 0,
            "housing_count": 0,
            "industry_count": 0,
            "population_grid_count": 0,
        }
        for data_type, (_model, _ct, field) in SPATIAL_SOURCES.items():
            counts[field] = per_type[data_type][scope][ring]
        return counts

    rings = [_ring_counts("exclusive", r) for r in RING_ORDER]
    cumulative = {r: _ring_counts("cumulative", r) for r in RING_ORDER}

    skipped_no_coord = {
        data_type: per_type[data_type]["skipped"] for data_type in SPATIAL_SOURCES
    }

    notes: list[str] = []
    if project.boundary_geojson:
        notes.append("已保存 boundary_geojson；本阶段核心圈仍按缓冲半径计算，红线多边形相交分析为后续阶段。")
    if (project.core_buffer_m or 0) <= 0:
        notes.append(f"core_buffer_m=0，核心圈采用兜底半径 {core_r}m。")
    if all(v == 0 for dt in per_type.values() for v in dt["exclusive"].values()):
        notes.append("当前 allowed_splits 内无落圈记录（可能 splits 未构建或范围内无 train/val 数据）。")

    return {
        "center_lng": center_lng,
        "center_lat": center_lat,
        "center_status": center_status,
        "has_boundary": bool(project.boundary_geojson),
        "include_test": include_test,
        "allowed_splits": allowed_splits,
        "rings": rings,
        "cumulative": cumulative,
        "by_split": {dt: per_type[dt]["by_split"] for dt in SPATIAL_SOURCES},
        "skipped_no_coord": skipped_no_coord,
        "notes": notes,
        "coordinate_system": project.coordinate_system or settings.coordinate_system,
    }


def collect_ring_records(
    db: Session,
    project: Project,
    data_type: str,
    include_test: bool = False,
) -> dict[str, Any]:
    """按圈层归集**落圈记录对象**（累计圆：core ⊆ nearby ⊆ radiation）。

    仅供四维分析服务在**服务内部**做明细级统计使用；返回的 ORM 行对象
    绝不直接出接口（接口只输出统计量/分类/评分/evidence_id）。

    返回：
    - center_lng/lat/status、radii、allowed_splits
    - records：{ring -> list[row]}，每个 ring 为以该半径为外缘的累计圆
    - skipped_no_coord：因坐标不可用被跳过的记录数
    """
    if data_type not in SPATIAL_SOURCES:
        raise SpatialError(f"未知 data_type: {data_type}")
    model, coord_type, _field = SPATIAL_SOURCES[data_type]

    center_lng, center_lat, center_status = _resolve_center(project)
    core_r, nearby_r, radiation_r = _ring_radii(project)
    allowed_splits = _allowed_splits(include_test)

    buckets: dict[str, list[Any]] = {r: [] for r in RING_ORDER}
    skipped = 0
    rows = db.query(model).filter(model.split.in_(allowed_splits)).all()
    for row in rows:
        coord = _usable_coord(row, coord_type)
        if coord is None:
            skipped += 1
            continue
        lng, lat = coord
        dist = geo_utils.haversine_m(center_lng, center_lat, lng, lat)
        if dist <= core_r:
            buckets[RING_CORE].append(row)
        if dist <= nearby_r:
            buckets[RING_NEARBY].append(row)
        if dist <= radiation_r:
            buckets[RING_RADIATION].append(row)

    return {
        "data_type": data_type,
        "center_lng": center_lng,
        "center_lat": center_lat,
        "center_status": center_status,
        "radii": {RING_CORE: core_r, RING_NEARBY: nearby_r, RING_RADIATION: radiation_r},
        "allowed_splits": allowed_splits,
        "include_test": include_test,
        "records": buckets,
        "skipped_no_coord": skipped,
    }


def ring_area_km2(radius_m: int) -> float:
    """圈层累计圆面积（km²），用于密度类指标。"""
    import math

    return math.pi * (radius_m / 1000.0) ** 2


def get_rings(db: Session, project: Project, include_test: bool = False) -> dict[str, Any]:
    """三圈层归集结果（互斥带 + 累计圆内）。"""
    core = _build_core(db, project, include_test)
    logger.info(
        "rings computed project_id=%s include_test=%s splits=%s",
        project.id, include_test, core["allowed_splits"],
    )
    return {
        "project_id": project.id,
        "coordinate_system": core["coordinate_system"],
        "center_lng": core["center_lng"],
        "center_lat": core["center_lat"],
        "center_status": core["center_status"],
        "has_boundary": core["has_boundary"],
        "include_test": core["include_test"],
        "allowed_splits": core["allowed_splits"],
        "rings": core["rings"],
        "cumulative": core["cumulative"],
        "skipped_no_coord": core["skipped_no_coord"],
        "notes": core["notes"],
    }


def get_spatial_summary(
    db: Session, project: Project, include_test: bool = False
) -> dict[str, Any]:
    """空间归集摘要：rings + 各数据类型按 split 分组数量。"""
    core = _build_core(db, project, include_test)
    logger.info(
        "spatial-summary computed project_id=%s include_test=%s",
        project.id, include_test,
    )
    return {
        "project_id": project.id,
        "coordinate_system": core["coordinate_system"],
        "center_lng": core["center_lng"],
        "center_lat": core["center_lat"],
        "center_status": core["center_status"],
        "has_boundary": core["has_boundary"],
        "include_test": core["include_test"],
        "allowed_splits": core["allowed_splits"],
        "rings": core["rings"],
        "cumulative": core["cumulative"],
        "by_split": core["by_split"],
        "skipped_no_coord": core["skipped_no_coord"],
        "notes": core["notes"],
    }
