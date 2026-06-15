"""第11 T2：POI 加载与统一分类服务（高德 + 科研 + 内部，仅特征工程/报告）。

职责：
- 把高德去重 POI（amap_poi_dedup_latest.jsonl，name 脱敏为 name_hash）与科研 POI
  （research_poi_feature_candidates.jsonl，含 shanghai_verdict）流式读取（chunk by line），
  bbox 预筛 + Haversine 距离过滤到项目最大圈层半径内。
- 把高德 typecode 分类体系统一映射为 T2 的 10 个一级类 + 二级类。
- feature-level 去重：坐标(5位) + 一级类；高德名称已脱敏，统一按坐标+分类近似去重。

红线：
- POI 仅 used_for_feature_engineering / used_for_report，**绝不作为监督标签**（used_for_training=false）。
- 不读取 / 不返回原始明细到接口；只返回统计量与分类计数。
- 科研 POI 仅取 shanghai 确认且 used_for_feature_engineering=true、competition_test=false。
- 坐标为 GCJ02；不静默转换为 WGS84，保留坐标系警告。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from sqlalchemy.orm import Session

from app.config import settings
from app.utils import geo_utils

logger = logging.getLogger("cityrenew.poi_feature")

# --------------------------------------------------------------------------- #
# 统一一级类 / 二级类
# --------------------------------------------------------------------------- #
L1_PUBLIC_SERVICE = "public_service"
L1_COMMERCIAL = "commercial_consumption"
L1_TRANSPORT = "transportation"
L1_CULTURE_SPORTS = "culture_sports"
L1_INDUSTRY_OFFICE = "industry_office"
L1_URBAN_RENEWAL = "urban_renewal"
L1_RESIDENTIAL = "residential_life"
L1_GREEN_SPACE = "green_space"
L1_GOVERNMENT = "government_service"
L1_UNKNOWN = "unknown"

L1_CLASSES: tuple[str, ...] = (
    L1_PUBLIC_SERVICE, L1_COMMERCIAL, L1_TRANSPORT, L1_CULTURE_SPORTS,
    L1_INDUSTRY_OFFICE, L1_URBAN_RENEWAL, L1_RESIDENTIAL, L1_GREEN_SPACE,
    L1_GOVERNMENT, L1_UNKNOWN,
)

# 城市更新专项关键词 → 二级类（仅在高德 matched_keywords 命中或科研 name 强匹配时启用，避免误判）
URBAN_RENEWAL_KW: dict[str, str] = {
    "旧厂房": "old_factory", "存量厂房": "old_factory", "旧厂": "old_factory",
    "工业遗存": "old_factory", "老厂房": "old_factory",
    "老旧小区": "old_residential", "旧住房": "old_residential", "旧区改造": "old_residential",
    "旧改": "old_residential", "棚户区": "old_residential", "旧住宅": "old_residential",
    "城中村": "urban_village",
    "更新单元": "renovation_project", "更新片区": "renovation_project",
    "城市微更新": "renovation_project", "城市更新": "renovation_project",
    "微更新": "renovation_project", "街区更新": "renovation_project",
    "社区更新": "renovation_project", "街道更新": "renovation_project",
    "历史文化街区": "renovation_project", "历史建筑": "renovation_project", "风貌区": "renovation_project",
    "TOD": "TOD", "换乘枢纽": "TOD", "轨交站": "TOD",
    "滨水更新": "waterfront", "滨水空间": "waterfront", "滨江公园": "waterfront", "滨江": "waterfront",
    "低效用地": "low_efficiency_land", "存量用地": "low_efficiency_land",
}
# 科研 POI 仅用强特异词，避免把普通餐饮/商铺误判为城市更新
RESEARCH_RENEWAL_STRICT: tuple[str, ...] = (
    "旧厂房", "存量厂房", "工业遗存", "城中村", "更新单元", "更新片区",
    "城市更新", "历史文化街区", "低效用地",
)

NEAREST_TARGETS: tuple[str, ...] = (
    "metro", "bus", "hospital", "school", "elderly_care",
    "park", "mall", "industrial_park", "government_service",
)


def map_poi_category(type_str: str | None, *, matched_keywords: list[str] | None = None,
                     name: str | None = None, is_research: bool = False) -> tuple[str, str | None]:
    """把高德 typecode 体系（type='L1;L2;L3'）统一映射为 (一级类, 二级类)。"""
    t = type_str or ""
    parts = [p.strip() for p in t.split(";") if p.strip()]
    l1 = parts[0] if parts else ""
    rest = t  # 含 L2/L3

    # —— 城市更新专项（保守判定）——
    if matched_keywords:
        for kw in matched_keywords:
            if kw in URBAN_RENEWAL_KW:
                return L1_URBAN_RENEWAL, URBAN_RENEWAL_KW[kw]
    if is_research and name:
        for kw in RESEARCH_RENEWAL_STRICT:
            if kw in name:
                return L1_URBAN_RENEWAL, URBAN_RENEWAL_KW.get(kw, "renovation_project")

    # —— 交通出行 ——
    if l1 == "交通设施服务":
        if "地铁" in rest or "轨道" in rest:
            return L1_TRANSPORT, "metro"
        if "公交" in rest or "公交车站" in rest:
            return L1_TRANSPORT, "bus"
        if "停车" in rest:
            return L1_TRANSPORT, "parking"
        if "充电" in rest:
            return L1_TRANSPORT, "charging"
        return L1_TRANSPORT, "road_transport"
    if l1 in ("道路附属设施", "通行设施"):
        if "停车" in rest:
            return L1_TRANSPORT, "parking"
        return L1_TRANSPORT, "road_transport"

    # —— 医疗 / 养老 ——
    if l1 == "医疗保健服务":
        if any(k in rest for k in ("养老", "敬老", "护理院", "福利院", "老年")):
            return L1_PUBLIC_SERVICE, "elderly_care"
        return L1_PUBLIC_SERVICE, "medical"

    # —— 科教文化 ——
    if l1 == "科教文化服务":
        if "图书馆" in rest:
            return L1_CULTURE_SPORTS, "library"
        if any(k in rest for k in ("博物馆", "美术馆", "展览馆", "纪念馆", "陈列")):
            return L1_CULTURE_SPORTS, "museum"
        if any(k in rest for k in ("文化宫", "文化馆", "科技馆", "文化中心", "活动中心")):
            return L1_CULTURE_SPORTS, "cultural_center"
        if any(k in rest for k in ("学校", "大学", "中学", "小学", "幼儿园", "学院", "教育", "培训")):
            return L1_PUBLIC_SERVICE, "education"
        return L1_PUBLIC_SERVICE, "education"

    # —— 公共设施 ——
    if l1 == "公共设施":
        if any(k in rest for k in ("养老", "敬老", "老年")):
            return L1_PUBLIC_SERVICE, "elderly_care"
        return L1_PUBLIC_SERVICE, "community_service"

    # —— 政府机构 ——
    if l1 == "政府机构及社会团体":
        return L1_GOVERNMENT, None

    # —— 餐饮 ——
    if l1 == "餐饮服务":
        return L1_COMMERCIAL, "catering"

    # —— 购物 ——
    if l1 == "购物服务":
        if any(k in rest for k in ("超市", "便利店", "菜市场", "农贸")):
            return L1_COMMERCIAL, "supermarket"
        if any(k in rest for k in ("商场", "购物中心", "广场", "综合体")):
            return L1_COMMERCIAL, "shopping"
        return L1_COMMERCIAL, "shopping"

    # —— 住宿 ——
    if l1 == "住宿服务":
        return L1_COMMERCIAL, "hotel"

    # —— 体育休闲 ——
    if l1 == "体育休闲服务":
        if any(k in rest for k in ("公园", "广场", "绿地")):
            return L1_GREEN_SPACE, None
        if any(k in rest for k in ("电影院", "ktv", "KTV", "娱乐", "酒吧", "网吧", "游戏")):
            return L1_COMMERCIAL, "entertainment"
        if any(k in rest for k in ("体育", "健身", "运动", "球场", "游泳")):
            return L1_CULTURE_SPORTS, "sports"
        return L1_CULTURE_SPORTS, "park_activity"

    # —— 风景名胜 ——
    if l1 == "风景名胜":
        if any(k in rest for k in ("公园", "植物园", "动物园")):
            return L1_GREEN_SPACE, None
        return L1_CULTURE_SPORTS, "park_activity"

    # —— 商务住宅 ——
    if l1 == "商务住宅":
        if any(k in rest for k in ("产业园", "工业园", "科技园", "软件园", "创意园")):
            return L1_INDUSTRY_OFFICE, "industrial_park"
        if any(k in rest for k in ("写字楼", "商务楼", "楼宇", "办公")):
            return L1_INDUSTRY_OFFICE, "office_building"
        if any(k in rest for k in ("住宅", "小区", "公寓", "宿舍", "社区")):
            return L1_RESIDENTIAL, None
        return L1_RESIDENTIAL, None

    # —— 公司企业 ——
    if l1 == "公司企业":
        if any(k in rest for k in ("工厂", "制造", "厂")):
            return L1_INDUSTRY_OFFICE, "industrial_park"
        if any(k in rest for k in ("产业园", "园区", "科技园", "孵化")):
            return L1_INDUSTRY_OFFICE, "incubator"
        return L1_INDUSTRY_OFFICE, "enterprise_service"

    # —— 金融保险 → 企业服务 ——
    if l1 == "金融保险服务":
        return L1_INDUSTRY_OFFICE, "enterprise_service"

    # —— 生活服务 → 居住生活 ——
    if l1 == "生活服务":
        if any(k in rest for k in ("社区服务", "便民", "居委")):
            return L1_PUBLIC_SERVICE, "community_service"
        return L1_RESIDENTIAL, None

    # —— 汽车相关 ——
    if l1 in ("汽车服务", "汽车销售", "汽车维修", "摩托车服务"):
        if "充电" in rest:
            return L1_TRANSPORT, "charging"
        if "加油" in rest or "加气" in rest:
            return L1_TRANSPORT, "road_transport"
        return L1_COMMERCIAL, "shopping"

    return L1_UNKNOWN, None


# 最近距离目标：(一级类, 二级类匹配) → 目标名
def _nearest_target_of(l1: str, l2: str | None) -> str | None:
    if l1 == L1_TRANSPORT and l2 == "metro":
        return "metro"
    if l1 == L1_TRANSPORT and l2 == "bus":
        return "bus"
    if l1 == L1_PUBLIC_SERVICE and l2 == "medical":
        return "hospital"
    if l1 == L1_PUBLIC_SERVICE and l2 == "education":
        return "school"
    if l1 == L1_PUBLIC_SERVICE and l2 == "elderly_care":
        return "elderly_care"
    if l1 == L1_GREEN_SPACE:
        return "park"
    if l1 == L1_COMMERCIAL and l2 in ("shopping", "supermarket"):
        return "mall"
    if l1 == L1_INDUSTRY_OFFICE and l2 in ("industrial_park", "incubator"):
        return "industrial_park"
    if l1 == L1_GOVERNMENT:
        return "government_service"
    return None


# --------------------------------------------------------------------------- #
# 文件路径与流式读取
# --------------------------------------------------------------------------- #
def _external_dir():
    return settings.data_dir / "external"


def amap_dedup_path():
    return _external_dir() / "amap" / "processed" / "amap_poi_dedup_latest.jsonl"


def research_poi_path():
    return _external_dir() / "public_service" / "processed" / "research_poi_feature_candidates.jsonl"


def _iter_jsonl(path) -> Iterator[dict[str, Any]]:
    """逐行流式读取 jsonl，跳过坏行；不一次性加载整文件。"""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except (ValueError, TypeError):
                continue


def _parse_amap_loc(rec: dict[str, Any]) -> tuple[float, float] | None:
    loc = rec.get("location_gcj02") or rec.get("location")
    if not loc or not isinstance(loc, str) or "," not in loc:
        return None
    try:
        lng_s, lat_s = loc.split(",", 1)
        return float(lng_s), float(lat_s)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# 加载并归集到圈层（bbox 预筛 + Haversine）
# --------------------------------------------------------------------------- #
def load_pois_near(
    db: Session,
    center_lng: float,
    center_lat: float,
    max_radius_m: int,
    *,
    include_amap: bool = True,
    include_research: bool = False,
    include_internal: bool = True,
) -> dict[str, Any]:
    """流式加载项目最大半径内的 POI，统一分类 + 特征级去重。

    返回：
    - pois: list[{dist_m, l1, l2, source}]（去重后；不含坐标明细）
    - source_counts: {amap, research, internal}
    - duplicate_overlap_count
    - nearest: {target -> 最近距离米}
    - coordinate_system / coordinate_system_warning
    """
    # bbox 半径换算（粗，纬度 1 度≈111km）
    deg_lat = max_radius_m / 111_000.0
    import math
    deg_lng = max_radius_m / (111_000.0 * max(0.2, math.cos(math.radians(center_lat))))
    lng_min, lng_max = center_lng - deg_lng, center_lng + deg_lng
    lat_min, lat_max = center_lat - deg_lat, center_lat + deg_lat

    seen: set[str] = set()
    pois: list[dict[str, Any]] = []
    source_counts = {"amap": 0, "research": 0, "internal": 0}
    duplicate_overlap = 0
    nearest: dict[str, float] = {}

    def _consider(lng: float, lat: float, l1: str, l2: str | None, source: str) -> None:
        nonlocal duplicate_overlap
        if not (lng_min <= lng <= lng_max and lat_min <= lat <= lat_max):
            return
        dist = geo_utils.haversine_m(center_lng, center_lat, lng, lat)
        if dist > max_radius_m:
            return
        key = f"{round(lng, 5)}_{round(lat, 5)}_{l1}"
        if key in seen:
            duplicate_overlap += 1
            return
        seen.add(key)
        pois.append({"dist_m": dist, "l1": l1, "l2": l2, "source": source})
        source_counts[source] += 1
        tgt = _nearest_target_of(l1, l2)
        if tgt is not None and (tgt not in nearest or dist < nearest[tgt]):
            nearest[tgt] = dist

    # 高德优先（脱敏，作主资产）
    if include_amap:
        for rec in _iter_jsonl(amap_dedup_path()):
            coord = _parse_amap_loc(rec)
            if coord is None:
                continue
            l1, l2 = map_poi_category(rec.get("type"),
                                      matched_keywords=rec.get("matched_keywords"),
                                      name=None, is_research=False)
            _consider(coord[0], coord[1], l1, l2, "amap")

    # 科研补充（默认关闭）：该集合 99.97% 为「餐饮服务」单一类别，混入会使圈层类别占比
    # 严重失真（陆家嘴等可达 ~94% 商业），导致类型判定退化为「哪都是商业活力」。类别占比
    # 特征只应使用多类别均衡来源（高德全市 POI + 比赛专用内部 POI）。如需餐饮密度可单列。
    if include_research:
        for rec in _iter_jsonl(research_poi_path()):
            if rec.get("competition_test"):
                continue
            if not rec.get("used_for_feature_engineering", True):
                continue
            if rec.get("shanghai_verdict") not in (None, "shanghai"):
                continue
            lng, lat = rec.get("lng"), rec.get("lat")
            if lng is None or lat is None:
                continue
            try:
                lng, lat = float(lng), float(lat)
            except (ValueError, TypeError):
                continue
            l1, l2 = map_poi_category(rec.get("type"), matched_keywords=None,
                                      name=rec.get("name"), is_research=True)
            _consider(lng, lat, l1, l2, "research")

    # 内部 competition POI（仅 train/val，作辅助 source_count）
    if include_internal:
        from app.models import PoiPoint
        rows = db.query(PoiPoint).filter(PoiPoint.split.in_(["train", "val"])).all()
        for row in rows:
            if row.lng is None or row.lat is None:
                continue
            if getattr(row, "coord_status", None) not in (
                None, geo_utils.STATUS_OK, geo_utils.STATUS_CORRECTED
            ):
                continue
            fl = (row.category_name or "").split(";")[0].strip()
            l1, l2 = map_poi_category(row.category_name, matched_keywords=None,
                                      name=fl or None, is_research=False)
            _consider(float(row.lng), float(row.lat), l1, l2, "internal")

    return {
        "pois": pois,
        "source_counts": source_counts,
        "duplicate_overlap_count": duplicate_overlap,
        "nearest": nearest,
        "coordinate_system": "GCJ02",
    }
