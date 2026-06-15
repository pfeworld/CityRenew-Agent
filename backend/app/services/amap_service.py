"""高德地图 Web 服务接入（第10B 实现，合规、有 Key 才采集）。

提供 POI 搜索 / 周边 / 多边形 / 地理编码 / 逆地理 / 行政区划 / 坐标转换 / 可达性占位
的统一封装。

红线（与项目规则 / 方案第十一节一致）：
- API Key 仅从 .env（settings.amap_key）读取，绝不写入代码、绝不返回 Key。
- 无 Key 返回 ``status=not_configured``，绝不伪造数据。
- coordinate_system=GCJ02；每次请求记录 params_hash（脱敏，不含 Key）。
- 不绕配额、不突破分页限制（受 amap_max_pages / amap_page_size 约束）、不换账号/IP。
- 使用本地 cache 复用重复请求；采集限流（amap_rate_limit_qps）；失败记 failed_reason。
- 不声称拿到高德全量 POI；仅样例/小范围合规采集。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger("cityrenew.amap")

_BASE = "https://restapi.amap.com/v3"
COORDINATE_SYSTEM = "GCJ02"
STATUS_NOT_CONFIGURED = "not_configured"
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_DEGRADED = "degraded"

# 进程内简单限流（最近一次请求时间戳）
_last_request_ts: float = 0.0


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _amap_dir() -> Path:
    return settings.data_dir / "external" / "amap"


def _cache_dir() -> Path:
    d = _amap_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _raw_dir() -> Path:
    d = _amap_dir() / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _processed_dir() -> Path:
    d = _amap_dir() / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(text: str) -> str:
    keep = "".join(c for c in (text or "") if c.isalnum() or c in "._-")
    return keep[:40] or "q"


def is_configured() -> bool:
    return bool(settings.amap_key and settings.amap_key.strip())


def _params_hash(api_name: str, params: dict[str, Any]) -> str:
    """对请求参数做稳定 hash（剔除 key，防泄露）。"""
    safe = {k: v for k, v in params.items() if k != "key"}
    blob = json.dumps({"api": api_name, "params": safe}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _cache_path(api_name: str, phash: str) -> Path:
    return _cache_dir() / f"{api_name}_{phash}.json"


def _read_cache(api_name: str, phash: str) -> dict[str, Any] | None:
    path = _cache_path(api_name, phash)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _write_cache(api_name: str, phash: str, raw: dict[str, Any]) -> str:
    path = _cache_path(api_name, phash)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
        return f"amap/cache/{path.name}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("amap cache write failed: %s", exc)
        return ""


def _rate_limit() -> None:
    global _last_request_ts
    qps = max(0.1, float(settings.amap_rate_limit_qps))
    min_interval = 1.0 / qps
    now = time.monotonic()
    delta = now - _last_request_ts
    if delta < min_interval:
        time.sleep(min_interval - delta)
    _last_request_ts = time.monotonic()


def _meta(api_name: str, params: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    """统一的脱敏元数据（不含 Key、不含原始点位明细）。"""
    out = {
        "api_name": api_name,
        "source": "amap",
        "status": status,
        "coordinate_system": COORDINATE_SYSTEM,
        "request_params_hash": _params_hash(api_name, params),
        "query_time": _utcnow_iso(),
        "returned_count": 0,
        "cleaned_count": 0,
        "failed_count": 0,
        "quota_status": "unknown",
        "cache_status": "miss",
        "cache_path": "",
        "license_status": "commercial_api_terms",
        "used_for_feature": True,
        "used_for_training": False,
        "used_for_report": True,
        "failed_reason": None,
    }
    out.update(extra)
    return out


def _not_configured(api_name: str, params: dict[str, Any]) -> dict[str, Any]:
    return _meta(
        api_name, params, STATUS_NOT_CONFIGURED,
        failed_reason="未配置 AMAP_KEY（仅从 .env 读取）；无 Key 不采集、不伪造数据。",
        quota_status="not_configured",
    )


def _fetch(api_name: str, path: str, params: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """执行一次高德请求（带 cache / 限流），返回 (原始响应, 脱敏元数据)。

    无 Key 返回 (None, not_configured)；requests 缺失或请求失败返回 (None, degraded/failed)。
    """
    if not is_configured():
        return None, _not_configured(api_name, params)

    phash = _params_hash(api_name, params)
    cached = _read_cache(api_name, phash)
    if cached is not None:
        meta = _summarize_response(api_name, params, cached)
        meta["cache_status"] = "hit"
        meta["cache_path"] = f"amap/cache/{_cache_path(api_name, phash).name}"
        return cached, meta

    try:
        import requests  # 懒加载：无 Key 时不需要该依赖
    except Exception as exc:  # noqa: BLE001
        return None, _meta(api_name, params, STATUS_DEGRADED,
                           failed_reason=f"requests 不可用：{type(exc).__name__}；未采集、未伪造数据。")

    full_params = dict(params)
    full_params["key"] = settings.amap_key
    try:
        _rate_limit()
        resp = requests.get(
            f"{_BASE}{path}", params=full_params,
            timeout=int(settings.amap_request_timeout_s),
        )
        http_status = resp.status_code
        raw = resp.json()
    except Exception as exc:  # noqa: BLE001
        return None, _meta(api_name, params, STATUS_FAILED,
                           failed_reason=f"请求失败：{type(exc).__name__}", failed_count=1,
                           error_type=type(exc).__name__, http_status=None)

    cache_path = _write_cache(api_name, phash, raw)
    meta = _summarize_response(api_name, params, raw)
    meta["cache_status"] = "miss"
    meta["cache_path"] = cache_path
    meta["http_status"] = http_status
    return raw, meta


def _request(api_name: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
    """执行一次高德请求（带 cache / 限流），仅返回脱敏元数据。"""
    return _fetch(api_name, path, params)[1]


def _summarize_response(api_name: str, params: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """把高德原始响应汇总为脱敏元数据（不返回任何点位/地址明细）。"""
    api_status = str(raw.get("status", "0"))
    ok = api_status == "1"
    # 高德常见列表字段
    pois = raw.get("pois") or raw.get("geocodes") or raw.get("districts") or []
    count = len(pois) if isinstance(pois, list) else 0
    meta = _meta(
        api_name, params, STATUS_OK if ok else STATUS_FAILED,
        returned_count=count,
        cleaned_count=count if ok else 0,
        failed_count=0 if ok else 1,
        quota_status="ok" if ok else str(raw.get("info", "unknown")),
        failed_reason=None if ok else str(raw.get("info", "amap_error")),
        amap_status=api_status,
        amap_info=str(raw.get("info", "")),
        amap_infocode=str(raw.get("infocode", "")),
    )
    return meta


# --------------------------------------------------------------------------- #
# 对外 API（无 Key 时全部返回 not_configured）
# --------------------------------------------------------------------------- #
def poi_search(keyword: str, city: str = "上海", page_size: int | None = None,
               page: int = 1) -> dict[str, Any]:
    ps = min(int(page_size or settings.amap_page_size), int(settings.amap_page_size))
    pg = min(max(1, int(page)), int(settings.amap_max_pages))
    params = {"keywords": keyword, "city": city, "offset": ps, "page": pg, "extensions": "base"}
    return _request("poi_search", "/place/text", params)


def around_search(location: str, keyword: str = "", radius: int = 1000,
                  page_size: int | None = None, page: int = 1) -> dict[str, Any]:
    """周边搜索。location 形如 "lng,lat"（GCJ02）。"""
    ps = min(int(page_size or settings.amap_page_size), int(settings.amap_page_size))
    pg = min(max(1, int(page)), int(settings.amap_max_pages))
    params = {"location": location, "keywords": keyword,
              "radius": min(int(radius), 50000), "offset": ps, "page": pg, "extensions": "base"}
    return _request("around_search", "/place/around", params)


def _aggregate_pois(pois: list[dict[str, Any]]) -> dict[str, Any]:
    """对 POI 列表做脱敏聚合：仅按一级类目计数与坐标有效性，不保留名称/地址明细。"""
    from collections import Counter

    by_category: Counter[str] = Counter()
    valid = 0
    for p in pois:
        if not isinstance(p, dict):
            continue
        loc = str(p.get("location", "")).strip()
        if loc and "," in loc:
            try:
                lng, lat = (float(x) for x in loc.split(",", 1))
                if -180 <= lng <= 180 and -90 <= lat <= 90:
                    valid += 1
            except Exception:  # noqa: BLE001
                pass
        cat = str(p.get("type", "")).split(";", 1)[0] or "未分类"
        by_category[cat] += 1
    return {"total": len(pois), "valid_location": valid, "by_category": dict(by_category)}


def collect_around_sample(location: str, keyword: str, radius: int = 1000,
                          max_pages: int = 1) -> dict[str, Any]:
    """周边小范围样例采集：落 raw（全量响应）/ processed（脱敏聚合）/ cache，返回脱敏元数据 + 路径。

    红线：受 amap_max_pages/amap_page_size 限制，不刷量、不绕配额、不绕分页、不声称全量；
    processed 仅含一级类目计数与坐标有效性（无名称/地址/个人信息）。
    """
    if not is_configured():
        return _not_configured("around_sample", {"keyword": keyword, "radius": radius})

    pages = max(1, min(int(max_pages), int(settings.amap_max_pages)))
    ps = int(settings.amap_page_size)
    all_pois: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    failed = 0
    quota_status = "ok"
    last_reason = None
    cache_hits = 0

    for pg in range(1, pages + 1):
        params = {"location": location, "keywords": keyword,
                  "radius": min(int(radius), 50000), "offset": ps, "page": pg,
                  "extensions": "base"}
        raw, meta = _fetch("around_search", "/place/around", params)
        if meta.get("cache_status") == "hit":
            cache_hits += 1
        if raw is None or meta.get("status") != STATUS_OK:
            failed += 1
            quota_status = meta.get("quota_status", "unknown")
            last_reason = meta.get("failed_reason")
            break
        raw_pages.append(raw)
        pois = raw.get("pois") or []
        if isinstance(pois, list):
            all_pois.extend(pois)
        quota_status = meta.get("quota_status", "ok")
        if len(pois) < ps:  # 已到末页，停止翻页（不突破分页）
            break

    phash = _params_hash("around_sample", {"keyword": keyword, "radius": radius, "location": location})
    raw_rel = processed_rel = None
    if raw_pages:
        raw_path = _raw_dir() / f"around_{_safe_name(keyword)}_{phash}.json"
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump({"keyword": keyword, "radius": radius, "pages": raw_pages},
                      f, ensure_ascii=False)
        raw_rel = f"amap/raw/{raw_path.name}"

        agg = _aggregate_pois(all_pois)
        proc_path = _processed_dir() / f"around_{_safe_name(keyword)}_{phash}.json"
        with proc_path.open("w", encoding="utf-8") as f:
            json.dump({"keyword": keyword, "radius": radius, "coordinate_system": COORDINATE_SYSTEM,
                       "query_time": _utcnow_iso(), **agg}, f, ensure_ascii=False, indent=2)
        processed_rel = f"amap/processed/{proc_path.name}"

    returned = len(all_pois)
    cleaned = _aggregate_pois(all_pois)["valid_location"] if all_pois else 0
    status = STATUS_OK if raw_pages else (STATUS_FAILED if failed else STATUS_OK)
    meta = _meta("around_sample",
                 {"keyword": keyword, "radius": radius, "location": location}, status,
                 returned_count=returned, cleaned_count=cleaned, failed_count=failed,
                 quota_status=quota_status,
                 cache_status="hit" if (cache_hits and cache_hits == len(raw_pages)) else "miss",
                 failed_reason=last_reason)
    meta["raw_path"] = raw_rel
    meta["processed_path"] = processed_rel
    meta["keyword"] = keyword
    meta["radius"] = radius
    return meta


def polygon_search(polygon: str, keyword: str = "", page_size: int | None = None,
                   page: int = 1) -> dict[str, Any]:
    ps = min(int(page_size or settings.amap_page_size), int(settings.amap_page_size))
    pg = min(max(1, int(page)), int(settings.amap_max_pages))
    params = {"polygon": polygon, "keywords": keyword, "offset": ps, "page": pg, "extensions": "base"}
    return _request("polygon_search", "/place/polygon", params)


def geocode(address: str, city: str = "上海") -> dict[str, Any]:
    return _request("geocode", "/geocode/geo", {"address": address, "city": city})


def regeocode(location: str) -> dict[str, Any]:
    return _request("regeocode", "/geocode/regeo", {"location": location})


def district(keyword: str = "上海", subdistrict: int = 1) -> dict[str, Any]:
    return _request("district", "/config/district",
                    {"keywords": keyword, "subdistrict": subdistrict})


def coordinate_convert(locations: str, coordsys: str = "gps") -> dict[str, Any]:
    """坐标转换为 GCJ02（高德 assistant/coordinate/convert）。"""
    return _request("coordinate_convert", "/assistant/coordinate/convert",
                    {"locations": locations, "coordsys": coordsys})


# --------------------------------------------------------------------------- #
# 正式批量采集（六大类关键词 + 采样点 + 多半径 + 去重 + 限流 + 停止条件）
# --------------------------------------------------------------------------- #
_DIRECTION_BEARINGS = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}

# 围绕城市更新前期策划场景的六大类外部 POI 关键词（非"高德全部分类"）
KEYWORD_CATEGORIES: dict[str, list[str]] = {
    "public_service": ["医院", "综合医院", "专科医院", "中医医院", "社区卫生服务中心", "卫生服务站",
                       "诊所", "药店", "学校", "幼儿园", "小学", "中学", "大学", "职业学校", "培训机构",
                       "养老院", "养老服务中心", "社区服务中心", "政务服务中心", "街道办", "居委会",
                       "派出所", "消防站", "公共厕所", "社区食堂", "福利院", "残疾人服务中心", "托育机构"],
    "commercial_consumption": ["商场", "购物中心", "百货商场", "超市", "便利店", "菜市场", "农贸市场",
                               "餐饮", "中餐", "西餐", "快餐", "咖啡", "奶茶", "酒店", "宾馆", "银行",
                               "ATM", "生活服务", "快递", "美容美发", "洗衣店", "家政", "维修", "宠物店",
                               "母婴店", "药妆店", "夜市", "商业街", "批发市场", "零售店"],
    "transport": ["地铁站", "公交站", "公交枢纽", "交通枢纽", "停车场", "加油站", "充电站", "出入口",
                  "道路", "客运站", "出租车站", "自行车租赁点", "轨道交通", "换乘站", "长途汽车站",
                  "桥梁", "隧道", "码头", "火车站"],
    "culture_sports": ["公园", "绿地", "城市公园", "口袋公园", "图书馆", "文化馆", "博物馆", "美术馆",
                       "体育馆", "健身房", "运动场", "篮球场", "足球场", "游泳馆", "电影院", "剧院",
                       "展览馆", "景点", "广场", "游乐场", "旅游景点", "文化中心", "社区文化活动中心"],
    "industry_office": ["写字楼", "商务楼", "办公楼", "园区", "公司企业", "创业园", "科技园", "产业园",
                        "工厂", "仓储", "物流园", "研发中心", "孵化器", "众创空间", "商务中心", "总部",
                        "产业基地", "软件园", "工业园", "企业园", "商务园", "工业区"],
    "urban_renewal": ["老旧小区", "社区商业", "历史建筑", "历史文化街区", "产业园区", "存量厂房",
                      "滨水空间", "公共空间", "商业街区", "街区更新", "TOD", "15分钟生活圈", "邻里中心",
                      "社区中心", "便民服务", "旧改", "更新片区", "城市公园", "慢行系统", "口袋公园",
                      "社区更新"],
}

# 六大类关键词补充（升级版，追加；不删除原关键词，自动去重）
_SUPPLEMENTAL_KEYWORDS: dict[str, list[str]] = {
    "industry_office": ["产业基地", "总部园区", "企业总部", "办公园区", "工业园区", "科技企业",
                        "研发机构", "制造企业", "商务办公", "办公园", "物流仓储", "创新园", "软件园",
                        "电子产业园", "智能制造园"],
    "transport": ["轨交站", "轨道交通站", "公交首末站", "换乘枢纽", "交通中心", "公共停车",
                  "地下停车场", "P+R停车场", "充电桩", "公交场站", "道路入口", "城市道路",
                  "主干道", "快速路入口"],
    "urban_renewal": ["更新单元", "旧区改造", "旧住房", "旧厂房", "旧仓库", "低效用地", "存量用地",
                      "社区更新", "街道更新", "滨水更新", "公共空间更新", "15分钟社区生活圈", "生活圈",
                      "邻里服务", "社区公共空间", "微更新", "城市微更新"],
    "culture_sports": ["社区体育", "全民健身", "体育公园", "运动公园", "滨江公园", "城市绿道", "步道",
                       "健身步道", "文化活动中心", "社区文化中心", "青少年活动中心", "老年活动室"],
    "public_service": ["社区事务受理中心", "党群服务中心", "社区卫生站", "托育园", "长者照护之家",
                       "日间照料中心", "社区养老", "便民服务中心"],
    "commercial_consumption": ["邻里商业", "社区商业", "商业综合体", "社区菜店", "生鲜超市", "品牌餐饮",
                               "商业广场", "商业中心", "生活广场"],
}
for _c, _kws in _SUPPLEMENTAL_KEYWORDS.items():
    for _k in _kws:
        if _k not in KEYWORD_CATEGORIES[_c]:
            KEYWORD_CATEGORIES[_c].append(_k)

# 类别均衡最低目标（去重 POI 条数；按类别轮询优先补缺口最大者）
CATEGORY_MIN_TARGETS: dict[str, int] = {
    "public_service": 8000, "commercial_consumption": 10000, "transport": 6000,
    "industry_office": 8000, "culture_sports": 6000, "urban_renewal": 3000,
}

# 中文类别名 → 内部 key（接受请求里的中文 category_min_targets / priority_categories）
CATEGORY_CN: dict[str, str] = {
    "公共服务类": "public_service", "商业消费类": "commercial_consumption",
    "交通出行类": "transport", "文化体育休闲类": "culture_sports",
    "产业办公类": "industry_office", "城市更新专项类": "urban_renewal",
}

# 上海市大致外包矩形（仅用于把网格采样点限制在上海范围内；GCJ02）
SH_BBOX = {"lng_min": 120.85, "lng_max": 122.20, "lat_min": 30.65, "lat_max": 31.90}


def build_grid_points(lng: float, lat: float, radius_m: int, spacing_m: int,
                      label: str | None = None) -> list[dict[str, Any]]:
    """围绕中心生成规则网格采样点（仅上海范围内、半径内；非真实项目点）。"""
    sp_type = label or ("grid_1km" if spacing_m <= 1000 else "grid_1_5km")
    steps = int(radius_m // spacing_m)
    pts: list[dict[str, Any]] = []
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            if i == 0 and j == 0:
                continue
            dn, de = i * spacing_m, j * spacing_m
            if (dn * dn + de * de) > radius_m * radius_m:
                continue
            ol = round(lng + de / (111320.0 * max(0.1, math.cos(math.radians(lat)))), 6)
            oa = round(lat + dn / 111320.0, 6)
            if not (SH_BBOX["lng_min"] <= ol <= SH_BBOX["lng_max"]
                    and SH_BBOX["lat_min"] <= oa <= SH_BBOX["lat_max"]):
                continue
            pts.append({"sample_point_type": sp_type, "dir": f"g{i}_{j}", "lng": ol, "lat": oa,
                        "dist": (dn * dn + de * de) ** 0.5})
    pts.sort(key=lambda p: p["dist"])  # 由近及远
    return pts

# 高德配额/限流相关 infocode 与关键词（命中即判 quota_limited）
_QUOTA_INFOCODES = {"10003", "10004", "10014", "10019", "10020", "10021", "10022", "10029", "10044", "10045"}
_QUOTA_TOKENS = ("LIMIT", "QUOTA", "EXCEEDED", "OUT_OF_SERVICE", "CUQPS")


def _is_quota_error(raw: dict[str, Any] | None, meta: dict[str, Any]) -> bool:
    if isinstance(raw, dict):
        code = str(raw.get("infocode", ""))
        info = str(raw.get("info", "")).upper()
        if code in _QUOTA_INFOCODES:
            return True
        if any(tok in info for tok in _QUOTA_TOKENS):
            return True
    q = str(meta.get("quota_status", "")).upper()
    return any(tok in q for tok in _QUOTA_TOKENS)


def _offset_point(lng: float, lat: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    """以 (lng,lat) 为原点，按方位角 bearing（自北顺时针）偏移 dist_m 米（等距近似）。"""
    dn = dist_m * math.cos(math.radians(bearing_deg))
    de = dist_m * math.sin(math.radians(bearing_deg))
    dlat = dn / 111320.0
    dlng = de / (111320.0 * max(0.1, math.cos(math.radians(lat))))
    return round(lng + dlng, 6), round(lat + dlat, 6)


def build_sample_points(lng: float, lat: float, distances: list[int],
                        direction_points: int = 8) -> list[dict[str, Any]]:
    """中心点 + 各环方向辅助采样点（仅采样点，非真实项目点）。"""
    dirs = list(_DIRECTION_BEARINGS.items())
    if direction_points and direction_points < len(dirs):
        dirs = dirs[:direction_points]
    pts: list[dict[str, Any]] = [{"sample_point_type": "center", "dir": "center", "lng": lng, "lat": lat}]
    for dist in distances:
        ring = f"ring_{int(dist)}"
        for name, brg in dirs:
            ol, oa = _offset_point(lng, lat, brg, float(dist))
            pts.append({"sample_point_type": ring, "dir": name, "lng": ol, "lat": oa})
    return pts


def _name_hash(name: str) -> str:
    return hashlib.sha1((name or "").encode("utf-8")).hexdigest()[:12]


def collect_formal(*, project_lng: float, project_lat: float, radii: list[int],
                   sample_points: list[dict[str, Any]], keywords_by_cat: dict[str, list[str]],
                   max_pages: int = 3, page_size: int = 20, max_total_requests: int = 2000,
                   soft_target: int = 1500, target: int = 3000, hard_target: int = 5000,
                   qps: float = 1.0, consecutive_fail_limit: int = 5) -> dict[str, Any]:
    """正式批量合规采集：受 max_total_requests/qps/quota/停止条件 严格约束，真实去重。

    返回脱敏统计 + 路径 + 各维度汇总；不返回原始名称/地址/坐标列表。
    """
    if not is_configured():
        return {"status": STATUS_NOT_CONFIGURED, "stopped_reason": "not_configured",
                "total_requests": 0, "total_returned": 0, "total_cleaned": 0,
                "total_deduplicated": 0, "total_failed": 0, "quota_status": "not_configured",
                "failed_reason": "未配置 AMAP_KEY（仅从 .env 读取）；未采集、未伪造数据。"}

    kw_cat: dict[str, str] = {}
    keywords: list[str] = []
    for cat, kws in keywords_by_cat.items():
        for kw in kws:
            if kw not in kw_cat:
                kw_cat[kw] = cat
                keywords.append(kw)

    page_size = min(int(page_size), 25)
    max_pages = max(1, min(int(max_pages), int(settings.amap_max_pages)))
    interval = 1.0 / max(0.1, float(qps))
    radii_sorted = sorted({int(r) for r in radii}, reverse=True)

    dedup: dict[str, dict[str, Any]] = {}
    raw_pages: list[dict[str, Any]] = []
    total_requests = total_returned = total_cleaned = total_failed = 0
    consec_fail = 0
    quota_status = "ok"
    stopped_reason: str | None = None
    last_failed_reason: str | None = None
    kw_summary: dict[str, int] = {}
    radius_summary: dict[str, int] = {}
    sp_summary: dict[str, int] = {}
    # 连续多个采样点 0 命中则跳过该关键词后续低价值请求
    kw_zero_streak: dict[str, int] = {kw: 0 for kw in keywords}

    def _targets_hit() -> str | None:
        if len(dedup) >= hard_target:
            return "hard_target_reached"
        if len(dedup) >= target:
            return "target_reached"
        return None

    def _one_request(loc: str, sp_type: str, sp_id: str, r: int, kw: str, page: int) -> int:
        """执行一次周边搜索，落 raw + 入 dedup，返回本次返回条数；-1 表示失败/限流。"""
        nonlocal total_requests, total_returned, total_cleaned, total_failed, consec_fail
        nonlocal quota_status, stopped_reason, last_failed_reason
        params = {"location": loc, "keywords": kw, "radius": min(int(r), 50000),
                  "offset": page_size, "page": page, "extensions": "base"}
        raw, meta = _fetch("around_search", "/place/around", params)
        total_requests += 1
        is_miss = meta.get("cache_status") == "miss"
        if raw is None or meta.get("status") != STATUS_OK:
            if _is_quota_error(raw, meta):
                stopped_reason = "quota_limited"
                quota_status = "limited"
                last_failed_reason = "高德返回配额/限流（quota/limit exceeded），已停止。"
            else:
                total_failed += 1
                consec_fail += 1
                last_failed_reason = meta.get("failed_reason")
                if consec_fail >= consecutive_fail_limit:
                    stopped_reason = "too_many_failures"
            if is_miss:
                time.sleep(interval)
            return -1
        consec_fail = 0
        pois = raw.get("pois") or []
        n = len(pois) if isinstance(pois, list) else 0
        total_returned += n
        kw_summary[kw] = kw_summary.get(kw, 0) + n
        radius_summary[str(r)] = radius_summary.get(str(r), 0) + n
        sp_summary[sp_type] = sp_summary.get(sp_type, 0) + n
        if n > 0:
            raw_pages.append({"sp": sp_id, "r": r, "kw": kw, "page": page, "resp": raw})
        for p in pois:
            if not isinstance(p, dict):
                continue
            loc_s = str(p.get("location", "")).strip()
            valid = False
            if loc_s and "," in loc_s:
                try:
                    lng2, lat2 = (float(x) for x in loc_s.split(",", 1))
                    valid = -180 <= lng2 <= 180 and -90 <= lat2 <= 90
                except Exception:  # noqa: BLE001
                    valid = False
            if valid:
                total_cleaned += 1
            pid = p.get("id") or f"{p.get('name')}|{loc_s}|{p.get('type')}"
            if pid in dedup:
                rec = dedup[pid]
                rec["matched_keywords"].add(kw)
                rec["rings_hit"].add(sp_type)
                rec["sample_points_hit"].add(sp_id)
            else:
                typ = str(p.get("type", ""))
                parts = typ.split(";")
                dedup[pid] = {
                    "poi_id": p.get("id"),
                    "name_hash": _name_hash(str(p.get("name", ""))),
                    "type": typ,
                    "typecode": p.get("typecode"),
                    "category_l1": parts[0] if parts else "",
                    "category_l2": parts[1] if len(parts) > 1 else "",
                    "category_l3": parts[2] if len(parts) > 2 else "",
                    "district": p.get("adname") or p.get("cityname") or "",
                    "location_gcj02": loc_s if valid else None,
                    "matched_keywords": {kw},
                    "rings_hit": {sp_type},
                    "sample_points_hit": {sp_id},
                    "first_seen_query": f"{kw}@{sp_id}@r{r}",
                    "source_id": "amap_poi",
                }
        if is_miss:
            time.sleep(interval)
        return n

    # 采集顺序：page-major（先把所有关键词第 1 页跑完→覆盖六大类，再逐页加深），
    # 避免在达成 target 前漏掉靠后的产业办公/城市更新类。
    for sp in sample_points:
        if stopped_reason:
            break
        loc = f"{sp['lng']},{sp['lat']}"
        sp_type = sp["sample_point_type"]
        sp_id = f"{sp_type}:{sp.get('dir', 'center')}"
        for r in radii_sorted:
            if stopped_reason:
                break
            active = [kw for kw in keywords if kw_zero_streak.get(kw, 0) < 4]
            sp_r_hit: dict[str, bool] = {}
            for page in range(1, max_pages + 1):
                if stopped_reason:
                    break
                next_active: list[str] = []
                for kw in active:
                    if total_requests >= max_total_requests:
                        stopped_reason = "request_limit_reached"
                        break
                    hit = _targets_hit()
                    if hit:
                        stopped_reason = hit
                        break
                    n = _one_request(loc, sp_type, sp_id, r, kw, page)
                    if stopped_reason:
                        break
                    if n < 0:
                        continue  # 失败：不再翻该 kw 的后续页
                    if n > 0:
                        sp_r_hit[kw] = True
                    if n >= page_size:  # 还有更多页，下一轮继续
                        next_active.append(kw)
                active = next_active
            # 该 (sp,r) 下零命中的关键词累计 streak（用于跨采样点跳过低价值请求）
            for kw in keywords:
                if sp_r_hit.get(kw):
                    kw_zero_streak[kw] = 0
                else:
                    kw_zero_streak[kw] = kw_zero_streak.get(kw, 0) + 1

    if stopped_reason is None:
        stopped_reason = "soft_target_reached" if len(dedup) >= soft_target else "completed"

    # ---- 落盘 raw / processed（external 已 gitignore）----
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records = []
    for rec in dedup.values():
        rec_out = dict(rec)
        rec_out["matched_keywords"] = sorted(rec["matched_keywords"])
        rec_out["rings_hit"] = sorted(rec["rings_hit"])
        rec_out["sample_points_hit"] = sorted(rec["sample_points_hit"])
        records.append(rec_out)

    raw_path = _raw_dir() / f"formal_batch_{ts}.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump({"collected_at": ts, "pages": raw_pages}, f, ensure_ascii=False)
    proc_path = _processed_dir() / f"formal_batch_{ts}.json"
    with proc_path.open("w", encoding="utf-8") as f:
        json.dump({"collected_at": ts, "coordinate_system": COORDINATE_SYSTEM,
                   "count": len(records), "records": records}, f, ensure_ascii=False, indent=2)

    # ---- 六大类 category_summary（按 matched_keywords 归类，记录可属多类）----
    category_summary: dict[str, int] = {cat: 0 for cat in keywords_by_cat}
    for rec in records:
        cats = {kw_cat.get(k) for k in rec["matched_keywords"] if kw_cat.get(k)}
        for c in cats:
            category_summary[c] = category_summary.get(c, 0) + 1

    return {
        "status": STATUS_OK if records else (STATUS_FAILED if total_failed else STATUS_OK),
        "total_requests": total_requests,
        "total_returned": total_returned,
        "total_cleaned": total_cleaned,
        "total_deduplicated": len(records),
        "total_failed": total_failed,
        "stopped_reason": stopped_reason,
        "quota_status": quota_status,
        "keyword_summary": kw_summary,
        "radius_summary": radius_summary,
        "sample_point_summary": sp_summary,
        "category_summary": category_summary,
        "sample_point_count": len(sample_points),
        "keyword_count": len(keywords),
        "radius_list": radii_sorted,
        "raw_path": f"amap/raw/{raw_path.name}",
        "processed_path": f"amap/processed/{proc_path.name}",
        "raw_dir": "amap/raw",
        "processed_dir": "amap/processed",
        "cache_dir": "amap/cache",
        "coordinate_system": COORDINATE_SYSTEM,
        "failed_reason": last_failed_reason,
    }


# --------------------------------------------------------------------------- #
# 5 万级分阶段 / 类别均衡 / 断点续采（合并去重，持久化 store + completed queries）
# --------------------------------------------------------------------------- #
def _store_dir() -> Path:
    d = _amap_dir() / "large_scale_store"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store_path() -> Path:
    return _store_dir() / "store.json"


def _legacy_store_path() -> Path:
    return _processed_dir() / "large_scale_store.json"


def _completed_path() -> Path:
    return _store_dir() / "completed_queries.json"


def _progress_path() -> Path:
    return _store_dir() / "progress.json"


def _kw_category_map(keywords_by_cat: dict[str, list[str]]) -> dict[str, str]:
    m: dict[str, str] = {}
    for cat, kws in keywords_by_cat.items():
        for kw in kws:
            m.setdefault(kw, cat)
    return m


def _rec_from_list(rec: dict[str, Any]) -> dict[str, Any]:
    rec["matched_keywords"] = set(rec.get("matched_keywords") or [])
    rec["rings_hit"] = set(rec.get("rings_hit") or [])
    rec["sample_points_hit"] = set(rec.get("sample_points_hit") or [])
    return rec


def _load_large_store(kw_cat: dict[str, str]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """加载已有去重 store（不存在则用 processed 下历史文件播种），返回 (dedup, cat_counts)。"""
    dedup: dict[str, dict[str, Any]] = {}
    sp = _store_path() if _store_path().exists() else _legacy_store_path()
    if sp.exists():
        try:
            with sp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for pid, rec in (data.get("records") or {}).items():
                dedup[pid] = _rec_from_list(rec)
        except Exception:  # noqa: BLE001
            dedup = {}
    if not dedup:  # 播种：聚合历史 processed/*.json（formal_batch / around 样例）
        for fp in _processed_dir().glob("*.json"):
            if fp.name == _legacy_store_path().name:
                continue
            try:
                with fp.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            recs = obj.get("records") if isinstance(obj, dict) else None
            if not isinstance(recs, list):
                continue
            for r in recs:
                pid = r.get("poi_id") or f"{r.get('name_hash')}|{r.get('location_gcj02')}|{r.get('type')}"
                if pid and pid not in dedup:
                    dedup[pid] = _rec_from_list(dict(r))
    cat_counts: dict[str, int] = {c: 0 for c in KEYWORD_CATEGORIES}
    for rec in dedup.values():
        cats = {kw_cat.get(k) for k in rec["matched_keywords"] if kw_cat.get(k)}
        rec["_cats"] = {c for c in cats if c}
        for c in rec["_cats"]:
            cat_counts[c] = cat_counts.get(c, 0) + 1
    return dedup, cat_counts


def _save_large_store(dedup: dict[str, dict[str, Any]]) -> None:
    out = {}
    for pid, rec in dedup.items():
        r = {k: v for k, v in rec.items() if k != "_cats"}
        r["matched_keywords"] = sorted(rec["matched_keywords"])
        r["rings_hit"] = sorted(rec["rings_hit"])
        r["sample_points_hit"] = sorted(rec["sample_points_hit"])
        out[pid] = r
    with _store_path().open("w", encoding="utf-8") as f:
        json.dump({"updated_at": _utcnow_iso(), "count": len(out), "records": out},
                  f, ensure_ascii=False)


def _load_completed() -> set[str]:
    p = _completed_path()
    if not p.exists():
        legacy = _amap_dir() / "cache_index" / "completed_queries.json"
        p = legacy if legacy.exists() else p
    if not p.exists():
        return set()
    try:
        with p.open("r", encoding="utf-8") as f:
            return set(json.load(f).get("hashes") or [])
    except Exception:  # noqa: BLE001
        return set()


def _save_completed(done: set[str]) -> None:
    with _completed_path().open("w", encoding="utf-8") as f:
        json.dump({"updated_at": _utcnow_iso(), "hashes": sorted(done)}, f, ensure_ascii=False)


def _bad_path() -> Path:
    return _store_dir() / "bad_queries.json"


def _failures_path() -> Path:
    return _store_dir() / "failures.json"


def _load_bad_queries() -> set[str]:
    p = _bad_path()
    if not p.exists():
        return set()
    try:
        with p.open("r", encoding="utf-8") as f:
            return set(json.load(f).get("hashes") or [])
    except Exception:  # noqa: BLE001
        return set()


def _save_bad_queries(bad: set[str]) -> None:
    with _bad_path().open("w", encoding="utf-8") as f:
        json.dump({"updated_at": _utcnow_iso(), "hashes": sorted(bad)}, f, ensure_ascii=False)


def _save_failures(failures: list[dict[str, Any]]) -> None:
    with _failures_path().open("w", encoding="utf-8") as f:
        json.dump({"updated_at": _utcnow_iso(), "count": len(failures),
                   "failures": failures[-200:]}, f, ensure_ascii=False, indent=2)


def _classify_failure(raw: dict[str, Any] | None, meta: dict[str, Any]) -> tuple[str, bool, str]:
    """返回 (error_type, retriable, suggested_action)。"""
    et = meta.get("error_type")
    if raw is None or et:  # 网络层异常（超时/连接失败/JSON 解析失败）
        return (et or "network_error", True, "transient_retry_next_resume")
    info = str(meta.get("amap_info", "") or (raw or {}).get("info", "")).upper()
    code = str(meta.get("amap_infocode", "") or (raw or {}).get("infocode", ""))
    if code.startswith("200") or "INVALID" in info or "PARAM" in info or "ILLEGAL" in info:
        return ("invalid_params", False, "skip_query_hash_bad_params")
    if code in ("10001", "10002", "10009"):  # key 相关
        return (f"key_error:{code}", False, "check_amap_key_config")
    return (f"amap_error:{info[:40]}", True, "retry_next_resume")


def collect_large_scale(*, project_lng: float, project_lat: float, radii: list[int],
                        sample_points: list[dict[str, Any]], keywords_by_cat: dict[str, list[str]],
                        category_min_targets: dict[str, int],
                        max_pages: int = 3, page_size: int = 20, max_total_requests: int = 20000,
                        stage_target: int = 0, target_total: int = 50000,
                        hard_target: int = 50000, qps: float = 1.0,
                        consecutive_fail_limit: int = 5, resume: bool = True,
                        dedup_merge_existing: bool = True, use_cache: bool = True,
                        stop_on_quota_limited: bool = True,
                        time_budget_s: float = 0.0, max_runtime_hours: float = 8.0,
                        priority_categories: list[str] | None = None,
                        do_not_stop_at_stage_target: bool = True,
                        skip_known_bad_queries: bool = False,
                        checkpoint_every: int = 50) -> dict[str, Any]:
    """类别均衡 + 断点续采 + 合并去重的大规模合规采集（QPS=1、并发=1、严格限流）。

    续采至 merged_dedup_total >= target_total 才停（无演示性提前停止）。进度按 checkpoint_every
    个请求增量落盘到 large_scale_store/（store + completed_queries + progress），中断后可断点续采。
    仅允许的 stopped_reason：target_reached / quota_limited / too_many_failures /
    request_limit_reached / runtime_limit_reached / all_combos_exhausted。
    """
    if not is_configured():
        return {"status": STATUS_NOT_CONFIGURED, "stopped_reason": "not_configured",
                "failed_reason": "未配置 AMAP_KEY（仅从 .env 读取）；未采集、未伪造数据。"}

    kw_cat = _kw_category_map(keywords_by_cat)
    cat_keywords = {c: list(kws) for c, kws in keywords_by_cat.items()}
    page_size = min(int(page_size), 25)
    max_pages = max(1, min(int(max_pages), int(settings.amap_max_pages)))
    interval = 1.0 / max(0.1, float(qps))
    radii_sorted = sorted({int(r) for r in radii}, reverse=True)

    dedup, cat_counts = (_load_large_store(kw_cat) if dedup_merge_existing else ({}, {c: 0 for c in KEYWORD_CATEGORIES}))
    category_before = dict(cat_counts)
    start_dedup = len(dedup)
    completed = _load_completed() if resume else set()

    raw_pages: list[dict[str, Any]] = []
    total_requests = total_returned = total_cleaned = total_failed = 0
    consec_fail = 0
    quota_status = "ok"
    stopped_reason: str | None = None
    last_failed_reason: str | None = None
    kw_returned: dict[str, int] = {}
    kw_attempts: dict[str, int] = {}
    failed_keywords: set[str] = set()
    attempted_keywords: set[str] = set()
    cat_kw_cursor: dict[str, int] = {c: 0 for c in keywords_by_cat}
    kw_combo_cursor: dict[str, int] = {}  # kw -> next combo index
    n_combos = len(sample_points) * len(radii_sorted)
    skipped_queries = 0
    skipped_bad_count = 0
    last_ckpt_req = 0
    bad_queries = _load_bad_queries() if skip_known_bad_queries else set()
    bad_seed_count = len(bad_queries)
    failures: list[dict[str, Any]] = []
    start_t = time.time()
    max_runtime_s = float(max_runtime_hours) * 3600.0 if max_runtime_hours else 0.0
    prio_rank = {c: i for i, c in enumerate(priority_categories or [])}

    def _checkpoint() -> None:
        _save_large_store(dedup)
        if resume:
            _save_completed(completed)
        if skip_known_bad_queries or bad_queries:
            _save_bad_queries(bad_queries)
        if failures:
            _save_failures(failures)
        try:
            with _progress_path().open("w", encoding="utf-8") as f:
                json.dump({
                    "updated_at": _utcnow_iso(),
                    "merged_dedup_total": len(dedup),
                    "previous_dedup_total": start_dedup,
                    "new_dedup": len(dedup) - start_dedup,
                    "total_requests_this_run": total_requests,
                    "total_returned": total_returned,
                    "elapsed_seconds": round(time.time() - start_t, 1),
                    "category_after": {c: cat_counts.get(c, 0) for c in keywords_by_cat},
                    "quota_status": quota_status,
                    "stopped_reason": stopped_reason,
                }, f, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            pass

    def _combo(idx: int) -> tuple[dict[str, Any], int]:
        sp = sample_points[idx // len(radii_sorted)]
        r = radii_sorted[idx % len(radii_sorted)]
        return sp, r

    def _qhash(sp_id: str, r: int, kw: str) -> str:
        return hashlib.sha1(f"{sp_id}|{r}|{kw}".encode("utf-8")).hexdigest()[:16]

    def _ingest(p: dict[str, Any], kw: str, sp_type: str, sp_id: str, r: int) -> None:
        nonlocal total_cleaned
        if not isinstance(p, dict):
            return
        loc_s = str(p.get("location", "")).strip()
        valid = False
        if loc_s and "," in loc_s:
            try:
                lng2, lat2 = (float(x) for x in loc_s.split(",", 1))
                valid = -180 <= lng2 <= 180 and -90 <= lat2 <= 90
            except Exception:  # noqa: BLE001
                valid = False
        if valid:
            total_cleaned += 1
        pid = p.get("id") or f"{p.get('name')}|{loc_s}|{p.get('type')}"
        cat = kw_cat.get(kw)
        if pid in dedup:
            rec = dedup[pid]
            rec["matched_keywords"].add(kw)
            rec["rings_hit"].add(sp_type)
            rec["sample_points_hit"].add(sp_id)
            if cat and cat not in rec.setdefault("_cats", set()):
                rec["_cats"].add(cat)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        else:
            typ = str(p.get("type", ""))
            parts = typ.split(";")
            dedup[pid] = {
                "poi_id": p.get("id"), "name_hash": _name_hash(str(p.get("name", ""))),
                "type": typ, "typecode": p.get("typecode"),
                "category_l1": parts[0] if parts else "",
                "category_l2": parts[1] if len(parts) > 1 else "",
                "category_l3": parts[2] if len(parts) > 2 else "",
                "district": p.get("adname") or p.get("cityname") or "",
                "location_gcj02": loc_s if valid else None,
                "matched_keywords": {kw}, "rings_hit": {sp_type}, "sample_points_hit": {sp_id},
                "first_seen_query": f"{kw}@{sp_id}@r{r}", "source_id": "amap_poi",
                "_cats": {cat} if cat else set(),
            }
            if cat:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

    def _pick_keyword(cat: str) -> str | None:
        """在该类别内挑下一个未走完 combo 的关键词（轮询）。"""
        kws = cat_keywords[cat]
        for _ in range(len(kws)):
            idx = cat_kw_cursor[cat] % len(kws)
            cat_kw_cursor[cat] = (cat_kw_cursor[cat] + 1) % len(kws)
            kw = kws[idx]
            if kw_combo_cursor.get(kw, 0) < n_combos:
                return kw
        return None

    def _next_combo(kw: str) -> tuple[dict[str, Any], str, str, int] | None:
        """取该关键词下一个未完成 combo（跳过 completed / 已诊断坏组合 / 超界）。"""
        nonlocal skipped_queries, skipped_bad_count
        while kw_combo_cursor.get(kw, 0) < n_combos:
            idx = kw_combo_cursor.get(kw, 0)
            kw_combo_cursor[kw] = idx + 1
            sp, r = _combo(idx)
            sp_type = sp["sample_point_type"]
            sp_id = f"{sp_type}:{sp.get('dir', 'center')}"
            qh = _qhash(sp_id, r, kw)
            if resume and qh in completed:
                skipped_queries += 1
                continue
            if skip_known_bad_queries and qh in bad_queries:
                skipped_bad_count += 1
                continue
            return sp, sp_id, sp_type, r
        return None

    # 主循环：按类别轮询（优先缺口最大、未达 min_target 的类别）
    while not stopped_reason:
        if total_requests >= max_total_requests:
            stopped_reason = "request_limit_reached"
            break
        if len(dedup) >= target_total or len(dedup) >= hard_target:
            stopped_reason = "target_reached"
            break
        if not do_not_stop_at_stage_target and stage_target and len(dedup) >= stage_target:
            stopped_reason = "stage_target_reached"
            break
        if time_budget_s and (time.time() - start_t) >= time_budget_s:
            stopped_reason = "time_budget_reached"
            break
        if max_runtime_s and (time.time() - start_t) >= max_runtime_s:
            stopped_reason = "runtime_limit_reached"
            break
        # 选类别：未达 min_target 的优先（按 priority_categories 顺序，其次缺口降序），
        # 已达标的类别排后（按缺口降序继续补总量）。
        def _cat_key(c: str) -> tuple[int, int, int]:
            gap = category_min_targets.get(c, 0) - cat_counts.get(c, 0)
            below = 0 if gap > 0 else 1  # 未达标排前
            rank = prio_rank.get(c, len(prio_rank))
            return (below, rank, -gap)
        order = sorted(keywords_by_cat.keys(), key=_cat_key)
        progressed = False
        for cat in order:
            if stopped_reason:
                break
            kw = _pick_keyword(cat)
            if kw is None:
                continue
            combo = _next_combo(kw)
            if combo is None:
                continue
            sp, sp_id, sp_type, r = combo
            loc = f"{sp['lng']},{sp['lat']}"
            attempted_keywords.add(kw)
            kw_attempts[kw] = kw_attempts.get(kw, 0) + 1
            progressed = True
            for page in range(1, max_pages + 1):
                if total_requests >= max_total_requests:
                    stopped_reason = "request_limit_reached"
                    break
                params = {"location": loc, "keywords": kw, "radius": min(int(r), 50000),
                          "offset": page_size, "page": page, "extensions": "base"}
                raw, meta = _fetch("around_search", "/place/around", params)
                total_requests += 1
                is_miss = meta.get("cache_status") == "miss"
                if raw is None or meta.get("status") != STATUS_OK:
                    qh = _qhash(sp_id, r, kw)
                    if _is_quota_error(raw, meta):
                        quota_status = "limited"
                        last_failed_reason = "高德返回配额/限流（quota/limit exceeded）。"
                        failures.append({
                            "failed_query_hash": qh, "keyword": kw, "category": kw_cat.get(kw),
                            "sample_point_type": sp_type, "radius": r,
                            "amap_status": meta.get("amap_status"), "amap_info": meta.get("amap_info"),
                            "http_status": meta.get("http_status"), "error_type": "quota_limited",
                            "failed_reason": last_failed_reason, "whether_retriable": False,
                            "suggested_action": "stop_and_wait_for_quota"})
                        if stop_on_quota_limited:
                            stopped_reason = "quota_limited"
                    else:
                        error_type, retriable, action = _classify_failure(raw, meta)
                        total_failed += 1
                        failed_keywords.add(kw)
                        last_failed_reason = meta.get("failed_reason")
                        failures.append({
                            "failed_query_hash": qh, "keyword": kw, "category": kw_cat.get(kw),
                            "sample_point_type": sp_type, "radius": r,
                            "amap_status": meta.get("amap_status"), "amap_info": meta.get("amap_info"),
                            "http_status": meta.get("http_status"), "error_type": error_type,
                            "failed_reason": last_failed_reason, "whether_retriable": retriable,
                            "suggested_action": action})
                        if retriable:
                            consec_fail += 1
                            if consec_fail >= consecutive_fail_limit:
                                stopped_reason = "too_many_failures"
                        else:  # 坏组合：登记跳过、不计入连续失败、不反复重试
                            bad_queries.add(qh)
                            consec_fail = 0
                    if is_miss:
                        time.sleep(interval)
                    break
                consec_fail = 0
                pois = raw.get("pois") or []
                n = len(pois) if isinstance(pois, list) else 0
                total_returned += n
                kw_returned[kw] = kw_returned.get(kw, 0) + n
                if n > 0:
                    raw_pages.append({"sp": sp_id, "r": r, "kw": kw, "page": page, "resp": raw})
                for p in pois:
                    _ingest(p, kw, sp_type, sp_id, r)
                if is_miss:
                    time.sleep(interval)
                if n < page_size:
                    break
            completed.add(_qhash(sp_id, r, kw))  # 该 combo 已走完（断点续采可跳过）
            if checkpoint_every and total_requests - last_ckpt_req >= checkpoint_every:
                last_ckpt_req = total_requests
                _checkpoint()
        if not progressed and not stopped_reason:
            stopped_reason = "all_combos_exhausted"
            break

    # ---- 持久化进度（即使 quota/失败也保存）----
    _save_large_store(dedup)
    if resume:
        _save_completed(completed)
    if skip_known_bad_queries or bad_queries:
        _save_bad_queries(bad_queries)
    if failures:
        _save_failures(failures)

    # 落 raw / processed 快照
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = _raw_dir() / f"large_scale_{ts}.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump({"collected_at": ts, "pages": raw_pages}, f, ensure_ascii=False)
    records = []
    for rec in dedup.values():
        r = {k: v for k, v in rec.items() if k != "_cats"}
        r["matched_keywords"] = sorted(rec["matched_keywords"])
        r["rings_hit"] = sorted(rec["rings_hit"])
        r["sample_points_hit"] = sorted(rec["sample_points_hit"])
        records.append(r)
    proc_path = _processed_dir() / f"large_scale_{ts}.json"
    with proc_path.open("w", encoding="utf-8") as f:
        json.dump({"collected_at": ts, "coordinate_system": COORDINATE_SYSTEM,
                   "count": len(records), "records": records}, f, ensure_ascii=False)

    category_after = {c: cat_counts.get(c, 0) for c in keywords_by_cat}
    category_gap = {c: max(0, category_min_targets.get(c, 0) - category_after.get(c, 0)) for c in keywords_by_cat}
    category_target_status = {c: ("met" if category_after.get(c, 0) >= category_min_targets.get(c, 0) else "below")
                              for c in keywords_by_cat}
    # natural_sparse / low_yield：尝试足够多但产出极低
    low_yield_keywords = sorted([kw for kw, a in kw_attempts.items()
                                 if a >= 3 and (kw_returned.get(kw, 0) / a) < 3])
    natural_sparse_categories = []
    for c, kws in cat_keywords.items():
        attempted = [kw for kw in kws if kw in kw_attempts]
        if attempted and category_target_status[c] == "below":
            low = [kw for kw in attempted if kw in low_yield_keywords]
            if len(low) >= max(1, len(attempted) // 2):
                natural_sparse_categories.append(c)

    new_dedup = len(dedup) - start_dedup
    duplicate_rate = round(1 - (new_dedup / total_returned), 4) if total_returned else None
    quality_score = round(new_dedup / total_returned, 4) if total_returned else None
    if stopped_reason is None:
        stopped_reason = "all_combos_exhausted"
    runtime_seconds = round(time.time() - start_t, 1)
    _checkpoint()  # 最终进度快照

    return {
        "status": STATUS_OK,
        "total_requests": total_requests, "total_returned": total_returned,
        "total_requests_this_run": total_requests,
        "total_cleaned": total_cleaned, "new_dedup": new_dedup,
        "new_returned": total_returned, "new_cleaned": total_cleaned,
        "new_deduplicated": new_dedup,
        "previous_dedup_total": start_dedup,
        "merged_dedup_total": len(dedup),
        "total_deduplicated": len(dedup), "total_failed": total_failed,
        "duplicate_rate": duplicate_rate, "quality_score": quality_score,
        "stopped_reason": stopped_reason, "quota_status": quota_status,
        "runtime_seconds": runtime_seconds,
        "target_total": target_total, "hard_target": hard_target,
        "target_reached": len(dedup) >= target_total,
        "category_before": category_before, "category_after": category_after,
        "category_gap": category_gap, "category_target_status": category_target_status,
        "natural_sparse_categories": natural_sparse_categories,
        "low_yield_keywords": low_yield_keywords,
        "attempted_keywords": sorted(attempted_keywords), "failed_keywords": sorted(failed_keywords),
        "keyword_count": len(kw_cat), "sample_point_count": len(sample_points),
        "radius_list": radii_sorted,
        "raw_path": f"amap/raw/{raw_path.name}", "processed_path": f"amap/processed/{proc_path.name}",
        "store_path": "amap/large_scale_store/store.json",
        "store_dir": "amap/large_scale_store",
        "completed_queries": len(completed), "skipped_queries": skipped_queries,
        "skipped_bad_count": skipped_bad_count, "bad_queries_seed": bad_seed_count,
        "bad_queries_total": len(bad_queries),
        "recent_failures": failures[-20:],
        "failure_summary": {et: sum(1 for f in failures if f.get("error_type") == et)
                            for et in sorted({f.get("error_type") for f in failures})},
        "coordinate_system": COORDINATE_SYSTEM, "failed_reason": last_failed_reason,
    }


def accessibility_placeholder(origin: str = "", destination: str = "") -> dict[str, Any]:
    """可达性占位：第10B 预留，不实际调用路线规划（避免额外配额消耗）。"""
    params = {"origin": origin, "destination": destination}
    if not is_configured():
        return _not_configured("accessibility_placeholder", params)
    meta = _meta("accessibility_placeholder", params, STATUS_DEGRADED,
                 failed_reason="可达性分析为第10B 预留，未实际采集（路线规划在后续阶段按配额开启）。")
    return meta
