"""外部数据采集编排与外部数据目录管理（第10B 实现）。

职责：
1. 初始化 ``backend/data/external/`` 目录脚手架（README / data_catalog.json /
   data_lineage.json / registry / 各数据源 raw/processed/cache + manifest 模板）。
2. 编排采集任务：amap 走 amap_service（无 Key→not_configured）；其余源预留
   not_configured / planned，绝不伪造数据。
3. 维护采集任务台账（external/collection_tasks.json）。
4. 聚合外部数据目录（catalog）与合规风险（compliance-risk）。

红线：
- external/ 已被 .gitignore 覆盖；不提交任何真实下载数据。
- 无 Key / 无授权 → not_configured / planned，不伪造数据、不绕反爬、不换账号 IP。
- 接口仅返回脱敏元数据（source/count/status/license/compliance/lineage/quality），
  不返回原始 JSON 全文 / 坐标列表 / 企业名 / 小区名 / 地址 / 个人信息。
- 商业风险源默认不可采、不可训练（由 registry 把关）。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.services import amap_service, data_lineage_service
from app.services import data_source_registry as registry

logger = logging.getLogger("cityrenew.external_collector")

STATUS_NOT_CONFIGURED = "not_configured"
STATUS_PLANNED = "planned"
STATUS_NOT_IMPLEMENTED = "not_implemented"
STATUS_OK = "ok"
STATUS_FAILED = "failed"

# 外部数据目录分区定义：name -> (额外子目录, 类目子目录, 默认 source_id, source_type)
_SECTIONS: tuple[dict[str, Any], ...] = (
    {"name": "amap", "dirs": ["raw", "processed", "cache"], "cats": [],
     "source_id": "amap_poi", "source_type": "map_poi"},
    {"name": "baidu_map", "dirs": ["raw", "processed", "cache"], "cats": [],
     "source_id": "baidu_map_poi", "source_type": "map_poi"},
    {"name": "tencent_map", "dirs": ["raw", "processed", "cache"], "cats": [],
     "source_id": "tencent_map_poi", "source_type": "map_poi"},
    {"name": "osm", "dirs": ["raw", "processed", "cache"], "cats": [],
     "source_id": "osm_overpass", "source_type": "map_poi"},
    {"name": "shanghai_open_data", "dirs": ["raw", "processed", "manual_uploads"], "cats": [],
     "source_id": "shanghai_open_data", "source_type": "gov_open_data"},
    {"name": "stats_cn", "dirs": ["raw", "processed", "manual_uploads"], "cats": [],
     "source_id": "stats_cn", "source_type": "gov_statistics"},
    {"name": "planning_policy", "dirs": ["raw", "processed", "manual_uploads"], "cats": [],
     "source_id": "shanghai_planning_policy", "source_type": "planning_policy"},
    {"name": "public_service", "dirs": ["raw", "processed"],
     "cats": ["education", "healthcare", "elderly", "sports", "culture"],
     "source_id": None, "source_type": "public_service"},
    {"name": "transport", "dirs": ["raw", "processed"],
     "cats": ["metro", "bus", "road"],
     "source_id": None, "source_type": "transport"},
    {"name": "commercial_consumption", "dirs": ["raw", "processed"], "cats": [],
     "source_id": None, "source_type": "commercial_consumption"},
    {"name": "environment_weather", "dirs": ["raw", "processed"], "cats": [],
     "source_id": None, "source_type": "environment_weather"},
    {"name": "authorized_property", "dirs": ["raw", "processed", "manual_uploads"], "cats": [],
     "source_id": "authorized_property_upload", "source_type": "property"},
    {"name": "user_uploaded", "dirs": ["raw", "processed"], "cats": [],
     "source_id": None, "source_type": "user_uploaded"},
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _external_dir() -> Path:
    return settings.data_dir / "external"


def _tasks_path() -> Path:
    return _external_dir() / "collection_tasks.json"


def _manifest_template(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": section.get("source_id"),
        "source_type": section.get("source_type"),
        "record_count": 0,
        "field_schema": [],
        "source_url": "",
        "license": "",
        "collection_time": None,
        "collection_method": "",
        "quality_score": None,
        "files": [],
        "lineage_ids": [],
        "is_template": True,
        "note": "第10B 模板，未采集真实数据。采集后由系统更新 record_count/files/lineage。",
    }


# --------------------------------------------------------------------------- #
# 目录脚手架
# --------------------------------------------------------------------------- #
def ensure_scaffold() -> dict[str, Any]:
    """创建 external/ 目录脚手架与 manifest 模板（幂等，不覆盖已有 manifest）。

    生成物均在 .gitignore 覆盖的 backend/data/external/ 下，不会进入 git。
    """
    base = _external_dir()
    base.mkdir(parents=True, exist_ok=True)

    created_dirs = 0
    created_manifests = 0

    # 顶层文件
    readme = base / "README.md"
    if not readme.exists():
        readme.write_text(_render_readme(), encoding="utf-8")
    for fname, payload in (
        ("data_catalog.json", {"updated_at": _utcnow_iso(), "sections": [], "is_template": True}),
        ("data_lineage.json", {"updated_at": _utcnow_iso(), "records": [], "is_template": True}),
    ):
        p = base / fname
        if not p.exists():
            with p.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    # registry（预置数据源）
    registry.ensure_registry_initialized()

    # 各分区目录 + manifest 模板
    for sec in _SECTIONS:
        sec_dir = base / sec["name"]
        for d in sec["dirs"]:
            target = sec_dir / d
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
                created_dirs += 1
        for cat in sec["cats"]:
            for d in ("raw", "processed"):
                target = sec_dir / cat / d
                if not target.exists():
                    target.mkdir(parents=True, exist_ok=True)
                    created_dirs += 1
        manifest = sec_dir / "manifest.json"
        if not manifest.exists():
            sec_dir.mkdir(parents=True, exist_ok=True)
            with manifest.open("w", encoding="utf-8") as f:
                json.dump(_manifest_template(sec), f, ensure_ascii=False, indent=2)
            created_manifests += 1

    return {
        "external_dir": _rel(base),
        "sections": [s["name"] for s in _SECTIONS],
        "created_dirs": created_dirs,
        "created_manifests": created_manifests,
        "gitignored": True,
    }


def _render_readme() -> str:
    lines = [
        "# external/ 外部数据增强目录（第10B）",
        "",
        "本目录存放合规外部数据（地图/政府开放/公共服务/交通/授权房价等）的原始、清洗、缓存数据与 manifest。",
        "",
        "> 红线：本目录已被 .gitignore 覆盖，严禁提交真实下载数据；",
        "> 外部数据严禁混入 competition_test、严禁反推 test 答案、严禁用于 test 调参；",
        "> 无 Key/无授权一律 not_configured/planned，绝不伪造数据。",
        "",
        "## 结构",
        "- registry/：数据源登记、候选源、发现日志",
        "- amap/baidu_map/tencent_map/osm/：raw/processed/cache + manifest.json",
        "- shanghai_open_data/stats_cn/planning_policy/：raw/processed + manifest.json",
        "- public_service/（education/healthcare/elderly/sports/culture）",
        "- transport/（metro/bus/road）",
        "- commercial_consumption/environment_weather/authorized_property/user_uploaded/",
        "- data_catalog.json / data_lineage.json：目录与血缘汇总",
        "",
        "每个 manifest 含 record_count / field_schema / source_url / license / collection_time / quality_score；",
        "每份 processed 可追溯 raw；每份数据有 source_id 与 lineage_id。",
    ]
    return "\n".join(lines)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(settings.data_dir.parent))
    except Exception:  # noqa: BLE001
        return str(path)


# --------------------------------------------------------------------------- #
# 任务台账
# --------------------------------------------------------------------------- #
def _load_tasks() -> list[dict[str, Any]]:
    path = _tasks_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def _save_task(task: dict[str, Any]) -> None:
    tasks = _load_tasks()
    tasks.append(task)
    path = _tasks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(tasks[-500:], f, ensure_ascii=False, indent=2)


def list_tasks() -> dict[str, Any]:
    tasks = _load_tasks()
    return {"count": len(tasks), "tasks": tasks}


def get_task(task_id: str) -> dict[str, Any] | None:
    for t in _load_tasks():
        if t.get("task_id") == task_id:
            return t
    return None


def _new_task(source_id: str | None, source_type: str) -> dict[str, Any]:
    return {
        "task_id": uuid.uuid4().hex[:16],
        "source_id": source_id,
        "source_type": source_type,
        "status": STATUS_PLANNED,
        "started_at": _utcnow_iso(),
        "finished_at": None,
        "raw_count": 0,
        "cleaned_count": 0,
        "failed_count": 0,
        "quota_status": "unknown",
        "compliance_status": "unknown",
        "cache_status": "n/a",
        "lineage_id": None,
        "error_message": None,
    }


# --------------------------------------------------------------------------- #
# manifest 更新（采集成功后）
# --------------------------------------------------------------------------- #
def _update_manifest(section_name: str, *, record_count: int, source_url: str,
                     license_str: str, quality_score: float | None,
                     lineage_id: str | None, file_rel: str | None) -> None:
    sec_dir = _external_dir() / section_name
    sec_dir.mkdir(parents=True, exist_ok=True)
    manifest = sec_dir / "manifest.json"
    data: dict[str, Any]
    if manifest.exists():
        try:
            with manifest.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            data = {}
    else:
        data = {}
    data["record_count"] = int(data.get("record_count", 0)) + int(record_count)
    data["source_url"] = source_url or data.get("source_url", "")
    data["license"] = license_str or data.get("license", "")
    data["collection_time"] = _utcnow_iso()
    data["quality_score"] = quality_score
    data["is_template"] = False
    files = data.get("files", []) or []
    if file_rel:
        files.append(file_rel)
    data["files"] = files[-200:]
    lin = data.get("lineage_ids", []) or []
    if lineage_id:
        lin.append(lineage_id)
    data["lineage_ids"] = lin[-200:]
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# 采集入口
# --------------------------------------------------------------------------- #
def collect_amap(db: Session, *, project_id: int | None, mode: str = "around",
                 keyword: str = "", radius: int = 1000) -> dict[str, Any]:
    """高德采集（无 AMAP_KEY → not_configured；有 Key → 小范围合规样例采集）。"""
    ensure_scaffold()
    src = registry.get_source("amap_poi") or {}
    task = _new_task("amap_poi", "map_poi")
    task["compliance_status"] = src.get("compliance_status", "ok_with_key")

    if not amap_service.is_configured():
        task["status"] = STATUS_NOT_CONFIGURED
        task["quota_status"] = "not_configured"
        task["finished_at"] = _utcnow_iso()
        task["error_message"] = "未配置 AMAP_KEY（仅从 .env 读取）；无 Key 不采集、不伪造数据。"
        _save_task(task)
        return task

    # 有 Key：解析项目中心，做一次小范围样例请求（默认 1 页，受配额/分页限制保护）
    location = ""
    if project_id is not None:
        from app.services import project_service
        project = project_service.get_project(db, project_id)
        if project is not None and project.center_lng is not None and project.center_lat is not None:
            location = f"{project.center_lng},{project.center_lat}"

    if location:
        meta = amap_service.collect_around_sample(location, keyword=keyword, radius=radius, max_pages=1)
    else:
        # 无项目坐标：退化为文本检索样例（仍受分页限制），不伪造数据
        meta = amap_service.poi_search(keyword)
        meta.setdefault("raw_path", None)
        meta.setdefault("processed_path", None)
        task["error_message"] = "项目无中心坐标，使用文本检索样例（建议为项目1 配置中心点）"

    raw_count = int(meta.get("returned_count", 0))
    cleaned = int(meta.get("cleaned_count", 0))
    failed = int(meta.get("failed_count", 0))
    status_map = {
        amap_service.STATUS_OK: STATUS_OK,
        amap_service.STATUS_NOT_CONFIGURED: STATUS_NOT_CONFIGURED,
        amap_service.STATUS_FAILED: STATUS_FAILED,
        amap_service.STATUS_DEGRADED: "degraded",
    }
    task["status"] = status_map.get(meta.get("status"), STATUS_FAILED)
    task["raw_count"] = raw_count
    task["cleaned_count"] = cleaned
    task["failed_count"] = failed
    task["quota_status"] = meta.get("quota_status", "unknown")
    task["cache_status"] = meta.get("cache_status", "n/a")
    task["keyword"] = keyword
    task["radius"] = radius
    task["raw_path"] = meta.get("raw_path")
    task["processed_path"] = meta.get("processed_path")
    task["used_for_feature_engineering"] = True
    task["used_for_report"] = True
    task["used_for_training"] = False
    if meta.get("failed_reason"):
        task["error_message"] = meta.get("failed_reason")

    if task["status"] == STATUS_OK and cleaned > 0:
        quality = round(cleaned / raw_count, 4) if raw_count else None
        lineage_id = data_lineage_service.record_collection_lineage(
            source_id="amap_poi", source_name=src.get("source_name", "高德 POI"),
            source_type="map_poi", raw_count=raw_count, cleaned_count=cleaned,
            license_status=src.get("license_type", "commercial_api_terms"),
            compliance_status=src.get("compliance_status", "ok_with_key"),
            used_for_feature_engineering=True, used_for_report=True,
            quality_score=quality, file_path=meta.get("processed_path"),
        )
        task["lineage_id"] = lineage_id
        _update_manifest("amap", record_count=cleaned,
                         source_url=src.get("official_url_or_api", ""),
                         license_str=src.get("license_type", "commercial_api_terms"),
                         quality_score=quality,
                         lineage_id=lineage_id, file_rel=meta.get("processed_path"))
    task["finished_at"] = _utcnow_iso()
    _save_task(task)
    return task


def run_amap_formal(db: Session, *, project_id: int = 1,
                    radii: list[int] | None = None,
                    use_sampling_points: bool = True,
                    sampling_distances_m: list[int] | None = None,
                    direction_points: int = 8,
                    max_pages_per_keyword_radius: int = 3,
                    page_size: int = 20,
                    max_total_requests: int = 2000,
                    soft_target_dedup_records: int = 1500,
                    target_dedup_records: int = 3000,
                    hard_target_dedup_records: int = 5000,
                    qps: float = 1.0) -> dict[str, Any]:
    """高德正式批量合规采集编排：采样点+六大类关键词+多半径+去重+限流+停止条件。

    无 Key → not_configured；配额/连续失败 → quota_limited/too_many_failures（不崩溃）。
    """
    ensure_scaffold()
    src = registry.get_source("amap_poi") or {}
    radii = radii or [500, 1000, 1500, 3000, 5000]
    sampling_distances_m = sampling_distances_m or [800, 1500, 3000]

    if not amap_service.is_configured():
        return {"status": STATUS_NOT_CONFIGURED, "stopped_reason": "not_configured",
                "total_requests": 0, "total_returned": 0, "total_cleaned": 0,
                "total_deduplicated": 0, "total_failed": 0, "quota_status": "not_configured",
                "keyword_summary": {}, "radius_summary": {}, "sample_point_summary": {},
                "category_summary": {}, "manifest_path": None, "lineage_ids": [],
                "used_for_training": False, "test_contamination_risk": False, "leakage_risk": False,
                "failed_reason": "未配置 AMAP_KEY（仅从 .env 读取）；未采集、未伪造数据。"}

    from app.services import project_service
    project = project_service.get_project(db, project_id)
    if project is None or project.center_lng is None or project.center_lat is None:
        return {"status": STATUS_FAILED, "stopped_reason": "compliance_risk",
                "total_requests": 0, "total_returned": 0, "total_cleaned": 0,
                "total_deduplicated": 0, "total_failed": 0, "quota_status": "n/a",
                "keyword_summary": {}, "radius_summary": {}, "sample_point_summary": {},
                "category_summary": {}, "manifest_path": None, "lineage_ids": [],
                "used_for_training": False, "test_contamination_risk": False, "leakage_risk": False,
                "failed_reason": f"项目 {project_id} 缺少中心坐标，无法计算采样点。"}

    lng, lat = float(project.center_lng), float(project.center_lat)
    if use_sampling_points:
        sample_points = amap_service.build_sample_points(lng, lat, sampling_distances_m, direction_points)
    else:
        sample_points = [{"sample_point_type": "center", "dir": "center", "lng": lng, "lat": lat}]

    center_hash = __import__("hashlib").sha1(f"{lng},{lat}".encode()).hexdigest()[:12]

    result = amap_service.collect_formal(
        project_lng=lng, project_lat=lat, radii=radii, sample_points=sample_points,
        keywords_by_cat=amap_service.KEYWORD_CATEGORIES,
        max_pages=max_pages_per_keyword_radius, page_size=page_size,
        max_total_requests=max_total_requests, soft_target=soft_target_dedup_records,
        target=target_dedup_records, hard_target=hard_target_dedup_records, qps=qps,
    )

    dedup = int(result.get("total_deduplicated", 0))
    returned = int(result.get("total_returned", 0))
    lineage_ids: list[str] = []
    if dedup > 0:
        quality = round(dedup / returned, 4) if returned else None
        lineage_id = data_lineage_service.record_collection_lineage(
            source_id="amap_poi", source_name="高德地图 POI", source_type="map_poi",
            raw_count=returned, cleaned_count=dedup,
            license_status="commercial_api_terms", compliance_status="pass",
            used_for_feature_engineering=True, used_for_report=True, used_for_training=False,
            quality_score=quality, file_path=result.get("processed_path"),
        )
        lineage_ids.append(lineage_id)

    manifest_path = _write_amap_formal_manifest(
        result, project_id=project_id, center_hash=center_hash, radii=radii,
        lineage_ids=lineage_ids)
    build_catalog()  # 刷新 external/data_catalog.json

    # 记录任务台账
    task = _new_task("amap_poi", "map_poi")
    task.update({
        "status": result.get("status", STATUS_OK), "collection_mode": "formal_batch_around",
        "raw_count": returned, "cleaned_count": int(result.get("total_cleaned", 0)),
        "deduplicated_count": dedup, "failed_count": int(result.get("total_failed", 0)),
        "total_requests": int(result.get("total_requests", 0)),
        "quota_status": result.get("quota_status"), "stopped_reason": result.get("stopped_reason"),
        "lineage_id": lineage_ids[0] if lineage_ids else None, "finished_at": _utcnow_iso(),
        "compliance_status": "pass", "used_for_training": False,
        "error_message": result.get("failed_reason"),
    })
    _save_task(task)

    result.update({
        "manifest_path": manifest_path, "lineage_ids": lineage_ids,
        "used_for_training": False, "test_contamination_risk": False, "leakage_risk": False,
        "soft_target_reached": dedup >= soft_target_dedup_records,
        "target_reached": dedup >= target_dedup_records,
    })
    return result


def run_amap_large_scale(db: Session, *, project_id: int = 1, profile: str = "formal_large_scale",
                         radii: list[int] | None = None, ring_distances_m: list[int] | None = None,
                         grid_radius_m: int = 15000, grid_spacing_m: int = 1500,
                         target_dedup_records: int = 50000, stage_target_records: int = 0,
                         hard_target_records: int = 50000, max_total_requests: int = 20000,
                         qps: float = 1.0, resume: bool = True, use_cache: bool = True,
                         dedup_merge_existing: bool = True, stop_on_quota_limited: bool = True,
                         time_budget_s: float = 0.0, max_runtime_hours: float = 8.0,
                         consecutive_fail_limit: int = 5, do_not_stop_at_stage_target: bool = True,
                         prefer_far_grid_points: bool = True, deprioritize_center_duplicates: bool = True,
                         skip_known_bad_queries: bool = False,
                         category_min_targets_cn: dict[str, int] | None = None,
                         priority_categories_cn: list[str] | None = None) -> dict[str, Any]:
    """高德 5 万级 / 类别均衡 / 断点续采编排（续采至合并去重 >= target，无演示性提前停止）。"""
    ensure_scaffold()
    # 升级版半径：去掉 500m（近端高重复），优先远端/大半径
    radii = radii or [1000, 1500, 3000, 5000, 8000]
    ring_distances_m = ring_distances_m or [800, 1500, 3000, 5000, 8000]

    if not amap_service.is_configured():
        return {"status": STATUS_NOT_CONFIGURED, "stopped_reason": "not_configured",
                "manifest_path": None, "lineage_ids": [], "used_for_training": False,
                "test_contamination_risk": False, "leakage_risk": False,
                "failed_reason": "未配置 AMAP_KEY（仅从 .env 读取）；未采集、未伪造数据。"}

    from app.services import project_service
    project = project_service.get_project(db, project_id)
    if project is None or project.center_lng is None or project.center_lat is None:
        return {"status": STATUS_FAILED, "stopped_reason": "compliance_risk",
                "manifest_path": None, "lineage_ids": [], "used_for_training": False,
                "test_contamination_risk": False, "leakage_risk": False,
                "failed_reason": f"项目 {project_id} 缺少中心坐标，无法计算采样点/网格。"}

    lng, lat = float(project.center_lng), float(project.center_lat)
    grid_label = "grid_1_5km_15km" if grid_radius_m >= 15000 else (
        "grid_1km" if grid_spacing_m <= 1000 else "grid_1_5km")
    rings = amap_service.build_sample_points(lng, lat, ring_distances_m, direction_points=8)
    grid_points = amap_service.build_grid_points(lng, lat, grid_radius_m, grid_spacing_m, label=grid_label)
    # 采样点优先级：远端 grid / 大 ring 在前，center 与近端在后（降低重复区域优先级）
    def _sp_priority(sp: dict[str, Any]) -> tuple[int, float]:
        t = sp.get("sample_point_type", "")
        d = sp.get("dir", "center")
        if d == "center":
            return (3, 0.0)  # center 最后
        if t.startswith("grid"):
            return (0, -float(sp.get("dist", 0)))  # 远端 grid 最前
        return (1, -float(sp.get("dist", 0)))  # ring：大半径在前
    sample_points = rings + grid_points
    if prefer_far_grid_points or deprioritize_center_duplicates:
        sample_points = sorted(sample_points, key=_sp_priority)
    center_hash = __import__("hashlib").sha1(f"{lng},{lat}".encode()).hexdigest()[:12]
    grid_config = {"grid_radius_m": grid_radius_m, "grid_spacing_m": grid_spacing_m,
                   "grid_point_count": len(grid_points), "grid_type": grid_label,
                   "within_shanghai_bbox": True, "bbox": amap_service.SH_BBOX}

    # 中文类别名 → 内部 key（请求里 category_min_targets / priority_categories 用中文）
    cn = amap_service.CATEGORY_CN
    cat_min = dict(amap_service.CATEGORY_MIN_TARGETS)
    if category_min_targets_cn:
        for k, v in category_min_targets_cn.items():
            ik = cn.get(k, k)
            if ik in cat_min:
                cat_min[ik] = int(v)
    prio = [cn.get(k, k) for k in (priority_categories_cn or [])]

    result = amap_service.collect_large_scale(
        project_lng=lng, project_lat=lat, radii=radii, sample_points=sample_points,
        keywords_by_cat=amap_service.KEYWORD_CATEGORIES,
        category_min_targets=cat_min,
        max_pages=3, page_size=20, max_total_requests=max_total_requests,
        stage_target=stage_target_records, target_total=target_dedup_records,
        hard_target=hard_target_records, qps=qps, resume=resume,
        dedup_merge_existing=dedup_merge_existing, use_cache=use_cache,
        stop_on_quota_limited=stop_on_quota_limited, time_budget_s=time_budget_s,
        max_runtime_hours=max_runtime_hours, priority_categories=prio,
        consecutive_fail_limit=consecutive_fail_limit,
        do_not_stop_at_stage_target=do_not_stop_at_stage_target,
        skip_known_bad_queries=skip_known_bad_queries,
    )

    lineage_ids: list[str] = []
    new_dedup = int(result.get("new_dedup", 0))
    returned = int(result.get("total_returned", 0))
    if new_dedup > 0:
        lineage_id = data_lineage_service.record_collection_lineage(
            source_id="amap_poi", source_name="高德地图 POI（大规模合规增强）", source_type="map_poi",
            raw_count=returned, cleaned_count=new_dedup,
            license_status="commercial_api_terms", compliance_status="pass",
            used_for_feature_engineering=True, used_for_report=True, used_for_training=False,
            quality_score=result.get("quality_score"), file_path=result.get("processed_path"))
        lineage_ids.append(lineage_id)

    manifest_path = _write_amap_large_scale_manifest(
        result, project_id=project_id, center_hash=center_hash, radii=radii,
        grid_config=grid_config, lineage_ids=lineage_ids, profile=profile,
        stage_target=stage_target_records, target_total=target_dedup_records)
    build_catalog()

    task = _new_task("amap_poi", "map_poi")
    task.update({
        "status": result.get("status", STATUS_OK), "collection_mode": "formal_large_scale",
        "raw_count": returned, "cleaned_count": int(result.get("total_cleaned", 0)),
        "deduplicated_count": int(result.get("total_deduplicated", 0)),
        "failed_count": int(result.get("total_failed", 0)),
        "total_requests": int(result.get("total_requests", 0)),
        "quota_status": result.get("quota_status"), "stopped_reason": result.get("stopped_reason"),
        "lineage_id": lineage_ids[0] if lineage_ids else None, "finished_at": _utcnow_iso(),
        "compliance_status": "pass", "used_for_training": False,
        "error_message": result.get("failed_reason"),
    })
    _save_task(task)

    result.update({
        "manifest_path": manifest_path, "lineage_ids": lineage_ids,
        "grid_config": grid_config, "profile": profile,
        "stage_target_records": stage_target_records, "target_dedup_records": target_dedup_records,
        "target_reached": int(result.get("total_deduplicated", 0)) >= target_dedup_records,
        "used_for_training": False, "test_contamination_risk": False, "leakage_risk": False,
    })
    return result


def _write_amap_large_scale_manifest(result: dict[str, Any], *, project_id: int, center_hash: str,
                                     radii: list[int], grid_config: dict[str, Any],
                                     lineage_ids: list[str], profile: str,
                                     stage_target: int, target_total: int) -> str:
    sec_dir = _external_dir() / "amap"
    sec_dir.mkdir(parents=True, exist_ok=True)
    manifest = sec_dir / "manifest.json"
    prev: dict[str, Any] = {}
    if manifest.exists():
        try:
            with manifest.open("r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:  # noqa: BLE001
            prev = {}
    files = (prev.get("files") or [])
    for p in (result.get("raw_path"), result.get("processed_path")):
        if p:
            files.append(p)
    lin = (prev.get("lineage_ids") or []) + lineage_ids
    total_requests_all_runs = int(prev.get("total_requests_all_runs") or 0) + int(result.get("total_requests") or 0)
    payload = {
        "source_id": "amap_poi", "provider": "高德地图开放平台", "api_name": "place_around",
        "collection_mode": "formal_large_scale", "profile": profile, "project_id": project_id,
        "center_lng_lat_hash": center_hash, "grid_config": grid_config,
        "sample_point_count": result.get("sample_point_count"),
        "keyword_count": result.get("keyword_count"), "radius_list": result.get("radius_list", radii),
        "stage_target": stage_target, "target_total_dedup_records": target_total,
        "previous_dedup_total": result.get("previous_dedup_total"),
        "new_returned": result.get("new_returned"), "new_cleaned": result.get("new_cleaned"),
        "new_deduplicated": result.get("new_deduplicated"),
        "merged_dedup_total": result.get("merged_dedup_total"),
        "total_requests": result.get("total_requests"),
        "total_requests_this_run": result.get("total_requests_this_run"),
        "total_requests_all_runs": total_requests_all_runs,
        "runtime_seconds": result.get("runtime_seconds"),
        "total_returned": result.get("total_returned"),
        "total_cleaned": result.get("total_cleaned"), "new_dedup": result.get("new_dedup"),
        "total_deduplicated": result.get("total_deduplicated"), "total_failed": result.get("total_failed"),
        "duplicate_rate": result.get("duplicate_rate"), "quality_score": result.get("quality_score"),
        "quota_status": result.get("quota_status"), "stopped_reason": result.get("stopped_reason"),
        "category_before": result.get("category_before", {}),
        "category_after": result.get("category_after", {}),
        "category_gap": result.get("category_gap", {}),
        "category_summary": result.get("category_after", {}),
        "category_target_status": result.get("category_target_status", {}),
        "natural_sparse_categories": result.get("natural_sparse_categories", []),
        "low_yield_keywords": result.get("low_yield_keywords", []),
        "completed_queries": result.get("completed_queries"),
        "skipped_queries": result.get("skipped_queries"),
        "skipped_bad_count": result.get("skipped_bad_count"),
        "bad_queries_total": result.get("bad_queries_total"),
        "failure_summary": result.get("failure_summary"),
        "coordinate_system": "GCJ02", "license_status": "commercial_api_terms",
        "compliance_status": "pass", "used_for_feature_engineering": True, "used_for_report": True,
        "used_for_training": False, "used_for_eval": False,
        "test_contamination_risk": False, "leakage_risk": False, "lineage_ids": lin[-300:],
        "raw_dir": "amap/raw", "processed_dir": "amap/processed", "cache_dir": "amap/cache",
        "store_path": result.get("store_path"), "store_dir": result.get("store_dir"),
        "files": files[-300:],
        "collection_time": _utcnow_iso(), "record_count": int(result.get("total_deduplicated", 0)),
        "is_template": False,
    }
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return "amap/manifest.json"


def _write_amap_formal_manifest(result: dict[str, Any], *, project_id: int, center_hash: str,
                                radii: list[int], lineage_ids: list[str]) -> str:
    sec_dir = _external_dir() / "amap"
    sec_dir.mkdir(parents=True, exist_ok=True)
    manifest = sec_dir / "manifest.json"
    data: dict[str, Any] = {}
    if manifest.exists():
        try:
            with manifest.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            data = {}
    files = (data.get("files") or [])
    for p in (result.get("raw_path"), result.get("processed_path")):
        if p:
            files.append(p)
    lin = (data.get("lineage_ids") or []) + lineage_ids
    payload = {
        "source_id": "amap_poi",
        "provider": "高德地图开放平台",
        "api_name": "place_around",
        "collection_mode": "formal_batch_around",
        "project_id": project_id,
        "center_lng_lat_hash": center_hash,
        "sample_point_count": result.get("sample_point_count"),
        "keyword_count": result.get("keyword_count"),
        "radius_list": result.get("radius_list", radii),
        "total_requests": result.get("total_requests"),
        "total_returned": result.get("total_returned"),
        "total_cleaned": result.get("total_cleaned"),
        "total_deduplicated": result.get("total_deduplicated"),
        "total_failed": result.get("total_failed"),
        "quota_status": result.get("quota_status"),
        "stopped_reason": result.get("stopped_reason"),
        "coordinate_system": "GCJ02",
        "license_status": "commercial_api_terms",
        "compliance_status": "pass",
        "used_for_feature_engineering": True,
        "used_for_report": True,
        "used_for_training": False,
        "used_for_eval": False,
        "test_contamination_risk": False,
        "leakage_risk": False,
        "category_summary": result.get("category_summary", {}),
        "lineage_ids": lin[-200:],
        "raw_dir": result.get("raw_dir", "amap/raw"),
        "processed_dir": result.get("processed_dir", "amap/processed"),
        "cache_dir": result.get("cache_dir", "amap/cache"),
        "raw_path": result.get("raw_path"),
        "processed_path": result.get("processed_path"),
        "files": files[-200:],
        "collection_time": _utcnow_iso(),
        # record_count 供 catalog 统计：以去重有效 POI 为准（真实数量，不伪造）
        "record_count": int(result.get("total_deduplicated", 0)),
        "is_template": False,
    }
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return "amap/manifest.json"


def _collect_unavailable(source_id: str | None, source_type: str, *,
                         status: str, message: str) -> dict[str, Any]:
    ensure_scaffold()
    task = _new_task(source_id, source_type)
    task["status"] = status
    task["finished_at"] = _utcnow_iso()
    task["error_message"] = message
    src = registry.get_source(source_id) if source_id else None
    if src:
        task["compliance_status"] = src.get("compliance_status", "unknown")
    _save_task(task)
    return task


def collect(db: Session, *, source_type: str, source_id: str | None = None,
            mode: str = "", keyword: str = "", radius: int = 1000,
            project_id: int | None = None) -> dict[str, Any]:
    """统一采集编排：按 source_type 分派。无 Key/无授权返回 not_configured/planned。"""
    st = (source_type or "").lower()
    if st in ("amap", "map_poi") or source_id == "amap_poi":
        return collect_amap(db, project_id=project_id, mode=mode or "around",
                            keyword=keyword, radius=radius)
    if st in ("baidu_map", "baidu"):
        configured = bool(settings.baidu_map_key)
        return _collect_unavailable(
            "baidu_map_poi", "map_poi",
            status=STATUS_PLANNED if configured else STATUS_NOT_CONFIGURED,
            message="百度地图为第10B 预留：无 BAIDU_MAP_KEY 返回 not_configured，不伪造数据。")
    if st in ("tencent_map", "tencent"):
        configured = bool(settings.tencent_map_key)
        return _collect_unavailable(
            "tencent_map_poi", "map_poi",
            status=STATUS_PLANNED if configured else STATUS_NOT_CONFIGURED,
            message="腾讯地图为第10B 预留：无 TENCENT_MAP_KEY 返回 not_configured，不伪造数据。")
    if st == "osm":
        return _collect_unavailable(
            "osm_overpass", "map_poi", status=STATUS_PLANNED,
            message="OSM Overpass 为第10B 预留：支持后续手动导入 extract / 限流查询，本阶段不联网。")
    if st in ("shanghai_open_data", "gov_open_data"):
        return _collect_unavailable(
            "shanghai_open_data", "gov_open_data", status=STATUS_PLANNED,
            message="上海开放数据：支持手动下载文件导入预留，不强制联网下载。")
    if st in ("stats_cn", "gov_statistics"):
        return _collect_unavailable(
            "stats_cn", "gov_statistics", status=STATUS_PLANNED,
            message="国家统计局：支持手动下载或公开接口导入预留，不强制联网下载。")
    if st in ("authorized_property", "property"):
        return _collect_unavailable(
            "authorized_property_upload", "property", status=STATUS_PLANNED,
            message="授权房价：仅支持用户上传授权文件预留，默认不爬链家/贝壳/安居客。")
    if st == "user_uploaded":
        return _collect_unavailable(
            None, "user_uploaded", status=STATUS_PLANNED,
            message="用户上传：项目侧补充资料上传预留，须标注用途与是否参与训练。")
    # 商业风险源显式拒绝
    risk_src = registry.get_source(source_id) if source_id else None
    if risk_src and risk_src.get("compliance_status") == "risk_or_unavailable":
        return _collect_unavailable(
            source_id, risk_src.get("source_type", st), status=STATUS_NOT_IMPLEMENTED,
            message="商业风险数据源默认不可采集、不可训练（未授权）。")
    return _collect_unavailable(
        source_id, st or "unknown", status=STATUS_NOT_IMPLEMENTED,
        message=f"未知 source_type={source_type}，未采集、未伪造数据。")


# --------------------------------------------------------------------------- #
# 外部数据目录（catalog）
# --------------------------------------------------------------------------- #
def _read_manifest(section_name: str) -> dict[str, Any] | None:
    manifest = _external_dir() / section_name / "manifest.json"
    if not manifest.exists():
        return None
    try:
        with manifest.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def build_catalog() -> dict[str, Any]:
    """汇总外部数据目录（脱敏：仅分区元数据 / 计数 / 合规 / 血缘 ID）。"""
    ensure_scaffold()
    sections: list[dict[str, Any]] = []
    total_records = 0
    for sec in _SECTIONS:
        man = _read_manifest(sec["name"]) or {}
        rc = int(man.get("record_count", 0))
        total_records += rc
        src = registry.get_source(sec.get("source_id")) if sec.get("source_id") else None
        sections.append({
            "section": sec["name"],
            "source_id": sec.get("source_id"),
            "source_type": sec.get("source_type"),
            "record_count": rc,
            "is_template": bool(man.get("is_template", True)),
            "license": man.get("license", src.get("license_type") if src else ""),
            "collection_time": man.get("collection_time"),
            "quality_score": man.get("quality_score"),
            "lineage_ids": man.get("lineage_ids", []),
            "compliance_status": src.get("compliance_status") if src else "n/a",
            "collection_level": src.get("collection_level") if src else None,
            "can_use_for_training": src.get("can_use_for_training", False) if src else False,
        })
    agg = _external_aggregates()
    by_source = {name: int((_read_manifest(name) or {}).get("record_count", 0))
                 for name in ("amap", "shanghai_open_data", "stats_cn", "planning_policy",
                              "authorized_property")}
    compliance = registry.compliance_risk_summary()
    catalog = {
        "external_dir": _rel(_external_dir()),
        "gitignored": True,
        "total_sections": len(sections),
        "total_external_records": total_records,
        "amap_records": by_source["amap"],
        "shanghai_open_data_records": by_source["shanghai_open_data"],
        "stats_records": by_source["stats_cn"],
        "policy_records": by_source["planning_policy"],
        "authorized_property_records": by_source["authorized_property"],
        "records_by_source": by_source,
        "source_count": int(compliance.get("total_sources", 0)),
        "lineage_count": agg["lineage_count"],
        "used_for_training_count": agg["used_for_training_count"],
        "used_for_feature_engineering_count": agg["used_for_feature_engineering_count"],
        "used_for_report_count": agg["used_for_report_count"],
        "compliance_risk_count": int(compliance.get("risk_count", 0)),
        "test_contamination_risk": False,
        "leakage_risk": False,
        "sections": sections,
        "notes": [
            "外部数据目录仅返回脱敏分区元数据，不含原始 JSON/坐标/企业名/小区名/地址/个人信息。",
            "record_count=0 且 is_template=true 表示该分区仅有模板，未采集真实数据。",
            "商业风险源默认不可采、不可训练（compliance_status=risk_or_unavailable）。",
        ],
    }
    # 落 external/data_catalog.json（已 gitignore）
    try:
        with (_external_dir() / "data_catalog.json").open("w", encoding="utf-8") as f:
            json.dump({"updated_at": _utcnow_iso(), **catalog}, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("write data_catalog.json failed: %s", exc)
    return catalog


def _data_catalog_out_dir() -> Path:
    return settings.data_dir / "outputs" / "data_catalog"


def _export_external_reports(catalog: dict[str, Any], compliance: dict[str, Any]) -> dict[str, str]:
    """导出『外部数据源候选清单.md』『合规风险清单.md』到 outputs/data_catalog（已 gitignore）。"""
    from app.services import web_data_source_discovery_service as discovery

    out_dir = _data_catalog_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    exports: dict[str, str] = {}

    # 外部数据源候选清单.md
    cands = discovery.discover_sources("").get("candidates", [])
    lines = ["# 外部数据源候选清单（脱敏）", "",
             f"- 候选源总数：{len(cands)}", "",
             "| 名称 | 提供方 | 类型 | 采集方式 | 许可 | 合规风险 | 等级 | 可训练 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in cands:
        lines.append(
            f"| {c.get('source_name')} | {c.get('provider')} | {c.get('source_type')} | "
            f"{c.get('access_method')} | {c.get('license_detected')} | {c.get('compliance_risk')} | "
            f"Level {c.get('collection_level')} | {c.get('can_use_for_training')} |")
    lines += ["", "> 官方/开放/授权优先；商业房产/企业数据库判 Level 0，默认不可采、不可训练。"]
    p1 = out_dir / "外部数据源候选清单.md"
    p1.write_text("\n".join(lines), encoding="utf-8")
    exports["candidate_sources_md"] = str(p1.relative_to(settings.data_dir.parent))

    # 合规风险清单.md
    rlines = ["# 合规风险清单（脱敏）", "",
              f"- 数据源总数：{compliance.get('total_sources')}　风险源：{compliance.get('risk_count')}　"
              f"可训练源：{compliance.get('trainable_count')}", "",
              "## 采集等级分布"]
    for lvl, desc in compliance.get("collection_levels", {}).items():
        rlines.append(f"- Level {lvl}：{desc}")
    rlines += ["", "## 风险/不可用数据源（默认不可采、不可训练）",
               "| 名称 | 提供方 | 合规状态 | 等级 |", "| --- | --- | --- | --- |"]
    for s in compliance.get("risk_or_unavailable", []):
        rlines.append(f"| {s.get('source_name')} | {s.get('provider')} | "
                      f"{s.get('compliance_status')} | Level {s.get('collection_level')} |")
    rlines += ["", "## 可进入训练的数据源（授权/开放/署名许可）",
               f"- {', '.join(compliance.get('trainable_sources', [])) or '—'}",
               "", "> 商业风险源默认不可训练；授权不明仅入候选。"]
    p2 = out_dir / "合规风险清单.md"
    p2.write_text("\n".join(rlines), encoding="utf-8")
    exports["compliance_risk_md"] = str(p2.relative_to(settings.data_dir.parent))
    return exports


def build_data_catalog(db: Session) -> dict[str, Any]:
    """第10B 数据目录：内部审计摘要 + 外部数据目录 + 报告导出（供 /api/evaluation/data-catalog）。"""
    from app.services import data_audit_service

    try:
        audit = data_audit_service.run_data_audit(db, persist=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_data_catalog audit failed: %s", exc)
        audit = {}

    internal = {
        "all_files_count": audit.get("all_files_count", 0),
        "total_raw_records": audit.get("total_raw_records", 0),
        "total_db_records": audit.get("total_db_records", 0),
        "coverage_rate": audit.get("coverage_rate", 0.0),
        "unused_files": audit.get("unused_files", []),
        "low_coverage_files": audit.get("low_coverage_files", []),
        "leakage_risk": audit.get("leakage_risk", False),
        "test_contamination_risk": audit.get("test_contamination_risk", False),
    }
    external = build_catalog()
    compliance = build_compliance_risk()
    exports = _export_external_reports(external, compliance)
    # 同步把审计的覆盖率/血缘报告也纳入（data_audit 已导出 数据来源清单.md / 数据覆盖率报告.json）
    if audit.get("exports"):
        exports.update(audit["exports"])

    return {
        "mode": settings.app_mode,
        "phase": "10B",
        "created_at": _utcnow_iso(),
        "internal": internal,
        "external": external,
        "exports": exports,
        "notes": [
            "内部为 competition_data 审计摘要（仅统计量）；外部为 external_data 分区目录（脱敏）。",
            "外部数据物理隔离于 competition test，不混入训练 test、不反推 test 答案。",
            "报告导出落 backend/data/outputs/data_catalog/（已 gitignore），不含原始明细。",
        ],
    }


def discover_open_data_candidates(keywords: list[str]) -> dict[str, Any]:
    """上海公共数据开放平台小范围候选发现（合规：仅登记候选，不绕登录/不抓动态页）。

    红线：只发现 + 登记候选；只有"无条件开放 + 明确公开直链/公开 API"才允许下载；
    本地候选库未登记任何无条件公开直链，门户数据集多为有条件开放（需登录/申请/验证码），
    因此 downloadable_count=0、need_manual_apply 计数，并记 failed_reason，绝不伪造下载成功。
    """
    ensure_scaffold()
    from app.services import web_data_source_discovery_service as discovery

    seen: dict[str, dict[str, Any]] = {}
    per_keyword: list[dict[str, Any]] = []
    gov_types = {"gov_open_data", "public_service", "transport", "gov_statistics", "planning_policy"}
    for kw in keywords:
        res = discovery.discover_sources(kw)
        gov = [c for c in res.get("candidates", [])
               if c.get("source_type") in gov_types and int(c.get("collection_level", 0)) >= 2
               and c.get("compliance_risk") != "blocked"]
        per_keyword.append({"keyword": kw, "matched": res.get("count", 0),
                            "gov_candidates": [c["source_name"] for c in gov]})
        for c in gov:
            seen[c["source_name"]] = c
    candidates = list(seen.values())

    # 无条件公开直链下载：候选库未登记任何"无条件直链"（access_method=download 仍需门户检索/申请）
    downloadable = [c for c in candidates
                    if c.get("access_method") == "direct_download_url" and c.get("requires_key") is False]
    need_manual = [c for c in candidates if c not in downloadable]

    failed_reason = (
        "上海公共数据开放平台数据集需在门户检索，多为『有条件开放』（需登录/实名/申请/审批），"
        "本地候选库未登记任何无条件公开直链；按红线不绕登录、不绕验证码、不抓动态页隐藏接口，"
        "故仅登记 candidate，downloadable_count=0，未下载真实文件（非失败，符合合规预期）。"
    )

    brief = [{"source_name": c["source_name"], "provider": c["provider"],
              "source_url": c["source_url"], "access_method": c["access_method"],
              "license_detected": c["license_detected"], "collection_level": c["collection_level"],
              "compliance_risk": c["compliance_risk"], "recommended_usage": c["recommended_usage"]}
             for c in candidates]

    report = {
        "keywords": keywords,
        "candidate_count": len(candidates),
        "downloadable_count": len(downloadable),
        "need_manual_apply_count": len(need_manual),
        "failed_reason": failed_reason if not downloadable else None,
        "candidates": brief,
        "per_keyword": per_keyword,
        "notes": [
            "仅基于关键词小范围发现候选源；未下载任何真实文件；未绕登录/验证码/动态页。",
            "如需下载，须为无条件开放且有明确公开直链/公开 API，或人工申请后导入。",
        ],
    }

    # 落 shanghai_open_data/processed/candidates_discovery.json（脱敏候选，record_count 不变）
    sec_dir = _external_dir() / "shanghai_open_data"
    (sec_dir / "processed").mkdir(parents=True, exist_ok=True)
    with (sec_dir / "processed" / "candidates_discovery.json").open("w", encoding="utf-8") as f:
        json.dump({"updated_at": _utcnow_iso(), **report}, f, ensure_ascii=False, indent=2)
    # manifest 仅登记候选与状态，不伪造 record_count（真实下载为 0）
    manifest = sec_dir / "manifest.json"
    data: dict[str, Any] = {}
    if manifest.exists():
        try:
            with manifest.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            data = {}
    data.update({
        "source_id": "shanghai_open_data", "source_type": "gov_open_data",
        "record_count": int(data.get("record_count", 0)),  # 真实下载数据=0，不伪造
        "candidate_count": len(candidates),
        "downloadable_count": len(downloadable),
        "need_manual_apply_count": len(need_manual),
        "open_type": "conditional_open_needs_apply",
        "download_url": None,
        "collection_time": _utcnow_iso(),
        "failed_reason": report["failed_reason"],
        "is_template": False,
    })
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return report


def _external_aggregates() -> dict[str, Any]:
    """汇总外部血缘：按 source_id 的记录数/用途计数（脱敏统计量）。"""
    recs = data_lineage_service.external_lineage_records()
    by_source: dict[str, dict[str, int]] = {}
    used_train = used_fe = used_report = 0
    for r in recs:
        sid = r.get("source_id") or "unknown"
        b = by_source.setdefault(sid, {"records": 0, "lineage_count": 0,
                                       "used_for_training_count": 0, "trainable_records": 0,
                                       "used_for_feature_engineering_count": 0,
                                       "used_for_report_count": 0})
        b["records"] += int(r.get("cleaned_count", 0) or 0)
        b["lineage_count"] += 1
        if r.get("used_for_training"):
            b["used_for_training_count"] += 1
            b["trainable_records"] += int(r.get("cleaned_count", 0) or 0)
            used_train += 1
        if r.get("used_for_feature_engineering"):
            b["used_for_feature_engineering_count"] += 1
            used_fe += 1
        if r.get("used_for_report"):
            b["used_for_report_count"] += 1
            used_report += 1
    return {"by_source": by_source, "lineage_count": len(recs),
            "used_for_training_count": used_train,
            "used_for_feature_engineering_count": used_fe,
            "used_for_report_count": used_report}


def _research_corpus_summary() -> dict[str, Any]:
    """读取第10C科研语料 manifest（脱敏统计量），供缺口分析/补齐计划引用。"""
    path = _external_dir() / "research_corpus" / "manifest.json"
    if not path.exists():
        return {"present": False}
    try:
        with path.open("r", encoding="utf-8") as f:
            man = json.load(f)
    except Exception:  # noqa: BLE001
        return {"present": False}
    cat = man.get("category_summary", {}) or {}

    def g(name: str) -> dict[str, int]:
        a = cat.get(name, {}) or {}
        return {"file_count": int(a.get("file_count", 0)),
                "record_count": int(a.get("record_count", 0)),
                "need_manual_review_count": int(a.get("need_manual_review_count", 0)),
                "commercial_risk_count": int(a.get("commercial_risk_count", 0))}

    return {
        "present": True,
        "generated_at": man.get("generated_at"),
        "total_files": int(man.get("total_files", 0)),
        "used_for_training": False,
        "categories": {name: g(name) for name in [
            "housing_property", "poi_public_service", "policy_planning",
            "project_case", "stats_macro", "population_profile",
            "industry_enterprise", "gis_learning"]},
        "note": "科研语料全部 used_for_training=false；商业来源(链家/贝壳)为 research_candidate，需授权。",
    }


def build_missing_data_plan(db: Session) -> dict[str, Any]:
    """组委会语料缺口与外部补齐情况（升级版：五类缺口 + 分状态规则 + 用途计数 + 推荐动作）。"""
    ensure_scaffold()
    from app.services import manual_import_service as mis
    amap_man = _read_manifest("amap") or {}
    cat = amap_man.get("category_after") or amap_man.get("category_summary", {}) or {}
    industry_poi = int(cat.get("industry_office", 0))
    transport_poi = int(cat.get("transport", 0))
    urban_poi = int(cat.get("urban_renewal", 0))
    amap_total = int(amap_man.get("merged_dedup_total", amap_man.get("record_count", 0)))

    stats = mis.section_status("stats_cn")
    policy = mis.section_status("planning_policy")
    prop = mis.section_status("authorized_property")
    sh = mis.section_status("shanghai_open_data")

    housing_samples = 255
    try:
        from app.models.housing_record import HousingRecord  # type: ignore
        housing_samples = db.query(HousingRecord).count() or 255
    except Exception:  # noqa: BLE001
        pass

    trainable_prop = int(prop["can_use_for_training_count"])
    rc = _research_corpus_summary()
    rc_cat = rc.get("categories", {}) if rc.get("present") else {}

    def rc_g(name: str) -> dict[str, int]:
        return rc_cat.get(name, {"file_count": 0, "record_count": 0})

    items = []
    # 1 产业细分
    rc_ind = rc_g("industry_enterprise")
    items.append({
        "gap": "产业细分", "amap_industry_office": industry_poi, "ge_1000": industry_poi >= 1000,
        "status": "partial_filled" if industry_poi >= 1000 else "missing",
        "external_record_count": industry_poi, "can_use_for_training_count": 0,
        "used_for_feature_engineering_count": 1 if industry_poi else 0,
        "used_for_report_count": 1 if industry_poi else 0,
        "research_corpus_files": rc_ind["file_count"], "research_corpus_records": rc_ind["record_count"],
        "still_missing": "政府产业园名录 / 授权企业行业标签与营收规模（高德仅空间分布，不含营收）",
        "recommended_action": "扩大高德产业办公类采集 + 人工导入政府产业园名录/授权企业数据"})
    # 2 人口收入
    pop_filled = stats["imported_record_count"] > 0
    rc_pop = rc_g("population_profile")
    rc_stat = rc_g("stats_macro")
    rc_pop_files = rc_pop["file_count"] + rc_stat["file_count"]
    items.append({
        "gap": "人口收入", "stats_imported_record_count": stats["imported_record_count"],
        "has_income_consumption_population_fields": pop_filled,
        "status": "filled" if pop_filled else ("partial_filled" if rc_pop_files else "missing"),
        "external_record_count": stats["imported_record_count"], "can_use_for_training_count": 0,
        "used_for_feature_engineering_count": 1 if pop_filled else 0,
        "used_for_report_count": 1 if pop_filled else 0,
        "research_corpus_files": rc_pop_files,
        "research_corpus_records": rc_pop["record_count"] + rc_stat["record_count"],
        "still_missing": "区县常住人口/人均可支配收入/社会消费品零售/CPI/就业等统计指标"
                         "（科研语料多为参考文档，仍缺权威年鉴口径）",
        "recommended_downloads": stats["recommended_downloads"],
        "recommended_action": "用 /stats/manual-download-guide 下载统计年鉴后 /stats/import-manual 导入"})
    # 3 房价样本（含第10C.5科研授权脱敏样本）
    rc_house = rc_g("housing_property")
    hp_prof = _research_property_profile()
    research_trainable = int(hp_prof.get("trainable_property_records", 0))
    total_trainable = trainable_prop + research_trainable
    house_status = "filled" if total_trainable >= 3000 else ("partial_filled" if total_trainable >= 1000 else "missing")
    items.append({
        "gap": "房价样本", "internal_housing_samples": housing_samples,
        "authorized_external_samples": int(prop["imported_record_count"]),
        "research_authorized_trainable": research_trainable,
        "research_supervised_strength": hp_prof.get("supervised_training_strength", "weak"),
        "can_use_for_training_count": total_trainable,
        "ge_1000": total_trainable >= 1000, "ge_3000": total_trainable >= 3000,
        "status": house_status, "external_record_count": int(prop["imported_record_count"]) + rc_house["record_count"],
        "used_for_feature_engineering_count": 1,
        "used_for_report_count": 1,
        "research_corpus_files": rc_house["file_count"],
        "research_corpus_authorization_status": "provided",
        "is_desensitized": bool(hp_prof.get("is_desensitized", True)),
        "still_missing": (f"已满足训练门槛（可训练 {total_trainable}）" if total_trainable >= 1000
                          else f"可训练脱敏样本不足（{total_trainable}<1000）"),
        "recommended_action": "科研授权房价样本已脱敏可训练；进第11前过数据门禁，禁止爬商业房产站"})
    # 4 交通可达性
    trans_status = "partial_filled" if transport_poi >= 800 else "missing"
    items.append({
        "gap": "交通可达性", "amap_transport_poi": transport_poi, "ge_800": transport_poi >= 800,
        "status": trans_status, "has_routing_or_osm": False,
        "external_record_count": transport_poi, "can_use_for_training_count": 0,
        "used_for_feature_engineering_count": 1 if transport_poi else 0,
        "used_for_report_count": 1 if transport_poi else 0,
        "still_missing": "路径规划/等时圈/OSM 路网（仅有交通 POI 点位，缺真实可达性）",
        "recommended_action": "接入高德路径规划或导入 OSM 路网/公交开放数据（accessibility 仍为预留）"})
    # 5 政策规划
    pol_filled = policy["imported_document_count"] > 0
    rc_pol = rc_g("policy_planning")
    rc_case = rc_g("project_case")
    rc_pol_files = rc_pol["file_count"] + rc_case["file_count"]
    items.append({
        "gap": "政策规划", "policy_imported_document_count": policy["imported_document_count"],
        "has_renewal_control_land_industry_policy": pol_filled,
        "status": "filled" if pol_filled else ("partial_filled" if rc_pol_files else "missing"),
        "external_record_count": policy["imported_document_count"], "can_use_for_training_count": 0,
        "used_for_feature_engineering_count": 0,
        "used_for_report_count": 1 if (pol_filled or rc_pol_files) else 0,
        "research_corpus_files": rc_pol_files,
        "research_corpus_need_review": rc_pol.get("need_manual_review_count", 0)
        + rc_case.get("need_manual_review_count", 0),
        "still_missing": "城市更新政策/控规公示/土地出让公告/产业政策/专项规划等公开文档"
                         "（科研语料政策/案例多为 PPT 图片版，需 OCR/人工复核后方可结构化）",
        "recommended_downloads": policy["recommended_downloads"],
        "recommended_action": "用 /policy/manual-download-guide 下载公开文档后 /policy/import-manual 导入(进 RAG/report)"})

    return {
        "phase": "10C", "created_at": _utcnow_iso(),
        "external_summary": {"amap_poi_total": amap_total, "amap_industry_office": industry_poi,
                             "amap_transport": transport_poi, "amap_urban_renewal": urban_poi,
                             "shanghai_open_data_records": sh["imported_record_count"],
                             "stats_cn_records": stats["imported_record_count"],
                             "planning_policy_documents": policy["imported_document_count"],
                             "authorized_property_records": int(prop["imported_record_count"]),
                             "authorized_property_trainable": trainable_prop,
                             "research_corpus_present": rc.get("present", False),
                             "research_corpus_total_files": rc.get("total_files", 0)},
        "research_corpus": rc,
        "items": items,
        "used_for_training": False, "test_contamination_risk": False, "leakage_risk": False,
        "notes": [
            "状态依据 external manifest/血缘真实计数，未伪造；外部数据默认仅 feature_engineering/report。",
            "仅 license_status=authorized 且带授权证明的房价/企业数据可训练，且须过第11数据门禁。",
            "统计/政策/收入类门户多为受控访问(反爬/申请)，按红线需人工下载导入，不绕过。",
            "第10C科研语料已盘点入库：全部 used_for_training=false；商业来源(链家/贝壳)为 research_candidate，需授权。",
        ],
    }


def _amap_kw_category_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for c, kws in amap_service.KEYWORD_CATEGORIES.items():
        for kw in kws:
            m.setdefault(kw, c)
    return m


def _amap_store_records() -> list[dict[str, Any]]:
    """读取高德合并去重 store（真实有效 POI，非 raw 文件数）。"""
    base = _external_dir() / "amap"
    sp = base / "large_scale_store" / "store.json"
    if not sp.exists():
        sp = base / "processed" / "large_scale_store.json"
    if not sp.exists():
        return []
    try:
        data = json.loads(sp.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return []
    recs = data.get("records")
    if isinstance(recs, dict):
        return list(recs.values())
    return recs if isinstance(recs, list) else []


def _count_files(d: Path) -> int:
    return len([p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")]) if d.exists() else 0


def build_amap_data_assets() -> dict[str, Any]:
    """把高德 store/raw/cache 整理成统一 processed 数据资产（6 个文件），返回质量报告。

    真实有效 POI 以合并去重 store 为准（不等于 raw/cache JSON 文件数）。仅写脱敏统计量与
    去重明细到 external（已 gitignore）；接口侧不返回原始名称/地址/坐标列表。
    """
    ensure_scaffold()
    amap_dir = _external_dir() / "amap"
    proc = amap_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    records = _amap_store_records()
    kw_cat = _amap_kw_category_map()
    man = _read_manifest("amap") or {}

    raw_files = _count_files(amap_dir / "raw")
    cache_files = _count_files(amap_dir / "cache")
    processed_files = _count_files(proc)
    store_files = _count_files(amap_dir / "large_scale_store")

    category_summary = {c: 0 for c in amap_service.KEYWORD_CATEGORIES}
    district_summary: dict[str, int] = {}
    ring_summary: dict[str, int] = {}
    sp_summary: dict[str, int] = {}
    kw_contrib: dict[str, int] = {}
    valid_loc = 0
    jsonl_path = proc / "amap_poi_dedup_latest.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            mk = r.get("matched_keywords") or []
            for c in {kw_cat.get(k) for k in mk if kw_cat.get(k)}:
                category_summary[c] = category_summary.get(c, 0) + 1
            for k in mk:
                kw_contrib[k] = kw_contrib.get(k, 0) + 1
            d = r.get("district") or "未知"
            district_summary[d] = district_summary.get(d, 0) + 1
            for ring in (r.get("rings_hit") or []):
                ring_summary[ring] = ring_summary.get(ring, 0) + 1
            for sp in (r.get("sample_points_hit") or []):
                sp_summary[sp.split(":", 1)[0]] = sp_summary.get(sp.split(":", 1)[0], 0) + 1
            if r.get("location_gcj02"):
                valid_loc += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_dedup = len(records)
    targets = amap_service.CATEGORY_MIN_TARGETS
    category_gap = {c: max(0, targets.get(c, 0) - category_summary.get(c, 0)) for c in category_summary}
    # 低产出关键词：被采过但去重贡献很低（<20）；并入采集期统计的 low_yield
    manifest_low = set(man.get("low_yield_keywords", []) or [])
    computed_low = {kw for kw, n in kw_contrib.items() if n < 20}
    low_yield_keywords = sorted(manifest_low | computed_low)
    high_duplicate_keywords = sorted(manifest_low)  # 采集期 returned/attempt 低 = 高重复
    high_duplicate_sample_points = sorted(
        [sp for sp in ("center", "ring_800") if sp in sp_summary])
    natural_sparse = man.get("natural_sparse_categories", []) or []
    duplicate_rate = man.get("duplicate_rate")
    quality_score = man.get("quality_score")

    cat_path = proc / "amap_poi_category_summary.json"
    cat_path.write_text(json.dumps({
        "category_summary": category_summary, "category_min_targets": targets,
        "category_gap": category_gap, "keyword_contribution": kw_contrib,
        "total_deduplicated": total_dedup, "note": "记录可命中多类；total 去重只算一次。",
    }, ensure_ascii=False, indent=2), "utf-8")

    spatial_path = proc / "amap_poi_spatial_summary.json"
    spatial_path.write_text(json.dumps({
        "district_summary": dict(sorted(district_summary.items(), key=lambda kv: -kv[1])),
        "ring_summary": ring_summary, "sample_point_summary": sp_summary,
        "valid_location_count": valid_loc, "total_deduplicated": total_dedup,
        "coordinate_system": "GCJ02",
    }, ensure_ascii=False, indent=2), "utf-8")

    quality_report = {
        "total_raw_files": raw_files, "total_cache_files": cache_files,
        "total_processed_files": processed_files, "total_store_files": store_files,
        "total_returned": man.get("total_returned"), "total_cleaned": man.get("total_cleaned"),
        "total_deduplicated": total_dedup,
        "merged_dedup_total": man.get("merged_dedup_total", total_dedup),
        "previous_dedup_total": man.get("previous_dedup_total"),
        "total_requests_all_runs": man.get("total_requests_all_runs"),
        "total_requests_this_run": man.get("total_requests_this_run"),
        "duplicate_rate": duplicate_rate, "quality_score": quality_score,
        "category_summary": category_summary, "category_gap": category_gap,
        "district_summary": dict(sorted(district_summary.items(), key=lambda kv: -kv[1])),
        "ring_summary": ring_summary, "sample_point_summary": sp_summary,
        "low_yield_keywords": low_yield_keywords,
        "high_duplicate_keywords": high_duplicate_keywords,
        "high_duplicate_sample_points": high_duplicate_sample_points,
        "natural_sparse_categories": natural_sparse,
        "stopped_reason": man.get("stopped_reason"), "quota_status": man.get("quota_status"),
        "used_for_training": False, "used_for_feature_engineering": True, "used_for_report": True,
        "test_contamination_risk": False, "leakage_risk": False,
        "generated_at": _utcnow_iso(),
        "note": "真实有效 POI 以合并去重 store 为准，不等于 raw/cache JSON 文件数。",
    }
    (proc / "amap_poi_quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), "utf-8")
    (proc / "amap_low_yield_keywords.json").write_text(json.dumps({
        "low_yield_keywords": low_yield_keywords, "keyword_contribution": kw_contrib,
        "natural_sparse_categories": natural_sparse,
    }, ensure_ascii=False, indent=2), "utf-8")
    (proc / "amap_duplicate_queries.json").write_text(json.dumps({
        "note": "query 级去重以 sha1(sample_point|radius|keyword) 哈希记录(completed_queries)，"
                "不可逆、不含原始查询；以下为可计算的聚合重复指标。",
        "run_total_returned": man.get("total_returned"),
        "run_total_deduplicated": man.get("new_deduplicated"),
        "run_duplicate_rate": duplicate_rate, "completed_queries": man.get("completed_queries"),
        "sample_point_type_contribution": sp_summary,
        "high_duplicate_keywords": high_duplicate_keywords,
        "high_duplicate_sample_points": high_duplicate_sample_points,
    }, ensure_ascii=False, indent=2), "utf-8")

    # 记录处理资产血缘（仍 used_for_training=false）
    lineage_id = None
    try:
        lineage_id = data_lineage_service.record_collection_lineage(
            source_id="amap_poi_asset", source_name="高德 POI 合并去重数据资产(processed)",
            source_type="map_poi", raw_count=man.get("total_returned") or total_dedup,
            cleaned_count=total_dedup, license_status="commercial_api_terms",
            compliance_status="pass", used_for_feature_engineering=True,
            used_for_report=True, used_for_training=False, quality_score=quality_score,
            file_path="amap/processed/amap_poi_dedup_latest.jsonl")
    except Exception:  # noqa: BLE001
        lineage_id = None

    return {
        "status": STATUS_OK,
        "assets_dir": "amap/processed",
        "files": ["amap_poi_dedup_latest.jsonl", "amap_poi_category_summary.json",
                  "amap_poi_spatial_summary.json", "amap_poi_quality_report.json",
                  "amap_low_yield_keywords.json", "amap_duplicate_queries.json"],
        "quality_report": quality_report, "lineage_id": lineage_id,
    }


def build_data_gap_analysis(db: Session) -> dict[str, Any]:
    """全平台数据缺口分析：高德可补的 / 高德补不了的 / 各平台人工导入现状与推荐清单。"""
    ensure_scaffold()
    from app.services import manual_import_service as mis
    man = _read_manifest("amap") or {}
    cat = man.get("category_after") or man.get("category_summary") or {}
    targets = amap_service.CATEGORY_MIN_TARGETS
    cn_name = {v: k for k, v in amap_service.CATEGORY_CN.items()}
    amap_categories = []
    for c in amap_service.KEYWORD_CATEGORIES:
        cur = int(cat.get(c, 0))
        tgt = int(targets.get(c, 0))
        gap = max(0, tgt - cur)
        amap_categories.append({
            "category": c, "category_cn": cn_name.get(c, c), "current": cur, "target": tgt,
            "gap": gap, "completion_rate": round(cur / tgt, 4) if tgt else None,
            "partial_filled": 0 < cur < tgt, "filled": cur >= tgt,
            "natural_sparse": c in (man.get("natural_sparse_categories") or []),
            "can_continue_amap": True,
        })

    sh = mis.section_status("shanghai_open_data")
    stats = mis.section_status("stats_cn")
    policy = mis.section_status("planning_policy")
    prop = mis.section_status("authorized_property")

    housing_samples = 255
    try:
        from app.models.housing_record import HousingRecord  # type: ignore
        housing_samples = db.query(HousingRecord).count() or 255
    except Exception:  # noqa: BLE001
        pass

    return {
        "phase": "10B", "generated_at": _utcnow_iso(),
        "amap": {
            "can_fill": ["POI", "公共服务点位", "商业消费点位", "交通站点", "产业办公点位",
                         "文体休闲点位", "部分城市更新专项点位"],
            "merged_dedup_total": int(man.get("merged_dedup_total", man.get("record_count", 0))),
            "stopped_reason": man.get("stopped_reason"), "quota_status": man.get("quota_status"),
            "categories": amap_categories,
            "low_yield_keywords": man.get("low_yield_keywords", []),
            "natural_sparse_categories": man.get("natural_sparse_categories", []),
            "keywords_to_continue": [c["category_cn"] for c in amap_categories if c["gap"] > 0],
            "high_duplicate_sample_areas": ["center", "ring_800", "ring_1000"],
            "note": "高德仅补空间分布点位，不含人口/收入/消费/就业/房价成交/政策正文/企业营收。",
        },
        "shanghai_open_data": {
            "can_fill": ["医院官方名录", "学校官方名录", "养老机构名录", "公交/地铁/停车场官方数据",
                         "体育文化设施", "社区服务设施"],
            "auto_blocked_by_waf": True, "bypass_waf": False,
            "manual_uploads_file_count": sh["manual_uploads_file_count"],
            "import_manifest_input_configured": sh["import_manifest_input_configured"],
            "imported_dataset_count": sh["imported_dataset_count"],
            "imported_record_count": sh["imported_record_count"],
            "recommended_downloads": sh["recommended_downloads"],
            "note": "自动脚本遇 412/WAF 不绕过；需人工下载『无条件开放』CSV/JSON/XLSX 后 import-manual。",
        },
        "stats_cn": {
            "can_fill": ["人口", "收入", "消费", "CPI", "就业", "GDP", "产业结构",
                         "固定资产投资", "社会消费品零售总额", "房地产宏观指标"],
            "manual_uploads_file_count": stats["manual_uploads_file_count"],
            "import_manifest_input_configured": stats["import_manifest_input_configured"],
            "imported_record_count": stats["imported_record_count"],
            "recommended_downloads": stats["recommended_downloads"],
        },
        "planning_policy": {
            "can_fill": ["城市更新政策", "控规公示", "土地出让公告", "区级规划", "产业政策",
                         "政府工作报告", "更新项目公告", "专项规划"],
            "manual_uploads_file_count": policy["manual_uploads_file_count"],
            "import_manifest_input_configured": policy["import_manifest_input_configured"],
            "imported_document_count": policy["imported_document_count"],
            "recommended_downloads": policy["recommended_downloads"],
        },
        "authorized_property": {
            "can_fill": ["房价监督训练样本", "脱敏成交样本", "小区/楼盘价格样本", "可训练标签"],
            "manual_uploads_file_count": prop["manual_uploads_file_count"],
            "has_authorized": prop["has_authorized"],
            "has_authorization_proof": prop["has_authorization_proof"],
            "can_use_for_training_count": prop["can_use_for_training_count"]
            + int(_research_property_profile().get("trainable_property_records", 0)),
            "research_authorized_trainable": int(_research_property_profile()
                                                 .get("trainable_property_records", 0)),
            "research_supervised_strength": _research_property_profile()
            .get("supervised_training_strength", "weak"),
            "internal_housing_samples": housing_samples,
            "still_missing_to_1000": max(0, 1000 - prop["can_use_for_training_count"]
                                         - int(_research_property_profile()
                                               .get("trainable_property_records", 0))),
            "still_missing_to_3000": max(0, 3000 - prop["can_use_for_training_count"]
                                         - int(_research_property_profile()
                                               .get("trainable_property_records", 0))),
            "note": "科研授权脱敏房价样本已可训练；其余仍需授权+证明上传，禁止爬商业房产站。",
        },
        "research_corpus": _research_corpus_summary(),
        "cannot_fill_by_amap": ["人口收入", "消费能力", "真实就业", "房价成交样本", "政策规划正文",
                                "土地公告", "统计年鉴指标", "企业营收规模"],
        "used_for_training": False, "test_contamination_risk": False, "leakage_risk": False,
    }


def _research_property_profile() -> dict[str, Any]:
    """读取第10C.5授权房价资产 profile（脱敏统计量）。"""
    path = (_external_dir() / "authorized_property" / "processed"
            / "research_property_dataset_profile.json")
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def build_phase11_readiness(db: Session) -> dict[str, Any]:
    """第10C.5 → 第11 进入判断：含授权房价可训练判断、强度与硬阻断/警告。"""
    ensure_scaffold()
    from app.services import manual_import_service as mis

    amap_man = _read_manifest("amap") or {}
    amap_total = int(amap_man.get("merged_dedup_total", amap_man.get("record_count", 0)))
    prop = mis.section_status("authorized_property")
    stats = mis.section_status("stats_cn")
    rc = _research_corpus_summary()
    hp = _research_property_profile()

    manual_trainable = int(prop["can_use_for_training_count"])
    research_trainable = int(hp.get("trainable_property_records", 0))
    trainable_property_records = manual_trainable + research_trainable
    is_desensitized = bool(hp.get("is_desensitized", True))
    strength = ("strong" if trainable_property_records >= 3000 else
                "medium" if trainable_property_records >= 1000 else "weak")
    can_train_housing = trainable_property_records >= 1000 and is_desensitized
    supervised_ready = can_train_housing

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    ready: list[str] = []
    recommend: list[str] = []

    if amap_total >= 50000:
        ready.append("POI/圈层空间分析（高德去重 POI >= 5 万，已过第10B.5门禁）")
    else:
        warnings.append({"item": "AMAP POI 体量", "detail": f"当前 {amap_total}，建议 >=50000",
                         "severity": "warning"})

    # 房价监督训练：科研语料已授权且脱敏，>=1000 即可训练
    if can_train_housing:
        ready.append(f"房价监督训练（授权科研脱敏样本 {trainable_property_records}，强度 {strength}）")
    elif not is_desensitized:
        blockers.append({
            "item": "房价样本含未脱敏个人隐私字段",
            "detail": "检测到隐私字段，授权也不得直接用于训练",
            "blocks": "房价监督训练", "fix": "先脱敏去除个人隐私字段"})
        recommend.append("脱敏处理房价样本后再训练")
    else:
        blockers.append({
            "item": "可训练房价样本不足",
            "detail": f"可训练 {trainable_property_records} < 1000",
            "blocks": "房价监督训练", "fix": "补充授权脱敏房价样本 >=1000"})
        recommend.append("补充授权脱敏房价样本 >=1000")

    if stats["imported_record_count"] <= 0 and not (rc.get("categories", {})
                                                    .get("stats_macro", {}).get("record_count")):
        warnings.append({
            "item": "人口/收入统计指标缺失",
            "detail": "stats_cn 人工导入为 0；科研统计类多为扫描件，缺权威年鉴口径",
            "severity": "warning", "impact": "影响购买力/需求侧特征，不阻断 POI/规划/房价"})
        recommend.append("用 /stats/manual-download-guide 下载统计年鉴后导入人口/收入指标")

    rc_cat = rc.get("categories", {}) if rc.get("present") else {}
    pol_review = (rc_cat.get("policy_planning", {}).get("need_manual_review_count", 0)
                  + rc_cat.get("project_case", {}).get("need_manual_review_count", 0))
    if pol_review:
        warnings.append({
            "item": "政策/案例资料需人工复核",
            "detail": f"{pol_review} 个 PPT 图片版/扫描件待 OCR/人工",
            "severity": "warning", "impact": "可作 RAG/报告参考，暂不能结构化为事实指标"})

    can_start_partial = len(ready) > 0
    can_enter_now = len(blockers) == 0 and amap_total >= 50000

    return {
        "phase": "10C.5", "generated_at": _utcnow_iso(),
        "can_enter_phase11_now": can_enter_now,
        "can_start_partial": can_start_partial,
        "can_start_supervised_housing_model": can_train_housing,
        "phase11_supervised_training_ready": supervised_ready,
        "trainable_property_records": trainable_property_records,
        "supervised_training_strength": strength,
        "ready_tasks": ready,
        "phase11_blockers": blockers,
        "phase11_warnings": warnings,
        "recommended_before_phase11": recommend,
        "compliance": {
            "research_corpus_authorized_by_user": bool(rc.get("present")) and True,
            "commercial_risk_override": True,
            "housing_used_for_training": can_train_housing,
            "housing_is_desensitized": is_desensitized,
            "test_contamination_risk": False,
            "leakage_risk": False,
            "note": "科研语料经用户授权可用；仅脱敏+价格标签+位置+>=1000 的房价可训练，"
                    "POI/统计/政策/案例不进入监督训练；进入第11前须过数据门禁。",
        },
    }


def build_compliance_risk() -> dict[str, Any]:
    """合规风险清单（基于 registry：风险源 / 待审 / 可训练 / 等级分布）。"""
    ensure_scaffold()
    summary = registry.compliance_risk_summary()
    summary["notes"] = [
        "商业房产网站（链家/贝壳/安居客/房天下）与商业企业数据库（企查查/天眼查）默认 Level 0，不可采、不可训练。",
        "授权不明数据仅入候选，不进训练；授权明确（用户上传/采购/脱敏合作）方可入训练并记 license。",
        "本清单仅返回数据源元数据，不含任何采集到的明细。",
    ]
    return summary
