"""第10B 合规外部数据增强接口。

数据源发现 / 登记 / 列表、采集任务、外部数据目录 / 血缘 / 合规风险、地图采集、人工上传预留。

红线：无 Key / 无授权 → not_configured / planned / not_implemented，绝不伪造数据；
不调用未授权外部 API；不返回原始明细；商业风险源默认不可采、不可训练；
外部数据不污染 competition_test。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.external import (
    AmapFormalRequest,
    AmapFormalResponse,
    AmapAssetsResponse,
    AmapLargeScaleRequest,
    AmapLargeScaleResponse,
    ComplianceRiskResponse,
    DataCatalogResponse,
    DataGapAnalysisResponse,
    DataLineageResponse,
    DiscoverResponse,
    ExternalCollectRequest,
    ExternalCollectResponse,
    ManualGuideResponse,
    ManualImportRequest,
    ManualImportResponse,
    MissingDataPlanResponse,
    Phase11ReadinessResponse,
    RegisterSourceRequest,
    RegisterSourceResponse,
    ShanghaiCollectRequest,
    ShanghaiCollectResponse,
    ShanghaiSearchResponse,
    SourceListResponse,
    TaskListResponse,
    UploadPlannedResponse,
)
from app.services import data_lineage_service
from app.services import data_source_registry as registry
from app.services import external_data_collector_service as collector
from app.services import manual_import_service
from app.services import shanghai_open_data_service as shanghai
from app.services import web_data_source_discovery_service as discovery

router = APIRouter(prefix="/api/external", tags=["external"])


# --------------------------------------------------------------------------- #
# 数据源发现 / 登记 / 列表
# --------------------------------------------------------------------------- #
@router.get("/discover-sources", response_model=DiscoverResponse)
def discover_sources(keyword: str = Query(default="", description="关键词，空格分隔")):
    """根据关键词发现候选数据源（本地规则，不联网、不采集）。"""
    return DiscoverResponse(**discovery.discover_sources(keyword))


@router.get("/sources", response_model=SourceListResponse)
def list_sources():
    """列出全部登记数据源（含预置）。"""
    sources = registry.list_sources()
    return SourceListResponse(
        count=len(sources),
        collection_levels=registry.COLLECTION_LEVELS,
        sources=sources,
        notes=[
            "数据源登记仅含元数据，不含任何 API Key；Key 一律从 .env 读取。",
            "商业风险源默认 compliance_status=risk_or_unavailable 且 can_use_for_training=false。",
        ],
    )


@router.post("/register-source", response_model=RegisterSourceResponse)
def register_source(payload: RegisterSourceRequest):
    """登记 / 更新一个数据源（按 source_id 去重）。"""
    try:
        result = registry.register_source(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RegisterSourceResponse(**result)


# --------------------------------------------------------------------------- #
# 采集任务
# --------------------------------------------------------------------------- #
@router.post("/collect", response_model=ExternalCollectResponse)
def collect(payload: ExternalCollectRequest, db: Session = Depends(get_db)):
    """统一采集编排（按 source_type 分派）；无 Key/无授权返回 not_configured/planned。"""
    task = collector.collect(
        db, source_type=payload.source_type, source_id=payload.source_id,
        mode=payload.mode, keyword=payload.keyword, radius=payload.radius,
        project_id=payload.project_id,
    )
    return ExternalCollectResponse(**task)


@router.get("/tasks", response_model=TaskListResponse)
def list_tasks():
    return TaskListResponse(**collector.list_tasks())


@router.get("/tasks/{task_id}", response_model=ExternalCollectResponse)
def get_task(task_id: str):
    task = collector.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"采集任务 {task_id} 不存在")
    return ExternalCollectResponse(**task)


# --------------------------------------------------------------------------- #
# 外部数据目录 / 血缘 / 合规风险
# --------------------------------------------------------------------------- #
@router.get("/catalog", response_model=DataCatalogResponse)
def catalog():
    return DataCatalogResponse(**collector.build_catalog())


@router.get("/lineage", response_model=DataLineageResponse)
def lineage(db: Session = Depends(get_db)):
    return DataLineageResponse(**data_lineage_service.build_lineage(db, export=False))


@router.get("/compliance-risk", response_model=ComplianceRiskResponse)
def compliance_risk():
    return ComplianceRiskResponse(**collector.build_compliance_risk())


# --------------------------------------------------------------------------- #
# 地图采集（无 Key → not_configured）
# --------------------------------------------------------------------------- #
@router.post("/amap/collect", response_model=ExternalCollectResponse)
def amap_collect(payload: ExternalCollectRequest, db: Session = Depends(get_db)):
    task = collector.collect_amap(
        db, project_id=payload.project_id, mode=payload.mode or "around",
        keyword=payload.keyword, radius=payload.radius,
    )
    return ExternalCollectResponse(**task)


@router.post("/amap/collect-formal", response_model=AmapFormalResponse)
def amap_collect_formal(payload: AmapFormalRequest, db: Session = Depends(get_db)):
    """高德正式批量合规采集：采样点+六大类关键词+多半径+去重+限流+停止条件。

    无 Key → not_configured；配额/连续失败 → quota_limited/too_many_failures（不崩溃、不伪造）。
    """
    result = collector.run_amap_formal(
        db, project_id=payload.project_id, radii=payload.radii,
        use_sampling_points=payload.use_sampling_points,
        sampling_distances_m=payload.sampling_distances_m,
        direction_points=payload.direction_points,
        max_pages_per_keyword_radius=payload.max_pages_per_keyword_radius,
        page_size=payload.page_size, max_total_requests=payload.max_total_requests,
        soft_target_dedup_records=payload.soft_target_dedup_records,
        target_dedup_records=payload.target_dedup_records,
        hard_target_dedup_records=payload.hard_target_dedup_records, qps=payload.qps,
    )
    return AmapFormalResponse(**result)


@router.post("/amap/collect-large-scale", response_model=AmapLargeScaleResponse)
def amap_collect_large_scale(payload: AmapLargeScaleRequest, db: Session = Depends(get_db)):
    """高德 5 万级分阶段 / 类别均衡 / 断点续采（围绕项目研究区的城市更新相关 POI 合规增强）。

    QPS=1、并发=1、严格限流、缓存、断点续采；quota_limited 保存进度后正常返回，不崩溃、不伪造。
    """
    gc = payload.grid_config or {}
    grid_radius_m = int(gc["radius_km"] * 1000) if gc.get("radius_km") else payload.grid_radius_m
    grid_spacing_m = int(gc.get("grid_spacing_m") or payload.grid_spacing_m)
    max_req = payload.max_total_requests_this_run or payload.max_total_requests
    time_budget = 0.0 if payload.disable_demo_time_budget else payload.time_budget_s
    result = collector.run_amap_large_scale(
        db, project_id=payload.project_id, profile=payload.profile, radii=payload.radii,
        ring_distances_m=payload.ring_distances_m, grid_radius_m=grid_radius_m,
        grid_spacing_m=grid_spacing_m, target_dedup_records=payload.target_dedup_records,
        stage_target_records=payload.stage_target_records, hard_target_records=payload.hard_target_records,
        max_total_requests=max_req, qps=payload.qps, resume=payload.resume,
        use_cache=payload.use_cache, dedup_merge_existing=payload.dedup_merge_existing,
        stop_on_quota_limited=payload.stop_on_quota_limited, time_budget_s=time_budget,
        max_runtime_hours=payload.max_runtime_hours,
        consecutive_fail_limit=payload.consecutive_fail_limit,
        do_not_stop_at_stage_target=payload.do_not_stop_at_stage_target,
        prefer_far_grid_points=payload.prefer_far_grid_points,
        deprioritize_center_duplicates=payload.deprioritize_center_duplicates,
        skip_known_bad_queries=payload.skip_known_bad_queries,
        category_min_targets_cn=payload.category_min_targets,
        priority_categories_cn=payload.priority_categories,
    )
    return AmapLargeScaleResponse(**result)


# --------------------------------------------------------------------------- #
# 上海公共数据：搜索 / 无条件开放真实下载（遇反爬不绕过，不伪造下载成功）
# --------------------------------------------------------------------------- #
@router.get("/shanghai-open-data/search", response_model=ShanghaiSearchResponse)
def shanghai_search(keyword: str = Query(..., description="检索关键词"),
                    max_pages: int = Query(default=2, ge=1, le=2)):
    return ShanghaiSearchResponse(**shanghai.search_datasets(keyword, max_pages=max_pages))


@router.post("/shanghai-open-data/collect-public", response_model=ShanghaiCollectResponse)
def shanghai_collect_public(payload: ShanghaiCollectRequest):
    result = shanghai.collect_by_keywords(
        payload.keywords, max_pages_per_keyword=payload.max_pages_per_keyword,
        max_datasets_per_keyword=payload.max_datasets_per_keyword,
        max_total_downloads=payload.max_total_downloads,
        stop_after_success=payload.stop_after_success,
        preferred_formats=payload.preferred_formats, only_unconditional=payload.only_unconditional,
    )
    collector.build_catalog()  # 刷新外部目录（若有真实下载则计入）
    return ShanghaiCollectResponse(**result)


@router.get("/shanghai-open-data/manual-download-guide", response_model=ManualGuideResponse)
def shanghai_manual_guide():
    return ManualGuideResponse(**shanghai.manual_download_guide())


@router.post("/shanghai-open-data/import-manual", response_model=ManualImportResponse)
def shanghai_import_manual(payload: ManualImportRequest | None = None):
    entries = payload.entries if payload else None
    result = shanghai.import_manual_files(entries)
    collector.build_catalog()
    return ManualImportResponse(**result)


# --------------------------------------------------------------------------- #
# 统计局 / 统计年鉴：人工下载指南 + 人工导入
# --------------------------------------------------------------------------- #
@router.get("/stats/manual-download-guide", response_model=ManualGuideResponse)
def stats_manual_guide():
    return ManualGuideResponse(**manual_import_service.guide("stats_cn"))


@router.post("/stats/import-manual", response_model=ManualImportResponse)
def stats_import_manual(payload: ManualImportRequest | None = None):
    entries = payload.entries if payload else None
    result = manual_import_service.import_manual("stats_cn", entries)
    collector.build_catalog()
    return ManualImportResponse(**result)


# --------------------------------------------------------------------------- #
# 政府规划 / 公告 / 政策：人工下载指南 + 人工导入（进 RAG/report）
# --------------------------------------------------------------------------- #
@router.get("/policy/manual-download-guide", response_model=ManualGuideResponse)
def policy_manual_guide():
    return ManualGuideResponse(**manual_import_service.guide("planning_policy"))


@router.post("/policy/import-manual", response_model=ManualImportResponse)
def policy_import_manual(payload: ManualImportRequest | None = None):
    entries = payload.entries if payload else None
    result = manual_import_service.import_manual("planning_policy", entries)
    collector.build_catalog()
    return ManualImportResponse(**result)


# --------------------------------------------------------------------------- #
# 授权房价：上传指南 + 授权导入（license!=authorized → can_use_for_training 强制 false）
# --------------------------------------------------------------------------- #
@router.get("/authorized-property/manual-upload-guide", response_model=ManualGuideResponse)
def authorized_property_guide():
    return ManualGuideResponse(**manual_import_service.guide("authorized_property"))


# --------------------------------------------------------------------------- #
# 组委会语料缺口补齐计划
# --------------------------------------------------------------------------- #
@router.get("/missing-data-plan", response_model=MissingDataPlanResponse)
def missing_data_plan(db: Session = Depends(get_db)):
    return MissingDataPlanResponse(**collector.build_missing_data_plan(db))


@router.get("/data-gap-analysis", response_model=DataGapAnalysisResponse)
def data_gap_analysis(db: Session = Depends(get_db)):
    """全平台数据缺口分析（高德六大类 gap + 上海公共数据/统计局/政策/授权房价人工导入现状）。"""
    return DataGapAnalysisResponse(**collector.build_data_gap_analysis(db))


@router.get("/phase11-readiness", response_model=Phase11ReadinessResponse)
def phase11_readiness(db: Session = Depends(get_db)):
    """第10C → 第11 进入判断（硬阻断/警告/已就绪任务 + 合规自检）。"""
    return Phase11ReadinessResponse(**collector.build_phase11_readiness(db))


@router.post("/amap/build-assets", response_model=AmapAssetsResponse)
def amap_build_assets():
    """把高德 store/raw/cache 整理成统一 processed 数据资产（去重 jsonl + 类别/空间/质量报告）。"""
    return AmapAssetsResponse(**collector.build_amap_data_assets())


@router.post("/baidu-map/collect", response_model=ExternalCollectResponse)
def baidu_collect(payload: ExternalCollectRequest, db: Session = Depends(get_db)):
    return ExternalCollectResponse(**collector.collect(db, source_type="baidu_map",
                                                       keyword=payload.keyword))


@router.post("/tencent-map/collect", response_model=ExternalCollectResponse)
def tencent_collect(payload: ExternalCollectRequest, db: Session = Depends(get_db)):
    return ExternalCollectResponse(**collector.collect(db, source_type="tencent_map",
                                                       keyword=payload.keyword))


@router.post("/osm/collect", response_model=ExternalCollectResponse)
def osm_collect(payload: ExternalCollectRequest, db: Session = Depends(get_db)):
    return ExternalCollectResponse(**collector.collect(db, source_type="osm",
                                                       keyword=payload.keyword))


# --------------------------------------------------------------------------- #
# 人工上传预留（返回 planned，不伪造成功保存）
# --------------------------------------------------------------------------- #
@router.post("/upload-authorized", response_model=UploadPlannedResponse)
def upload_authorized():
    return UploadPlannedResponse(
        status="planned", source_type="user_uploaded",
        message="授权文件上传为第10B 预留：须填 data_owner/license/allowed_usage 并脱敏，"
                "授权明确方可入训练；本阶段不实际保存文件，不伪造成功。",
    )


@router.post("/import-open-data", response_model=UploadPlannedResponse)
def import_open_data():
    return UploadPlannedResponse(
        status="planned", source_id="shanghai_open_data", source_type="gov_open_data",
        message="开放数据手动导入为第10B 预留：支持下载文件导入与 source 注册，"
                "记录集名/下载时间/来源/字段/授权；本阶段不强制联网下载，不伪造成功。",
    )


@router.post("/upload-authorized-property", response_model=ManualImportResponse)
def upload_authorized_property(payload: ManualImportRequest | None = None):
    """导入用户人工放入 authorized_property/manual_uploads 的授权房价文件。

    license_status!=authorized → can_use_for_training 强制 false；authorized 且带 authorization_proof
    方可训练（仍须过第11数据门禁）；默认不爬链家/贝壳/安居客；无文件返回 waiting_for_manual_upload。
    """
    entries = payload.entries if payload else None
    result = manual_import_service.import_manual("authorized_property", entries)
    collector.build_catalog()
    return ManualImportResponse(**result)
