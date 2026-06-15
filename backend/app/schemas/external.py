"""第10B 合规外部数据增强响应模型。

红线：仅返回脱敏元数据（source_name/provider/type/count/status/license/compliance/
lineage_id/quality_score/summary）；不含原始 JSON 全文/坐标列表/企业名/小区名/地址/个人信息/
chunk_text/profile_json。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DataSourceItem(BaseModel):
    """数据源登记条目（与 data_source_registry.SOURCE_SCHEMA 对齐）。"""

    source_id: str
    source_name: str = ""
    source_type: str = ""
    provider: str = ""
    official_url_or_api: str = ""
    license_type: str = "unknown"
    collection_method: str = "manual"
    api_required: bool = False
    api_key_env_name: str | None = None
    allowed_usage: list[str] = Field(default_factory=list)
    forbidden_usage: list[str] = Field(default_factory=list)
    update_frequency: str = "unknown"
    coordinate_system: str = "unknown"
    privacy_level: str = "none"
    compliance_status: str = "needs_review"
    collection_level: int = 1
    can_use_for_training: bool = False
    can_use_for_feature_engineering: bool = False
    can_use_for_report: bool = False
    can_use_for_eval: bool = False
    notes: str = ""


class SourceListResponse(BaseModel):
    phase: str = "10B"
    count: int = 0
    collection_levels: dict[int, str] = Field(default_factory=dict)
    sources: list[DataSourceItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RegisterSourceRequest(BaseModel):
    source_id: str
    source_name: str = ""
    source_type: str = ""
    provider: str = ""
    official_url_or_api: str = ""
    license_type: str = "unknown"
    collection_method: str = "manual"
    api_required: bool = False
    api_key_env_name: str | None = None
    allowed_usage: list[str] = Field(default_factory=list)
    forbidden_usage: list[str] = Field(default_factory=list)
    update_frequency: str = "unknown"
    coordinate_system: str = "unknown"
    privacy_level: str = "none"
    compliance_status: str = "needs_review"
    collection_level: int = 1
    can_use_for_training: bool = False
    can_use_for_feature_engineering: bool = False
    can_use_for_report: bool = False
    can_use_for_eval: bool = False
    notes: str = ""


class RegisterSourceResponse(BaseModel):
    registered: bool = False
    replaced: bool = False
    source_id: str = ""
    source: DataSourceItem | None = None


class CandidateSourceItem(BaseModel):
    source_name: str = ""
    source_url: str = ""
    provider: str = ""
    source_type: str = ""
    data_category: str = ""
    access_method: str = "unknown"
    license_detected: str = "unknown"
    robots_policy_status: str = "unknown"
    api_available: bool = False
    requires_key: bool = False
    estimated_update_frequency: str = "unknown"
    expected_fields: list[str] = Field(default_factory=list)
    expected_record_count: str = "unknown"
    collection_feasibility: str = "unknown"
    compliance_risk: str = "needs_review"
    collection_level: int = 1
    recommended_usage: list[str] = Field(default_factory=list)
    can_use_for_training: bool = False
    can_use_for_feature_engineering: bool = False
    can_use_for_report: bool = False
    can_use_for_eval: bool = False
    match_keywords: list[str] = Field(default_factory=list)
    notes: str = ""


class DiscoverResponse(BaseModel):
    keyword: str = ""
    online_search: bool = False
    count: int = 0
    blocked_count: int = 0
    candidates: list[CandidateSourceItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ExternalCollectRequest(BaseModel):
    source_type: str = ""
    source_id: str | None = None
    project_id: int | None = None
    mode: str = ""
    keyword: str = ""
    radius: int = 1000


class CollectionTaskResponse(BaseModel):
    task_id: str
    source_id: str | None = None
    source_type: str = ""
    status: str = "planned"
    started_at: str | None = None
    finished_at: str | None = None
    raw_count: int = 0
    cleaned_count: int = 0
    failed_count: int = 0
    quota_status: str = "unknown"
    compliance_status: str = "unknown"
    cache_status: str = "n/a"
    lineage_id: str | None = None
    error_message: str | None = None
    # ---- 第10B 补充：采集明细（脱敏路径与用途；不含原始点位/名称/地址）----
    keyword: str | None = None
    radius: int | None = None
    raw_path: str | None = None
    processed_path: str | None = None
    used_for_feature_engineering: bool = False
    used_for_report: bool = False
    used_for_training: bool = False


class ExternalCollectResponse(CollectionTaskResponse):
    """采集响应（与任务结构一致，便于前端统一处理）。"""


class TaskListResponse(BaseModel):
    count: int = 0
    tasks: list[CollectionTaskResponse] = Field(default_factory=list)


class CatalogSectionItem(BaseModel):
    section: str
    source_id: str | None = None
    source_type: str = ""
    record_count: int = 0
    is_template: bool = True
    license: str = ""
    collection_time: str | None = None
    quality_score: float | None = None
    lineage_ids: list[str] = Field(default_factory=list)
    compliance_status: str = "n/a"
    collection_level: int | None = None
    can_use_for_training: bool = False


class DataCatalogResponse(BaseModel):
    external_dir: str = ""
    gitignored: bool = True
    total_sections: int = 0
    total_external_records: int = 0
    amap_records: int = 0
    shanghai_open_data_records: int = 0
    stats_records: int = 0
    policy_records: int = 0
    authorized_property_records: int = 0
    records_by_source: dict[str, int] = Field(default_factory=dict)
    source_count: int = 0
    lineage_count: int = 0
    used_for_training_count: int = 0
    used_for_feature_engineering_count: int = 0
    used_for_report_count: int = 0
    compliance_risk_count: int = 0
    test_contamination_risk: bool = False
    leakage_risk: bool = False
    sections: list[CatalogSectionItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DataLineageResponse(BaseModel):
    mode: str = "eval"
    phase: str = "10B"
    created_at: str = ""
    schema_fields: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    answers: dict[str, Any] = Field(default_factory=dict)
    exports: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ComplianceRiskResponse(BaseModel):
    total_sources: int = 0
    collection_levels: dict[int, str] = Field(default_factory=dict)
    by_collection_level: dict[str, int] = Field(default_factory=dict)
    risk_or_unavailable: list[dict[str, Any]] = Field(default_factory=list)
    needs_review_or_planned: list[dict[str, Any]] = Field(default_factory=list)
    trainable_sources: list[str] = Field(default_factory=list)
    risk_count: int = 0
    trainable_count: int = 0
    notes: list[str] = Field(default_factory=list)


class ScaffoldResponse(BaseModel):
    external_dir: str = ""
    sections: list[str] = Field(default_factory=list)
    created_dirs: int = 0
    created_manifests: int = 0
    gitignored: bool = True


class AmapFormalRequest(BaseModel):
    """高德正式批量采集请求（受 max_total_requests/qps/quota/停止条件 约束）。"""

    project_id: int = 1
    radii: list[int] = Field(default_factory=lambda: [500, 1000, 1500, 3000, 5000])
    use_sampling_points: bool = True
    sampling_distances_m: list[int] = Field(default_factory=lambda: [800, 1500, 3000])
    direction_points: int = 8
    max_pages_per_keyword_radius: int = 3
    page_size: int = 20
    max_total_requests: int = 2000
    soft_target_dedup_records: int = 1500
    target_dedup_records: int = 3000
    hard_target_dedup_records: int = 5000
    qps: float = 1.0


class AmapFormalResponse(BaseModel):
    status: str = "ok"
    total_requests: int = 0
    total_returned: int = 0
    total_cleaned: int = 0
    total_deduplicated: int = 0
    total_failed: int = 0
    stopped_reason: str | None = None
    quota_status: str = "unknown"
    keyword_summary: dict[str, int] = Field(default_factory=dict)
    radius_summary: dict[str, int] = Field(default_factory=dict)
    sample_point_summary: dict[str, int] = Field(default_factory=dict)
    category_summary: dict[str, int] = Field(default_factory=dict)
    sample_point_count: int | None = None
    keyword_count: int | None = None
    radius_list: list[int] = Field(default_factory=list)
    manifest_path: str | None = None
    lineage_ids: list[str] = Field(default_factory=list)
    soft_target_reached: bool = False
    target_reached: bool = False
    raw_path: str | None = None
    processed_path: str | None = None
    used_for_training: bool = False
    test_contamination_risk: bool = False
    leakage_risk: bool = False
    failed_reason: str | None = None


class ShanghaiSearchResponse(BaseModel):
    status: str = "ok"
    keyword: str = ""
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    blocked: bool = False
    failed_reason: str | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)


class ShanghaiCollectRequest(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    max_pages_per_keyword: int = 2
    max_datasets_per_keyword: int = 10
    max_total_downloads: int = 80
    stop_after_success: int = 30
    preferred_formats: list[str] = Field(default_factory=lambda: ["csv", "json", "xlsx"])
    only_unconditional: bool = True


class ShanghaiCollectResponse(BaseModel):
    searched_keywords: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    unconditional_count: int = 0
    conditional_count: int = 0
    downloadable_count: int = 0
    downloaded_count: int = 0
    failed_count: int = 0
    need_manual_apply_count: int = 0
    total_raw_records: int = 0
    total_cleaned_records: int = 0
    downloaded_datasets: list[dict[str, Any]] = Field(default_factory=list)
    failed_datasets: list[dict[str, Any]] = Field(default_factory=list)
    need_manual_apply_datasets: list[dict[str, Any]] = Field(default_factory=list)
    manifest_path: str | None = None
    lineage_ids: list[str] = Field(default_factory=list)
    blocked_by_anti_crawler: bool = False
    blocked_by_waf: bool = False
    can_auto_download: bool = True
    can_manual_import: bool = True
    failed_reason: str | None = None
    manual_import_endpoint: str | None = None
    catalog_paths: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AmapLargeScaleRequest(BaseModel):
    """高德 5 万级分阶段 / 类别均衡 / 断点续采请求。"""

    project_id: int = 1
    profile: str = "formal_large_scale"
    radii: list[int] | None = None
    ring_distances_m: list[int] = Field(default_factory=lambda: [800, 1500, 3000, 5000, 8000])
    grid_radius_m: int = 10000
    grid_spacing_m: int = 1500
    grid_config: dict[str, Any] | None = None
    target_dedup_records: int = 50000
    stage_target_records: int = 0
    hard_target_records: int = 50000
    max_total_requests: int = 20000
    max_total_requests_this_run: int | None = None
    max_runtime_hours: float = 8.0
    qps: float = 1.0
    concurrency: int = 1
    resume: bool = True
    use_cache: bool = True
    dedup_merge_existing: bool = True
    stop_on_quota_limited: bool = True
    consecutive_fail_limit: int = 5
    time_budget_s: float = 0.0
    disable_demo_time_budget: bool = True
    do_not_stop_at_stage_target: bool = True
    prefer_far_grid_points: bool = True
    deprioritize_center_duplicates: bool = True
    skip_known_bad_queries: bool = False
    category_min_targets: dict[str, int] | None = None
    priority_categories: list[str] | None = None


class AmapLargeScaleResponse(BaseModel):
    status: str = "ok"
    profile: str = "formal_large_scale"
    total_requests: int = 0
    total_returned: int = 0
    total_cleaned: int = 0
    new_dedup: int = 0
    total_deduplicated: int = 0
    total_failed: int = 0
    previous_dedup_total: int = 0
    merged_dedup_total: int = 0
    new_returned: int = 0
    new_cleaned: int = 0
    new_deduplicated: int = 0
    total_requests_this_run: int = 0
    total_requests_all_runs: int | None = None
    runtime_seconds: float | None = None
    duplicate_rate: float | None = None
    quality_score: float | None = None
    stopped_reason: str | None = None
    quota_status: str = "unknown"
    stage_target_records: int = 0
    target_dedup_records: int = 0
    target_reached: bool = False
    stage_reached: bool = False
    skipped_queries: int | None = None
    category_before: dict[str, int] = Field(default_factory=dict)
    category_after: dict[str, int] = Field(default_factory=dict)
    category_gap: dict[str, int] = Field(default_factory=dict)
    category_target_status: dict[str, str] = Field(default_factory=dict)
    natural_sparse_categories: list[str] = Field(default_factory=list)
    low_yield_keywords: list[str] = Field(default_factory=list)
    attempted_keywords: list[str] = Field(default_factory=list)
    failed_keywords: list[str] = Field(default_factory=list)
    keyword_count: int | None = None
    sample_point_count: int | None = None
    radius_list: list[int] = Field(default_factory=list)
    grid_config: dict[str, Any] = Field(default_factory=dict)
    completed_queries: int | None = None
    skipped_bad_count: int | None = None
    bad_queries_total: int | None = None
    recent_failures: list[dict[str, Any]] = Field(default_factory=list)
    failure_summary: dict[str, int] = Field(default_factory=dict)
    manifest_path: str | None = None
    store_path: str | None = None
    lineage_ids: list[str] = Field(default_factory=list)
    used_for_training: bool = False
    test_contamination_risk: bool = False
    leakage_risk: bool = False
    failed_reason: str | None = None


class ManualImportRequest(BaseModel):
    """人工下载文件导入请求；entries 为空时读取该分区 import_manifest_input.json。"""

    entries: list[dict[str, Any]] | None = None


class ManualImportResponse(BaseModel):
    section: str = ""
    status: str = "waiting_for_manual_upload"
    imported_count: int = 0
    failed_count: int = 0
    total_raw_records: int = 0
    total_cleaned_records: int = 0
    imported_datasets: list[dict[str, Any]] = Field(default_factory=list)
    failed_datasets: list[dict[str, Any]] = Field(default_factory=list)
    manifest_path: str | None = None
    lineage_ids: list[str] = Field(default_factory=list)
    used_for_training: bool = False
    test_contamination_risk: bool = False
    leakage_risk: bool = False
    guide: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)


class ManualGuideResponse(BaseModel):
    section: str = ""
    provider: str = ""
    portal: str = ""
    manual_uploads_dir: str = ""
    import_manifest_input: str = ""
    required_fields: list[str] = Field(default_factory=list)
    extra_meta: list[str] = Field(default_factory=list)
    supported_formats: list[str] = Field(default_factory=list)
    recommended_downloads: list[str] = Field(default_factory=list)
    existing_files: list[str] = Field(default_factory=list)
    status: str = "waiting_for_manual_upload"
    notes: list[str] = Field(default_factory=list)


class MissingDataPlanResponse(BaseModel):
    phase: str = "10B"
    created_at: str = ""
    external_summary: dict[str, Any] = Field(default_factory=dict)
    research_corpus: dict[str, Any] = Field(default_factory=dict)
    items: list[dict[str, Any]] = Field(default_factory=list)
    used_for_training: bool = False
    test_contamination_risk: bool = False
    leakage_risk: bool = False
    notes: list[str] = Field(default_factory=list)


class DataGapAnalysisResponse(BaseModel):
    phase: str = "10B"
    generated_at: str = ""
    amap: dict[str, Any] = Field(default_factory=dict)
    shanghai_open_data: dict[str, Any] = Field(default_factory=dict)
    stats_cn: dict[str, Any] = Field(default_factory=dict)
    planning_policy: dict[str, Any] = Field(default_factory=dict)
    authorized_property: dict[str, Any] = Field(default_factory=dict)
    research_corpus: dict[str, Any] = Field(default_factory=dict)
    cannot_fill_by_amap: list[str] = Field(default_factory=list)
    used_for_training: bool = False
    test_contamination_risk: bool = False
    leakage_risk: bool = False


class Phase11ReadinessResponse(BaseModel):
    """第10C → 第11 进入判断（基于真实缺口与合规计数）。"""

    phase: str = "10C"
    generated_at: str = ""
    can_enter_phase11_now: bool = False
    can_start_partial: bool = False
    can_start_supervised_housing_model: bool = False
    phase11_supervised_training_ready: bool = False
    trainable_property_records: int = 0
    supervised_training_strength: str = "weak"
    ready_tasks: list[str] = Field(default_factory=list)
    phase11_blockers: list[dict[str, Any]] = Field(default_factory=list)
    phase11_warnings: list[dict[str, Any]] = Field(default_factory=list)
    recommended_before_phase11: list[str] = Field(default_factory=list)
    compliance: dict[str, Any] = Field(default_factory=dict)


class AmapAssetsResponse(BaseModel):
    status: str = "ok"
    assets_dir: str = ""
    files: list[str] = Field(default_factory=list)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    lineage_id: str | None = None


class UploadPlannedResponse(BaseModel):
    """人工上传预留接口的统一响应（planned / not_implemented，不伪造成功）。"""

    status: str = "planned"
    source_id: str | None = None
    source_type: str = ""
    message: str = ""
    accepted: bool = False
