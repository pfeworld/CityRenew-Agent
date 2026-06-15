"""特征工程响应模型（第10A阶段）。

红线：仅特征名/特征值/分组/来源计数/evidence_id；不含 raw_json/原始明细。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeatureBuildRequest(BaseModel):
    """T2 特征构建请求（POST body，可选；不传则用默认全开）。"""

    include_external_poi: bool = True
    include_research_poi: bool = False  # 科研餐饮库默认关闭，避免类别占比失真（见 poi_feature_service）
    include_internal_poi: bool = True
    dry_run: bool = False


class FeatureResponse(BaseModel):
    """项目级特征向量（build 与 latest 共用）。"""

    project_id: int
    created_at: str | None = None
    status: str | None = None
    feature_version: str | None = None
    feature_vector: list[float | None] = Field(default_factory=list)
    feature_names: list[str] = Field(default_factory=list)
    feature_values: dict[str, Any] = Field(default_factory=dict)
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)
    missing_features: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    data_lineage_ids: list[str] = Field(default_factory=list)
    used_source_counts: dict[str, int] = Field(default_factory=dict)
    feature_coverage_rate: float = 0.0
    overall_coverage_rate: float | None = None
    ring_summary: dict[str, Any] = Field(default_factory=dict)
    category_summary: dict[str, Any] = Field(default_factory=dict)
    short_board_vector: dict[str, Any] = Field(default_factory=dict)
    renewal_type_feature_vector: list[float] = Field(default_factory=list)
    poi_feature_quality: dict[str, Any] = Field(default_factory=dict)
    coordinate_system: str | None = None
    distance_method: str | None = None
    feature_build_log: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    allowed_splits: list[str] = Field(default_factory=list)
    used_test: bool = False
    test_used: bool = False
    include_external: bool = False
    notes: list[str] = Field(default_factory=list)


class PoiSummaryResponse(BaseModel):
    """POI/圈层脱敏摘要。"""

    project_id: int
    available: bool = True
    message: str | None = None
    feature_version: str | None = None
    coordinate_system: str | None = None
    distance_method: str | None = None
    used_source_counts: dict[str, int] = Field(default_factory=dict)
    ring_summary: dict[str, Any] = Field(default_factory=dict)
    category_summary: dict[str, Any] = Field(default_factory=dict)
    short_board_vector: dict[str, Any] = Field(default_factory=dict)
    poi_feature_quality: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class FeatureQualityResponse(BaseModel):
    """T2 特征质量门禁结果。"""

    project_id: int
    feature_version: str | None = None
    quality_status: str
    pass_: list[str] = Field(default_factory=list, alias="pass")
    warning: list[str] = Field(default_factory=list)
    fail: list[str] = Field(default_factory=list)
    feature_coverage_rate: float | None = None
    poi_total_count_1500m: int | None = None
    l1_covered: int | None = None
    data_lineage_ids_count: int | None = None
    can_enter_t3: bool | None = None
    reasons: list[str] = Field(default_factory=list)
    recommended_next_action: str | None = None

    model_config = {"populate_by_name": True}
