"""第11 阶段模型相关 Pydantic schema（T1：训练护栏 guard）。

仅含 guard-only 契约：guard-check 请求/响应、training-readiness 响应。
不含真实训练接口（T3 实现）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GuardCheckRequest(BaseModel):
    """训练护栏检查请求（dry-run，只检查不训练）。"""

    training_task: str = Field(
        default="housing_price_regression",
        description="训练任务：housing_price_regression / poi_feature_engineering 等",
    )
    project_id: int | None = Field(default=1, description="项目 ID（可选）")
    use_authorized_property: bool = Field(default=True, description="是否使用授权房价作训练标签源")
    use_poi_features: bool = Field(default=True, description="是否使用 POI 作特征（恒不进监督训练）")
    use_policy_as_label: bool = Field(
        default=False, description="仿真：误用政策文本作监督标签（自测护栏拦截用）"
    )
    use_poi_as_label: bool = Field(
        default=False, description="仿真：误用 POI 作监督标签（自测护栏拦截用）"
    )
    inject_test_records: bool = Field(
        default=False, description="仿真：训练源混入 split=test（自测护栏拦截用，仅元数据不读明细）"
    )
    requested_splits: list[str] = Field(
        default_factory=lambda: ["train", "val"], description="请求训练用 split（禁止含 test）"
    )
    housing_overrides: dict[str, Any] | None = Field(
        default=None, description="仿真：覆盖房价合规属性以自测 fail 场景（不改真实数据）"
    )
    dry_run: bool = Field(default=True, description="恒为 dry_run：本接口只检查不训练")


class GuardCheckResponse(BaseModel):
    """训练护栏检查结果。"""

    status: str  # pass / fail
    can_train: bool
    training_task: str
    requested_splits: list[str] = Field(default_factory=list)
    allowed_training_sources: list[str] = Field(default_factory=list)
    rejected_sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[dict[str, Any]] = Field(default_factory=list)
    data_usage_audit: dict[str, Any] = Field(default_factory=dict)
    test_used_for_training: bool = False
    supervised_training_strength: str | None = None
    dry_run: bool = True
    generated_at: str | None = None
    policy: str | None = None


class TrainingReadinessResponse(BaseModel):
    """训练就绪概览（默认房价请求 guard + 第10C.5 readiness 引用）。"""

    phase: str
    generated_at: str
    guard: GuardCheckResponse
    phase11_readiness: dict[str, Any] = Field(default_factory=dict)
    next_step: str | None = None


class TrainRequest(BaseModel):
    """T3 房价监督训练请求（接口必先跑 guard）。"""

    training_task: str = Field(default="housing_price_regression")
    project_id: int | None = Field(default=1)
    use_authorized_property: bool = Field(default=True, description="使用授权房价作训练标签源")
    use_t2_features: bool = Field(default=True, description="使用 T2/区级 POI 特征")
    use_poi_features: bool = Field(default=True, description="POI 作特征（恒不进监督标签）")
    use_t3_housing_model: bool = Field(default=True, description="T5：纳入 T3 房价模型输出")
    use_t4_project_type: bool = Field(default=True, description="T5：纳入 T4 类型识别输出")
    dry_run: bool = Field(default=False, description="true 仅跑护栏+数据审计不训练")


class TrainResponse(BaseModel):
    """T3 训练结果（含质量门禁）。"""

    status: str
    trained: bool = False
    training_task: str | None = None
    guard_status: str | None = None
    best_model: str | None = None
    model_comparison: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    overfit_gap: dict[str, Any] = Field(default_factory=dict)
    feature_importance: list[dict[str, Any]] = Field(default_factory=list)
    feature_importance_source: str | None = None
    feature_groups_used: dict[str, Any] = Field(default_factory=dict)
    missing_features: list[str] = Field(default_factory=list)
    skipped_models: list[dict[str, Any]] = Field(default_factory=list)
    degraded: bool = False
    partial_degraded: bool = False
    trainable_record_count: int | None = None
    modeled_record_count: int | None = None
    excluded_unit_inconsistent: int | None = None
    train_count: int | None = None
    val_count: int | None = None
    test_count: int | None = None
    test_used_for_training: bool = False
    data_lineage_ids: list[str] = Field(default_factory=list)
    data_usage_audit: dict[str, Any] = Field(default_factory=dict)
    model_card: dict[str, Any] = Field(default_factory=dict)
    training_log: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    created_at: str | None = None
    training_quality: dict[str, Any] = Field(default_factory=dict)


class ModelArtifactResponse(BaseModel):
    """latest/audit/feature-importance/training-log 通用响应。"""

    available: bool
    data: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class ProjectTypeTrainRequest(BaseModel):
    """T4 项目类型识别（弱监督）训练请求。"""

    training_task: str = Field(default="project_type_classification")
    project_id: int | None = Field(default=1)
    use_weak_labels: bool = Field(default=True, description="使用规则弱标签（weak_label=true）")
    use_t2_features: bool = Field(default=True, description="使用 T2/POI 组成特征")
    dry_run: bool = Field(default=False)


class ProjectTypeTrainResponse(BaseModel):
    """T4 训练结果（弱监督，含质量门禁）。"""

    status: str
    trained: bool = False
    degraded: bool = False
    training_task: str | None = None
    guard_status: str | None = None
    weak_label: bool = True
    label_source: str | None = None
    trained_models: list[str] = Field(default_factory=list)
    selected_model: str | None = None
    weak_label_accuracy_on_val: float | None = None
    agreement_rate_with_rules: float | None = None
    consistency_rate: float | None = None
    type_model_comparison: list[dict[str, Any]] = Field(default_factory=list)
    type_feature_importance: list[dict[str, Any]] = Field(default_factory=list)
    class_distribution: dict[str, Any] = Field(default_factory=dict)
    train_count: int | None = None
    val_count: int | None = None
    test_count: int | None = None
    test_used_for_training: bool = False
    pseudo_profile: bool = False
    not_real_project: bool = False
    synthetic_label: bool = False
    weak_label_audit: dict[str, Any] = Field(default_factory=dict)
    data_lineage_ids: list[str] = Field(default_factory=list)
    model_card: dict[str, Any] = Field(default_factory=dict)
    training_log: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reason: str | None = None
    pseudo_project_count: int | None = None
    created_at: str | None = None
    type_training_quality: dict[str, Any] = Field(default_factory=dict)


class ScoreCalibrateRequest(BaseModel):
    """T5 评分校准请求（接口必先跑 guard；仅 train/val，不用 test）。"""

    training_task: str = Field(default="score_calibration")
    project_id: int | None = Field(default=1)
    use_t2_features: bool = Field(default=True, description="使用 T2 POI/圈层特征")
    use_t3_housing_model: bool = Field(default=True, description="纳入 T3 区级房价水平")
    use_t4_project_type: bool = Field(default=True, description="纳入 T4 类型识别输出")
    dry_run: bool = Field(default=False, description="true 仅装配校准样本不出分")


class ScoreCalibrateResponse(BaseModel):
    """T5 评分校准结果（含质量门禁）。"""

    status: str
    trained: bool = False
    training_task: str | None = None
    guard_status: str | None = None
    score_version: str | None = None
    test_used: bool = False
    comprehensive_recomputable: bool | None = None
    score_result: dict[str, Any] = Field(default_factory=dict)
    score_contributions: list[dict[str, Any]] = Field(default_factory=list)
    calibration_card: dict[str, Any] = Field(default_factory=dict)
    calibration_report: dict[str, Any] = Field(default_factory=dict)
    weight_config: dict[str, Any] = Field(default_factory=dict)
    data_lineage_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    quality_status: str | None = None
    score_calibration_quality: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    reason: str | None = None
    created_at: str | None = None
