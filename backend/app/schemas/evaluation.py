"""阶段性评估基线响应模型（第1-5阶段轻量评估）。

红线：仅统计量与阶段状态，不含原文/原始明细。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetricStatus(BaseModel):
    """单项门禁指标状态。"""

    metric_name: str
    current_value: Any = None
    threshold: str
    status: str  # pass / warning / fail / not_ready
    explanation: str


class StageBaselineResponse(BaseModel):
    mode: str
    phase: str
    # ---- 质量门禁汇总 ----
    overall_status: str = "pass"  # pass / warning / fail
    can_enter_next_stage: bool = True
    metrics_status: list[MetricStatus] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    next_required_actions: list[str] = Field(default_factory=list)
    # ---- 统计明细 ----
    data_import_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, Any] = Field(default_factory=dict)
    rag_chunks: dict[str, Any] = Field(default_factory=dict)
    evidence_chain_count: int = 0
    analysis: dict[str, Any] = Field(default_factory=dict)
    housing_model: dict[str, Any] = Field(default_factory=dict)
    default_allowed_splits: list[str] = Field(default_factory=list)
    default_allowed_is_train_val: bool = True
    desensitization_check: dict[str, Any] = Field(default_factory=dict)
    pending_metrics: list[dict[str, str]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 第6.5阶段质量门禁（类型识别 / 综合评分 / 策略建议）
# 红线：仅统计量/分数/枚举/规则名/脱敏短语；不含原文/raw_json/原始明细。
# --------------------------------------------------------------------------- #
class Phase6CoreResults(BaseModel):
    project_type: str | None = None
    project_type_confidence: float | None = None
    matched_rules_count: int = 0
    P_score: float | None = None
    H_score: float | None = None
    L_score: float | None = None
    I_score: float | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    F_score: float | None = None
    score_level: str | None = None
    strategy_count: int = 0
    evidence_ids_count: int = 0
    allowed_splits: list[str] = Field(default_factory=list)
    used_test: bool = False


class ModelAuditConclusions(BaseModel):
    """房价模型训练审计结论（布尔判定）。"""

    training_uses_only_train: bool = False
    validation_uses_only_val: bool = False
    test_used_in_training: bool = True
    metrics_recomputed: bool = False
    metrics_match_saved: bool = False


class ModelAuditResponse(BaseModel):
    """第5阶段房价基线模型训练审计（只读，仅 train/val 训练；test 仅计数）。

    红线：仅统计量 / 指标 / 脱敏 hash / 结论；不含原始房源/小区/地址/坐标/raw_json。
    """

    mode: str = "eval"
    phase: str = "5-audit"
    overall_status: str = "pass"  # pass / warning / fail
    can_trust_val_metrics: bool = False
    metrics_status: list[MetricStatus] = Field(default_factory=list)
    # ---- split 分布（仅计数）----
    split_counts: dict[str, int] = Field(default_factory=dict)
    # ---- 独立重算 vs 已保存 ----
    model_recomputed: dict[str, Any] = Field(default_factory=dict)
    saved_metrics: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)
    # ---- 脱敏 hash（仅 id 集合指纹，不含明细）----
    hashes: dict[str, Any] = Field(default_factory=dict)
    # ---- 结论 ----
    conclusions: ModelAuditConclusions = Field(default_factory=ModelAuditConclusions)
    hard_fail_items: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    next_required_actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Phase6GateResponse(BaseModel):
    mode: str = "eval"
    phase: str = "6.5"
    target_project_id: int | None = None
    overall_status: str = "pass"  # pass / warning / fail
    can_enter_next_stage: bool = False
    metrics_status: list[MetricStatus] = Field(default_factory=list)
    core_results: Phase6CoreResults = Field(default_factory=Phase6CoreResults)
    not_ready_metrics: list[dict[str, str]] = Field(default_factory=list)
    hard_fail_items: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    next_required_actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 第9阶段：最终自评（final-eval）与交付材料导出（export-delivery）
# 红线：仅汇总指标/门禁结果/结论/风险；test 仅用于最终评估；不含原文/raw_json/原始明细。
# --------------------------------------------------------------------------- #
class FinalEvalResponse(BaseModel):
    """第9阶段最终自评结构（字段为脱敏汇总，不含原文/原始明细）。"""

    mode: str = "eval"
    phase: str = "9"
    overall_status: str = "fail"  # pass / warning / fail
    can_submit: bool = False
    core_metrics: dict[str, Any] = Field(default_factory=dict)
    extended_metrics: dict[str, Any] = Field(default_factory=dict)
    test_isolation_check: dict[str, Any] = Field(default_factory=dict)
    model_test_metrics: dict[str, Any] = Field(default_factory=dict)
    report_quality_metrics: dict[str, Any] = Field(default_factory=dict)
    retrieval_quality_metrics: dict[str, Any] = Field(default_factory=dict)
    delivery_checklist: list[dict[str, Any]] = Field(default_factory=list)
    delivery_checklist_complete: bool = False
    manual_pending_items: list[str] = Field(default_factory=list)
    external_api_calls: int = 0
    llm_used_for_scoring: bool = False
    report_export_success: bool = False
    frontend_structure_check: str = "fail"
    frontend_structure_detail: dict[str, Any] = Field(default_factory=dict)
    frontend_demo_status_runtime: str = "not_verified_runtime"
    leakage_check: dict[str, Any] = Field(default_factory=dict)
    final_pass_thresholds: dict[str, Any] = Field(default_factory=dict)
    blocking_fail: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    final_summary: str = ""
    test_policy: str = ""
    notes: list[str] = Field(default_factory=list)


class Phase105GateResponse(BaseModel):
    """第10.5 数据覆盖率与特征质量门禁响应（只读，仅统计量/门禁结论，无原文/原始明细）。"""

    mode: str = "eval"
    phase: str = "10.5"
    target_project_id: int | None = None
    overall_status: str = "fail"  # pass / warning / fail
    can_enter_next_stage: bool = False
    metrics_status: list[MetricStatus] = Field(default_factory=list)
    data_audit_gate: dict[str, Any] = Field(default_factory=dict)
    feature_quality_gate: dict[str, Any] = Field(default_factory=dict)
    training_usage_gate: dict[str, Any] = Field(default_factory=dict)
    external_data_gate: dict[str, Any] = Field(default_factory=dict)
    leakage_gate: dict[str, Any] = Field(default_factory=dict)
    gitignore_gate: dict[str, Any] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    next_required_actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Phase10b5GateResponse(BaseModel):
    """第10B.5 外部数据增强门禁响应（只读，仅统计量/门禁结论）。"""

    mode: str = "eval"
    phase: str = "10B.5"
    overall_status: str = "fail"
    can_enter_next_stage: bool = False
    merged_dedup_total: int = 0
    amap_volume_gate: dict[str, Any] = Field(default_factory=dict)
    category_coverage_gate: dict[str, Any] = Field(default_factory=dict)
    data_asset_gate: dict[str, Any] = Field(default_factory=dict)
    compliance_gate: dict[str, Any] = Field(default_factory=dict)
    git_safety_gate: dict[str, Any] = Field(default_factory=dict)
    non_amap_gap_gate: dict[str, Any] = Field(default_factory=dict)
    pass_items: list[str] = Field(default_factory=list)
    warning_items: list[str] = Field(default_factory=list)
    fail_items: list[str] = Field(default_factory=list)
    recommend_commit: bool = False
    recommended_commit_message: str | None = None
    used_for_training: bool = False
    test_contamination_risk: bool = False
    leakage_risk: bool = False
    notes: list[str] = Field(default_factory=list)


class DataAuditResponse(BaseModel):
    """第10A 全量数据资产审计响应（脱敏：仅统计量与结论，无原文/原始明细）。"""

    mode: str = "eval"
    phase: str = "10A"
    created_at: str = ""
    overall_status: str = "pass"  # pass / warning / fail
    all_files_count: int = 0
    total_raw_records: int = 0
    total_parsed_records: int = 0
    total_db_records: int = 0
    total_used_records: int = 0
    coverage_rate: float = 0.0
    split_built: bool = False
    files: list[dict[str, Any]] = Field(default_factory=list)
    unused_files: list[str] = Field(default_factory=list)
    unused_media_count: int = 0
    low_coverage_files: list[str] = Field(default_factory=list)
    skipped_summary: dict[str, Any] = Field(default_factory=dict)
    leakage_risk: bool = False
    test_contamination_risk: bool = False
    external_data: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    exports: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class EvalDataCatalogResponse(BaseModel):
    """第10B 数据目录（内部审计摘要 + 外部数据目录，脱敏）。"""

    mode: str = "eval"
    phase: str = "10B"
    created_at: str = ""
    internal: dict[str, Any] = Field(default_factory=dict)
    external: dict[str, Any] = Field(default_factory=dict)
    exports: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class EvalDataLineageResponse(BaseModel):
    """第10B 全量数据血缘（内部 + 外部，回答血缘 13 问，脱敏）。"""

    mode: str = "eval"
    phase: str = "10B"
    created_at: str = ""
    schema_fields: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    answers: dict[str, Any] = Field(default_factory=dict)
    exports: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class DeliveryExportResponse(BaseModel):
    """交付材料导出结果（仅文件清单/大小/路径与门禁结论，不含涉密内容）。"""

    mode: str = "eval"
    phase: str = "9"
    export_success: bool = False
    output_dir: str = ""
    gitignore_covered: bool = True
    files: list[dict[str, Any]] = Field(default_factory=list)
    skipped_files: list[dict[str, Any]] = Field(default_factory=list)
    leakage_check: dict[str, Any] = Field(default_factory=dict)
    overall_status: str = "fail"
    can_submit: bool = False
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 第11 T6：知识检索匹配准确率评测
# --------------------------------------------------------------------------- #
class RetrievalBuildBenchmarkRequest(BaseModel):
    """构建检索评测集请求（仅 train/val，登记冻结 test manifest）。"""

    splits: list[str] = Field(default_factory=lambda: ["train", "val"])
    include_test_manifest: bool = Field(default=True, description="登记并冻结 test（不用于调参）")
    use_test: bool = Field(default=False, description="恒 false：评测集禁止含 test")


class RetrievalRunRequest(BaseModel):
    """运行检索评测请求（默认 val + 默认策略；tune_mode 下禁止 test）。"""

    split: str = Field(default="val", description="train/val；test 仅 tune_mode=false 才允许")
    strategy: str = Field(default="hybrid_plus_rerank")
    top_k: int = Field(default=5, ge=1, le=50)
    use_test: bool = Field(default=False)
    tune_mode: bool = Field(default=True, description="true 时禁止 split=test")


class RetrievalEvalResponse(BaseModel):
    """检索评测响应（指标 + 策略对比 + 门禁 + metric_card）。"""

    status: str
    available: bool = True
    version: str | None = None
    split: str | None = None
    requested_strategy: str | None = None
    best_strategy: str | None = None
    top_k: int | None = None
    tune_mode: bool | None = None
    use_test: bool = False
    test_used_for_tuning: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)
    strategy_comparison: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_quality_status: str | None = None
    retrieval_quality: dict[str, Any] = Field(default_factory=dict)
    metric_card: dict[str, Any] = Field(default_factory=dict)
    failed_case_count: int | None = None
    message: str | None = None
    created_at: str | None = None


class RetrievalBenchmarkResponse(BaseModel):
    """构建评测集响应（仅统计量，不含题目原文明细）。"""

    status: str
    available: bool = True
    version: str | None = None
    train_sample_count: int = 0
    val_sample_count: int = 0
    total_sample_count: int | None = None
    test_manifest_count: int | None = None
    topics_covered: dict[str, Any] = Field(default_factory=dict)
    doc_types_covered: dict[str, Any] = Field(default_factory=dict)
    difficulty_distribution: dict[str, Any] = Field(default_factory=dict)
    used_test_for_tuning: bool = False
    use_test: bool = False
    contamination_check: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    created_at: str | None = None


class RetrievalArtifactResponse(BaseModel):
    """latest / fail-cases / metric-card 通用响应。"""

    available: bool
    data: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


# --------------------------------------------------------------------------- #
# 第11 T7：报告结构完整率门禁
# --------------------------------------------------------------------------- #
class ReportStructureRunRequest(BaseModel):
    """运行报告结构完整率评测请求（默认 use_test=false；允许自动生成报告）。"""

    project_id: int = Field(default=1)
    report_id: str | None = Field(default=None)
    use_latest_report: bool = Field(default=True)
    generate_if_missing: bool = Field(default=True, description="无报告时自动生成一次（仅 train/val）")
    use_test: bool = Field(default=False, description="恒 false：结构门禁禁止使用 test")


class ReportStructureEvalResponse(BaseModel):
    """报告结构完整率评测响应（八类完整率 + checklist + 门禁）。"""

    status: str
    available: bool = True
    version: str | None = None
    project_id: int | None = None
    report_id: str | None = None
    report_generated_now: bool | None = None
    test_used: bool = False
    rates: dict[str, Any] = Field(default_factory=dict)
    checklist_summary: dict[str, Any] = Field(default_factory=dict)
    section_check: list[dict[str, Any]] = Field(default_factory=list)
    table_check: list[dict[str, Any]] = Field(default_factory=list)
    placeholder_check: list[dict[str, Any]] = Field(default_factory=list)
    section_lineage: dict[str, Any] = Field(default_factory=dict)
    failed_items: list[dict[str, Any]] = Field(default_factory=list)
    repair_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    metric_card: dict[str, Any] = Field(default_factory=dict)
    report_structure_quality_status: str | None = None
    report_structure_quality: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    created_at: str | None = None


class ReportStructureArtifactResponse(BaseModel):
    """latest / fail-items / metric-card 通用响应。"""

    available: bool
    data: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


# --------------------------------------------------------------------------- #
# 第11 T8：生成内容与底层数据一致性门禁
# --------------------------------------------------------------------------- #
class ReportConsistencyRunRequest(BaseModel):
    """运行报告内容一致性评测请求（默认 use_test=false；允许自动生成报告）。"""

    project_id: int = Field(default=1)
    report_id: str | None = Field(default=None)
    use_latest_report: bool = Field(default=True)
    generate_if_missing: bool = Field(default=True, description="无报告时自动生成一次（仅 train/val）")
    use_test: bool = Field(default=False, description="恒 false：一致性门禁禁止使用 test")


class ReportConsistencyEvalResponse(BaseModel):
    """报告内容一致性评测响应（四率 + 加权 overall + claims + 门禁）。"""

    status: str
    available: bool = True
    version: str | None = None
    project_id: int | None = None
    report_id: str | None = None
    report_generated_now: bool | None = None
    test_used: bool = False
    rates: dict[str, Any] = Field(default_factory=dict)
    claims_summary: dict[str, Any] = Field(default_factory=dict)
    numeric_check: dict[str, Any] = Field(default_factory=dict)
    conclusion_check: list[dict[str, Any]] = Field(default_factory=list)
    evidence_check: dict[str, Any] = Field(default_factory=dict)
    limitation_check: list[dict[str, Any]] = Field(default_factory=list)
    inconsistent_items: list[dict[str, Any]] = Field(default_factory=list)
    repair_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    metric_card: dict[str, Any] = Field(default_factory=dict)
    report_consistency_quality_status: str | None = None
    report_consistency_quality: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    created_at: str | None = None


class ReportConsistencyArtifactResponse(BaseModel):
    """latest / inconsistent-items / metric-card 通用响应。"""

    available: bool
    data: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


# --------------------------------------------------------------------------- #
# 第11.5：总门禁与三大硬指标自评包
# --------------------------------------------------------------------------- #
class Phase115GateResponse(BaseModel):
    """第11.5 总门禁响应。"""

    mode: str | None = None
    phase: str | None = None
    version: str | None = None
    project_id: int | None = None
    overall_status: str
    can_enter_phase12: bool
    three_hard_metrics_status: dict[str, Any] = Field(default_factory=dict)
    model_metrics_status: dict[str, Any] = Field(default_factory=dict)
    safety_status: dict[str, Any] = Field(default_factory=dict)
    lineage_status: dict[str, Any] = Field(default_factory=dict)
    final_test_status: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_before_submission: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    eval_card: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    created_at: str | None = None


class Phase115ArtifactResponse(BaseModel):
    """eval-card / risk-summary / final-test-status 通用响应。"""

    available: bool
    data: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


# --------------------------------------------------------------------------- #
# 第11.6：final 10% test 最终评估
# --------------------------------------------------------------------------- #
class FinalTestRunRequest(BaseModel):
    """final 10% test 运行请求（只读冻结；禁止调参）。"""

    use_frozen_test_manifest: bool = Field(default=True)
    eval_mode: bool = Field(default=True)
    allow_tuning: bool = Field(default=False, description="恒 false：final test 禁止调参")
    write_results: bool = Field(default=True)


class FinalTestEvalResponse(BaseModel):
    """final 10% test 评估响应（final 三大硬指标 + 阶段对比 + test 隔离）。"""

    status: str
    available: bool = True
    version: str | None = None
    mode: str | None = None
    eval_mode: bool | None = None
    final_eval_mode: bool | None = None
    use_frozen_test_manifest: bool | None = None
    allow_tuning: bool | None = None
    final_test_used_for_tuning: bool | None = None
    results_backflow_to_training_or_tuning: bool | None = None
    final_test_manifest_exists: bool | None = None
    final_test_manifest_id: Any | None = None
    final_test_sample_count: Any | None = None
    test_sample_count_note: str | None = None
    split_info: dict[str, Any] = Field(default_factory=dict)
    three_hard_metrics: dict[str, Any] = Field(default_factory=dict)
    train_val_vs_final_comparison: dict[str, Any] = Field(default_factory=dict)
    test_isolation_check: dict[str, Any] = Field(default_factory=dict)
    housing_test_metrics: dict[str, Any] = Field(default_factory=dict)
    fail_cases: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    underlying_final_eval_overall: str | None = None
    underlying_final_eval_can_submit: bool | None = None
    required_action_if_fail: str | None = None
    notes: list[str] = Field(default_factory=list)
    message: str | None = None
    created_at: str | None = None


class FinalTestArtifactResponse(BaseModel):
    """latest / metric-card / fail-cases 通用响应。"""

    available: bool
    data: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
