"""报告生成 / 报告质量门禁 响应模型（第7阶段）。

红线：仅暴露结构化章节、脱敏指标、证据ID、门禁统计量；
**不含** raw_json、原始坐标/点位、企业名、小区名、地址明细。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReportMetric(BaseModel):
    """单条章节指标（数字均来自 AnalysisResult / full analysis / Project）。"""

    key: str
    label: str
    value: Any = None
    unit: str | None = None
    evidence_id: str | None = None


class ReportSection(BaseModel):
    """报告章节（固定 9 章，每章 7 字段齐全；缺数据章节亦保留）。"""

    section_id: str
    title: str
    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    metrics: list[ReportMetric] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    data_limitations: list[str] = Field(default_factory=list)


class ReportContentResponse(BaseModel):
    """结构化报告内容 + 第7.5门禁预留字段。"""

    # ---- 报告主体 ----
    report_id: str
    project_id: int
    project_name: str | None = None
    project_type: str | None = None
    generated_at: str
    sections: list[ReportSection] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    # ---- 第7.5质量门禁预留字段 ----
    sections_count: int = 0
    required_sections_count: int = 9
    report_completeness: float = 0.0
    data_consistency: float = 0.0
    evidence_coverage: float = 0.0
    allowed_splits: list[str] = Field(default_factory=list)
    used_test: bool = False
    evidence_ids_count: int = 0
    leakage_check: dict[str, Any] = Field(default_factory=dict)
    quality_status: str = "pending"  # pass / warning / fail
    can_enter_next_stage: bool = False


class ReportMetricStatus(BaseModel):
    """单项门禁指标状态（对齐 evaluation 门禁风格）。"""

    metric_name: str
    current_value: Any = None
    threshold: str
    status: str  # pass / warning / fail / not_ready
    explanation: str


class ReportQualityResponse(BaseModel):
    """报告质量门禁结果（第7.5门禁准备）。"""

    mode: str = "eval"
    phase: str = "7.5"
    report_id: str | None = None
    project_id: int
    overall_status: str = "pass"  # pass / warning / fail
    can_enter_next_stage: bool = False
    metrics_status: list[ReportMetricStatus] = Field(default_factory=list)
    # ---- 关键指标值 ----
    report_completeness: float = 0.0
    data_consistency: float = 0.0
    evidence_coverage: float = 0.0
    leakage_check: dict[str, Any] = Field(default_factory=dict)
    test_usage_check: dict[str, Any] = Field(default_factory=dict)
    # ---- 风险 / 建议 / 下一步 ----
    hard_fail_items: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    next_required_actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class MutationTestResult(BaseModel):
    """单个 mutation test 结果（反作弊：变异后门禁必须 fail）。"""

    name: str
    description: str
    expected_fail_metric: str
    got_status: str  # 变异副本的 overall_status（期望 fail）
    triggered_metrics: list[str] = Field(default_factory=list)
    passed: bool = False  # 变异确实被判 fail 则 passed=true


class Phase75GateResponse(BaseModel):
    """第7.5阶段独立质量门禁 + 反作弊校验结果。"""

    mode: str = "eval"
    phase: str = "7.5"
    report_id: str | None = None
    project_id: int
    overall_status: str = "pass"  # pass / warning / fail
    can_enter_next_stage: bool = False
    metrics_status: list[ReportMetricStatus] = Field(default_factory=list)
    # ---- 关键指标值 ----
    report_completeness: float = 0.0
    data_consistency: float = 0.0
    evidence_coverage: float = 0.0
    number_traceability: float = 0.0
    independent_consistency_check: str = "fail"  # pass / fail
    leakage_check: dict[str, Any] = Field(default_factory=dict)
    test_usage_check: dict[str, Any] = Field(default_factory=dict)
    used_test: bool = False
    allowed_splits: list[str] = Field(default_factory=list)
    # ---- 反作弊 mutation tests ----
    mutation_tests_pass: bool = False
    mutation_tests: list[MutationTestResult] = Field(default_factory=list)
    # ---- 风险 / 建议 / 下一步 ----
    hard_fail_items: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    next_required_actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReportExportResponse(BaseModel):
    """docx 导出结果（文件落在 gitignored 的 outputs 目录）。"""

    report_id: str
    project_id: int
    file_name: str
    file_path: str
    size_bytes: int = 0
    download_url: str | None = None
    notes: list[str] = Field(default_factory=list)
