"""四维核心分析接口响应模型（第5阶段）。

红线：仅暴露统计量、分类、评分、置信度、notes、evidence_id；
**不含** raw_json、原始坐标、POI/企业/小区名称与地址明细。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BaseAnalysisResponse(BaseModel):
    """四维分析响应公共字段。"""

    project_id: int
    dimension: str
    score: float = Field(description="维度评分 0-100（可解释经验权重，非最终综合评分）")
    confidence: float = Field(description="数据充分度置信度 0-1，数据越缺越低")
    allowed_splits: list[str] = []
    include_test: bool = False
    used_test: bool = False
    center_status: str | None = None
    evidence_ids: list[str] = []
    notes: list[str] = []


# --------------------------------------------------------------------------- #
# POI / L 维度
# --------------------------------------------------------------------------- #
class PoiRingStat(BaseModel):
    ring: str
    radius_m: int
    total: int = 0
    commercial: int = 0  # 商业服务类
    public: int = 0  # 公共服务类
    convenience: int = 0  # 便民生活类
    transport: int = 0  # 交通类
    other: int = 0
    mix_index: float = 0.0  # 功能混合度（归一化香农熵 0-1）


class PoiAnalysisResponse(BaseAnalysisResponse):
    rings: list[PoiRingStat] = []
    category_top: dict[str, int] = Field(default_factory=dict, description="一级类目计数（辐射圈，脱敏分类）")
    shortboards_top5: list[str] = Field(default_factory=list, description="配套短板 top5（类别名）")
    recommend_top5: list[str] = Field(default_factory=list, description="推荐补充业态 top5（类别名）")


# --------------------------------------------------------------------------- #
# 人口 / P 维度
# --------------------------------------------------------------------------- #
class PopulationRingStat(BaseModel):
    ring: str
    radius_m: int
    grid_count: int = 0
    residential: int = 0
    worker: int = 0
    job_housing_ratio: float | None = None  # 职住比 worker/residential


class StructureSummary(BaseModel):
    """结构占比摘要（来自画像计数汇总；缺失则为空并在 notes 标注）。"""

    available: bool = False
    base_count: int = 0
    ratios: dict[str, float] = Field(default_factory=dict)


class PopulationAnalysisResponse(BaseAnalysisResponse):
    rings: list[PopulationRingStat] = []
    age_structure: StructureSummary = Field(default_factory=StructureSummary)
    consumption_structure: StructureSummary = Field(default_factory=StructureSummary)
    education_structure: StructureSummary = Field(default_factory=StructureSummary)
    car_ownership: StructureSummary = Field(default_factory=StructureSummary)
    income_structure: str = "数据缺失/不适用"
    main_segment: str | None = None  # 主力客群判断


# --------------------------------------------------------------------------- #
# 房价 / H 维度
# --------------------------------------------------------------------------- #
class HousingRingStat(BaseModel):
    ring: str
    radius_m: int
    sample_count: int = 0
    avg_unit_price: float | None = None  # 元/㎡
    median_unit_price: float | None = None
    avg_area: float | None = None  # ㎡
    room_type_dist: dict[str, int] = Field(default_factory=dict)
    year_summary: dict[str, float | None] = Field(default_factory=dict)


class HousingModelMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str  # gradient_boosting / random_forest / median_baseline
    train_count: int = 0
    val_count: int = 0
    val_mape: float | None = None
    val_mae: float | None = None
    degraded: bool = False
    note: str | None = None


class HousingBaselineInterval(BaseModel):
    low: float | None = None
    mid: float | None = None
    high: float | None = None
    unit: str = "元/㎡"


class HousingAnalysisResponse(BaseAnalysisResponse):
    model_config = ConfigDict(protected_namespaces=())

    rings: list[HousingRingStat] = []
    price_gradient: dict[str, float | None] = Field(
        default_factory=dict, description="圈层均价梯度（core/nearby/radiation）"
    )
    baseline_interval: HousingBaselineInterval = Field(default_factory=HousingBaselineInterval)
    model_metrics: HousingModelMetrics | None = None


# --------------------------------------------------------------------------- #
# 产业 / I 维度
# --------------------------------------------------------------------------- #
class IndustryRingStat(BaseModel):
    ring: str
    radius_m: int
    enterprise_count: int = 0
    density_per_km2: float | None = None
    diversity_index: float = 0.0  # 归一化香农熵（单一类目 ≈ 0）


class IndustryAnalysisResponse(BaseAnalysisResponse):
    rings: list[IndustryRingStat] = []
    category_dist: dict[str, int] = Field(default_factory=dict)
    dominant_industry: str | None = None
    adaptation_suggestions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 四维一键 / 汇总
# --------------------------------------------------------------------------- #
class FourDimensionResponse(BaseModel):
    project_id: int
    allowed_splits: list[str] = []
    include_test: bool = False
    used_test: bool = False
    scores: dict[str, float] = Field(default_factory=dict, description="P/H/L/I 维度分")
    confidence: dict[str, float] = Field(default_factory=dict)
    poi: PoiAnalysisResponse | None = None
    population: PopulationAnalysisResponse | None = None
    housing: HousingAnalysisResponse | None = None
    industry: IndustryAnalysisResponse | None = None
    evidence_ids: list[str] = []
    notes: list[str] = []


class DimensionSummary(BaseModel):
    dimension: str
    score: float | None = None
    metric_count: int = 0
    evidence_count: int = 0


class AnalysisSummaryResponse(BaseModel):
    project_id: int
    dimensions: list[DimensionSummary] = []
    scores: dict[str, float] = Field(default_factory=dict)
    total_metrics: int = 0
    total_evidence: int = 0
    notes: list[str] = []


# --------------------------------------------------------------------------- #
# 第6阶段：类型识别 / 综合评分 / 策略 / 一键完整分析
# 红线：仅暴露枚举/分数/权重/规则名/脱敏短语/evidence_id；不含 raw_json/原始明细。
# --------------------------------------------------------------------------- #
class MatchedRule(BaseModel):
    rule: str
    weight: float
    detail: str = Field(description="命中依据（脱敏短语，含词典词/阈值，不含原文整段）")


class TypeCandidate(BaseModel):
    project_type: str
    raw_score: float
    matched_rule_count: int = 0


class ProjectTypeResponse(BaseModel):
    project_id: int
    project_type: str
    confidence: float = Field(description="类型识别置信度 0-1（含数据完整度与四维置信度修正）")
    matched_rules: list[MatchedRule] = []
    reason: str = ""
    candidates: list[TypeCandidate] = []
    data_sufficiency: float = 0.0
    missing_fields: list[str] = []
    allowed_splits: list[str] = []
    include_test: bool = False
    used_test: bool = False
    evidence_ids: list[str] = []
    notes: list[str] = []


class ScoreContribution(BaseModel):
    dimension: str  # P / H / L / I
    score_key: str  # P_score / H_score / L_score / I_score
    label: str
    score: float
    weight: float
    contribution: float
    confidence: float | None = None


class ScoreResponse(BaseModel):
    project_id: int
    project_type: str
    scores: dict[str, float] = Field(default_factory=dict, description="P/H/L/I 原始分")
    weights: dict[str, float] = Field(default_factory=dict, description="按类型切换的四维权重")
    contributions: list[ScoreContribution] = []
    F_score: float
    score_level: str = Field(description="高 / 中高 / 中 / 中低 / 低")
    explanation: str = ""
    allowed_splits: list[str] = []
    include_test: bool = False
    used_test: bool = False
    evidence_ids: list[str] = []
    notes: list[str] = []


class StrategyResponse(BaseModel):
    project_id: int
    project_type: str
    update_positioning: str = ""
    key_opportunities: list[str] = []
    key_risks: list[str] = []
    recommended_directions: list[str] = []
    priority_actions: list[str] = []
    data_limitations: list[str] = []
    strategy_count: int = 0
    allowed_splits: list[str] = []
    include_test: bool = False
    used_test: bool = False
    evidence_ids: list[str] = []
    notes: list[str] = []


class FullAnalysisResponse(BaseModel):
    """一键完整分析结果（含第6.5 门禁可扫描的扁平字段 + 各阶段完整结果）。"""

    project_id: int
    # ---- 扁平门禁字段 ----
    project_type: str
    project_type_confidence: float
    matched_rules: list[MatchedRule] = []
    F_score: float
    scores: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    score_level: str
    strategy_count: int = 0
    allowed_splits: list[str] = []
    include_test: bool = False
    used_test: bool = False
    evidence_ids: list[str] = []
    # ---- 各阶段完整结果 ----
    four_dimension: FourDimensionResponse
    project_type_result: ProjectTypeResponse
    score_result: ScoreResponse
    strategy_result: StrategyResponse
    notes: list[str] = []


class FullSummaryResponse(BaseModel):
    """已落库的完整分析汇总（不重算）。"""

    project_id: int
    project_type: str | None = None
    project_type_confidence: float | None = None
    matched_rules: list[str] = []
    F_score: float | None = None
    score_level: str | None = None
    weights: dict[str, float] | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    strategy_count: int | None = None
    dimensions: list[DimensionSummary] = []
    total_metrics: int = 0
    total_evidence: int = 0
    has_full_analysis: bool = False
    notes: list[str] = []
