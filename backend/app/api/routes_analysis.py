"""四维核心分析接口（第5阶段）。

POST /api/analysis/{project_id}/poi                 区位配套（L）
POST /api/analysis/{project_id}/population           人口画像（P）
POST /api/analysis/{project_id}/housing              房价价值（H，含基线模型）
POST /api/analysis/{project_id}/industry             产业经济（I）
POST /api/analysis/{project_id}/run-four-dimensions  一键四维
GET  /api/analysis/{project_id}/summary              读取已落库汇总

第6阶段新增（决策层）：
POST /api/analysis/{project_id}/project-type         项目类型识别（规则+指标）
POST /api/analysis/{project_id}/score                综合评分 F_score
POST /api/analysis/{project_id}/strategy             结构化策略建议
POST /api/analysis/{project_id}/run-full             一键完整分析流水线
GET  /api/analysis/{project_id}/full-summary         读取已落库完整汇总

红线：默认排除 test（include_test 默认 false）；单步接口缺四维结果时自动补跑四维（仍默认
不含 test）；不调用外部 API、不使用大模型打分；仅返回统计量/分类/评分/权重/置信度/notes/
evidence_id，不返回 raw_json/原始明细/企业名/小区名/地址。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.analysis import (
    AnalysisSummaryResponse,
    FourDimensionResponse,
    FullAnalysisResponse,
    FullSummaryResponse,
    HousingAnalysisResponse,
    IndustryAnalysisResponse,
    PoiAnalysisResponse,
    PopulationAnalysisResponse,
    ProjectTypeResponse,
    ScoreResponse,
    StrategyResponse,
)
from app.services import (
    analysis_orchestrator,
    housing_analysis_service,
    industry_analysis_service,
    poi_analysis_service,
    population_analysis_service,
    project_service,
    spatial_service,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

_INCLUDE_TEST_DESC = "是否纳入 test（默认 false；仅评估场景使用）"


def _get_project_or_404(db: Session, project_id: int):
    project = project_service.get_project(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"项目 {project_id} 不存在")
    return project


@router.post("/{project_id}/poi", response_model=PoiAnalysisResponse)
def analyze_poi(
    project_id: int,
    include_test: bool = Query(default=False, description=_INCLUDE_TEST_DESC),
    db: Session = Depends(get_db),
) -> PoiAnalysisResponse:
    project = _get_project_or_404(db, project_id)
    try:
        return PoiAnalysisResponse(**poi_analysis_service.analyze(db, project, include_test))
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{project_id}/population", response_model=PopulationAnalysisResponse)
def analyze_population(
    project_id: int,
    include_test: bool = Query(default=False, description=_INCLUDE_TEST_DESC),
    db: Session = Depends(get_db),
) -> PopulationAnalysisResponse:
    project = _get_project_or_404(db, project_id)
    try:
        return PopulationAnalysisResponse(
            **population_analysis_service.analyze(db, project, include_test)
        )
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{project_id}/housing", response_model=HousingAnalysisResponse)
def analyze_housing(
    project_id: int,
    include_test: bool = Query(default=False, description=_INCLUDE_TEST_DESC),
    db: Session = Depends(get_db),
) -> HousingAnalysisResponse:
    project = _get_project_or_404(db, project_id)
    try:
        return HousingAnalysisResponse(
            **housing_analysis_service.analyze(db, project, include_test)
        )
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{project_id}/industry", response_model=IndustryAnalysisResponse)
def analyze_industry(
    project_id: int,
    include_test: bool = Query(default=False, description=_INCLUDE_TEST_DESC),
    db: Session = Depends(get_db),
) -> IndustryAnalysisResponse:
    project = _get_project_or_404(db, project_id)
    try:
        return IndustryAnalysisResponse(
            **industry_analysis_service.analyze(db, project, include_test)
        )
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{project_id}/run-four-dimensions", response_model=FourDimensionResponse)
def run_four_dimensions(
    project_id: int,
    include_test: bool = Query(default=False, description=_INCLUDE_TEST_DESC),
    db: Session = Depends(get_db),
) -> FourDimensionResponse:
    project = _get_project_or_404(db, project_id)
    try:
        return FourDimensionResponse(
            **analysis_orchestrator.run_four_dimension_analysis(db, project, include_test)
        )
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}/summary", response_model=AnalysisSummaryResponse)
def get_summary(project_id: int, db: Session = Depends(get_db)) -> AnalysisSummaryResponse:
    project = _get_project_or_404(db, project_id)
    return AnalysisSummaryResponse(**analysis_orchestrator.get_summary(db, project))


# --------------------------------------------------------------------------- #
# 第6阶段：类型识别 / 综合评分 / 策略 / 一键完整分析
# --------------------------------------------------------------------------- #
@router.post("/{project_id}/project-type", response_model=ProjectTypeResponse)
def identify_project_type(
    project_id: int,
    include_test: bool = Query(default=False, description=_INCLUDE_TEST_DESC),
    db: Session = Depends(get_db),
) -> ProjectTypeResponse:
    project = _get_project_or_404(db, project_id)
    try:
        return ProjectTypeResponse(
            **analysis_orchestrator.run_project_type(db, project, include_test)
        )
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{project_id}/score", response_model=ScoreResponse)
def compute_score(
    project_id: int,
    include_test: bool = Query(default=False, description=_INCLUDE_TEST_DESC),
    db: Session = Depends(get_db),
) -> ScoreResponse:
    project = _get_project_or_404(db, project_id)
    try:
        return ScoreResponse(**analysis_orchestrator.run_score(db, project, include_test))
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{project_id}/strategy", response_model=StrategyResponse)
def build_strategy(
    project_id: int,
    include_test: bool = Query(default=False, description=_INCLUDE_TEST_DESC),
    db: Session = Depends(get_db),
) -> StrategyResponse:
    project = _get_project_or_404(db, project_id)
    try:
        return StrategyResponse(**analysis_orchestrator.run_strategy(db, project, include_test))
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{project_id}/run-full", response_model=FullAnalysisResponse)
def run_full(
    project_id: int,
    include_test: bool = Query(default=False, description=_INCLUDE_TEST_DESC),
    db: Session = Depends(get_db),
) -> FullAnalysisResponse:
    project = _get_project_or_404(db, project_id)
    try:
        return FullAnalysisResponse(
            **analysis_orchestrator.run_full_analysis(db, project, include_test)
        )
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}/full-summary", response_model=FullSummaryResponse)
def get_full_summary(project_id: int, db: Session = Depends(get_db)) -> FullSummaryResponse:
    project = _get_project_or_404(db, project_id)
    return FullSummaryResponse(**analysis_orchestrator.get_full_summary(db, project))
