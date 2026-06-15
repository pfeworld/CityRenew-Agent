"""阶段性评估接口。

GET /api/evaluation/stage-baseline  汇总第1-5阶段质量指标（只读，仅 train/val）。
GET /api/evaluation/phase6-gate     第6.5阶段质量门禁（类型/评分/策略，只读，仅 train/val）。

红线：不读取 test 内容、不调用外部 API、不生成报告、不返回原文/raw_json/原始明细。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.evaluation import (
    DataAuditResponse,
    DeliveryExportResponse,
    EvalDataCatalogResponse,
    EvalDataLineageResponse,
    FinalEvalResponse,
    ModelAuditResponse,
    Phase6GateResponse,
    FinalTestArtifactResponse,
    FinalTestEvalResponse,
    FinalTestRunRequest,
    Phase105GateResponse,
    Phase10b5GateResponse,
    Phase115ArtifactResponse,
    Phase115GateResponse,
    ReportConsistencyArtifactResponse,
    ReportConsistencyEvalResponse,
    ReportConsistencyRunRequest,
    ReportStructureArtifactResponse,
    ReportStructureEvalResponse,
    ReportStructureRunRequest,
    RetrievalArtifactResponse,
    RetrievalBenchmarkResponse,
    RetrievalBuildBenchmarkRequest,
    RetrievalEvalResponse,
    RetrievalRunRequest,
    StageBaselineResponse,
)
from app.services import (
    data_audit_service,
    data_lineage_service,
    delivery_export_service,
    external_data_collector_service,
    final_eval_service,
    final_test_eval_service,
    model_audit_service,
    phase6_eval_service,
    phase105_gate_service,
    phase10b5_gate_service,
    phase115_gate_service,
    report_consistency_eval_service,
    report_structure_eval_service,
    retrieval_eval_service,
    stage_eval_service,
)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/stage-baseline", response_model=StageBaselineResponse)
def stage_baseline(db: Session = Depends(get_db)) -> StageBaselineResponse:
    return StageBaselineResponse(**stage_eval_service.get_stage_baseline(db))


@router.get("/phase6-gate", response_model=Phase6GateResponse)
def phase6_gate(db: Session = Depends(get_db)) -> Phase6GateResponse:
    return Phase6GateResponse(**phase6_eval_service.run_phase6_gate(db))


@router.get("/phase10-5-gate", response_model=Phase105GateResponse)
def phase10_5_gate(db: Session = Depends(get_db)) -> Phase105GateResponse:
    """第10.5 数据覆盖率与特征质量门禁：复用第10A 的 data-audit 与 feature-engineering，
    校验覆盖率/特征质量/训练使用/外部数据/泄露/gitignore 七大门禁（只读，仅 train/val）。

    红线：不调用外部 API；不采集外部数据；test 永不参与；不生成未被 gitignore 覆盖的数据文件；
    输出仅含统计量与脱敏结论，不含原文/raw_json/原始明细。
    """
    return Phase105GateResponse(**phase105_gate_service.run_phase105_gate(db))


@router.get("/phase10b-5-gate", response_model=Phase10b5GateResponse)
def phase10b_5_gate(db: Session = Depends(get_db)) -> Phase10b5GateResponse:
    """第10B.5 外部数据增强门禁：高德数据量/类别覆盖/数据资产/合规/git 安全/非高德缺口（只读）。

    红线：只读本地 manifest/store/processed/.gitignore；不调用外部 API、不采集数据。
    """
    return Phase10b5GateResponse(**phase10b5_gate_service.run_phase10b5_gate(db))


@router.get("/data-audit", response_model=DataAuditResponse)
def data_audit(db: Session = Depends(get_db)) -> DataAuditResponse:
    """第10A 全量数据资产审计：原始/解析/入库/特征/训练使用追踪 + test 隔离/泄露检查（只读统计）。

    红线：仅 train/val 计入特征工程/训练；test 永不计入；不返回原文/raw_json/原始明细；
    导出落 backend/data/outputs/data_catalog/（已 gitignore）。
    """
    return DataAuditResponse(**data_audit_service.run_data_audit(db))


@router.get("/data-catalog", response_model=EvalDataCatalogResponse)
def data_catalog(db: Session = Depends(get_db)) -> EvalDataCatalogResponse:
    """第10B 数据目录：内部审计摘要 + 外部数据目录 + 报告导出（只读，脱敏）。

    回答：系统用了哪些数据、来自哪里、是否合法、多少条、进入哪些环节、是否 test 污染/泄露。
    导出落 backend/data/outputs/data_catalog/（已 gitignore）。
    """
    return EvalDataCatalogResponse(**external_data_collector_service.build_data_catalog(db))


@router.get("/data-lineage", response_model=EvalDataLineageResponse)
def data_lineage(db: Session = Depends(get_db)) -> EvalDataLineageResponse:
    """第10B 全量数据血缘：内部（审计派生）+ 外部（采集登记），回答血缘 13 问（只读，脱敏）。

    红线：外部数据物理隔离于 competition test；test 仅最终评估；不含原文/原始明细。
    """
    return EvalDataLineageResponse(**data_lineage_service.build_lineage(db, export=True))


@router.get("/model-audit", response_model=ModelAuditResponse)
def model_audit(db: Session = Depends(get_db)) -> ModelAuditResponse:
    """第5阶段房价模型训练审计：仅 train/val 重算 + test 隔离 + 指标可复现验证（只读）。"""
    return ModelAuditResponse(**model_audit_service.run_model_audit(db))


@router.get("/final-eval", response_model=FinalEvalResponse)
def final_eval(db: Session = Depends(get_db)) -> FinalEvalResponse:
    """第9阶段最终自评：聚合三大核心指标 + 扩展指标 + test 隔离/泄露检查 + 交付清单。

    红线：test 仅在 housing_test_mape 处用于最终评估；不重训/不改规则/不调参；
    不调外部 API；不使用大模型生成结论；不返回原文/raw_json/原始明细。
    """
    return FinalEvalResponse(**final_eval_service.run_final_eval(db))


@router.post("/export-delivery", response_model=DeliveryExportResponse)
def export_delivery(db: Session = Depends(get_db)) -> DeliveryExportResponse:
    """生成最终自评并导出交付材料到 backend/data/outputs/final_eval（已 gitignore）。"""
    return DeliveryExportResponse(**delivery_export_service.export_delivery(db))


# --------------------------------------------------------------------------- #
# 第11 T6：知识检索匹配准确率评测（仅 train/val 调优；test 冻结不参与）
# --------------------------------------------------------------------------- #
@router.post("/retrieval/build-benchmark", response_model=RetrievalBenchmarkResponse)
def retrieval_build_benchmark(payload: RetrievalBuildBenchmarkRequest) -> RetrievalBenchmarkResponse:
    """构建检索评测集（仅 train/val 文档 RAG chunks 自检索题），并登记冻结 test manifest。"""
    return RetrievalBenchmarkResponse(**retrieval_eval_service.build_benchmark(
        splits=payload.splits, include_test_manifest=payload.include_test_manifest,
        use_test=False))


@router.post("/retrieval/run", response_model=RetrievalEvalResponse)
def retrieval_run(payload: RetrievalRunRequest) -> RetrievalEvalResponse:
    """运行检索评测：策略对比 + 指标计算 + 门禁；tune_mode 下禁止 test，默认不跑 test。"""
    return RetrievalEvalResponse(**retrieval_eval_service.run_eval(
        split=payload.split, strategy=payload.strategy, top_k=payload.top_k,
        use_test=payload.use_test, tune_mode=payload.tune_mode))


@router.get("/retrieval/latest", response_model=RetrievalArtifactResponse)
def retrieval_latest() -> RetrievalArtifactResponse:
    """最近一次检索评测结果。"""
    data = retrieval_eval_service.get_latest()
    if data is None:
        return RetrievalArtifactResponse(available=False,
                                         message="尚无评测，请先 POST /api/evaluation/retrieval/run")
    return RetrievalArtifactResponse(available=True, data=data)


@router.get("/retrieval/fail-cases", response_model=RetrievalArtifactResponse)
def retrieval_fail_cases() -> RetrievalArtifactResponse:
    """最近一次评测失败用例。"""
    data = retrieval_eval_service.get_fail_cases()
    if data is None:
        return RetrievalArtifactResponse(available=False, message="尚无失败用例记录")
    return RetrievalArtifactResponse(available=True, data=data)


@router.get("/retrieval/metric-card", response_model=RetrievalArtifactResponse)
def retrieval_metric_card() -> RetrievalArtifactResponse:
    """检索指标卡（口径/权重/通过线/best_strategy）。"""
    data = retrieval_eval_service.get_metric_card()
    if data is None:
        return RetrievalArtifactResponse(available=False, message="尚无 metric_card")
    return RetrievalArtifactResponse(available=True, data=data)


# --------------------------------------------------------------------------- #
# 第11 T7：报告结构完整率门禁（仅 train/val 报告；禁止 test）
# --------------------------------------------------------------------------- #
@router.post("/report-structure/run", response_model=ReportStructureEvalResponse)
def report_structure_run(payload: ReportStructureRunRequest,
                         db: Session = Depends(get_db)) -> ReportStructureEvalResponse:
    """运行报告结构完整率评测：9 章 + 必备表格/指标/证据/血缘/占位检查 + 门禁（禁止 test）。"""
    return ReportStructureEvalResponse(**report_structure_eval_service.evaluate_report_structure(
        db, project_id=payload.project_id, report_id=payload.report_id,
        use_latest_report=payload.use_latest_report,
        generate_if_missing=payload.generate_if_missing, use_test=False))


@router.get("/report-structure/latest", response_model=ReportStructureArtifactResponse)
def report_structure_latest() -> ReportStructureArtifactResponse:
    """最近一次报告结构评测结果。"""
    data = report_structure_eval_service.get_latest()
    if data is None:
        return ReportStructureArtifactResponse(
            available=False, message="尚无评测，请先 POST /api/evaluation/report-structure/run")
    return ReportStructureArtifactResponse(available=True, data=data)


@router.get("/report-structure/fail-items", response_model=ReportStructureArtifactResponse)
def report_structure_fail_items() -> ReportStructureArtifactResponse:
    """最近一次结构评测缺失项与修复建议。"""
    data = report_structure_eval_service.get_failed_items()
    if data is None:
        return ReportStructureArtifactResponse(available=False, message="尚无缺失项记录")
    return ReportStructureArtifactResponse(available=True, data=data)


@router.get("/report-structure/metric-card", response_model=ReportStructureArtifactResponse)
def report_structure_metric_card() -> ReportStructureArtifactResponse:
    """报告结构完整率指标卡（口径/通过线/八类完整率）。"""
    data = report_structure_eval_service.get_metric_card()
    if data is None:
        return ReportStructureArtifactResponse(available=False, message="尚无 metric_card")
    return ReportStructureArtifactResponse(available=True, data=data)


# --------------------------------------------------------------------------- #
# 第11 T8：生成内容与底层数据一致性门禁（仅 train/val 报告；禁止 test）
# --------------------------------------------------------------------------- #
@router.post("/report-consistency/run", response_model=ReportConsistencyEvalResponse)
def report_consistency_run(payload: ReportConsistencyRunRequest,
                           db: Session = Depends(get_db)) -> ReportConsistencyEvalResponse:
    """运行内容一致性评测：数字/结论/证据/限制四类回比底层 ground truth + 门禁（禁止 test）。"""
    return ReportConsistencyEvalResponse(**report_consistency_eval_service.evaluate_report_consistency(
        db, project_id=payload.project_id, report_id=payload.report_id,
        use_latest_report=payload.use_latest_report,
        generate_if_missing=payload.generate_if_missing, use_test=False))


@router.get("/report-consistency/latest", response_model=ReportConsistencyArtifactResponse)
def report_consistency_latest() -> ReportConsistencyArtifactResponse:
    """最近一次内容一致性评测结果。"""
    data = report_consistency_eval_service.get_latest()
    if data is None:
        return ReportConsistencyArtifactResponse(
            available=False, message="尚无评测，请先 POST /api/evaluation/report-consistency/run")
    return ReportConsistencyArtifactResponse(available=True, data=data)


@router.get("/report-consistency/inconsistent-items", response_model=ReportConsistencyArtifactResponse)
def report_consistency_inconsistent_items() -> ReportConsistencyArtifactResponse:
    """最近一次评测的不一致项与修复建议。"""
    data = report_consistency_eval_service.get_inconsistent_items()
    if data is None:
        return ReportConsistencyArtifactResponse(available=False, message="尚无不一致项记录")
    return ReportConsistencyArtifactResponse(available=True, data=data)


@router.get("/report-consistency/metric-card", response_model=ReportConsistencyArtifactResponse)
def report_consistency_metric_card() -> ReportConsistencyArtifactResponse:
    """内容一致性指标卡（口径/权重/通过线/四率）。"""
    data = report_consistency_eval_service.get_metric_card()
    if data is None:
        return ReportConsistencyArtifactResponse(available=False, message="尚无 metric_card")
    return ReportConsistencyArtifactResponse(available=True, data=data)


# --------------------------------------------------------------------------- #
# 第11.5：总门禁与三大硬指标自评包（纯只读汇总；不触碰 test）
# --------------------------------------------------------------------------- #
@router.get("/phase11-5-gate", response_model=Phase115GateResponse)
def phase11_5_gate(project_id: int = 1, db: Session = Depends(get_db)) -> Phase115GateResponse:
    """第11总门禁：汇总三大硬指标 + 模型/特征 + 合规安全/血缘 + final test 状态。"""
    return Phase115GateResponse(**phase115_gate_service.build_phase115_gate_result(db, project_id))


@router.get("/phase11-eval-card", response_model=Phase115ArtifactResponse)
def phase11_eval_card(project_id: int = 1, db: Session = Depends(get_db)) -> Phase115ArtifactResponse:
    """第11自评卡（KupasEval 自评材料）。"""
    data = phase115_gate_service.get_eval_card()
    if data is None:
        phase115_gate_service.build_phase115_gate_result(db, project_id)
        data = phase115_gate_service.get_eval_card()
    if data is None:
        return Phase115ArtifactResponse(available=False, message="尚无自评卡")
    return Phase115ArtifactResponse(available=True, data=data)


@router.get("/phase11-risk-summary", response_model=Phase115ArtifactResponse)
def phase11_risk_summary(project_id: int = 1, db: Session = Depends(get_db)) -> Phase115ArtifactResponse:
    """第11风险摘要（blockers + warnings）。"""
    data = phase115_gate_service.get_risk_summary()
    if data is None:
        phase115_gate_service.build_phase115_gate_result(db, project_id)
        data = phase115_gate_service.get_risk_summary()
    if data is None:
        return Phase115ArtifactResponse(available=False, message="尚无风险摘要")
    return Phase115ArtifactResponse(available=True, data=data)


@router.get("/phase11-final-test-status", response_model=Phase115ArtifactResponse)
def phase11_final_test_status() -> Phase115ArtifactResponse:
    """final 10% test 状态（区分阶段指标与最终成绩，不伪装）。"""
    data = phase115_gate_service.get_final_test_status()
    if data is None:
        data = phase115_gate_service.evaluate_final_test_status()
    return Phase115ArtifactResponse(available=True, data=data)


# --------------------------------------------------------------------------- #
# 第11.6：final 10% test 最终评估（只读冻结；禁止调参；不回流）
# --------------------------------------------------------------------------- #
@router.post("/final-test/run", response_model=FinalTestEvalResponse)
def final_test_run(payload: FinalTestRunRequest,
                   db: Session = Depends(get_db)) -> FinalTestEvalResponse:
    """运行 final 10% test 最终评估：final 三大硬指标 + 阶段对比 + test 隔离（禁止调参）。"""
    return FinalTestEvalResponse(**final_test_eval_service.run_final_test(
        db, use_frozen_test_manifest=payload.use_frozen_test_manifest,
        eval_mode=payload.eval_mode, allow_tuning=False,
        write_results=payload.write_results))


@router.get("/final-test/latest", response_model=FinalTestArtifactResponse)
def final_test_latest() -> FinalTestArtifactResponse:
    """最近一次 final test 评估结果。"""
    data = final_test_eval_service.get_latest()
    if data is None:
        return FinalTestArtifactResponse(
            available=False, message="尚无 final test，请先 POST /api/evaluation/final-test/run")
    return FinalTestArtifactResponse(available=True, data=data)


@router.get("/final-test/metric-card", response_model=FinalTestArtifactResponse)
def final_test_metric_card() -> FinalTestArtifactResponse:
    """final test 指标卡（final 三大硬指标 + 阶段对比）。"""
    data = final_test_eval_service.get_metric_card()
    if data is None:
        return FinalTestArtifactResponse(available=False, message="尚无 metric_card")
    return FinalTestArtifactResponse(available=True, data=data)


@router.get("/final-test/fail-cases", response_model=FinalTestArtifactResponse)
def final_test_fail_cases() -> FinalTestArtifactResponse:
    """final test 失败用例（仅供人工复核，不得回流调参）。"""
    data = final_test_eval_service.get_fail_cases()
    if data is None:
        return FinalTestArtifactResponse(available=False, message="尚无 fail-cases")
    return FinalTestArtifactResponse(available=True, data=data)
