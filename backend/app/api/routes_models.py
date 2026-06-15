"""第11 阶段模型接口（T1：训练入口安全护栏 guard-only）。

GET  /api/models/training-readiness  训练就绪概览（默认房价请求 guard + readiness）
POST /api/models/guard-check         训练护栏检查（dry-run，只检查不训练）

红线：本组接口**只检查训练条件，不训练任何模型、不生成模型文件、不读取 test 明细**。
真实训练（T3）须在 fit() 前调用 training_guard_service.assert_training_allowed。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from fastapi import HTTPException

from app.database import get_db
from app.schemas.models import (
    GuardCheckRequest,
    GuardCheckResponse,
    ModelArtifactResponse,
    ProjectTypeTrainRequest,
    ProjectTypeTrainResponse,
    ScoreCalibrateRequest,
    ScoreCalibrateResponse,
    TrainingReadinessResponse,
    TrainRequest,
    TrainResponse,
)
from app.services import (
    housing_price_training_service,
    housing_robustness_service,
    model_training_service,
    project_type_training_service,
    score_calibration_service,
    training_guard_service,
)


def _score_response(result: dict) -> ScoreCalibrateResponse:
    """把 T5 校准结果统一封装为响应（附质量门禁）。"""
    quality = score_calibration_service.score_calibration_quality(
        result if result.get("trained") else None)
    sr = result.get("score_result", {}) or {}
    return ScoreCalibrateResponse(
        status=result.get("status", "unknown"),
        trained=result.get("trained", False),
        training_task=result.get("training_task", "score_calibration"),
        guard_status=result.get("guard_status"),
        score_version=result.get("score_version"),
        test_used=result.get("test_used", False),
        comprehensive_recomputable=result.get("comprehensive_recomputable"),
        score_result=sr,
        score_contributions=sr.get("contributions", []),
        calibration_card=result.get("calibration_card", {}),
        calibration_report=result.get("calibration_report", {}),
        weight_config=result.get("weight_config", {}),
        data_lineage_ids=result.get("data_lineage_ids", []),
        evidence_ids=sr.get("evidence_ids", []),
        quality_status=quality.get("score_calibration_quality_status"),
        score_calibration_quality=quality,
        warnings=result.get("warnings", []),
        reason=result.get("reason"),
        created_at=result.get("created_at"),
    )

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/training-readiness", response_model=TrainingReadinessResponse)
def training_readiness(db: Session = Depends(get_db)) -> TrainingReadinessResponse:
    """训练就绪概览：默认房价监督训练请求的护栏结果 + 第10C.5 readiness（不训练）。"""
    return TrainingReadinessResponse(**training_guard_service.build_training_readiness(db))


@router.post("/guard-check", response_model=GuardCheckResponse)
def guard_check(payload: GuardCheckRequest, db: Session = Depends(get_db)) -> GuardCheckResponse:
    """训练护栏检查（dry-run）：检查 split 隔离 / test 阻断 / 外部训练标记 / 房价合规 / 禁止源。

    只检查不训练；fail 时返回 status=fail、can_train=false、blockers 列表，不抛错。
    """
    result = training_guard_service.validate_training_request(
        db, payload.model_dump(), raise_on_violation=False
    )
    return GuardCheckResponse(**result)


@router.post("/train")
def train_model(payload: TrainRequest, db: Session = Depends(get_db)):
    """第11 训练统一入口（先 guard 再 fit；fail 即阻断 HTTP 400）。

    - housing_price_regression → T3 房价监督训练（TrainResponse）
    - project_type_classification → T4 项目类型弱监督（ProjectTypeTrainResponse）
    - score_calibration → T5 评分校准（ScoreCalibrateResponse）
    """
    task = payload.training_task
    try:
        result = model_training_service.guarded_training_entry(db, payload.model_dump())
    except training_guard_service.TrainingGuardError as exc:
        raise HTTPException(status_code=400, detail=f"训练护栏未通过：{exc}") from exc

    if task in {"project_type_classification", "project_type_weak_classifier"}:
        quality = project_type_training_service.type_training_quality(
            result if result.get("trained") else None)
        return ProjectTypeTrainResponse(**result, type_training_quality=quality)

    if task in {"score_calibration", "score_calibrator"}:
        return _score_response(result)

    quality = housing_price_training_service.training_quality(
        result if result.get("trained") else None)
    return TrainResponse(**result, training_quality=quality)


@router.post("/calibrate-score", response_model=ScoreCalibrateResponse)
def calibrate_score(payload: ScoreCalibrateRequest, db: Session = Depends(get_db)) -> ScoreCalibrateResponse:
    """第11 T5：评分校准（原始 10 维 → train/val 分位校准 → 加权综合；先 guard，不读 test）。"""
    req = payload.model_dump()
    req["training_task"] = "score_calibration"
    result = score_calibration_service.train(db, req)
    return _score_response(result)


@router.get("/score/latest", response_model=ModelArtifactResponse)
def score_latest() -> ModelArtifactResponse:
    """最近一次评分校准结果。"""
    data = score_calibration_service.get_latest()
    if data is None:
        return ModelArtifactResponse(available=False, message="尚未校准，请先 POST /api/models/calibrate-score")
    return ModelArtifactResponse(available=True, data=data)


@router.get("/score/explain/{project_id}", response_model=ModelArtifactResponse)
def score_explain(project_id: int, db: Session = Depends(get_db)) -> ModelArtifactResponse:
    """指定项目的可解释评分（raw/calibrated/contribution/drivers）。"""
    data = score_calibration_service.explain_project(db, project_id)
    return ModelArtifactResponse(available=data.get("available", True), data=data,
                                 message=data.get("message"))


@router.post("/train-project-type", response_model=ProjectTypeTrainResponse)
def train_project_type(payload: ProjectTypeTrainRequest, db: Session = Depends(get_db)) -> ProjectTypeTrainResponse:
    """第11 T4：项目类型识别弱监督辅助模型（weak_label=true；先 guard，不读 test）。"""
    result = project_type_training_service.train(db, payload.model_dump())
    quality = project_type_training_service.type_training_quality(
        result if result.get("trained") else None)
    return ProjectTypeTrainResponse(**result, type_training_quality=quality)


@router.get("/project-type/latest", response_model=ModelArtifactResponse)
def project_type_latest() -> ModelArtifactResponse:
    """最近一次项目类型弱监督训练结果。"""
    data = project_type_training_service.get_latest()
    if data is None:
        return ModelArtifactResponse(available=False, message="尚未训练，请先 POST /api/models/train-project-type")
    return ModelArtifactResponse(available=True, data=data)


@router.get("/project-type/explain/{project_id}", response_model=ModelArtifactResponse)
def project_type_explain(project_id: int, db: Session = Depends(get_db)) -> ModelArtifactResponse:
    """项目类型可解释预测（rule_based + model_assisted + reason_codes）。"""
    data = project_type_training_service.explain_project_type_prediction(db, project_id)
    return ModelArtifactResponse(available=data.get("available", True), data=data,
                                 message=data.get("message"))


@router.get("/latest", response_model=ModelArtifactResponse)
def latest_model() -> ModelArtifactResponse:
    """最近一次训练结果（脱敏摘要）。"""
    data = housing_price_training_service.get_latest()
    if data is None:
        return ModelArtifactResponse(available=False, message="尚未训练，请先 POST /api/models/train")
    return ModelArtifactResponse(available=True, data=data)


@router.get("/audit", response_model=ModelArtifactResponse)
def model_audit() -> ModelArtifactResponse:
    """最近一次训练的数据使用审计。"""
    data = housing_price_training_service.get_audit()
    if data is None:
        return ModelArtifactResponse(available=False, message="尚无训练审计")
    return ModelArtifactResponse(available=True, data=data)


@router.get("/feature-importance", response_model=ModelArtifactResponse)
def model_feature_importance() -> ModelArtifactResponse:
    """最近一次训练的特征重要性。"""
    data = housing_price_training_service.get_feature_importance()
    if data is None:
        return ModelArtifactResponse(available=False, message="尚无特征重要性")
    return ModelArtifactResponse(available=True, data=data)


@router.get("/training-log", response_model=ModelArtifactResponse)
def model_training_log() -> ModelArtifactResponse:
    """最近一次训练日志。"""
    data = housing_price_training_service.get_training_log()
    if data is None:
        return ModelArtifactResponse(available=False, message="尚无训练日志")
    return ModelArtifactResponse(available=True, data=data)


@router.post("/robustness-run", response_model=ModelArtifactResponse)
def robustness_run(db: Session = Depends(get_db)) -> ModelArtifactResponse:
    """第11 T3.5：运行真实性复核与防记忆验证（A–G 实验，仅 train/val）。"""
    data = housing_robustness_service.run_robustness(db)
    return ModelArtifactResponse(available=True, data=data)


@router.get("/robustness", response_model=ModelArtifactResponse)
def robustness_report() -> ModelArtifactResponse:
    """最近一次真实性复核报告。"""
    data = housing_robustness_service.get_robustness()
    if data is None:
        return ModelArtifactResponse(available=False, message="尚无复核报告，请先 POST /api/models/robustness-run")
    return ModelArtifactResponse(available=True, data=data)


@router.get("/leakage-check", response_model=ModelArtifactResponse)
def leakage_check() -> ModelArtifactResponse:
    """标签泄漏检查结果。"""
    data = housing_robustness_service.get_leakage_check()
    if data is None:
        return ModelArtifactResponse(available=False, message="尚无泄漏检查，请先 POST /api/models/robustness-run")
    return ModelArtifactResponse(available=True, data=data)


@router.get("/ablation-study", response_model=ModelArtifactResponse)
def ablation_study() -> ModelArtifactResponse:
    """特征消融研究结果。"""
    data = housing_robustness_service.get_ablation_study()
    if data is None:
        return ModelArtifactResponse(available=False, message="尚无消融研究，请先 POST /api/models/robustness-run")
    return ModelArtifactResponse(available=True, data=data)
