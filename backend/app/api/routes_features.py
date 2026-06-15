"""特征工程接口（第10A阶段）。

POST /api/features/{project_id}/build   构建项目级特征向量（仅 train/val，落库）
GET  /api/features/{project_id}/latest  读取最近一次特征向量

红线：include_test 不可开启（特征工程固定仅 train/val，used_test=false）；
不调用外部 API；不返回 raw_json/原始点位/企业名/小区名/地址/坐标。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.features import (
    FeatureBuildRequest,
    FeatureQualityResponse,
    FeatureResponse,
    PoiSummaryResponse,
)
from app.services import feature_engineering_service, project_service, spatial_service

router = APIRouter(prefix="/api/features", tags=["features"])


def _get_project_or_404(db: Session, project_id: int):
    project = project_service.get_project(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"项目 {project_id} 不存在")
    return project


@router.post("/{project_id}/build", response_model=FeatureResponse)
def build_features(
    project_id: int,
    payload: FeatureBuildRequest | None = None,
    include_external: bool = Query(
        default=True, description="是否纳入外部 POI（第11 T2 默认开启）"
    ),
    db: Session = Depends(get_db),
) -> FeatureResponse:
    """构建项目级特征向量（第11 T2：圈层 POI 空间特征工程）。"""
    project = _get_project_or_404(db, project_id)
    req = payload or FeatureBuildRequest()
    try:
        return FeatureResponse(
            **feature_engineering_service.build_features(
                db, project, include_external,
                include_external_poi=req.include_external_poi,
                include_research_poi=req.include_research_poi,
                include_internal_poi=req.include_internal_poi,
            )
        )
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}/latest", response_model=FeatureResponse)
def latest_features(project_id: int, db: Session = Depends(get_db)) -> FeatureResponse:
    _get_project_or_404(db, project_id)
    result = feature_engineering_service.get_latest(db, project_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"项目 {project_id} 尚未构建特征，请先调用 build"
        )
    return FeatureResponse(**result)


@router.get("/{project_id}/poi-summary", response_model=PoiSummaryResponse)
def poi_summary(project_id: int, db: Session = Depends(get_db)) -> PoiSummaryResponse:
    """最近一次特征构建的 POI/圈层脱敏摘要。"""
    _get_project_or_404(db, project_id)
    return PoiSummaryResponse(**feature_engineering_service.build_poi_summary(db, project_id))


@router.get("/{project_id}/quality", response_model=FeatureQualityResponse)
def feature_quality(project_id: int, db: Session = Depends(get_db)) -> FeatureQualityResponse:
    """T2 特征质量门禁（pass / warning / fail）。"""
    _get_project_or_404(db, project_id)
    return FeatureQualityResponse(**feature_engineering_service.build_feature_quality(db, project_id))
