"""空间圈层分析接口（第4阶段）。

GET /api/projects/{id}/rings            三圈层归集数量（core/nearby/radiation）
GET /api/projects/{id}/spatial-summary  圈层 + 各数据类型按 split 分组数量

红线：默认排除 test（include_test 默认 false）；仅返回统计数量与归集摘要，
不返回 raw_json / 原始坐标列表 / 小区/企业名等原始明细。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.spatial import RingsResponse, SpatialSummaryResponse
from app.services import project_service, spatial_service

router = APIRouter(prefix="/api/projects", tags=["spatial"])


@router.get("/{project_id}/rings", response_model=RingsResponse)
def get_rings(
    project_id: int,
    include_test: bool = Query(default=False, description="是否纳入 test（默认 false）"),
    db: Session = Depends(get_db),
) -> RingsResponse:
    project = project_service.get_project(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"项目 {project_id} 不存在")
    try:
        return RingsResponse(**spatial_service.get_rings(db, project, include_test))
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{project_id}/spatial-summary", response_model=SpatialSummaryResponse)
def get_spatial_summary(
    project_id: int,
    include_test: bool = Query(default=False, description="是否纳入 test（默认 false）"),
    db: Session = Depends(get_db),
) -> SpatialSummaryResponse:
    project = project_service.get_project(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"项目 {project_id} 不存在")
    try:
        return SpatialSummaryResponse(
            **spatial_service.get_spatial_summary(db, project, include_test)
        )
    except spatial_service.SpatialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
