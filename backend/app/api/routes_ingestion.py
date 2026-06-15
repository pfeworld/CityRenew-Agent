"""资料导入接口（第2阶段）。

POST /api/ingestion/run            执行本地结构化资料导入
GET  /api/ingestion/status         查询导入状态（各表计数）
GET  /api/ingestion/quality-report 查看数据质量报告（仅统计量）

红线：返回内容仅含统计量/文件名/计数，不含语料原文。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.ingestion import IngestionRunResponse, IngestionStatusResponse
from app.services import ingestion_service

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])


@router.post("/run", response_model=IngestionRunResponse)
def run_ingestion(db: Session = Depends(get_db)) -> IngestionRunResponse:
    try:
        report = ingestion_service.run_ingestion(db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IngestionRunResponse(**report)


@router.get("/status", response_model=IngestionStatusResponse)
def ingestion_status(db: Session = Depends(get_db)) -> IngestionStatusResponse:
    return IngestionStatusResponse(**ingestion_service.get_status(db))


@router.get("/quality-report")
def quality_report() -> dict:
    report = ingestion_service.get_quality_report()
    if report is None:
        raise HTTPException(status_code=404, detail="质量报告不存在，请先执行 /api/ingestion/run")
    return report
