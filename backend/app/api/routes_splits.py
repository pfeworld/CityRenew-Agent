"""数据集切分接口（第2阶段）。

POST /api/splits/build   生成并冻结 train/val/test，写 manifest 并回写数据表
GET  /api/splits/status  查看各 data_type 的 train/val/test 计数与比例
GET  /api/splits/verify  校验 manifest（结构/hash/重复/数据库一致/空间组整组性）

红线：test split 仅生成与冻结，本阶段不被训练/分析路径读取。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.splits import SplitBuildResponse, SplitStatusResponse, SplitVerifyResponse
from app.services import split_manager

router = APIRouter(prefix="/api/splits", tags=["splits"])


@router.post("/build", response_model=SplitBuildResponse)
def build_splits(db: Session = Depends(get_db)) -> SplitBuildResponse:
    result = split_manager.build_splits(db)
    return SplitBuildResponse(**result)


@router.get("/status", response_model=SplitStatusResponse)
def split_status() -> SplitStatusResponse:
    return SplitStatusResponse(**split_manager.get_split_summary())


@router.get("/verify", response_model=SplitVerifyResponse)
def verify_splits(db: Session = Depends(get_db)) -> SplitVerifyResponse:
    return SplitVerifyResponse(**split_manager.verify_manifest(db))
