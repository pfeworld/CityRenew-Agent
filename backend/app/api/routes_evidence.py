"""证据链接口（第3阶段）。

GET /api/evidence/{evidence_id}  按 evidence_id 查询脱敏证据摘要
GET /api/evidence/stats          证据链覆盖率基础统计

红线：仅返回 source_file / summary / metadata / confidence，不含原文整段。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.evidence import EvidenceResponse, EvidenceStatsResponse
from app.services import evidence_service

router = APIRouter(prefix="/api/evidence", tags=["evidence"])


@router.get("/stats", response_model=EvidenceStatsResponse)
def evidence_stats(db: Session = Depends(get_db)) -> EvidenceStatsResponse:
    return EvidenceStatsResponse(**evidence_service.coverage_stats(db))


@router.get("/{evidence_id}", response_model=EvidenceResponse)
def get_evidence(evidence_id: str, db: Session = Depends(get_db)) -> EvidenceResponse:
    ev = evidence_service.get_evidence(db, evidence_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="evidence_id 不存在")
    return EvidenceResponse(**ev)
