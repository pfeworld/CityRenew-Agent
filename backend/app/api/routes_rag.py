"""RAG 知识库接口（第3阶段）。

POST /api/rag/build    解析知识源、入库并构建本地 BM25 索引
POST /api/rag/query    本地关键词检索（默认仅 train/val，排除 test）
GET  /api/rag/status   索引状态

红线：返回仅含摘要 + 限长片段 + 来源 + score + evidence_id，不含原文整段。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.rag import (
    RagBuildResponse,
    RagQueryRequest,
    RagQueryResponse,
    RagStatusResponse,
)
from app.services import rag_service

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post("/build", response_model=RagBuildResponse)
def build_index(db: Session = Depends(get_db)) -> RagBuildResponse:
    return RagBuildResponse(**rag_service.build_index(db))


@router.post("/query", response_model=RagQueryResponse)
def query_index(payload: RagQueryRequest) -> RagQueryResponse:
    result = rag_service.query(payload.query, top_k=payload.top_k)
    return RagQueryResponse(**result)


@router.get("/status", response_model=RagStatusResponse)
def status(db: Session = Depends(get_db)) -> RagStatusResponse:
    return RagStatusResponse(**rag_service.get_status(db))
