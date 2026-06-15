"""RAG 接口请求/响应模型（第3阶段）。

响应仅含脱敏字段（摘要 + 限长片段 + 来源 + score + evidence_id），不含原文整段。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索问题")
    top_k: int | None = Field(default=None, ge=1, le=50)


class RagResultItem(BaseModel):
    chunk_id: str
    source_file: str
    source_type: str
    section: str | None = None
    page_no: int | None = None
    score: float
    evidence_id: str
    summary: str | None = None
    snippet: str | None = None


class RagQueryResponse(BaseModel):
    query: str
    top_k: int | None = None
    splits: list[str] = []
    count: int = 0
    results: list[RagResultItem] = []
    message: str | None = None


class RagBuildResponse(BaseModel):
    built_at: str
    mode: str
    backend: str
    total_chunks: int
    by_source_type: dict[str, int] = {}
    allowed_splits: list[str] = []
    files: list[dict[str, Any]] = []
    index_path: str
    notes: list[str] = []


class RagStatusResponse(BaseModel):
    index_built: bool
    backend: str
    mode: str
    total_chunks: int
    by_source_type: dict[str, int] = {}
    by_split: dict[str, int] = {}
    allowed_splits: list[str] = []
    built_at: str | None = None
