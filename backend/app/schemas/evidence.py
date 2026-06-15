"""证据链接口响应模型（第3阶段）。

仅返回脱敏字段：source_file / summary / metadata / confidence，不含原文整段。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class EvidenceResponse(BaseModel):
    evidence_id: str
    data_type: str | None = None
    source_file: str | None = None
    summary: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = {}


class EvidenceStatsResponse(BaseModel):
    total_chunks: int
    chunks_with_evidence: int
    chunks_evidence_linked: int
    evidence_coverage: float
    total_evidence_records: int
    by_source_type: dict[str, int] = {}
    by_split: dict[str, int] = {}
