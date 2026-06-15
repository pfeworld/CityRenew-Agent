"""证据链服务（第3阶段）。

职责：
- 生成稳定 evidence_id：{data_type}:{source_hash8}:{record_or_chunk_id}（对齐 docs/06）。
- 维护 EvidenceChain 映射，支持按 evidence_id 查询证据摘要。
- 默认只返回 source_file / summary / metadata / confidence，不返回原文整段。
- 提供 evidence coverage 基础统计。

红线：查询/统计响应均为脱敏字段；不暴露 chunk_text 原文。
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from typing import Any

from sqlalchemy.orm import Session

from app.models import EvidenceChain, KnowledgeChunk

logger = logging.getLogger("cityrenew.evidence")


def source_hash8(source_file: str) -> str:
    return hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:8]


def make_evidence_id(data_type: str, source_file: str, record_or_chunk_id: str) -> str:
    """生成稳定 evidence_id：{data_type}:{source_hash8}:{record_or_chunk_id}。"""
    return f"{data_type}:{source_hash8(source_file)}:{record_or_chunk_id}"


def upsert_evidence(
    db: Session,
    *,
    evidence_id: str,
    data_type: str | None,
    source_file: str | None,
    record_ref: str | None,
    summary: str | None,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceChain:
    """写入或更新一条证据记录（幂等，按 evidence_id）。"""
    obj = (
        db.query(EvidenceChain)
        .filter(EvidenceChain.evidence_id == evidence_id)
        .first()
    )
    metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
    if obj is None:
        obj = EvidenceChain(evidence_id=evidence_id)
        db.add(obj)
    obj.data_type = data_type
    obj.source_file = source_file
    obj.record_ref = record_ref
    obj.summary = summary
    obj.confidence = confidence
    obj.metadata_json = metadata_json
    return obj


def clear_document_evidence(db: Session) -> int:
    """清理文档型（知识库）证据，便于重建索引时幂等。

    仅删除 data_type 属于文档知识源的证据，不动结构化分析证据。
    """
    doc_types = {"policy", "template", "case", "spec"}
    deleted = (
        db.query(EvidenceChain)
        .filter(EvidenceChain.data_type.in_(doc_types))
        .delete(synchronize_session=False)
    )
    return int(deleted or 0)


def get_evidence(db: Session, evidence_id: str) -> dict[str, Any] | None:
    """按 evidence_id 返回脱敏证据（不含原文整段）。"""
    obj = (
        db.query(EvidenceChain)
        .filter(EvidenceChain.evidence_id == evidence_id)
        .first()
    )
    if obj is None:
        return None
    metadata = json.loads(obj.metadata_json) if obj.metadata_json else {}
    return {
        "evidence_id": obj.evidence_id,
        "data_type": obj.data_type,
        "source_file": obj.source_file,
        "summary": obj.summary,
        "confidence": obj.confidence,
        "metadata": metadata,
    }


def coverage_stats(db: Session) -> dict[str, Any]:
    """证据链覆盖率基础统计（针对知识块）。

    coverage = 带有效 evidence 的知识块数 / 知识块总数。
    """
    total_chunks = db.query(KnowledgeChunk).count()
    chunks_with_evidence = (
        db.query(KnowledgeChunk)
        .filter(KnowledgeChunk.evidence_id.isnot(None))
        .count()
    )

    # 知识块引用的 evidence 是否真实存在于 EvidenceChain
    valid_evidence_ids = {
        eid for (eid,) in db.query(EvidenceChain.evidence_id).all()
    }
    linked = 0
    by_source_type: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    for ev_id, src_type, split in db.query(
        KnowledgeChunk.evidence_id,
        KnowledgeChunk.source_type,
        KnowledgeChunk.split,
    ).all():
        if ev_id and ev_id in valid_evidence_ids:
            linked += 1
        by_source_type[src_type or "unknown"] += 1
        by_split[split or "unknown"] += 1

    coverage = round(linked / total_chunks, 4) if total_chunks else 0.0
    total_evidence = db.query(EvidenceChain).count()
    return {
        "total_chunks": total_chunks,
        "chunks_with_evidence": chunks_with_evidence,
        "chunks_evidence_linked": linked,
        "evidence_coverage": coverage,
        "total_evidence_records": total_evidence,
        "by_source_type": dict(by_source_type),
        "by_split": dict(by_split),
    }
