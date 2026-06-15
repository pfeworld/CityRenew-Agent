"""RAG 知识库服务（第3阶段 MVP）。

实现本地关键词检索（BM25 + jieba 中文分词），不依赖任何外部 embedding/API。

能力：
- build_index()：解析知识源 → 写 KnowledgeChunk + EvidenceChain → 构建 BM25
  → 持久化到 backend/data/index/（bm25_index.pkl + chunks_meta.json）。
- query()：分词检索，返回 chunk 摘要 / source_file / source_type / score / evidence_id，
  默认只用 train/val，显式排除 test，不返回原文整段。
- get_status()：索引状态。

红线：
- 仅 train/val 入库；test 不进知识库（本阶段文档源均为 train）。
- 接口返回仅摘要 + 限长片段，不含 chunk_text 原文。
- 日志只含计数；不输出原文。
- 预留 backend 开关，后续可扩展 Chroma/FAISS。
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.database import init_db
from app.models import KnowledgeChunk
from app.services import document_parser_service as parser
from app.services import evidence_service

logger = logging.getLogger("cityrenew.rag")

ALLOWED_SPLITS = ("train", "val")  # 红线：test 不进知识库
_INDEX_CACHE: dict[str, Any] | None = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> list[str]:
    try:
        import jieba

        return [t.strip() for t in jieba.cut(text) if t.strip()]
    except Exception:  # noqa: BLE001  回退
        import re

        return re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", text)


def _snippet(summary: str | None) -> str:
    if not summary:
        return ""
    limit = settings.rag_snippet_max_chars
    return summary if len(summary) <= limit else summary[:limit] + "…"


# --------------------------------------------------------------------------- #
# 构建索引
# --------------------------------------------------------------------------- #
def build_index(db: Session) -> dict[str, Any]:
    """解析知识源、入库并构建本地 BM25 索引。"""
    global _INDEX_CACHE
    init_db()

    # 解析（仅本地读取）
    chunks, file_reports = parser.parse_knowledge_sources()

    # 红线：仅 train/val 入库
    chunks = [c for c in chunks if c.split in ALLOWED_SPLITS]

    # 幂等：清理旧的知识块与文档型证据
    db.query(KnowledgeChunk).delete()
    evidence_service.clear_document_evidence(db)
    db.flush()

    meta: list[dict[str, Any]] = []
    corpus_tokens: list[list[str]] = []

    for pos, c in enumerate(chunks):
        db.add(
            KnowledgeChunk(
                source_file=c.source_file,
                source_type=c.source_type,
                chunk_id=c.chunk_id,
                section=c.section,
                page_no=c.page_no,
                chunk_text=c.chunk_text,
                chunk_summary=c.chunk_summary,
                summary=c.chunk_summary,
                keywords=json.dumps(c.keywords, ensure_ascii=False),
                metadata_json=json.dumps(c.metadata, ensure_ascii=False),
                is_sensitive=c.is_sensitive,
                split=c.split,
                evidence_id=c.evidence_id,
                vector_ref=str(pos),
            )
        )
        evidence_service.upsert_evidence(
            db,
            evidence_id=c.evidence_id,
            data_type=c.data_type,
            source_file=c.source_file,
            record_ref=c.chunk_id,
            summary=c.chunk_summary,
            confidence=None,
            metadata={
                "source_type": c.source_type,
                "section": c.section,
                "page_no": c.page_no,
            },
        )
        # 检索语料：摘要 + 关键词加权（不入原文亦可，但用 chunk_text 提升召回）
        corpus_tokens.append(_tokenize(c.chunk_text))
        meta.append(
            {
                "pos": pos,
                "chunk_id": c.chunk_id,
                "source_file": c.source_file,
                "source_type": c.source_type,
                "section": c.section,
                "page_no": c.page_no,
                "summary": c.chunk_summary,
                "keywords": c.keywords,
                "evidence_id": c.evidence_id,
                "split": c.split,
            }
        )

    db.commit()

    # 构建 BM25
    built = _persist_index(corpus_tokens, meta)
    _INDEX_CACHE = None  # 失效缓存，下次 query 重新加载

    by_type: Counter[str] = Counter(m["source_type"] for m in meta)
    result = {
        "built_at": _utcnow_iso(),
        "mode": settings.app_mode,
        "backend": "bm25",
        "total_chunks": len(meta),
        "by_source_type": dict(by_type),
        "allowed_splits": list(ALLOWED_SPLITS),
        "files": file_reports,
        "index_path": str(settings.bm25_index_path.relative_to(settings.data_dir.parent)),
        "notes": [
            "知识库仅含 train/val 文档，未引入 test。",
            "接口返回仅摘要与限长片段，不含原文整段。",
        ],
    }
    logger.info(
        "rag index built: chunks=%s types=%s", len(meta), dict(by_type)
    )
    return result


def _persist_index(
    corpus_tokens: list[list[str]], meta: list[dict[str, Any]]
) -> bool:
    from rank_bm25 import BM25Okapi

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
    payload = {
        "built_at": _utcnow_iso(),
        "backend": "bm25",
        "bm25": bm25,
    }
    with settings.bm25_index_path.open("wb") as f:
        pickle.dump(payload, f)
    with settings.chunks_meta_path.open("w", encoding="utf-8") as f:
        json.dump({"built_at": payload["built_at"], "meta": meta}, f, ensure_ascii=False)
    return True


# --------------------------------------------------------------------------- #
# 加载与查询
# --------------------------------------------------------------------------- #
def _load_index() -> dict[str, Any] | None:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    if not settings.bm25_index_path.exists() or not settings.chunks_meta_path.exists():
        return None
    with settings.bm25_index_path.open("rb") as f:
        payload = pickle.load(f)
    with settings.chunks_meta_path.open("r", encoding="utf-8") as f:
        meta_doc = json.load(f)
    _INDEX_CACHE = {
        "bm25": payload.get("bm25"),
        "meta": meta_doc.get("meta", []),
        "built_at": meta_doc.get("built_at"),
    }
    return _INDEX_CACHE


def query(
    q: str,
    top_k: int | None = None,
    splits: tuple[str, ...] = ALLOWED_SPLITS,
) -> dict[str, Any]:
    """关键词检索。返回脱敏结果（摘要 + 限长片段 + 来源 + score + evidence_id）。"""
    top_k = top_k or settings.rag_default_top_k
    # 红线：禁止把 test 纳入检索范围
    splits = tuple(s for s in splits if s in ALLOWED_SPLITS) or ALLOWED_SPLITS

    index = _load_index()
    if index is None or index.get("bm25") is None:
        return {"query": q, "results": [], "message": "索引不存在，请先 /api/rag/build"}

    bm25 = index["bm25"]
    meta = index["meta"]
    tokens = _tokenize(q)
    if not tokens:
        return {"query": q, "results": [], "message": "查询为空或无有效分词"}

    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(meta)), key=lambda i: scores[i], reverse=True)

    results: list[dict[str, Any]] = []
    for i in ranked:
        m = meta[i]
        if m.get("split") not in splits:
            continue
        if scores[i] <= 0:
            continue
        results.append(
            {
                "chunk_id": m["chunk_id"],
                "source_file": m["source_file"],
                "source_type": m["source_type"],
                "section": m.get("section"),
                "page_no": m.get("page_no"),
                "score": round(float(scores[i]), 4),
                "evidence_id": m["evidence_id"],
                "summary": m.get("summary"),
                "snippet": _snippet(m.get("summary")),
                "keywords": m.get("keywords") or [],
            }
        )
        if len(results) >= top_k:
            break

    return {
        "query": q,
        "top_k": top_k,
        "splits": list(splits),
        "count": len(results),
        "results": results,
    }


def get_status(db: Session) -> dict[str, Any]:
    """索引状态（脱敏统计）。"""
    index_exists = (
        settings.bm25_index_path.exists() and settings.chunks_meta_path.exists()
    )
    total_chunks = db.query(KnowledgeChunk).count()
    by_type: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    for src_type, split in db.query(
        KnowledgeChunk.source_type, KnowledgeChunk.split
    ).all():
        by_type[src_type or "unknown"] += 1
        by_split[split or "unknown"] += 1

    built_at = None
    if settings.chunks_meta_path.exists():
        with settings.chunks_meta_path.open("r", encoding="utf-8") as f:
            built_at = json.load(f).get("built_at")

    return {
        "index_built": index_exists,
        "backend": "bm25",
        "mode": settings.app_mode,
        "total_chunks": total_chunks,
        "by_source_type": dict(by_type),
        "by_split": dict(by_split),
        "allowed_splits": list(ALLOWED_SPLITS),
        "built_at": built_at,
    }
