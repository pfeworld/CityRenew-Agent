"""KnowledgeChunk RAG 知识块表（第3阶段填充）。

存储文档解析后的结构化知识块。`chunk_text` 仅本地入库用于检索，
接口/前端/日志默认不返回原文整段（符合涉密保护，只暴露摘要与限长片段）。

字段说明：
- source_file  来源文件名（不含敏感路径）
- source_type  知识源类型：policy / template / case_report / field_spec / dataset_spec
- chunk_id     块内稳定标识（用于拼 evidence_id 与索引引用）
- section      章节 / 条款 / sheet 名
- page_no      页码（PDF）或行号（xlsx），无则 None
- chunk_text   块原文（仅本地存，默认不外发）
- chunk_summary / summary  脱敏摘要（本地规则生成，非 LLM）
- keywords     关键词列表（JSON 文本）
- metadata_json 其它元数据（JSON 文本）
- is_sensitive 是否涉密（来源于涉密语料则为 True）
- split        仅 train/val 入库；test 不进知识库
- evidence_id  证据链标识
- vector_ref   索引内部引用 id（BM25 文档下标 / 后续向量库 id）
"""

from __future__ import annotations

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class KnowledgeChunk(Base, TimestampMixin):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    chunk_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 块原文：仅本地入库，接口默认不返回
    chunk_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 脱敏摘要（非原文整段）
    chunk_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 关键词与元数据（JSON 文本）
    keywords: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=True)
    # 仅 train/val 入库
    split: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    evidence_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    # 索引中的引用（BM25 文档下标 / 后续 chroma id），非向量本身
    vector_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
