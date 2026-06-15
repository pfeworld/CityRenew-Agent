"""EvidenceChain 证据链表（第3/5阶段填充）。

evidence_id 规则：{data_type}:{source_file_hash8}:{record_id|chunk_id}（见 docs/06）。
报告/前端只展示 evidence_id + 来源文件名 + 摘要，不展开原文。
"""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class EvidenceChain(Base, TimestampMixin):
    __tablename__ = "evidence_chains"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evidence_id: Mapped[str] = mapped_column(
        String(128), unique=True, index=True, nullable=False
    )
    data_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 指向底层记录或计算过程的引用（record_id / chunk_id / 计算签名）
    record_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 脱敏摘要
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 证据置信度（检索/计算可信度，0~1）
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 其它元数据（JSON 文本）
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
