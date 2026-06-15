"""AnalysisResult 分析结果表（第5阶段四维分析填充）。

报告中所有数字的权威来源（禁止编造数据原则）。
按 圈层(ring) × 维度(dimension) × 指标(metric) 记录，并携带 evidence_id。
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class AnalysisResult(Base, TimestampMixin):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id"), nullable=True, index=True
    )
    # 圈层：core / nearby / radiation
    ring: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # 维度：poi / population / house / industry
    dimension: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metric_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 非数值型指标（如主导产业名）放这里
    metric_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    evidence_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
