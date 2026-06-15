"""EvaluationResult 自评结果表（第9阶段填充）。

记录 eval 模式下基于 test split 的指标结果，可导出为 KupasEval 自评材料。
"""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class EvaluationResult(Base, TimestampMixin):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # 运行模式：eval / demo
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # 指标名：retrieval_accuracy / report_completeness / data_consistency /
    #          type_f1 / house_mape / evidence_coverage / hallucination_rate
    metric_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 评估所用 split（应为 test）
    split: Mapped[str | None] = mapped_column(String(16), nullable=True)
    dataset_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 明细（JSON 字符串）
    detail_json: Mapped[str | None] = mapped_column(Text, nullable=True)
