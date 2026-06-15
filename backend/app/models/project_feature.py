"""ProjectFeature 项目级特征向量表（第10A阶段填充）。

把 POI / 人口 / 房价 / 产业 / 项目字段（后续可含合规外部数据）汇聚成项目级
特征向量，作为多模型训练 / 聚类 / 相似度学习的统一输入来源。

红线：
- 默认仅 train/val（used_test 固定 false）；test 不参与特征工程。
- 仅存储派生统计量与特征值，不存任何语料原文/原始点位明细；
  payload_json 内仅含特征名、特征值、分组、缺失项、来源计数与 evidence_id。
"""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class ProjectFeature(Base, TimestampMixin):
    __tablename__ = "project_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id"), nullable=True, index=True
    )
    # 是否纳入外部数据（第10A 恒为 false：外部数据在第10B 才接入）
    include_external: Mapped[bool] = mapped_column(Boolean, default=False)
    # 参与特征工程的 split（应为 "train,val"）
    allowed_splits: Mapped[str | None] = mapped_column(String(32), nullable=True)
    used_test: Mapped[bool] = mapped_column(Boolean, default=False)
    feature_count: Mapped[int] = mapped_column(Integer, default=0)
    missing_count: Mapped[int] = mapped_column(Integer, default=0)
    feature_coverage_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 完整特征载荷（脱敏 JSON）：feature_names/feature_values/feature_groups/
    # missing_features/evidence_ids/used_source_counts/feature_vector
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
