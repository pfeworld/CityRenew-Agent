"""IndustryPoint 产业布局表（第2阶段导入填充）。"""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class IndustryPoint(Base, TimestampMixin):
    __tablename__ = "industry_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 产业 9 大类之一
    category_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    district_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    address: Mapped[str | None] = mapped_column(String(512), nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    coord_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    split: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 本地溯源用原始记录（涉密，仅落库不外发）
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)
