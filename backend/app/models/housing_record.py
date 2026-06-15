"""HousingRecord 房价历史交易表（第2阶段导入填充）。

字段做自适应映射的目标 schema；缺失字段允许为空并在导入时标注。
"""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class HousingRecord(Base, TimestampMixin):
    __tablename__ = "housing_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)  # 总价(元)
    unit_price: Mapped[float | None] = mapped_column(Float, nullable=True)  # 元/㎡
    area: Mapped[float | None] = mapped_column(Float, nullable=True)  # ㎡
    direction: Mapped[str | None] = mapped_column(String(32), nullable=True)  # 朝向
    room_type: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 户型
    residence: Mapped[str | None] = mapped_column(String(255), nullable=True)  # 小区
    building_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 建成年代
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    coord_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    split: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 本地溯源用原始记录（涉密，仅落库不外发）
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)
