"""PopulationProfile 区域人口总量+画像表（第2阶段导入填充）。

人口为网格数据：coordinates 为网格边界框两点（D1 中点/框语义不同）。
画像 30+ 字段以 JSON 字符串存于 profile_json，避免列爆炸且便于自适应映射。
"""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class PopulationProfile(Base, TimestampMixin):
    __tablename__ = "population_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grid_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    residential: Mapped[int | None] = mapped_column(Integer, nullable=True)
    worker: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 网格中心点（bbox 两角点均值，便于后续圈层归集）
    center_lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 网格边界框 GeoJSON 字符串（[[lng,lat],[lng,lat]] 派生）
    bbox_geojson: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 画像 30+ 字段（年龄/性别/教育/消费/资产）以 JSON 存储
    profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    coord_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    split: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # 本地溯源用原始记录（涉密，仅落库不外发）
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)
