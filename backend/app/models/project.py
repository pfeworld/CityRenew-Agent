"""Project 项目表。

按 docs/11 决策预留：
- coordinate_system（D1）
- boundary_geojson 红线（D2，支持手动中心点 + 后续上传红线）
- core_buffer_m / nearby_buffer_m / radiation_buffer_m 圈层口径（D4，500/1500）
"""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._mixins import TimestampMixin


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # 行政区划与地址（第4阶段补充，幂等 ALTER 新增，均可空）
    city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    district: Mapped[str | None] = mapped_column(String(128), nullable=True)
    street: Mapped[str | None] = mapped_column(String(128), nullable=True)
    address: Mapped[str | None] = mapped_column(String(512), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 项目输入：手动中心点（lng, lat 顺序见 D1）
    center_lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_lat: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 坐标系（预留，默认 WGS84，待确认）
    coordinate_system: Mapped[str] = mapped_column(String(32), default="WGS84")

    # 红线（预留，后续支持上传边界多边形 GeoJSON 字符串）
    boundary_geojson: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 圈层口径（米），默认对齐报告模板：近邻 500 / 辐射 1500
    # core_buffer_m 为中心点核心缓冲（无红线时使用），默认 0 表示以红线为准
    core_buffer_m: Mapped[int] = mapped_column(Integer, default=0)
    nearby_buffer_m: Mapped[int] = mapped_column(Integer, default=500)
    radiation_buffer_m: Mapped[int] = mapped_column(Integer, default=1500)

    # 项目业务属性（第4阶段补充，幂等 ALTER 新增，均可空）
    land_use: Mapped[str | None] = mapped_column(String(128), nullable=True)
    project_area: Mapped[float | None] = mapped_column(Float, nullable=True)  # 用地面积 ㎡
    building_area: Mapped[float | None] = mapped_column(Float, nullable=True)  # 建筑面积 ㎡
    build_year: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 建成年代
    update_demand: Mapped[str | None] = mapped_column(Text, nullable=True)  # 更新诉求
    expected_direction: Mapped[str | None] = mapped_column(Text, nullable=True)  # 期望方向

    # 项目类型（第6阶段识别填充）
    project_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="created")
