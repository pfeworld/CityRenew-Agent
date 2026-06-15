"""项目管理接口请求/响应模型（第4阶段）。

字段别名（对齐用户口径）：
- project_name ↔ name
- lon ↔ center_lng
- lat ↔ center_lat

红线：响应模型不含 raw_json / 原始明细，仅项目自身录入字段。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    """创建项目入参，接受 name/project_name、center_lng/lon、center_lat/lat 等别名。"""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        min_length=1,
        validation_alias=AliasChoices("name", "project_name"),
        description="项目名称",
    )
    city: str | None = None
    district: str | None = None
    street: str | None = None
    address: str | None = None
    description: str | None = None

    center_lng: float | None = Field(
        default=None, validation_alias=AliasChoices("center_lng", "lon", "lng")
    )
    center_lat: float | None = Field(
        default=None, validation_alias=AliasChoices("center_lat", "lat")
    )
    coordinate_system: str | None = None
    boundary_geojson: str | None = None

    core_buffer_m: int | None = Field(default=None, ge=0)
    nearby_buffer_m: int | None = Field(default=None, ge=0)
    radiation_buffer_m: int | None = Field(default=None, ge=0)

    land_use: str | None = None
    project_area: float | None = Field(default=None, ge=0)
    building_area: float | None = Field(default=None, ge=0)
    build_year: int | None = None
    update_demand: str | None = None
    expected_direction: str | None = None
    status: str | None = None


class ProjectUpdate(BaseModel):
    """更新项目入参，所有字段可选；同样支持别名。"""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(
        default=None, validation_alias=AliasChoices("name", "project_name")
    )
    city: str | None = None
    district: str | None = None
    street: str | None = None
    address: str | None = None
    description: str | None = None

    center_lng: float | None = Field(
        default=None, validation_alias=AliasChoices("center_lng", "lon", "lng")
    )
    center_lat: float | None = Field(
        default=None, validation_alias=AliasChoices("center_lat", "lat")
    )
    coordinate_system: str | None = None
    boundary_geojson: str | None = None

    core_buffer_m: int | None = Field(default=None, ge=0)
    nearby_buffer_m: int | None = Field(default=None, ge=0)
    radiation_buffer_m: int | None = Field(default=None, ge=0)

    land_use: str | None = None
    project_area: float | None = Field(default=None, ge=0)
    building_area: float | None = Field(default=None, ge=0)
    build_year: int | None = None
    update_demand: str | None = None
    expected_direction: str | None = None
    status: str | None = None


class ProjectOut(BaseModel):
    """项目响应（不含 raw_json / 原始明细）。

    同时输出规范字段与常用别名（project_name / lon / lat），便于前端消费。
    """

    id: int
    name: str
    project_name: str | None = None
    city: str | None = None
    district: str | None = None
    street: str | None = None
    address: str | None = None
    description: str | None = None

    center_lng: float | None = None
    center_lat: float | None = None
    lon: float | None = None
    lat: float | None = None
    coordinate_system: str | None = None
    boundary_geojson: str | None = None

    core_buffer_m: int | None = None
    nearby_buffer_m: int | None = None
    radiation_buffer_m: int | None = None

    land_use: str | None = None
    project_area: float | None = None
    building_area: float | None = None
    build_year: int | None = None
    update_demand: str | None = None
    expected_direction: str | None = None

    project_type: str | None = None
    status: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ProjectListResponse(BaseModel):
    total: int
    items: list[ProjectOut] = []
