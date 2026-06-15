"""空间圈层分析接口响应模型（第4阶段）。

红线：仅返回统计数量与空间归集摘要，**不含** raw_json、原始坐标列表、
小区/企业名等原始明细。
"""

from __future__ import annotations

from pydantic import BaseModel


class RingCounts(BaseModel):
    """单个圈层带内的四类数据归集数量。"""

    ring: str  # core / nearby / radiation
    radius_m: int  # 该圈层外缘半径（米）
    poi_count: int = 0
    housing_count: int = 0
    industry_count: int = 0
    population_grid_count: int = 0


class RingsResponse(BaseModel):
    """三圈层归集结果（互斥带计数 + 累计圆内计数）。"""

    project_id: int
    coordinate_system: str
    center_lng: float
    center_lat: float
    center_status: str  # ok / corrected
    has_boundary: bool
    include_test: bool
    allowed_splits: list[str] = []
    # 互斥圈层带：core / nearby(core~500) / radiation(500~1500)
    rings: list[RingCounts] = []
    # 累计圆内（落入对应半径圆的总量），便于后续四维分析
    cumulative: dict[str, RingCounts] = {}
    skipped_no_coord: dict[str, int] = {}
    notes: list[str] = []


class SpatialSummaryResponse(BaseModel):
    """空间归集摘要：rings + 各数据类型按 split 分组数量。"""

    project_id: int
    coordinate_system: str
    center_lng: float
    center_lat: float
    center_status: str
    has_boundary: bool
    include_test: bool
    allowed_splits: list[str] = []
    rings: list[RingCounts] = []
    cumulative: dict[str, RingCounts] = {}
    # data_type -> {ring -> {split -> count}}，仅落入 radiation 范围内的记录参与分 split 统计
    by_split: dict[str, dict[str, dict[str, int]]] = {}
    skipped_no_coord: dict[str, int] = {}
    notes: list[str] = []
