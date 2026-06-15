"""健康检查与系统信息的响应模型。"""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    system: str = "CityRenew Agent"


class RingConfig(BaseModel):
    """圈层口径（对齐 docs/11 D4）。"""

    core: str = "红线内 / 中心点核心缓冲"
    nearby_buffer_m: int
    radiation_buffer_m: int


class SystemInfoResponse(BaseModel):
    name: str
    version: str
    environment: str
    mode: str  # eval / demo
    database: str  # ok / error
    coordinate_system: str
    rings: RingConfig
    data_security_notice: str
