"""系统信息服务：拼装 /api/system/info 的内容。"""

from __future__ import annotations

from app.config import settings
from app.database import check_db_connection
from app.schemas.system import RingConfig, SystemInfoResponse


def build_system_info() -> SystemInfoResponse:
    db_ok = check_db_connection()
    return SystemInfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        mode=settings.app_mode,
        database="ok" if db_ok else "error",
        coordinate_system=settings.coordinate_system,
        rings=RingConfig(
            nearby_buffer_m=settings.nearby_buffer_m,
            radiation_buffer_m=settings.radiation_buffer_m,
        ),
        data_security_notice=settings.data_security_notice,
    )
