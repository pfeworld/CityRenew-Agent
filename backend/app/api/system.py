"""系统信息接口。"""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.system import SystemInfoResponse
from app.services.system_service import build_system_info

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/info", response_model=SystemInfoResponse)
def system_info() -> SystemInfoResponse:
    return build_system_info()
