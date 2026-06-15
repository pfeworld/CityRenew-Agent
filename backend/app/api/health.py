"""健康检查接口。"""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.system import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", system="CityRenew Agent")
