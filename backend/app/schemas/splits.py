"""数据集切分相关响应模型（第2阶段）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SplitBuildResponse(BaseModel):
    manifest_path: str
    seed: int
    ratios: dict[str, float]
    total_records: int
    per_type: dict[str, Any]
    warnings: list[str] = []


class SplitStatusResponse(BaseModel):
    built: bool
    message: str | None = None
    version: str | None = None
    seed: int | None = None
    mode: str | None = None
    ratios: dict[str, float] | None = None
    created_at: str | None = None
    per_type: dict[str, Any] | None = None
    total_records: int | None = None


class SplitVerifyResponse(BaseModel):
    ok: bool
    checks: list[dict[str, Any]]
    summary: dict[str, Any] | None = None
