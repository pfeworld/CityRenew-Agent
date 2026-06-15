"""资料导入相关响应模型（第2阶段）。

仅承载统计量/文件名/计数，不含任何语料原文。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class IngestionFileReport(BaseModel):
    file_name: str
    data_type: str
    source_file: str
    file_hash: str
    record_count_raw: int
    record_count_written: int
    skipped: int
    missing_field_counts: dict[str, int] = {}
    missing_optional_counts: dict[str, int] = {}
    coordinate_stats: dict[str, int] = {}
    warnings: list[str] = []
    note: str | None = None
    sheets: list[dict[str, Any]] | None = None


class IngestionRunResponse(BaseModel):
    created_at: str
    mode: str
    corpus_dir_name: str
    files: list[IngestionFileReport]
    totals: dict[str, Any]
    notes: list[str] = []


class IngestionStatusResponse(BaseModel):
    ingested: bool
    table_counts: dict[str, int]
    data_files: int
    quality_report_exists: bool
    quality_report_path: str
    mode: str
