"""字段自适应映射服务（第2阶段）。

职责：
- 从 ``field_mapping.yaml`` 加载标准字段 ← 多别名映射。
- ``map_record`` 把原始记录归一化为标准字段，缺失字段标 null 并返回缺失列表。
- 保留 ``raw_json`` 便于本地溯源（仅落库，不外发）。

红线：缺失字段绝不编造（对齐 .cursor/rules 第9条）。
本模块不打印任何字段原值；仅返回结构化结果供上层聚合统计。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_MAPPING_PATH = Path(__file__).resolve().parent.parent / "resources" / "field_mapping.yaml"

# 人口画像字段前缀（动态保留至 profile_json）
_PROFILE_PREFIXES = ("residential_", "worker_")


@lru_cache
def load_mapping() -> dict[str, Any]:
    """加载并缓存字段映射配置。"""
    with _MAPPING_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class MappingResult:
    """单条记录的字段映射结果。"""

    data_type: str
    mapped: dict[str, Any]
    missing_fields: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    raw_json: str = ""


def _first_present(raw: dict[str, Any], aliases: list[str]) -> Any:
    for key in aliases:
        if key in raw:
            val = raw[key]
            if val is not None and val != "":
                return val
    return None


def map_record(data_type: str, raw: dict[str, Any]) -> MappingResult:
    """按 data_type 归一化单条记录。

    返回标准字段值、缺失字段（必需）与缺失可选字段（spec_optional）。
    """
    spec = load_mapping().get(data_type)
    if spec is None:
        raise KeyError(f"未配置的 data_type: {data_type}")

    aliases: dict[str, list[str]] = spec.get("aliases", {})
    spec_optional: list[str] = spec.get("spec_optional", [])

    mapped: dict[str, Any] = {}
    missing: list[str] = []
    for std_field, alias_list in aliases.items():
        value = _first_present(raw, alias_list)
        mapped[std_field] = value
        if value is None:
            missing.append(std_field)

    missing_optional: list[str] = []
    for opt in spec_optional:
        if _first_present(raw, [opt]) is None:
            missing_optional.append(opt)

    return MappingResult(
        data_type=data_type,
        mapped=mapped,
        missing_fields=missing,
        missing_optional=missing_optional,
        raw_json=json.dumps(raw, ensure_ascii=False),
    )


def extract_profile_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """抽取人口画像 residential_* / worker_* 全部字段（不含 coordinates）。

    不枚举具体档位，保留原始键值，避免与口径表漂移；不编造缺失档位。
    """
    return {
        k: v
        for k, v in raw.items()
        if isinstance(k, str) and k.startswith(_PROFILE_PREFIXES)
    }
