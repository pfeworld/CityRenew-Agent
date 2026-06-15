"""数据集切分管理（split_manager，第2阶段）。

职责：
- 按 data_type 策略生成 train / val / test，默认比例 70/15/15，固定 seed=42。
- 输出 backend/data/splits/split_manifest.json（含 file_path/record_id/data_type/
  split/is_sensitive/hash/created_at）。
- 将 split 回写到对应数据表。
- 提供 verify_manifest() 与 get_split_summary()。

切分策略（对齐 docs/08 + 用户确认）：
- house_price：按记录切分。
- poi / industry：按坐标粗网格 cell 整组切分（防同区近邻泄露；不拆散同一 cell）。
- population：按 grid（网格）整组切分（总量与画像同 grid 同 split）。

红线：
- test split 仅在此生成并冻结，本阶段不被任何训练/调参/规则/Prompt 路径读取。
- manifest 与回写均不含语料原文；日志仅含计数。
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    DataFile,
    HousingRecord,
    IndustryPoint,
    PoiPoint,
    PopulationProfile,
)
from app.utils import geo_utils

logger = logging.getLogger("cityrenew.split")

DEFAULT_SEED = 42
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
# 实际比例与目标偏差超过该阈值（绝对值）则在质量报告中 warning
RATIO_WARN_THRESHOLD = 0.10

# data_type -> (Model, 分组策略, 对应 DataFile 文件名)
SPLIT_TABLES: dict[str, tuple[Any, str, str]] = {
    "poi": (PoiPoint, "cell", "POI兴趣点分布数据.json"),
    "industry": (IndustryPoint, "cell", "产业布局数据.json"),
    "house_price": (HousingRecord, "record", "房价历史交易数据.json"),
    "population": (PopulationProfile, "grid", "区域人口总量.json"),
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_hash(data_type: str, record_id: str, file_path: str) -> str:
    payload = f"{data_type}|{record_id}|{file_path}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _group_key(strategy: str, row: Any) -> str:
    if strategy == "record":
        return f"rec_{row.id}"
    if strategy == "grid":
        return row.grid_id or f"grid_{row.id}"
    # cell：坐标粗网格整组；无坐标记录各自成组，避免聚簇
    if row.lng is not None and row.lat is not None:
        return geo_utils.cell_key(row.lng, row.lat, settings.split_cell_size_deg)
    return f"nocoord_{row.id}"


def _record_id(data_type: str, row: Any) -> str:
    if data_type == "population":
        return row.grid_id or f"population_{row.id}"
    return row.source_id or f"{data_type}_{row.id}"


def _assign_groups(
    group_keys: list[str], ratios: dict[str, float], seed: int, data_type: str
) -> dict[str, str]:
    """确定性地把分组整组分配到 train/val/test。"""
    keys = sorted(set(group_keys))
    rng = random.Random(f"{seed}:{data_type}")
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    mapping: dict[str, str] = {}
    for i, k in enumerate(keys):
        if i < n_train:
            mapping[k] = "train"
        elif i < n_train + n_val:
            mapping[k] = "val"
        else:
            mapping[k] = "test"
    return mapping


def build_splits(
    db: Session,
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    """生成 train/val/test，回写数据表，写 manifest，并更新质量报告。"""
    ratios = ratios or DEFAULT_RATIOS

    manifest_records: list[dict[str, Any]] = []
    per_type_summary: dict[str, Any] = {}
    warnings: list[str] = []
    created_at = _utcnow_iso()

    for data_type, (model, strategy, file_name) in SPLIT_TABLES.items():
        rows = db.query(model).all()
        if not rows:
            per_type_summary[data_type] = {
                "train": 0, "val": 0, "test": 0, "n_groups": 0,
                "actual_ratios": {"train": 0, "val": 0, "test": 0},
            }
            continue

        group_keys = {row.id: _group_key(strategy, row) for row in rows}
        group_to_split = _assign_groups(
            list(group_keys.values()), ratios, seed, data_type
        )

        split_counter: Counter[str] = Counter()
        for row in rows:
            split = group_to_split[group_keys[row.id]]
            row.split = split
            split_counter[split] += 1
            file_path = row.source_file or f"{settings.corpus_path.name}/{file_name}"
            rid = _record_id(data_type, row)
            manifest_records.append(
                {
                    "file_path": file_path,
                    "record_id": rid,
                    "data_type": data_type,
                    "split": split,
                    "is_sensitive": True,
                    "hash": _record_hash(data_type, rid, file_path),
                    "created_at": created_at,
                }
            )

        total = sum(split_counter.values())
        actual = {
            s: round(split_counter.get(s, 0) / total, 4) if total else 0
            for s in ("train", "val", "test")
        }
        n_groups = len(set(group_keys.values()))
        per_type_summary[data_type] = {
            "train": split_counter.get("train", 0),
            "val": split_counter.get("val", 0),
            "test": split_counter.get("test", 0),
            "n_groups": n_groups,
            "actual_ratios": actual,
        }

        for split_name in ("val", "test"):
            deviation = abs(actual[split_name] - ratios[split_name])
            if deviation > RATIO_WARN_THRESHOLD:
                warnings.append(
                    f"{data_type} 的 {split_name} 实际比例 {actual[split_name]:.2f} "
                    f"偏离目标 {ratios[split_name]:.2f}（仅 {n_groups} 个空间组，"
                    f"不拆散同一 cell/grid）"
                )

        # 回写 DataFile.split_summary
        df = (
            db.query(DataFile)
            .filter(DataFile.file_name == file_name, DataFile.source == "corpus")
            .first()
        )
        if df is not None:
            df.split_summary = json.dumps(per_type_summary[data_type], ensure_ascii=False)

    db.commit()

    manifest = {
        "version": "1.0",
        "created_at": created_at,
        "seed": seed,
        "mode": settings.app_mode,
        "ratios": ratios,
        "records": manifest_records,
    }
    settings.splits_dir.mkdir(parents=True, exist_ok=True)
    with settings.split_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _update_quality_report(created_at, seed, ratios, per_type_summary, warnings)

    logger.info(
        "splits built: seed=%s total_records=%s warnings=%s",
        seed, len(manifest_records), len(warnings),
    )
    return {
        "manifest_path": str(settings.split_manifest_path.relative_to(settings.data_dir.parent)),
        "seed": seed,
        "ratios": ratios,
        "total_records": len(manifest_records),
        "per_type": per_type_summary,
        "warnings": warnings,
    }


def _update_quality_report(
    created_at: str,
    seed: int,
    ratios: dict[str, float],
    per_type: dict[str, Any],
    warnings: list[str],
) -> None:
    """把切分检查结果（含比例 warning）写回质量报告。"""
    qr = settings.quality_report_path
    if not qr.exists():
        return
    with qr.open("r", encoding="utf-8") as f:
        report = json.load(f)
    report["split_check"] = {
        "created_at": created_at,
        "seed": seed,
        "ratios": ratios,
        "per_type": per_type,
        "warnings": warnings,
    }
    with qr.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _load_manifest() -> dict[str, Any] | None:
    path = settings.split_manifest_path
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_split_summary() -> dict[str, Any]:
    """返回各 data_type 的 train/val/test 计数与实际比例。"""
    manifest = _load_manifest()
    if manifest is None:
        return {"built": False, "message": "split_manifest.json 不存在，请先执行 build"}

    counts: dict[str, Counter] = defaultdict(Counter)
    for rec in manifest["records"]:
        counts[rec["data_type"]][rec["split"]] += 1

    summary: dict[str, Any] = {}
    for data_type, c in counts.items():
        total = sum(c.values())
        summary[data_type] = {
            "train": c.get("train", 0),
            "val": c.get("val", 0),
            "test": c.get("test", 0),
            "total": total,
            "actual_ratios": {
                s: round(c.get(s, 0) / total, 4) if total else 0
                for s in ("train", "val", "test")
            },
        }
    return {
        "built": True,
        "version": manifest.get("version"),
        "seed": manifest.get("seed"),
        "mode": manifest.get("mode"),
        "ratios": manifest.get("ratios"),
        "created_at": manifest.get("created_at"),
        "per_type": summary,
        "total_records": len(manifest["records"]),
    }


def verify_manifest(db: Session | None = None) -> dict[str, Any]:
    """校验 manifest 结构、hash、重复与（可选）数据库一致性。"""
    manifest = _load_manifest()
    checks: list[dict[str, Any]] = []

    if manifest is None:
        return {"ok": False, "checks": [{"name": "manifest_exists", "passed": False,
                "detail": "split_manifest.json 不存在"}]}

    records = manifest.get("records", [])
    required = {"file_path", "record_id", "data_type", "split", "is_sensitive", "hash", "created_at"}

    # 1. 字段完整性
    field_ok = all(required.issubset(r.keys()) for r in records) and len(records) > 0
    checks.append({"name": "fields_complete", "passed": field_ok,
                   "detail": f"{len(records)} 条记录字段齐全" if field_ok else "存在字段缺失或无记录"})

    # 2. 比例合法
    ratios = manifest.get("ratios", {})
    ratio_ok = abs(sum(ratios.values()) - 1.0) < 1e-6
    checks.append({"name": "ratios_sum_to_1", "passed": ratio_ok,
                   "detail": f"ratios={ratios}"})

    # 3. seed 存在
    checks.append({"name": "seed_present", "passed": manifest.get("seed") is not None,
                   "detail": f"seed={manifest.get('seed')}"})

    # 4. (data_type, record_id) 无重复跨 split
    seen: dict[tuple, str] = {}
    dup = 0
    for r in records:
        key = (r["data_type"], r["record_id"])
        if key in seen and seen[key] != r["split"]:
            dup += 1
        seen[key] = r["split"]
    checks.append({"name": "no_split_leakage_duplicate", "passed": dup == 0,
                   "detail": f"{dup} 个记录被分到多个 split" if dup else "无记录跨 split"})

    # 5. hash 可复算
    hash_bad = 0
    for r in records:
        if _record_hash(r["data_type"], r["record_id"], r["file_path"]) != r["hash"]:
            hash_bad += 1
    checks.append({"name": "hash_recomputable", "passed": hash_bad == 0,
                   "detail": f"{hash_bad} 条 hash 不匹配" if hash_bad else "全部 hash 可复算"})

    # 6.（可选）DB 回写一致 + 空间组整组性
    if db is not None:
        db_mismatch = _verify_db_consistency(db)
        checks.append(db_mismatch)
        checks.append(_verify_group_integrity(db))

    ok = all(c["passed"] for c in checks)
    data_types = sorted({k[0] for k in seen})
    return {"ok": ok, "checks": checks,
            "summary": {"records": len(records), "data_types": data_types}}


def _verify_db_consistency(db: Session) -> dict[str, Any]:
    """校验数据库各表的 split 与 manifest 是否一致（按记录数）。"""
    manifest = _load_manifest() or {"records": []}
    manifest_counts: dict[str, Counter] = defaultdict(Counter)
    for r in manifest["records"]:
        manifest_counts[r["data_type"]][r["split"]] += 1

    mismatch = 0
    for data_type, (model, _strategy, _file) in SPLIT_TABLES.items():
        db_counts: Counter = Counter()
        for (split_val,) in db.query(model.split).all():
            db_counts[split_val] += 1
        for split_name in ("train", "val", "test"):
            if db_counts.get(split_name, 0) != manifest_counts[data_type].get(split_name, 0):
                mismatch += 1
    return {"name": "db_split_matches_manifest", "passed": mismatch == 0,
            "detail": f"{mismatch} 处计数不一致" if mismatch else "数据库 split 与 manifest 一致"}


def _verify_group_integrity(db: Session) -> dict[str, Any]:
    """校验 cell/grid 未被拆散到多个 split（空间防泄露）。"""
    broken = 0
    for data_type, (model, strategy, _file) in SPLIT_TABLES.items():
        if strategy == "record":
            continue
        group_split: dict[str, set] = defaultdict(set)
        for row in db.query(model).all():
            group_split[_group_key(strategy, row)].add(row.split)
        broken += sum(1 for splits in group_split.values() if len(splits) > 1)
    return {"name": "spatial_group_not_split", "passed": broken == 0,
            "detail": f"{broken} 个空间组被拆散" if broken else "空间组均未被拆散"}
