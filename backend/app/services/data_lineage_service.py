"""数据血缘服务（第10A 内部血缘派生 + 第10B 外部血缘与全量血缘）。

第10A：从 data_audit_service 的审计结果派生内部数据血缘
（文件 → 解析 → 入库 → train/val 特征工程 → 房价监督训练）。
第10B：登记外部数据源/采集任务血缘（external/data_lineage.json），并提供"全量血缘"
聚合（内部 competition_data + 外部 external_data），导出脱敏血缘报告。

红线：每份数据可追溯来源/数量/用途/是否进入训练/是否 test 污染；仅统计量，不含原文；
外部数据 source_group=external_data，与 competition_test 物理隔离，永不混入训练 test。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings

logger = logging.getLogger("cityrenew.data_lineage")

LINEAGE_SCHEMA: tuple[str, ...] = (
    "lineage_id",
    "source_id",
    "source_name",
    "source_type",
    "file_path",
    "raw_count",
    "parsed_count",
    "cleaned_count",
    "db_count",
    "feature_count",
    "training_count",
    "validation_count",
    "test_count",
    "used_for_training",
    "used_for_feature_engineering",
    "used_for_clustering",
    "used_for_similarity",
    "used_for_report",
    "used_for_eval",
    "skipped_count",
    "skipped_reason",
    "coverage_rate",
    "leakage_risk",
    "license_status",
    "compliance_status",
    "quality_score",
    "created_at",
    "updated_at",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _external_dir() -> Path:
    return settings.data_dir / "external"


def _lineage_path() -> Path:
    return _external_dir() / "data_lineage.json"


def _data_catalog_dir() -> Path:
    return settings.data_dir / "outputs" / "data_catalog"


def _blank_record(**kwargs: Any) -> dict[str, Any]:
    rec = {k: None for k in LINEAGE_SCHEMA}
    rec.update({
        "raw_count": 0, "parsed_count": 0, "cleaned_count": 0, "db_count": 0,
        "feature_count": 0, "training_count": 0, "validation_count": 0, "test_count": 0,
        "used_for_training": False, "used_for_feature_engineering": False,
        "used_for_clustering": False, "used_for_similarity": False,
        "used_for_report": False, "used_for_eval": False,
        "skipped_count": 0, "skipped_reason": [],
        "leakage_risk": False, "created_at": _utcnow_iso(), "updated_at": _utcnow_iso(),
    })
    rec.update(kwargs)
    return rec


# --------------------------------------------------------------------------- #
# 内部血缘（从审计派生）
# --------------------------------------------------------------------------- #
def derive_from_audit(audit_result: dict[str, Any]) -> list[dict[str, Any]]:
    """从审计结果派生内部数据血缘条目（仅 competition_data 结构化文件）。"""
    out: list[dict[str, Any]] = []
    created = audit_result.get("created_at")
    for f in audit_result.get("files", []):
        if f.get("source_group") != "competition_data":
            continue
        out.append(
            {
                "lineage_id": f"lin:{f['file_name']}",
                "source_id": f["file_name"],
                "source_name": f["file_name"],
                "source_type": "competition_data",
                "file_path": f["file_path"],
                "raw_count": f.get("raw_record_count"),
                "parsed_count": f.get("parsed_record_count"),
                "cleaned_count": f.get("db_inserted_count"),
                "db_count": f.get("db_inserted_count"),
                "feature_count": f.get("used_in_feature_engineering_count", 0),
                "training_count": f.get("used_in_training_count", 0),
                "validation_count": f.get("split_val_count", 0),
                "test_count": f.get("split_test_count", 0),
                "used_for_training": f.get("used_in_training_count", 0) > 0,
                "used_for_feature_engineering": f.get("used_in_feature_engineering_count", 0) > 0,
                "used_for_clustering": False,
                "used_for_similarity": False,
                "used_for_report": False,
                "used_for_eval": f.get("split_test_count", 0) > 0,
                "skipped_count": f.get("skipped_count", 0),
                "skipped_reason": f.get("skipped_reason", []),
                "coverage_rate": f.get("coverage_rate"),
                "leakage_risk": f.get("leakage_risk", False),
                "license_status": "competition_official",
                "compliance_status": "ok",
                "quality_score": f.get("coverage_rate"),
                "created_at": created,
                "updated_at": created,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 外部血缘（采集任务写入 external/data_lineage.json）
# --------------------------------------------------------------------------- #
def _load_external_lineage() -> list[dict[str, Any]]:
    path = _lineage_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("records", []) or []
        if isinstance(data, list):
            return data
    except Exception:  # noqa: BLE001
        return []
    return []


def _save_external_lineage(records: list[dict[str, Any]]) -> None:
    path = _lineage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"updated_at": _utcnow_iso(), "records": records[-1000:]},
                  f, ensure_ascii=False, indent=2)


def remove_external_by_source_prefix(prefix: str) -> int:
    """删除 source_id 以 prefix 开头的外部血缘记录（用于重算前去重），返回删除数量。"""
    records = _load_external_lineage()
    kept = [r for r in records if not str(r.get("source_id", "")).startswith(prefix)]
    removed = len(records) - len(kept)
    if removed:
        _save_external_lineage(kept)
        logger.info("removed %s external lineage records with prefix %s", removed, prefix)
    return removed


def patch_external_by_source(source_id: str, patch: dict[str, Any]) -> bool:
    """就地更新某个 source_id 的外部血缘记录字段（用于上海范围校验后回填）。"""
    records = _load_external_lineage()
    hit = False
    for r in records:
        if r.get("source_id") == source_id:
            r.update(patch)
            r["updated_at"] = _utcnow_iso()
            hit = True
    if hit:
        _save_external_lineage(records)
    return hit


def record_collection_lineage(*, source_id: str, source_name: str, source_type: str,
                              raw_count: int = 0, cleaned_count: int = 0,
                              license_status: str = "unknown",
                              compliance_status: str = "unknown",
                              used_for_training: bool = False,
                              used_for_feature_engineering: bool = False,
                              used_for_report: bool = False,
                              quality_score: float | None = None,
                              file_path: str | None = None,
                              extra: dict[str, Any] | None = None) -> str:
    """登记一条外部采集血缘，返回 lineage_id。

    外部数据恒不进入 competition test（test_count=0）；used_for_training 由合规把关，
    默认 False（仅授权数据可为 True）。
    """
    lineage_id = f"ext:{source_id}:{uuid.uuid4().hex[:8]}"
    rec = _blank_record(
        lineage_id=lineage_id,
        source_id=source_id,
        source_name=source_name,
        source_type=source_type,
        file_path=file_path,
        raw_count=int(raw_count),
        parsed_count=int(cleaned_count),
        cleaned_count=int(cleaned_count),
        db_count=int(cleaned_count),
        feature_count=int(cleaned_count) if used_for_feature_engineering else 0,
        training_count=int(cleaned_count) if used_for_training else 0,
        used_for_training=used_for_training,
        used_for_feature_engineering=used_for_feature_engineering,
        used_for_report=used_for_report,
        coverage_rate=round(cleaned_count / raw_count, 4) if raw_count else None,
        leakage_risk=False,
        license_status=license_status,
        compliance_status=compliance_status,
        quality_score=quality_score,
    )
    if extra:
        rec.update(extra)
    records = _load_external_lineage()
    records.append(rec)
    _save_external_lineage(records)
    logger.info("external lineage recorded: %s (%s records)", lineage_id, len(records))
    return lineage_id


def external_lineage_records() -> list[dict[str, Any]]:
    return _load_external_lineage()


# --------------------------------------------------------------------------- #
# 全量血缘（内部 + 外部）
# --------------------------------------------------------------------------- #
def build_lineage(db: Session, *, export: bool = True) -> dict[str, Any]:
    """聚合内部（审计派生）+ 外部（采集登记）血缘，回答数据血缘 13 问。"""
    from app.services import data_audit_service

    try:
        audit = data_audit_service.run_data_audit(db, persist=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_lineage audit failed: %s", exc)
        audit = {}

    internal = derive_from_audit(audit) if audit else []
    external = external_lineage_records()
    records = internal + external

    used_for_training = [r["source_id"] for r in records if r.get("used_for_training")]
    used_for_fe = [r["source_id"] for r in records if r.get("used_for_feature_engineering")]
    used_for_clustering = [r["source_id"] for r in records if r.get("used_for_clustering")]
    used_for_similarity = [r["source_id"] for r in records if r.get("used_for_similarity")]
    used_for_report = [r["source_id"] for r in records if r.get("used_for_report")]
    unused = [
        {"source_id": r["source_id"],
         "reason": (r.get("skipped_reason") or
                    ["未进入特征工程/训练（如说明表/文档类仅作口径核对或佐证）"])}
        for r in records
        if not r.get("used_for_feature_engineering") and not r.get("used_for_training")
        and not r.get("used_for_report")
    ]
    leakage_risk = bool(audit.get("leakage_risk", False)) if audit else False
    test_contamination = bool(audit.get("test_contamination_risk", False)) if audit else False

    summary = {
        "internal_record_count": len(internal),
        "external_record_count": len(external),
        "total_records": len(records),
        "used_for_training": used_for_training,
        "used_for_feature_engineering": used_for_fe,
        "used_for_clustering": used_for_clustering,
        "used_for_similarity": used_for_similarity,
        "used_for_report": used_for_report,
        "unused_sources": unused,
        "leakage_risk": leakage_risk,
        "test_contamination_risk": test_contamination,
        "external_in_competition_test": False,
    }
    answers = {
        "1_used_data": [r["source_id"] for r in records],
        "2_origin": {r["source_id"]: r.get("source_type") for r in records},
        "3_legal": {r["source_id"]: r.get("compliance_status") for r in records},
        "4_counts": {r["source_id"]: r.get("db_count") for r in records},
        "5_stages": "见各记录 used_for_* 字段",
        "6_training": used_for_training,
        "7_feature_engineering": used_for_fe,
        "8_clustering": used_for_clustering,
        "9_similarity": used_for_similarity,
        "10_report_only": [r["source_id"] for r in records
                           if r.get("used_for_report") and not r.get("used_for_training")
                           and not r.get("used_for_feature_engineering")],
        "11_unused_with_reason": unused,
        "12_test_contamination": test_contamination,
        "13_leakage_risk": leakage_risk,
    }

    result = {
        "mode": settings.app_mode,
        "phase": "10B",
        "created_at": _utcnow_iso(),
        "schema_fields": list(LINEAGE_SCHEMA),
        "records": records,
        "summary": summary,
        "answers": answers,
        "notes": [
            "内部 competition_data 血缘由审计派生（仅 train/val 计入特征工程/训练，test 不计入）。",
            "外部 external_data 血缘来自采集登记；外部数据物理隔离于 competition test，永不混入训练 test。",
            "本响应仅含统计量与脱敏结论，不含原文/raw_json/坐标/企业名/小区名/地址/个人信息。",
        ],
    }
    if export:
        result["exports"] = _export_lineage_report(result)
    return result


def _export_lineage_report(result: dict[str, Any]) -> dict[str, str]:
    out_dir = _data_catalog_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "数据血缘报告.md"
    path.write_text(_render_lineage_md(result), encoding="utf-8")
    rel = str(path.relative_to(settings.data_dir.parent))
    return {"lineage_md": rel}


def _render_lineage_md(result: dict[str, Any]) -> str:
    s = result["summary"]
    lines = [
        "# 数据血缘报告（内部 + 外部，脱敏）",
        "",
        f"- 生成时间：{result['created_at']}",
        f"- 内部记录：{s['internal_record_count']}　外部记录：{s['external_record_count']}　合计：{s['total_records']}",
        f"- test 污染风险：{s['test_contamination_risk']}　泄露风险：{s['leakage_risk']}",
        f"- 外部数据进入 competition_test：{s['external_in_competition_test']}",
        "",
        "| 来源 | 类型 | 原始 | 入库 | 特征 | 训练 | val | test | 用途(报告) | 合规 | 许可 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result["records"]:
        lines.append(
            f"| {r.get('source_id')} | {r.get('source_type')} | {r.get('raw_count')} | "
            f"{r.get('db_count')} | {r.get('feature_count')} | {r.get('training_count')} | "
            f"{r.get('validation_count')} | {r.get('test_count')} | {r.get('used_for_report')} | "
            f"{r.get('compliance_status')} | {r.get('license_status')} |"
        )
    lines += [
        "",
        f"- 用于训练：{', '.join(s['used_for_training']) or '（仅房价监督训练）'}",
        f"- 用于特征工程：{', '.join(s['used_for_feature_engineering']) or '—'}",
        f"- 仅用于报告佐证：{', '.join(result['answers']['10_report_only']) or '—'}",
        "- 说明：外部数据物理隔离于 competition test，不反推 test 答案、不用于 test 调参。",
    ]
    return "\n".join(lines)
