"""第11 阶段 T1：训练入口安全护栏（training guard）。

本服务是所有监督训练的**强制前置门禁**，只做"检查"，不训练任何模型、不生成模型文件、
不读取大体量明细、不触碰 competition_test 明细。

红线（对齐 docs/第11执行前检查清单.md 与 .cursor/rules）：
1. train 训练 / val 选模型调参 / test 仅最终评估；训练数据严禁出现 split=test。
2. 外部/科研数据必须检查 used_for_training；只有授权+脱敏+上海确认的房价样本可监督训练。
3. 高德 POI / 科研 POI / 政策 / 案例 / 统计 / 公共数据不得作为监督标签（used_for_training 恒 False）。
4. 任一红线不满足 → can_train=False、status=fail，并给出 blockers；绝不写假指标。

核心函数：
- validate_training_request：训练请求合法性总编排。
- validate_dataset_splits：train/val/test 隔离检查（请求 split 不得含 test）。
- assert_no_test_records：训练数据中出现 split=test 立即阻断（可选抛异常）。
- validate_external_training_usage：外部/科研数据 used_for_training 标记检查。
- validate_housing_training_data：房价训练数据合规校验。
- validate_non_training_sources：禁止源不得进入监督训练。
- build_data_usage_audit：训练数据使用审计。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.services import data_lineage_service

logger = logging.getLogger("cityrenew.training_guard")

# --------------------------------------------------------------------------- #
# 常量：训练隔离与合规白/黑名单
# --------------------------------------------------------------------------- #
ALLOWED_TRAIN_SPLITS: frozenset[str] = frozenset({"train", "val"})
FORBIDDEN_TRAIN_SPLITS: frozenset[str] = frozenset({"test"})

# 仅以下源/类型允许作为监督训练标签源（授权脱敏房价）
ALLOWED_TRAINING_SOURCE_IDS: frozenset[str] = frozenset(
    {"research_housing_property", "authorized_property"}
)
ALLOWED_TRAINING_SOURCE_TYPES: frozenset[str] = frozenset({"authorized_property"})

# 以下源/类型严禁进入监督训练（只能作特征/报告/RAG），除非未来明确变更为合法训练源
NON_TRAINING_SOURCE_IDS: frozenset[str] = frozenset(
    {
        "amap_poi",
        "research_poi_public_service",
        "policy_planning",
        "project_case",
        "stats_macro",
        "shanghai_open_data",
        "public_service",
        "government_policy",
        "urban_renewal_cases",
    }
)
NON_TRAINING_SOURCE_TYPES: frozenset[str] = frozenset(
    {
        "amap",
        "poi",
        "osm",
        "gov_open_data",
        "stats_cn",
        "planning_policy",
        "urban_renewal_cases",
        "public_service",
    }
)

VALID_AUTHORIZATION_STATUS: frozenset[str] = frozenset(
    {"provided", "authorized", "authorized_candidate"}
)
MIN_TRAINABLE_RECORDS = 1000
REQUIRED_CITY_SCOPE = "上海市"

# 需要监督标签的训练任务（必须有授权房价标签源）
SUPERVISED_TASKS: frozenset[str] = frozenset(
    {"housing_price_regression", "house_price_regressor"}
)


class TrainingGuardError(RuntimeError):
    """训练护栏硬阻断异常：用于真实训练入口（raise_on_violation=True）。"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _external_dir():
    return settings.data_dir / "external"


# --------------------------------------------------------------------------- #
# 数据源装配（从真实脱敏统计量派生，不读明细；支持仿真覆盖供自测）
# --------------------------------------------------------------------------- #
def _lineage_ids_for(source_id_prefix: str) -> list[str]:
    """从外部血缘登记中收集某来源的 lineage_id（仅 ID，不含明细）。"""
    try:
        records = data_lineage_service.external_lineage_records()
    except Exception:  # noqa: BLE001
        return []
    return [
        str(r.get("lineage_id"))
        for r in records
        if str(r.get("source_id", "")).startswith(source_id_prefix) and r.get("lineage_id")
    ]


def _load_housing_source(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """从第10C.5 授权房价 profile 派生房价训练源描述（脱敏统计量）。

    overrides 仅供自测注入 fail 场景，不改动任何真实数据文件。
    """
    from app.services import external_data_collector_service as collector

    hp = collector._research_property_profile()  # noqa: SLF001 复用脱敏统计读取
    trainable = int(hp.get("trainable_property_records", 0))
    src = {
        "source_id": "research_housing_property",
        "source_name": "科研授权脱敏房价样本（上海确认）",
        "source_type": "authorized_property",
        "role": "label",
        "split": "train_val",  # 仅 train/val，不含 test
        "used_for_training": True,
        "used_for_feature_engineering": bool(hp.get("trainable_property_records", 0) > 0),
        "used_for_report": True,
        "record_count": int(hp.get("total_records", 0)),
        "trainable_record_count": trainable,
        "city_scope": hp.get("city_scope", REQUIRED_CITY_SCOPE if trainable else ""),
        "shanghai_verified": bool(hp.get("shanghai_verified", False)),
        "is_desensitized": bool(hp.get("is_desensitized", False)),
        "authorization_status": hp.get("authorization_status", "unknown"),
        "license_status": hp.get("license_status", "unknown"),
        "has_price_label": bool(trainable > 0),
        "test_contamination_risk": bool(hp.get("test_contamination_risk", False)),
        "leakage_risk": bool(hp.get("leakage_risk", False)),
        "data_lineage_ids": _lineage_ids_for("research_housing_property"),
    }
    if overrides:
        src.update(overrides)
        src["_simulated_override"] = True
    return src


def _load_poi_feature_source() -> dict[str, Any]:
    """高德/科研 POI 特征源（仅特征工程/报告，恒不进监督训练）。"""
    from app.services import external_data_collector_service as collector

    amap_man = collector._read_manifest("amap") or {}  # noqa: SLF001
    total = int(amap_man.get("merged_dedup_total", amap_man.get("record_count", 0)))
    return {
        "source_id": "amap_poi",
        "source_name": "高德上海去重 POI（特征工程用）",
        "source_type": "amap",
        "role": "feature",
        "split": None,
        "used_for_training": False,
        "used_for_feature_engineering": True,
        "used_for_report": True,
        "record_count": total,
        "trainable_record_count": 0,
        "city_scope": REQUIRED_CITY_SCOPE,
        "shanghai_verified": True,
        "is_desensitized": True,
        "authorization_status": "provided",
        "license_status": amap_man.get("license_status", "commercial_api_terms"),
        "has_price_label": False,
        "test_contamination_risk": False,
        "leakage_risk": False,
        "data_lineage_ids": _lineage_ids_for("amap_poi"),
    }


def _poi_as_label_source() -> dict[str, Any]:
    """仿真：误把 POI 当监督标签（自测场景3用），不读真实明细。"""
    src = _load_poi_feature_source()
    src.update({"role": "label", "used_for_training": True, "_simulated_misuse": True})
    return src


def _source_as_label(source_id: str, source_type: str, name: str) -> dict[str, Any]:
    """仿真：误把禁止源（政策/案例等）当监督标签。"""
    return {
        "source_id": source_id, "source_name": name, "source_type": source_type,
        "role": "label", "split": None, "used_for_training": True,
        "used_for_feature_engineering": False, "used_for_report": True,
        "record_count": 0, "trainable_record_count": 0, "city_scope": "",
        "shanghai_verified": False, "is_desensitized": True,
        "authorization_status": "provided", "license_status": "provided_by_research_partner",
        "has_price_label": False, "test_contamination_risk": False, "leakage_risk": False,
        "data_lineage_ids": [], "_simulated_misuse": True,
    }


def _test_split_source() -> dict[str, Any]:
    """仿真：训练源混入 split=test（自测场景2用）。仅构造元数据，不读取任何 test 明细。"""
    return {
        "source_id": "competition_housing_test_leak", "source_name": "（仿真）混入的 test 记录",
        "source_type": "competition_data", "role": "label", "split": "test",
        "used_for_training": True, "used_for_feature_engineering": False,
        "used_for_report": False, "record_count": 1, "trainable_record_count": 1,
        "city_scope": REQUIRED_CITY_SCOPE, "shanghai_verified": True, "is_desensitized": True,
        "authorization_status": "provided", "license_status": "competition_official",
        "has_price_label": True, "test_contamination_risk": True, "leakage_risk": True,
        "data_lineage_ids": [], "_simulated_misuse": True,
    }


def assemble_training_sources(db: Session, req: dict[str, Any]) -> list[dict[str, Any]]:
    """根据训练请求装配训练数据源描述列表（从脱敏统计量派生）。

    仿真字段（use_poi_as_label / use_policy_as_label / inject_test_records / housing_overrides）
    仅用于自测护栏拦截能力，不改动/不读取任何真实 test 明细或大体量 jsonl。
    """
    task = req.get("training_task", "housing_price_regression")
    needs_label = task in SUPERVISED_TASKS
    sources: list[dict[str, Any]] = []

    if needs_label and req.get("use_authorized_property", True):
        sources.append(_load_housing_source(req.get("housing_overrides")))

    if req.get("use_poi_features", True):
        sources.append(_load_poi_feature_source())

    # ---- 以下均为仿真注入（自测护栏拦截），默认关闭 ----
    if req.get("use_poi_as_label"):
        sources.append(_poi_as_label_source())
    if req.get("use_policy_as_label"):
        sources.append(_source_as_label("policy_planning", "planning_policy", "（仿真）政策文本当标签"))
    if req.get("inject_test_records"):
        sources.append(_test_split_source())

    return sources


# --------------------------------------------------------------------------- #
# 核心校验函数
# --------------------------------------------------------------------------- #
def validate_dataset_splits(requested_splits: list[str]) -> list[dict[str, Any]]:
    """检查请求的 split 不得包含 test（train 训练 / val 选模型 / test 仅最终评估）。"""
    blockers: list[dict[str, Any]] = []
    bad = [s for s in (requested_splits or []) if s in FORBIDDEN_TRAIN_SPLITS]
    if bad:
        blockers.append({
            "code": "split_includes_test",
            "item": "请求的训练 split 含 test",
            "detail": f"requested_splits={requested_splits}；test 仅用于最终评估，禁止训练/调参",
            "fix": "训练只允许 train/val",
        })
    unknown = [s for s in (requested_splits or [])
               if s not in ALLOWED_TRAIN_SPLITS and s not in FORBIDDEN_TRAIN_SPLITS]
    if unknown:
        blockers.append({
            "code": "split_unknown",
            "item": "未知 split",
            "detail": f"无法识别的 split: {unknown}",
            "fix": "split 只允许 train/val",
        })
    return blockers


def assert_no_test_records(
    sources: list[dict[str, Any]], *, raise_on_violation: bool = False
) -> list[dict[str, Any]]:
    """训练数据中出现 split=test 或 test 污染风险立即阻断。

    raise_on_violation=True（真实训练入口）时抛 TrainingGuardError；
    False（dry-run guard-check）时返回 blockers 列表。
    """
    blockers: list[dict[str, Any]] = []
    for s in sources:
        sid = s.get("source_id")
        if s.get("split") in FORBIDDEN_TRAIN_SPLITS:
            blockers.append({
                "code": "test_record_in_training",
                "item": "训练数据混入 split=test",
                "detail": f"来源 {sid} 标记 split=test，严禁进入训练",
                "fix": "从训练集中剔除全部 test 记录",
            })
        if s.get("test_contamination_risk"):
            blockers.append({
                "code": "test_contamination_risk",
                "item": "test 污染风险",
                "detail": f"来源 {sid} test_contamination_risk=true",
                "fix": "排查并隔离 competition_test",
            })
    if blockers and raise_on_violation:
        logger.error("training guard blocked (test records): %s", blockers)
        raise TrainingGuardError(f"训练数据混入 test，已阻断：{blockers}")
    return blockers


def validate_external_training_usage(
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """检查外部/科研数据 used_for_training 标记：只有白名单源可为 True。"""
    blockers: list[dict[str, Any]] = []
    allowed: list[str] = []
    rejected: list[dict[str, Any]] = []
    for s in sources:
        sid = s.get("source_id", "")
        stype = s.get("source_type", "")
        if not s.get("used_for_training"):
            continue
        is_allowed = sid in ALLOWED_TRAINING_SOURCE_IDS or stype in ALLOWED_TRAINING_SOURCE_TYPES
        if is_allowed:
            allowed.append(sid)
        else:
            rejected.append({"source_id": sid, "source_type": stype,
                             "reason": "非白名单源，used_for_training 必须为 false"})
            blockers.append({
                "code": "external_used_for_training_not_allowed",
                "item": "外部/科研数据被误标为可训练",
                "detail": f"来源 {sid}（{stype}）used_for_training=true，但不在训练白名单",
                "fix": "将 used_for_training 置 false，或先完成合法训练源变更评审",
            })
    return {"blockers": blockers, "allowed_training_sources": allowed,
            "rejected_sources": rejected}


def validate_non_training_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """禁止源（POI/政策/案例/统计/公共数据等）不得作为监督标签或进入训练。"""
    blockers: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for s in sources:
        sid = s.get("source_id", "")
        stype = s.get("source_type", "")
        is_forbidden = sid in NON_TRAINING_SOURCE_IDS or stype in NON_TRAINING_SOURCE_TYPES
        if not is_forbidden:
            continue
        if s.get("used_for_training") or s.get("role") == "label":
            rejected.append({"source_id": sid, "source_type": stype,
                             "reason": "禁止源不得进入监督训练（仅特征/报告/RAG）"})
            blockers.append({
                "code": "non_training_source_used_as_label",
                "item": "禁止源被用作监督标签/训练",
                "detail": f"来源 {sid}（{stype}）属禁止训练源，不能作为标签或进入训练",
                "fix": "仅将其用于特征工程/报告/RAG，used_for_training 置 false",
            })
    return {"blockers": blockers, "rejected_sources": rejected}


def validate_housing_training_data(housing: dict[str, Any] | None) -> dict[str, Any]:
    """房价监督训练数据合规校验（全部条件必须满足）。"""
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not housing:
        blockers.append({
            "code": "housing_source_missing",
            "item": "缺少房价训练标签源",
            "detail": "监督房价训练需授权脱敏上海确认房价样本，但未提供",
            "fix": "use_authorized_property=true 并确保授权房价资产已就绪",
        })
        return {"blockers": blockers, "warnings": warnings}

    checks = [
        ("city_scope", housing.get("city_scope") == REQUIRED_CITY_SCOPE,
         f"city_scope 必须为 {REQUIRED_CITY_SCOPE}，当前={housing.get('city_scope')!r}"),
        ("shanghai_verified", housing.get("shanghai_verified") is True,
         "shanghai_verified 必须为 true"),
        ("is_desensitized", housing.get("is_desensitized") is True,
         "is_desensitized 必须为 true（未脱敏严禁训练）"),
        ("authorization_status",
         housing.get("authorization_status") in VALID_AUTHORIZATION_STATUS,
         f"authorization_status 必须 ∈ {sorted(VALID_AUTHORIZATION_STATUS)}，"
         f"当前={housing.get('authorization_status')!r}"),
        ("has_price_label", housing.get("has_price_label") is True,
         "has_price_label 必须为 true"),
        ("trainable_record_count",
         int(housing.get("trainable_record_count", 0)) >= MIN_TRAINABLE_RECORDS,
         f"trainable_record_count 必须 >= {MIN_TRAINABLE_RECORDS}，"
         f"当前={housing.get('trainable_record_count', 0)}"),
        ("test_contamination_risk", housing.get("test_contamination_risk") is False,
         "test_contamination_risk 必须为 false"),
        ("leakage_risk", housing.get("leakage_risk") is False,
         "leakage_risk 必须为 false"),
    ]
    for code, ok, detail in checks:
        if not ok:
            blockers.append({
                "code": f"housing_{code}_invalid",
                "item": f"房价训练数据不满足：{code}",
                "detail": detail,
                "fix": "修正房价数据合规属性或补充授权脱敏上海样本",
            })

    strength = ("strong" if int(housing.get("trainable_record_count", 0)) >= 3000
                else "medium" if int(housing.get("trainable_record_count", 0)) >= 1000
                else "weak")
    return {"blockers": blockers, "warnings": warnings, "supervised_training_strength": strength}


def build_data_usage_audit(
    sources: list[dict[str, Any]],
    *,
    rejected_sources: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    """训练数据使用审计（脱敏统计量，含每源用途与合规标记）。"""
    audit_sources = []
    for s in sources:
        audit_sources.append({
            "source_id": s.get("source_id"),
            "source_name": s.get("source_name"),
            "source_type": s.get("source_type"),
            "role": s.get("role"),
            "record_count": s.get("record_count"),
            "trainable_record_count": s.get("trainable_record_count"),
            "used_for_training": bool(s.get("used_for_training")),
            "used_for_feature_engineering": bool(s.get("used_for_feature_engineering")),
            "used_for_report": bool(s.get("used_for_report")),
            "city_scope": s.get("city_scope"),
            "shanghai_verified": bool(s.get("shanghai_verified")),
            "authorization_status": s.get("authorization_status"),
            "license_status": s.get("license_status"),
            "test_contamination_risk": bool(s.get("test_contamination_risk")),
            "leakage_risk": bool(s.get("leakage_risk")),
            "data_lineage_ids": s.get("data_lineage_ids", []),
            "simulated": bool(s.get("_simulated_override") or s.get("_simulated_misuse")),
        })
    return {
        "generated_at": _utcnow_iso(),
        "sources": audit_sources,
        "training_sources": [s["source_id"] for s in audit_sources if s["used_for_training"]],
        "feature_only_sources": [s["source_id"] for s in audit_sources
                                 if s["used_for_feature_engineering"] and not s["used_for_training"]],
        "rejected_sources": rejected_sources,
        "warnings": warnings,
        "blockers": blockers,
        "test_used_for_training": any(s.get("split") in FORBIDDEN_TRAIN_SPLITS
                                      or s.get("test_contamination_risk") for s in sources),
    }


# --------------------------------------------------------------------------- #
# 总编排
# --------------------------------------------------------------------------- #
def validate_training_request(
    db: Session, req: dict[str, Any], *, raise_on_violation: bool = False
) -> dict[str, Any]:
    """训练请求合法性总编排：装配数据源 → 逐项校验 → 汇总 guard 结果。

    raise_on_violation=True 用于真实训练入口（fail 即抛 TrainingGuardError）。
    """
    task = req.get("training_task", "housing_price_regression")
    requested_splits = req.get("requested_splits") or ["train", "val"]
    sources = assemble_training_sources(db, req)

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    blockers += validate_dataset_splits(requested_splits)
    blockers += assert_no_test_records(sources, raise_on_violation=False)

    ext = validate_external_training_usage(sources)
    blockers += ext["blockers"]
    rejected += ext["rejected_sources"]
    allowed_training_sources = ext["allowed_training_sources"]

    nontrain = validate_non_training_sources(sources)
    blockers += nontrain["blockers"]
    rejected += nontrain["rejected_sources"]

    strength = None
    if task in SUPERVISED_TASKS:
        housing = next((s for s in sources if s.get("role") == "label"
                        and s.get("source_type") == "authorized_property"), None)
        hr = validate_housing_training_data(housing)
        blockers += hr["blockers"]
        warnings += hr.get("warnings", [])
        strength = hr.get("supervised_training_strength")

    # 仅特征工程任务的提示
    if task not in SUPERVISED_TASKS:
        warnings.append({
            "item": "非监督训练任务", "severity": "info",
            "detail": f"任务 {task} 仅做特征工程/无标签流程，所有源 used_for_training 应为 false",
        })

    audit = build_data_usage_audit(
        sources, rejected_sources=rejected, warnings=warnings, blockers=blockers
    )

    can_train = len(blockers) == 0
    status = "pass" if can_train else "fail"
    test_used = audit["test_used_for_training"]

    if not can_train and raise_on_violation:
        logger.error("training guard FAIL for task=%s: %s", task, blockers)
        raise TrainingGuardError(f"训练护栏未通过（task={task}）：{blockers}")

    result = {
        "status": status,
        "can_train": can_train,
        "training_task": task,
        "requested_splits": requested_splits,
        "allowed_training_sources": allowed_training_sources,
        "rejected_sources": rejected,
        "warnings": warnings,
        "blockers": blockers,
        "data_usage_audit": audit,
        "test_used_for_training": test_used,
        "supervised_training_strength": strength,
        "dry_run": bool(req.get("dry_run", True)),
        "generated_at": _utcnow_iso(),
        "policy": "train 训练 / val 选模型 / test 仅最终评估；POI/政策/案例/统计/公共数据不进监督训练；"
                  "仅授权+脱敏+上海确认+>=1000 房价可训练。",
    }
    return result


def assert_training_allowed(db: Session, req: dict[str, Any]) -> dict[str, Any]:
    """真实训练入口必须调用：fail 则抛 TrainingGuardError，pass 返回 guard 结果。

    第11 T3 实现真实房价训练时，须在任何 fit() 之前调用本函数。
    """
    return validate_training_request(db, req, raise_on_violation=True)


def build_training_readiness(db: Session) -> dict[str, Any]:
    """训练就绪概览：默认房价监督训练请求的 guard 结果 + 第10C.5 readiness 引用。"""
    from app.services import external_data_collector_service as collector

    default_req = {
        "training_task": "housing_price_regression",
        "project_id": 1,
        "use_authorized_property": True,
        "use_poi_features": True,
        "use_policy_as_label": False,
        "requested_splits": ["train", "val"],
        "dry_run": True,
    }
    guard = validate_training_request(db, default_req, raise_on_violation=False)
    try:
        readiness = collector.build_phase11_readiness(db)
    except Exception as exc:  # noqa: BLE001
        logger.warning("phase11 readiness failed: %s", exc)
        readiness = {"error": "phase11_readiness_unavailable"}

    return {
        "phase": "11-T1",
        "generated_at": _utcnow_iso(),
        "guard": guard,
        "phase11_readiness": readiness,
        "next_step": "guard pass 后方可进入 T2（POI/圈层特征工程）与 T3（房价监督训练）。",
    }
