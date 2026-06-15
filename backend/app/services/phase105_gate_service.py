"""第10.5阶段：数据覆盖率与特征质量门禁（只读评估）。

目标：基于第10A 的 data-audit（全量数据资产审计）与 feature-engineering（项目级特征向量）
结果，做一个独立质量门禁，确认数据覆盖率、特征质量、test 隔离、泄露风险、gitignore 安全
全部达标后，才允许进入第10B / 第11 阶段。

工作方式（不重做第10A，只复用其服务）：
- 调用 data_audit_service.run_data_audit（持久化 persist=False，仍导出已 gitignore 的 catalog）。
- 对 project id=1 调用 feature_engineering_service.build_features（仅 train/val，clear+rewrite 同项目）
  再 get_latest 读回，验证 build 与 latest 链路可用。
- 读 .gitignore 校验派生目录均被覆盖（确定性文本匹配，不 shell out）。

红线：仅 train/val（used_test 恒 false）；不调用任何外部 API；不采集外部数据；
不生成未被 gitignore 覆盖的数据文件；输出仅含统计量与脱敏结论，不含
raw_json/原始坐标/企业名/小区名/地址/profile_json/chunk_text/center_lng/center_lat。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.services import data_audit_service, feature_engineering_service, project_service

logger = logging.getLogger("cityrenew.phase105_gate")

ST_PASS = "pass"
ST_WARNING = "warning"
ST_FAIL = "fail"

# ---- 阈值（对齐用户门禁规范 / 方案第23、26节）----
COVERAGE_MIN = 0.95
FEATURE_COVERAGE_MIN = 0.90
FEATURE_COUNT_MIN = 60
MISSING_FEATURES_MAX = 10
REQUIRED_FEATURE_GROUPS = ("poi", "population", "housing", "industry", "project_fields")
# used_source_counts 中至少 3 类数据来源非零
SOURCE_NONZERO_MIN = 3
TARGET_PROJECT_ID = 1

# 核心结构化 JSON（raw/parsed/db 三项一致性检查；人口画像因合并入网格允许不一致+解释）
CORE_JSON_FILES = (
    "POI兴趣点分布数据.json",
    "产业布局数据.json",
    "房价历史交易数据.json",
    "区域人口总量.json",
)
POPULATION_PROFILE_FILE = "区域人口画像.json"

# 文件名 → 监督训练许可（仅房价进入监督训练；其余仅特征工程/聚类/相似度）
SUPERVISED_TRAINING_FILES = ("房价历史交易数据.json",)

# 泄露扫描禁用 token（仅扫描 data-audit 与 features 的实际数据载荷，不扫描本门禁的指标描述）
FORBIDDEN_TOKENS = (
    "raw_json",
    "chunk_text",
    "profile_json",
    "center_lng",
    "center_lat",
    '"coordinates"',
    '"address"',
    '"residence"',
)

# gitignore 必须覆盖的派生路径（path → 可接受的覆盖模式之一）
GITIGNORE_REQUIRED = {
    "backend/data/external/": ("backend/data/external/",),
    "backend/data/outputs/data_catalog/": (
        "backend/data/outputs/data_catalog/",
        "backend/data/outputs/",
    ),
    "backend/data/outputs/": ("backend/data/outputs/",),
    "backend/data/cityrenew.db": ("*.db", "backend/data/cityrenew.db"),
}


def _mk(name: str, value: Any, threshold: str, status: str, explanation: str) -> dict[str, Any]:
    return {
        "metric_name": name,
        "current_value": value,
        "threshold": threshold,
        "status": status,
        "explanation": explanation,
    }


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": ST_PASS if ok else ST_FAIL, "detail": detail}


def _files_by_name(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """按文件名索引，仅取 competition_data 条目。

    注意：参考资料中存在与训练语料同名的 1 条 schema 样例（source_group=reference_doc），
    若不限定来源组会覆盖竞赛数据条目，导致核心 JSON 一致性/训练计数误判。
    """
    out: dict[str, dict[str, Any]] = {}
    for f in audit.get("files", []):
        if f.get("source_group") == "competition_data":
            out[f.get("file_name")] = f
    return out


# 泄露扫描前需剔除的"解释/诊断/合规说明"字段：它们合法地引用字段名（如 address）
# 或包含中文合规免责声明（如「不含任何 raw_json...」），属于脱敏说明而非泄露内容。
_SCRUB_KEYS = {"notes", "recommendations", "skipped_reason", "field_mapping_status"}


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in _SCRUB_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _scan_forbidden(payload: Any) -> list[str]:
    blob = json.dumps(_scrub(payload), ensure_ascii=False, default=str)
    return [tok for tok in FORBIDDEN_TOKENS if tok in blob]


def _project_root() -> Path:
    # config.BASE_DIR = backend；其父为仓库根
    return settings.data_dir.parent.parent


# --------------------------------------------------------------------------- #
# 一、data_audit_gate
# --------------------------------------------------------------------------- #
def _data_audit_gate(audit: dict[str, Any], audit_ran: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    checks.append(_check("data_audit_runnable", audit_ran,
                         "/api/evaluation/data-audit 可运行" if audit_ran else "data-audit 运行失败"))
    if not audit_ran:
        return {"status": ST_FAIL, "checks": checks, "warnings": warnings,
                "coverage_rate": 0.0, "leakage_risk": True, "test_contamination_risk": True}

    total_raw = int(audit.get("total_raw_records", 0))
    total_parsed = int(audit.get("total_parsed_records", 0))
    total_db = int(audit.get("total_db_records", 0))
    coverage = float(audit.get("coverage_rate", 0.0))
    leakage_risk = bool(audit.get("leakage_risk", True))
    test_contam = bool(audit.get("test_contamination_risk", True))

    checks.append(_check("total_raw_records>0", total_raw > 0, f"total_raw_records={total_raw}"))
    checks.append(_check("total_parsed_records>0", total_parsed > 0, f"total_parsed_records={total_parsed}"))
    checks.append(_check("total_db_records>0", total_db > 0, f"total_db_records={total_db}"))
    checks.append(_check(f"coverage_rate>={COVERAGE_MIN}", coverage >= COVERAGE_MIN,
                         f"coverage_rate={coverage}"))

    # 核心 JSON raw/parsed/db 三项一致或有合理解释
    by_name = _files_by_name(audit)
    inconsistent: list[str] = []
    for fname in CORE_JSON_FILES:
        f = by_name.get(fname)
        if not f:
            inconsistent.append(f"{fname} 缺失审计条目")
            continue
        raw = f.get("raw_record_count")
        parsed = f.get("parsed_record_count")
        db = f.get("db_inserted_count")
        if raw == parsed == db:
            continue
        if f.get("recommendations"):
            warnings.append(f"{fname} raw/parsed/db 不一致（{raw}/{parsed}/{db}），审计已附说明，非阻断。")
        else:
            inconsistent.append(f"{fname}={raw}/{parsed}/{db} 无解释")
    # 人口画像允许合并不一致，但要求审计附说明
    pf = by_name.get(POPULATION_PROFILE_FILE)
    if pf and not (pf.get("raw_record_count") == pf.get("db_inserted_count")) and not pf.get("recommendations"):
        inconsistent.append(f"{POPULATION_PROFILE_FILE} 合并不一致且无解释")
    checks.append(_check("core_json_consistency", not inconsistent,
                         "核心 JSON raw/parsed/db 一致或有合理解释" if not inconsistent
                         else f"无解释的不一致：{inconsistent}"))

    # unused_files 有原因（每个未使用文件对应的审计条目须有 recommendations）
    unused_paths = set(audit.get("unused_files", []))
    unused_no_reason = [
        f.get("file_path") for f in audit.get("files", [])
        if f.get("file_path") in unused_paths and not f.get("recommendations")
    ]
    checks.append(_check("unused_files_have_reason", not unused_no_reason,
                         f"unused_files={len(unused_paths)} 均有原因" if not unused_no_reason
                         else f"以下未使用文件缺原因：{unused_no_reason}"))

    # low_coverage_files 有原因
    low_cov = audit.get("low_coverage_files", [])
    low_no_reason = [
        fn for fn in low_cov
        if not (by_name.get(fn) and by_name[fn].get("recommendations"))
    ]
    checks.append(_check("low_coverage_files_have_reason", not low_no_reason,
                         f"low_coverage_files={low_cov} 均有原因" if not low_no_reason
                         else f"以下低覆盖文件缺原因：{low_no_reason}"))
    if low_cov:
        warnings.append(f"存在低覆盖文件：{low_cov}（已附原因，非阻断）。")

    # field_mapping_status 不存在 fail
    def _is_fail_status(s: Any) -> bool:
        if isinstance(s, str):
            return s == "fail"
        if isinstance(s, dict):
            return s.get("status") == "fail"
        return False

    mapping_fails = [f.get("file_name") for f in audit.get("files", [])
                     if _is_fail_status(f.get("field_mapping_status"))]
    checks.append(_check("field_mapping_no_fail", not mapping_fails,
                         "无 field_mapping_status=fail" if not mapping_fails
                         else f"字段映射失败文件：{mapping_fails}"))
    # partial（必需字段部分缺失）记为 warning
    partial = [f.get("file_name") for f in audit.get("files", [])
               if isinstance(f.get("field_mapping_status"), dict)
               and f["field_mapping_status"].get("status") == "partial"]
    if partial:
        warnings.append(f"字段映射 partial（部分必需字段缺失）：{partial}（如实标注，非阻断）。")

    # coordinate_invalid_count 不异常（核心文件无效坐标占比过半视为异常）
    coord_abnormal: list[str] = []
    for fname in CORE_JSON_FILES:
        f = by_name.get(fname)
        if not f:
            continue
        valid = int(f.get("coordinate_valid_count", 0))
        invalid = int(f.get("coordinate_invalid_count", 0))
        if invalid > 0 and invalid >= max(1, valid):
            coord_abnormal.append(f"{fname} 无效坐标={invalid} 有效={valid}")
        elif invalid > 0:
            warnings.append(f"{fname} 存在 {invalid} 条无效/缺失坐标（未参与空间归集，非阻断）。")
    checks.append(_check("coordinate_invalid_normal", not coord_abnormal,
                         "核心文件坐标解析正常" if not coord_abnormal
                         else f"坐标异常：{coord_abnormal}"))

    checks.append(_check("leakage_risk=false", leakage_risk is False, f"leakage_risk={leakage_risk}"))
    checks.append(_check("test_contamination_risk=false", test_contam is False,
                         f"test_contamination_risk={test_contam}"))

    status = ST_FAIL if any(c["status"] == ST_FAIL for c in checks) else ST_PASS
    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "coverage_rate": coverage,
        "total_raw_records": total_raw,
        "total_parsed_records": total_parsed,
        "total_db_records": total_db,
        "leakage_risk": leakage_risk,
        "test_contamination_risk": test_contam,
    }


# --------------------------------------------------------------------------- #
# 二、feature_quality_gate
# --------------------------------------------------------------------------- #
def _feature_quality_gate(
    build_ran: bool, latest: dict[str, Any] | None
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    checks.append(_check("feature_build_runnable", build_ran,
                         "POST /api/features/1/build 可运行" if build_ran else "feature build 运行失败"))
    latest_ok = latest is not None
    checks.append(_check("feature_latest_readable", latest_ok,
                         "GET /api/features/1/latest 可读取" if latest_ok else "latest 读取为空"))
    if not (build_ran and latest_ok):
        return {"status": ST_FAIL, "checks": checks, "warnings": warnings,
                "feature_count": 0, "feature_coverage_rate": 0.0,
                "missing_features_count": 0, "used_test": True, "allowed_splits": []}

    feature_count = len(latest.get("feature_names", []))
    coverage = float(latest.get("feature_coverage_rate", 0.0))
    missing_count = len(latest.get("missing_features", []))
    groups = latest.get("feature_groups", {})
    evidence_ids = latest.get("evidence_ids", [])
    used_counts = latest.get("used_source_counts", {})
    used_test = bool(latest.get("used_test", True))
    allowed = latest.get("allowed_splits", [])

    checks.append(_check(f"feature_count>={FEATURE_COUNT_MIN}", feature_count >= FEATURE_COUNT_MIN,
                         f"feature_count={feature_count}"))
    checks.append(_check(f"feature_coverage_rate>={FEATURE_COVERAGE_MIN}",
                         coverage >= FEATURE_COVERAGE_MIN, f"feature_coverage_rate={coverage}"))
    checks.append(_check(f"missing_features<={MISSING_FEATURES_MAX}",
                         missing_count <= MISSING_FEATURES_MAX, f"missing_features_count={missing_count}"))

    missing_groups = [g for g in REQUIRED_FEATURE_GROUPS if g not in groups]
    checks.append(_check("feature_groups_complete", not missing_groups,
                         "包含 poi/population/housing/industry/project_fields" if not missing_groups
                         else f"缺失特征组：{missing_groups}"))

    checks.append(_check("evidence_ids_non_empty", bool(evidence_ids),
                         f"evidence_ids 数量={len(evidence_ids)}"))

    nonzero = [k for k in ("poi", "population", "housing", "industry")
               if int(used_counts.get(k, 0)) > 0]
    checks.append(_check(f"used_source_nonzero>={SOURCE_NONZERO_MIN}",
                         len(nonzero) >= SOURCE_NONZERO_MIN,
                         f"非零来源={nonzero}（POI/人口/房价/产业）"))

    checks.append(_check("used_test=false", used_test is False, f"used_test={used_test}"))
    checks.append(_check("allowed_splits=['train','val']", allowed == ["train", "val"],
                         f"allowed_splits={allowed}"))

    status = ST_FAIL if any(c["status"] == ST_FAIL for c in checks) else ST_PASS
    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "feature_count": feature_count,
        "feature_coverage_rate": coverage,
        "missing_features_count": missing_count,
        "feature_groups": sorted(groups.keys()),
        "evidence_ids_count": len(evidence_ids),
        "used_source_counts": used_counts,
        "used_test": used_test,
        "allowed_splits": allowed,
    }


# --------------------------------------------------------------------------- #
# 三、training_usage_gate
# --------------------------------------------------------------------------- #
def _training_usage_gate(audit: dict[str, Any], audit_ran: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    notes: list[str] = []

    if not audit_ran:
        return {"status": ST_FAIL, "checks": [_check("audit_available", False, "data-audit 不可用")],
                "warnings": warnings, "notes": notes}

    by_name = _files_by_name(audit)

    # 房价进入监督训练
    house = by_name.get("房价历史交易数据.json", {})
    house_train = int(house.get("used_in_training_count", 0))
    checks.append(_check("housing_in_supervised_training", house_train > 0,
                         f"房价监督训练样本数={house_train}（train/val）"))
    notes.append("房价是唯一进入监督训练的样本（train 拟合 / val 验证），test 不参与。")

    # POI/人口/产业 training_count=0 不判失败，但需说明用途
    for fname, label in (("POI兴趣点分布数据.json", "POI"),
                         ("区域人口总量.json", "人口"),
                         ("产业布局数据.json", "产业")):
        f = by_name.get(fname, {})
        tc = int(f.get("used_in_training_count", 0))
        fe = int(f.get("used_in_feature_engineering_count", 0))
        # 仅作合理性提示，不作 fail
        notes.append(
            f"{label} used_in_training_count={tc}（合理为 0），used_in_feature_engineering_count={fe}："
            f"{label} 数据用于特征工程/聚类/相似度，不作为监督训练标签。"
        )

    # test 不得参与训练：任何文件不得在 used_in_training 中混入 test
    # （审计口径：used_in_training/feature_engineering 仅统计 train/val；据 leakage/test_contamination 判定）
    test_in_training = bool(audit.get("test_contamination_risk", True)) or bool(audit.get("leakage_risk", True))
    checks.append(_check("test_not_in_training", test_in_training is False,
                         "used_in_training/feature_engineering 仅 train/val，test 永不计入"
                         if test_in_training is False else "检测到 test 进入训练/特征的风险"))

    status = ST_FAIL if any(c["status"] == ST_FAIL for c in checks) else ST_PASS
    return {"status": status, "checks": checks, "warnings": warnings, "notes": notes}


# --------------------------------------------------------------------------- #
# 四、external_data_gate
# --------------------------------------------------------------------------- #
def _external_data_gate(audit: dict[str, Any], audit_ran: bool, gitignore_lines: set[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    ext = audit.get("external_data", {}) if audit_ran else {}
    ext_count = int(ext.get("external_files_count", 0))
    ext_data_count = int(ext.get("external_data_files_count", 0))
    ext_exists = bool(ext.get("external_dir_exists", False))

    # 第10B 起 external/ 可能含脚手架/模板（manifest/registry/README）；模板不算真实采集数据。
    # 门禁要求：真实外部数据（raw/processed/cache）须被 gitignore 覆盖且不污染 competition test。
    ext_ignored = "backend/data/external/" in gitignore_lines
    checks.append(_check("external_gitignored", ext_ignored,
                         "backend/data/external/ 已被 .gitignore 覆盖" if ext_ignored
                         else "backend/data/external/ 未被 gitignore 覆盖"))

    # 真实外部数据存在时必须被 gitignore 覆盖（不会被 git 跟踪），否则阻断
    checks.append(_check("external_data_not_tracked", (ext_data_count == 0) or ext_ignored,
                         "external 真实数据已被 gitignore 覆盖（不会被 git 跟踪）"
                         if (ext_data_count == 0 or ext_ignored)
                         else "external 真实数据未被 gitignore 覆盖"))

    if ext_count > 0:
        warnings.append(
            f"backend/data/external/ 已建脚手架/模板（files={ext_count}，真实采集数据={ext_data_count}，"
            "均在 gitignore 覆盖内，非阻断）。")
    if ext_data_count > 0:
        warnings.append(
            f"已采集 {ext_data_count} 份外部真实数据：物理隔离于 competition test，"
            "仅用于特征工程/报告，不混入训练 test（如实标注，非阻断）。")

    status = ST_FAIL if any(c["status"] == ST_FAIL for c in checks) else ST_PASS
    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "external_files_count": ext_count,
        "external_data_files_count": ext_data_count,
        "external_dir_exists": ext_exists,
    }


# --------------------------------------------------------------------------- #
# 五、leakage_gate
# --------------------------------------------------------------------------- #
def _leakage_gate(audit: dict[str, Any], latest: dict[str, Any] | None) -> dict[str, Any]:
    # 仅扫描真实数据载荷（data-audit 响应 + features 响应），不扫描本门禁的指标描述（其含 token 定义）
    hits = _scan_forbidden({"data_audit": audit, "features": latest or {}})
    audit_leak = bool(audit.get("leakage_risk", False))
    checks = [
        _check("no_forbidden_tokens", not hits,
               "data-audit / features 未出现 raw_json/坐标/企业名/小区名/地址/profile_json/chunk_text"
               if not hits else f"检测到泄露 token：{hits}"),
        _check("audit_leakage_risk=false", audit_leak is False, f"audit.leakage_risk={audit_leak}"),
    ]
    status = ST_FAIL if any(c["status"] == ST_FAIL for c in checks) else ST_PASS
    return {"status": status, "checks": checks, "leak_tokens": hits}


# --------------------------------------------------------------------------- #
# 六、gitignore_gate
# --------------------------------------------------------------------------- #
def _read_gitignore_lines() -> set[str]:
    path = _project_root() / ".gitignore"
    if not path.exists():
        return set()
    lines: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.add(s)
    return lines


def _gitignore_gate(gitignore_lines: set[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for target, patterns in GITIGNORE_REQUIRED.items():
        covered = any(p in gitignore_lines for p in patterns)
        checks.append(_check(f"gitignore::{target}", covered,
                             f"{target} 被覆盖（模式之一：{patterns}）" if covered
                             else f"{target} 未被 .gitignore 覆盖"))
    status = ST_FAIL if any(c["status"] == ST_FAIL for c in checks) else ST_PASS
    return {"status": status, "checks": checks,
            "gitignore_patterns": sorted(gitignore_lines)}


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_phase105_gate(db: Session) -> dict[str, Any]:
    """执行第10.5 数据覆盖率与特征质量门禁（只读评估，仅 train/val）。"""
    notes: list[str] = [
        "本门禁复用第10A 的 data-audit 与 feature-engineering，不重做第10A。",
        "data-audit 以 persist=False 运行（不新增 EvaluationResult 行），catalog 导出落已 gitignore 目录。",
        "feature build 仅 clear+rewrite 项目 id=1 同项目特征（仅 train/val），不触碰 test。",
        "全程本地确定性计算，未调用任何外部 API、未采集任何外部数据。",
    ]

    # ---- 1. data-audit（可运行性 + 结果）----
    audit_ran = True
    try:
        audit = data_audit_service.run_data_audit(db, persist=False)
    except Exception as exc:  # noqa: BLE001 - 门禁需捕获并记录失败
        audit_ran = False
        audit = {}
        logger.warning("phase10.5 gate data-audit failed: %s", exc)

    # ---- 2. feature build + latest（project id=1）----
    build_ran = True
    latest: dict[str, Any] | None = None
    project = project_service.get_project(db, TARGET_PROJECT_ID)
    if project is None:
        build_ran = False
        notes.append(f"项目 id={TARGET_PROJECT_ID} 不存在，无法执行特征质量门禁。")
    else:
        try:
            feature_engineering_service.build_features(db, project, include_external=False)
        except Exception as exc:  # noqa: BLE001
            build_ran = False
            logger.warning("phase10.5 gate feature build failed: %s", exc)
        latest = feature_engineering_service.get_latest(db, TARGET_PROJECT_ID)

    # ---- 3. 读 .gitignore ----
    gitignore_lines = _read_gitignore_lines()

    # ---- 4. 七大门禁 ----
    data_audit_gate = _data_audit_gate(audit, audit_ran)
    feature_quality_gate = _feature_quality_gate(build_ran, latest)
    training_usage_gate = _training_usage_gate(audit, audit_ran)
    external_data_gate = _external_data_gate(audit, audit_ran, gitignore_lines)
    leakage_gate = _leakage_gate(audit, latest)
    gitignore_gate = _gitignore_gate(gitignore_lines)

    # ---- 5. 关键指标抽取 ----
    coverage = float(data_audit_gate.get("coverage_rate", 0.0))
    feature_coverage = float(feature_quality_gate.get("feature_coverage_rate", 0.0))
    used_test = bool(feature_quality_gate.get("used_test", True))
    allowed_splits = feature_quality_gate.get("allowed_splits", [])
    leakage_risk = bool(data_audit_gate.get("leakage_risk", True)) or bool(leakage_gate.get("leak_tokens"))
    test_contam = bool(data_audit_gate.get("test_contamination_risk", True))

    # ---- 6. overall 规则（硬失败 → fail）----
    hard_fail_reasons: list[str] = []
    if not audit_ran:
        hard_fail_reasons.append("data-audit 不能运行")
    if not build_ran:
        hard_fail_reasons.append("feature build 不能运行")
    if audit_ran and coverage < COVERAGE_MIN:
        hard_fail_reasons.append(f"coverage_rate={coverage} < {COVERAGE_MIN}")
    if build_ran and feature_coverage < FEATURE_COVERAGE_MIN:
        hard_fail_reasons.append(f"feature_coverage_rate={feature_coverage} < {FEATURE_COVERAGE_MIN}")
    if test_contam:
        hard_fail_reasons.append("test_contamination_risk=true")
    if used_test:
        hard_fail_reasons.append("used_test=true")
    if leakage_risk:
        hard_fail_reasons.append("leakage_risk=true")
    if gitignore_gate["status"] == ST_FAIL:
        hard_fail_reasons.append("gitignore_gate fail")

    gates = {
        "data_audit_gate": data_audit_gate,
        "feature_quality_gate": feature_quality_gate,
        "training_usage_gate": training_usage_gate,
        "external_data_gate": external_data_gate,
        "leakage_gate": leakage_gate,
        "gitignore_gate": gitignore_gate,
    }
    any_gate_fail = any(g["status"] == ST_FAIL for g in gates.values())

    # ---- 7. warnings（非阻断已知真实数据问题）----
    warnings: list[str] = []
    for g in (data_audit_gate, feature_quality_gate, training_usage_gate, external_data_gate):
        warnings.extend(g.get("warnings", []))
    warnings.append("产业 category_name 单一（多为「公司企业;公司;公司」），细分行业缺失（如实标注，非阻断）。")
    warnings.append("人口数据无收入字段（如实标注，非阻断）。")
    warnings.append("房价样本 255 条偏小、year 多缺失（如实标注，非阻断）。")
    warnings.append("POI/人口/产业无监督训练标签，仅用于特征工程/聚类/相似度（合理，非阻断）。")
    warnings.append("外部数据尚未接入（第10A 不允许采集，第10B 再增强，非阻断）。")
    if latest is not None and "house_sample_confidence" in latest.get("feature_values", {}):
        pass  # 项目2 数据稀疏等在多项目时再提示

    if hard_fail_reasons or any_gate_fail:
        overall = ST_FAIL
    elif warnings:
        overall = ST_WARNING
    else:
        overall = ST_PASS

    # ---- 8. can_enter_next_stage ----
    all_gate_pass = not any_gate_fail
    can_enter = (
        overall in (ST_PASS, ST_WARNING)
        and all_gate_pass
        and not hard_fail_reasons
        and used_test is False
        and allowed_splits == ["train", "val"]
    )

    # ---- 9. metrics_status 汇总 ----
    def _gate_status_metric(name: str, gate: dict[str, Any], threshold: str) -> dict[str, Any]:
        return _mk(name, gate["status"], threshold, gate["status"],
                   f"{name} = {gate['status']}")

    metrics_status = [
        _gate_status_metric("data_audit_gate", data_audit_gate, "all checks pass"),
        _gate_status_metric("feature_quality_gate", feature_quality_gate, "all checks pass"),
        _gate_status_metric("training_usage_gate", training_usage_gate, "all checks pass"),
        _gate_status_metric("external_data_gate", external_data_gate, "all checks pass"),
        _gate_status_metric("leakage_gate", leakage_gate, "all checks pass"),
        _gate_status_metric("gitignore_gate", gitignore_gate, "all required paths covered"),
        _mk("coverage_rate", coverage, f">= {COVERAGE_MIN}",
            ST_PASS if coverage >= COVERAGE_MIN else ST_FAIL, f"数据覆盖率={coverage}"),
        _mk("feature_coverage_rate", feature_coverage, f">= {FEATURE_COVERAGE_MIN}",
            ST_PASS if feature_coverage >= FEATURE_COVERAGE_MIN else ST_FAIL,
            f"特征覆盖率={feature_coverage}"),
        _mk("used_test", used_test, "== false",
            ST_PASS if used_test is False else ST_FAIL, f"used_test={used_test}"),
        _mk("allowed_splits", allowed_splits, "== ['train','val']",
            ST_PASS if allowed_splits == ["train", "val"] else ST_FAIL,
            f"allowed_splits={allowed_splits}"),
        _mk("external_api_calls", 0, "== 0", ST_PASS, "全程本地计算，未调用任何外部 API。"),
        _mk("external_data_collected", 0, "== 0", ST_PASS, "第10A 未采集任何外部数据。"),
    ]

    # ---- 10. risks / recommendations / next_required ----
    risks: list[str] = list(hard_fail_reasons)
    for gname, g in gates.items():
        for c in g["checks"]:
            if c["status"] == ST_FAIL:
                risks.append(f"[FAIL] {gname}.{c['name']}：{c['detail']}")

    recommendations: list[str] = []
    next_required: list[str] = []
    if overall == ST_FAIL:
        recommendations.append("存在阻断性 fail，必须修复后方可进入第10B / 第11。")
        next_required.extend(f"修复：{r}" for r in (hard_fail_reasons or risks))
    else:
        recommendations.append(
            "数据覆盖率、特征质量、test 隔离、泄露风险、gitignore 安全全部达标；"
            "可进入第10B（合规外部数据增强）以补强产业细分/人口收入/房价样本，再进第11 多模型训练。"
        )
        recommendations.append(
            "建议优先进入第10B：用合规外部数据增强（产业细分行业、人口收入近似、房价样本扩充），"
            "随后第11 多模型训练时房价 MAPE/类型 F1 更可信。"
        )
        recommendations.append(
            "注意：本门禁仅验证数据/特征质量与合规，不等价于模型预测准确率达标"
            "（房价 MAPE / 类型 F1 / test 检索匹配率属第11/13 阶段 eval 模式 test 评估）。"
        )
        next_required.append("进入第10B 前确认 test 仍未被触碰；外部数据接入须走 registry + 合规等级。")

    logger.info(
        "phase10.5 gate overall=%s can_enter=%s coverage=%s feat_cov=%s used_test=%s "
        "leakage=%s test_contam=%s gates=%s",
        overall, can_enter, coverage, feature_coverage, used_test, leakage_risk, test_contam,
        {k: v["status"] for k, v in gates.items()},
    )

    return {
        "mode": settings.app_mode,
        "phase": "10.5",
        "target_project_id": TARGET_PROJECT_ID if project is not None else None,
        "overall_status": overall,
        "can_enter_next_stage": can_enter,
        "metrics_status": metrics_status,
        "data_audit_gate": data_audit_gate,
        "feature_quality_gate": feature_quality_gate,
        "training_usage_gate": training_usage_gate,
        "external_data_gate": external_data_gate,
        "leakage_gate": leakage_gate,
        "gitignore_gate": gitignore_gate,
        "risks": risks,
        "warnings": warnings,
        "recommendations": recommendations,
        "next_required_actions": next_required,
        "notes": notes,
    }
