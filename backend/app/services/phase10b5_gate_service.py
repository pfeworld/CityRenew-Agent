"""第10B.5阶段：外部数据增强门禁（只读评估）。

确认高德合规 POI 增强后的数据量、类别覆盖、数据资产、合规与 git 安全是否达标，
并把"非高德能解决的缺口"（人口收入/政策/房价训练样本）如实标为 warning（不阻塞 10B.5）。

红线：仅读取本地 manifest/store/processed/血缘与 .gitignore；不调用任何外部 API、不采集数据、
不绕 WAF、不爬未授权商业数据；输出仅含统计量与脱敏结论。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.services import amap_service
from app.services import external_data_collector_service as collector
from app.services import manual_import_service as mis

logger = logging.getLogger("cityrenew.phase10b5_gate")

ST_PASS = "pass"
ST_WARNING = "warning"
ST_FAIL = "fail"

AMAP_TARGET = 50000
AMAP_WARN_MIN = 30000
# 五大强制类别最低目标（中文名→内部 key 见 amap_service.CATEGORY_CN）
MANDATORY_TARGETS = {
    "public_service": 8000, "commercial_consumption": 10000, "transport": 6000,
    "industry_office": 8000, "culture_sports": 6000,
}
URBAN_WARN_MIN = 500  # 城市更新专项 >=500 即 warning/pass；<3000 但 natural_sparse 不阻塞

REQUIRED_ASSETS = (
    "amap_poi_dedup_latest.jsonl", "amap_poi_category_summary.json",
    "amap_poi_spatial_summary.json", "amap_poi_quality_report.json",
)
GITIGNORE_REQUIRED = {
    "backend/.env": (".env", "backend/.env"),
    "backend/data/external/": ("backend/data/external/",),
    "backend/data/outputs/": ("backend/data/outputs/", "backend/data/outputs/data_catalog/"),
}


def _check(name: str, status: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail}


def _cn(key: str) -> str:
    return {v: k for k, v in amap_service.CATEGORY_CN.items()}.get(key, key)


def _project_root() -> Path:
    return settings.data_dir.parent.parent


def _read_gitignore_lines() -> set[str]:
    path = _project_root() / ".gitignore"
    if not path.exists():
        return set()
    out: set[str] = set()
    for raw in path.read_text("utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def _amap_volume_gate(man: dict[str, Any]) -> dict[str, Any]:
    total = int(man.get("merged_dedup_total", man.get("record_count", 0)))
    if total >= AMAP_TARGET:
        status = ST_PASS
    elif total >= AMAP_WARN_MIN:
        status = ST_WARNING
    else:
        status = ST_FAIL
    return {"status": status, "merged_dedup_total": total,
            "checks": [_check("merged_dedup_total>=50000", status,
                              f"merged_dedup_total={total}（>=50000 pass / 30000-50000 warning / <30000 fail）")]}


def _category_gate(man: dict[str, Any]) -> dict[str, Any]:
    cat = man.get("category_after") or man.get("category_summary") or {}
    sparse = set(man.get("natural_sparse_categories") or [])
    checks: list[dict[str, Any]] = []
    blocking_fail = False
    for key, tgt in MANDATORY_TARGETS.items():
        cur = int(cat.get(key, 0))
        if cur >= tgt:
            st = ST_PASS
        elif cur >= int(tgt * 0.95):  # 近达标（95%内）记 warning，不硬阻塞
            st = ST_WARNING
        else:
            st = ST_FAIL
            blocking_fail = True
        checks.append(_check(f"{_cn(key)}>={tgt}", st, f"{_cn(key)}={cur}/{tgt}"))
    # 城市更新专项：>=500 warning/pass；<3000 但 natural_sparse=true 不阻塞
    urban = int(cat.get("urban_renewal", 0))
    urban_sparse = "urban_renewal" in sparse
    if urban >= 3000:
        ust = ST_PASS
    elif urban >= URBAN_WARN_MIN:
        ust = ST_WARNING
    else:
        ust = ST_WARNING if urban_sparse else ST_FAIL
        if not urban_sparse:
            blocking_fail = True
    checks.append(_check("城市更新专项类>=500(自然稀疏不阻塞)", ust,
                         f"城市更新专项={urban}/3000，natural_sparse={urban_sparse}"))
    status = ST_FAIL if blocking_fail else (
        ST_WARNING if any(c["status"] == ST_WARNING for c in checks) else ST_PASS)
    return {"status": status, "checks": checks, "category_after": cat,
            "urban_renewal_natural_sparse": urban_sparse}


def _asset_gate(man: dict[str, Any]) -> dict[str, Any]:
    proc = settings.data_dir / "external" / "amap" / "processed"
    checks: list[dict[str, Any]] = []
    for fn in REQUIRED_ASSETS:
        exists = (proc / fn).exists()
        checks.append(_check(f"asset::{fn}", ST_PASS if exists else ST_FAIL,
                             "存在" if exists else "缺失"))
    manifest_exists = (settings.data_dir / "external" / "amap" / "manifest.json").exists()
    checks.append(_check("manifest 存在", ST_PASS if manifest_exists else ST_FAIL,
                         "amap/manifest.json"))
    lineage_ok = bool(man.get("lineage_ids"))
    checks.append(_check("lineage 存在", ST_PASS if lineage_ok else ST_FAIL,
                         f"lineage_ids={len(man.get('lineage_ids') or [])}"))
    status = ST_FAIL if any(c["status"] == ST_FAIL for c in checks) else ST_PASS
    return {"status": status, "checks": checks}


def _compliance_gate(man: dict[str, Any]) -> dict[str, Any]:
    checks = [
        _check("used_for_training=false", ST_PASS if man.get("used_for_training") is False else ST_FAIL,
               f"used_for_training={man.get('used_for_training')}"),
        _check("used_for_report=true", ST_PASS if man.get("used_for_report") else ST_FAIL,
               f"used_for_report={man.get('used_for_report')}"),
        _check("used_for_feature_engineering=true",
               ST_PASS if man.get("used_for_feature_engineering") else ST_FAIL,
               f"used_for_feature_engineering={man.get('used_for_feature_engineering')}"),
        _check("test_contamination_risk=false",
               ST_PASS if man.get("test_contamination_risk") is False else ST_FAIL,
               f"test_contamination_risk={man.get('test_contamination_risk')}"),
        _check("leakage_risk=false", ST_PASS if man.get("leakage_risk") is False else ST_FAIL,
               f"leakage_risk={man.get('leakage_risk')}"),
        _check("no_waf_bypass", ST_PASS, "上海公共数据遇 WAF 不绕过，仅人工导入（设计保证）"),
        _check("no_unauthorized_commercial_scraping", ST_PASS,
               "不爬未授权商业房产站/企业数据库（registry Level0 拦截，设计保证）"),
    ]
    status = ST_FAIL if any(c["status"] == ST_FAIL for c in checks) else ST_PASS
    return {"status": status, "checks": checks}


def _git_gate(gitignore_lines: set[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for target, patterns in GITIGNORE_REQUIRED.items():
        covered = any(p in gitignore_lines for p in patterns)
        checks.append(_check(f"gitignore::{target}", ST_PASS if covered else ST_FAIL,
                             f"覆盖模式之一={patterns}" if covered else "未覆盖"))
    status = ST_FAIL if any(c["status"] == ST_FAIL for c in checks) else ST_PASS
    return {"status": status, "checks": checks,
            "note": "git status 是否含数据/密钥由自测命令 git status 复核（数据/输出/.env 均在 gitignore 覆盖内）。"}


def _non_amap_gap_gate() -> dict[str, Any]:
    sh = mis.section_status("shanghai_open_data")
    stats = mis.section_status("stats_cn")
    policy = mis.section_status("planning_policy")
    prop = mis.section_status("authorized_property")
    checks = [
        _check("上海公共数据 imported", ST_WARNING, f"imported_record={sh['imported_record_count']}（人工下载，非阻塞）"),
        _check("统计局/年鉴 imported", ST_WARNING, f"imported_record={stats['imported_record_count']}（人工下载，非阻塞）"),
        _check("政府规划/政策 imported", ST_WARNING, f"imported_document={policy['imported_document_count']}（人工下载，非阻塞）"),
        _check("授权房价 imported", ST_WARNING,
               f"trainable={prop['can_use_for_training_count']}（授权上传，非阻塞）"),
    ]
    return {
        "status": ST_WARNING, "checks": checks,
        "notes": [
            "高德已补：POI 空间点位（公共服务/商业/交通/产业/文体/部分城市更新）。",
            "人口收入仍需统计局/统计年鉴人工下载导入（高德补不了）。",
            "政策规划仍需政府公开文件人工导入（高德补不了）。",
            "房价训练样本仍需授权脱敏上传（禁止爬商业房产站，高德补不了）。",
            "以上为非高德能解决的数据，对第10B.5 不阻塞；进入第11 监督训练前需补齐。",
        ],
    }


def run_phase10b5_gate(db: Session) -> dict[str, Any]:
    """执行第10B.5 外部数据增强门禁（只读，不调用外部 API、不采集）。"""
    man = collector._read_manifest("amap") or {}

    volume = _amap_volume_gate(man)
    category = _category_gate(man)
    asset = _asset_gate(man)
    compliance = _compliance_gate(man)
    gitignore_lines = _read_gitignore_lines()
    git = _git_gate(gitignore_lines)
    non_amap = _non_amap_gap_gate()

    gates = {
        "amap_volume_gate": volume, "category_coverage_gate": category,
        "data_asset_gate": asset, "compliance_gate": compliance,
        "git_safety_gate": git, "non_amap_gap_gate": non_amap,
    }
    # 阻塞性 gate（非高德缺口门禁恒 warning，不阻塞）
    blocking = (volume, category, asset, compliance, git)
    any_fail = any(g["status"] == ST_FAIL for g in blocking)
    any_warn = any(g["status"] == ST_WARNING for g in gates.values())

    if any_fail:
        overall = ST_FAIL
    elif any_warn:
        overall = ST_WARNING
    else:
        overall = ST_PASS
    can_enter = not any_fail

    pass_items = [k for k, g in gates.items() if g["status"] == ST_PASS]
    warning_items = [k for k, g in gates.items() if g["status"] == ST_WARNING]
    fail_items = [k for k, g in gates.items() if g["status"] == ST_FAIL]

    recommend_commit = overall in (ST_PASS, ST_WARNING) and not any_fail
    total = volume["merged_dedup_total"]
    commit_msg = (f"feat(phase10b): AMAP compliant POI enhancement ({total} dedup) "
                  f"+ data assets/gap-analysis/missing-data-plan + phase10b.5 gate")

    return {
        "mode": settings.app_mode, "phase": "10B.5",
        "overall_status": overall, "can_enter_next_stage": can_enter,
        "merged_dedup_total": total,
        "amap_volume_gate": volume, "category_coverage_gate": category,
        "data_asset_gate": asset, "compliance_gate": compliance,
        "git_safety_gate": git, "non_amap_gap_gate": non_amap,
        "pass_items": pass_items, "warning_items": warning_items, "fail_items": fail_items,
        "recommend_commit": recommend_commit,
        "recommended_commit_message": commit_msg if recommend_commit else None,
        "used_for_training": False, "test_contamination_risk": False, "leakage_risk": False,
        "notes": [
            "本门禁只读本地 manifest/store/processed/.gitignore，未调用外部 API、未采集数据。",
            "非高德缺口（人口收入/政策/房价训练样本）为 warning，不阻塞第10B.5；进入第11 前需补齐。",
            "城市更新专项类天然稀疏（关键词命中 POI 少），不伪造、不阻塞。",
        ],
    }
