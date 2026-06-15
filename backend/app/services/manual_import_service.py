"""合规人工下载导入服务（第10B 升级版）。

用于把用户**人工**从官方/公开渠道下载的文件（上海公共数据无条件开放、统计局/统计年鉴、
政府规划/公告政策、授权房价）导入系统，做结构化统计与血缘登记。

红线：
- 不绕反爬/不绕验证码/不绕登录/不抓隐藏接口/不模拟 WAF/不提取浏览器 Cookie；
- 仅导入用户人工放入 ``manual_uploads/`` 的文件，且必须填写 source_url 与开放/授权类型；
- 外部数据默认 ``used_for_training=false``；仅 ``license_status=authorized`` 且带授权证明的
  房价/企业数据可置 can_use_for_training=true（仍须过第11数据门禁）；
- 不混入 competition_test；接口只返回统计量/字段schema/路径，不返回原始明细；
- 文件均在 ``backend/data/external/`` 下（已 gitignore），不提交 git；失败记 failed_reason，不伪造。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services import data_lineage_service

logger = logging.getLogger("cityrenew.manual_import")

_DATA_FORMATS = {"csv", "json", "xlsx", "xml", "rdf"}
_DOC_FORMATS = {"pdf", "doc", "docx", "html", "htm", "txt"}

# 各分区导入配置：source_type / 默认许可 / 必填字段 / 推荐下载清单 / guide 说明
SECTION_CONFIGS: dict[str, dict[str, Any]] = {
    "shanghai_open_data": {
        "source_type": "shanghai_open_data", "provider": "上海市公共数据开放平台",
        "default_license": "open_unconditional", "used_for_rag": False,
        "required": ["dataset_name", "source_url", "file_path", "file_format"],
        "require_open_type": "无条件开放",
        "portal": "https://data.sh.gov.cn",
        "recommended": ["上海市医疗机构名录", "上海市学校名录", "上海市养老服务机构",
                        "上海市公共交通站点", "上海市文化体育设施", "上海市人口/经济统计指标"],
        "guide_notes": "门户对脚本访问启用 412/WAF/JS 反爬，按红线不绕过；请在浏览器人工下载"
                       "『无条件开放』数据集的 CSV/JSON/XLSX，放入 manual_uploads 后导入。",
    },
    "stats_cn": {
        "source_type": "gov_statistics", "provider": "国家统计局 / 上海统计年鉴",
        "default_license": "public_open_data", "used_for_rag": False,
        "required": ["dataset_name", "source_url", "file_path", "file_format"],
        "extra_meta": ["year", "region", "indicator_group"],
        "portal": "https://www.stats.gov.cn / https://tjj.sh.gov.cn",
        "recommended": ["分区县常住人口", "居民人均可支配收入", "社会消费品零售总额", "CPI居民消费价格指数",
                        "城镇登记失业率/就业", "地区生产总值GDP", "三次产业结构", "固定资产投资",
                        "房地产开发与销售宏观指标"],
        "guide_notes": "优先下载公开 Excel/CSV/HTML 表格；未被反爬阻断时可小范围公开页面采集；"
                       "禁止绕反爬/验证码/破解接口/高频爬取/伪造指标。",
    },
    "planning_policy": {
        "source_type": "planning_policy", "provider": "政府公开规划/公告/政策",
        "default_license": "public_gov_document", "used_for_rag": True,
        "required": ["document_title", "source_url", "file_path", "file_format"],
        "extra_meta": ["publish_date", "region", "policy_type"],
        "portal": "上海市政府/区政府/规划资源局/住建/公共资源交易等官方公开栏目",
        "recommended": ["城市更新政策/管理办法", "控规公示", "土地出让公告", "区级国土空间规划",
                        "产业政策", "政府工作报告", "城市更新项目公告", "专项规划"],
        "guide_notes": "仅下载官方公开发布的 PDF/Word/HTML；写入 RAG/report/evidence；"
                       "禁止绕登录/验证码/反爬、禁止扒取非公开数据、禁止伪造来源。",
    },
    "authorized_property": {
        "source_type": "property", "provider": "授权房价/成交数据",
        "default_license": "needs_review", "used_for_rag": False,
        "required": ["file_path", "file_format", "data_owner", "license_status", "allowed_usage"],
        "extra_meta": ["privacy_level", "source_url_or_contract", "is_desensitized", "authorization_proof"],
        "portal": "用户上传授权文件 / 政府公开成交统计 / 已采购或客户授权脱敏样本",
        "recommended": ["授权脱敏房价成交样本", "政府公开商品房成交统计", "区级二手房成交均价"],
        "guide_notes": "默认不爬链家/贝壳/安居客/房天下/诸葛找房；license_status!=authorized 时 "
                       "can_use_for_training 强制 false；authorized 且带 authorization_proof 方可训练，"
                       "且须过第11数据门禁；禁止采集个人隐私、禁止伪造样本。",
    },
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sec_dir(section: str) -> Path:
    return settings.data_dir / "external" / section


def _manual_dir(section: str) -> Path:
    return _sec_dir(section) / "manual_uploads"


def _manifest_input_path(section: str) -> Path:
    return _sec_dir(section) / "import_manifest_input.json"


def ensure_section_dirs(section: str) -> None:
    base = _sec_dir(section)
    for d in ("manual_uploads", "raw", "processed"):
        (base / d).mkdir(parents=True, exist_ok=True)


def _write_input_template(section: str) -> str:
    """若不存在则写一份导入清单模板（仅模板，不含真实数据）。"""
    cfg = SECTION_CONFIGS[section]
    path = _manifest_input_path(section)
    if not path.exists():
        sample: dict[str, Any] = {f: "" for f in cfg["required"]}
        for f in cfg.get("extra_meta", []):
            sample[f] = ""
        sample.update({"used_for_feature_engineering": True, "used_for_report": True,
                       "used_for_training": False, "used_for_eval": False})
        if section == "shanghai_open_data":
            sample["open_type"] = "无条件开放"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([sample], f, ensure_ascii=False, indent=2)
    return str(path.relative_to(settings.data_dir.parent))


def guide(section: str) -> dict[str, Any]:
    """返回某分区的人工下载/导入指南，并确保目录与导入清单模板就绪。"""
    cfg = SECTION_CONFIGS[section]
    ensure_section_dirs(section)
    template_rel = _write_input_template(section)
    existing = sorted(p.name for p in _manual_dir(section).iterdir()
                      if p.is_file() and not p.name.startswith(".")) if _manual_dir(section).exists() else []
    status = "ready_to_import" if existing else "waiting_for_manual_upload"
    return {
        "section": section, "provider": cfg["provider"], "portal": cfg.get("portal", ""),
        "manual_uploads_dir": str(_manual_dir(section).relative_to(settings.data_dir.parent)),
        "import_manifest_input": template_rel,
        "required_fields": cfg["required"], "extra_meta": cfg.get("extra_meta", []),
        "supported_formats": sorted(_DATA_FORMATS | _DOC_FORMATS),
        "recommended_downloads": cfg.get("recommended", []),
        "existing_files": existing, "status": status,
        "notes": [cfg["guide_notes"],
                  "外部数据默认仅用于 feature_engineering / report，不进入监督训练、不混入 competition_test。",
                  "目录在 backend/data/external/ 下，已 gitignore，不会提交 git。"],
    }


def section_status(section: str) -> dict[str, Any]:
    """某分区人工导入现状（脱敏统计量）：上传文件数 / 清单是否配置 / 已导入数据集与记录数。"""
    cfg = SECTION_CONFIGS.get(section, {})
    md = _manual_dir(section)
    files = sorted(p.name for p in md.iterdir()
                   if p.is_file() and not p.name.startswith(".")) if md.exists() else []
    # 导入清单是否已配置（存在且至少一行填了非空 file_path）
    input_path = _manifest_input_path(section)
    input_configured = False
    if input_path.exists():
        try:
            entries = json.loads(input_path.read_text("utf-8"))
            input_configured = any(str((e or {}).get("file_path", "")).strip()
                                   for e in (entries if isinstance(entries, list) else []))
        except Exception:  # noqa: BLE001
            input_configured = False
    manifest = _sec_dir(section) / "manifest.json"
    man: dict[str, Any] = {}
    if manifest.exists():
        try:
            man = json.loads(manifest.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            man = {}
    datasets = man.get("datasets") or []
    is_doc = cfg.get("used_for_rag") or section == "planning_policy"
    trainable = sum(int(d.get("cleaned_count", 0) or 0) for d in datasets
                    if d.get("license_status") == "authorized" and d.get("used_for_training"))
    return {
        "section": section,
        "manual_uploads_file_count": len(files),
        "manual_uploads_files": files[:50],
        "import_manifest_input_configured": input_configured,
        "imported_dataset_count": int(man.get("imported_count", 0)),
        "imported_record_count": int(man.get("record_count", 0)),
        "imported_document_count": int(man.get("imported_count", 0)) if is_doc else 0,
        "license_status": man.get("license_status", cfg.get("default_license")),
        "has_authorized": any(d.get("license_status") == "authorized" for d in datasets),
        "has_authorization_proof": any(d.get("authorization_proof") for d in datasets),
        "can_use_for_training_count": trainable,
        "recommended_downloads": cfg.get("recommended", []),
        "portal": cfg.get("portal", ""),
    }


def _parse_file(file_path: Path, fmt: str) -> dict[str, Any]:
    """解析文件，返回 raw_count/parsed_count/cleaned_count/field_schema/char_count（仅统计量）。"""
    fmt = fmt.lower()
    if fmt == "json":
        obj = json.loads(file_path.read_text("utf-8", "ignore"))
        rows = obj if isinstance(obj, list) else (obj.get("data") or obj.get("records") or [])
        rows = rows if isinstance(rows, list) else []
        schema = sorted({k for r in rows[:80] if isinstance(r, dict) for k in r})[:60]
        return {"raw_count": len(rows), "parsed_count": len(rows), "cleaned_count": len(rows),
                "field_schema": schema}
    if fmt == "csv":
        text = file_path.read_text("utf-8-sig", "ignore")
        rows = list(csv.reader(io.StringIO(text)))
        header = rows[0] if rows else []
        n = max(0, len(rows) - 1)
        return {"raw_count": n, "parsed_count": n, "cleaned_count": n, "field_schema": header[:60]}
    if fmt == "xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        header = [str(c) for c in rows[0]] if rows else []
        n = max(0, len(rows) - 1)
        return {"raw_count": n, "parsed_count": n, "cleaned_count": n, "field_schema": header[:60]}
    if fmt in ("xml", "rdf"):
        root = ET.fromstring(file_path.read_text("utf-8", "ignore"))
        children = list(root)
        tags = sorted({c.tag.split('}')[-1] for c in children})[:60]
        n = len(children)
        return {"raw_count": n, "parsed_count": n, "cleaned_count": n, "field_schema": tags}
    if fmt in _DOC_FORMATS:  # 文档类：登记为 1 篇，仅记字符数，不存全文
        try:
            size = file_path.stat().st_size
        except Exception:  # noqa: BLE001
            size = 0
        return {"raw_count": 1, "parsed_count": 1, "cleaned_count": 1, "field_schema": [],
                "char_count": size, "is_document": True}
    raise ValueError(f"不支持的格式：{fmt}")


def _resolve_path(file_path: str) -> Path:
    p = Path(file_path)
    if p.is_absolute():
        return p
    # 允许相对项目根 或 仅文件名（落在该分区 manual_uploads）
    cand = settings.data_dir.parent / file_path
    return cand


def import_manual(section: str, entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """导入人工下载文件；entries 为空时读取该分区 import_manifest_input.json。"""
    if section not in SECTION_CONFIGS:
        return {"status": "error", "failed_reason": f"未知分区 {section}"}
    cfg = SECTION_CONFIGS[section]
    ensure_section_dirs(section)
    _write_input_template(section)

    if entries is None:
        ip = _manifest_input_path(section)
        if ip.exists():
            try:
                entries = json.loads(ip.read_text("utf-8"))
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "failed_reason": f"导入清单解析失败：{exc}"}
        else:
            entries = []
    entries = [e for e in (entries or []) if isinstance(e, dict)
               and any(str(e.get(k, "")).strip() for k in cfg["required"])]

    if not entries:
        guide_payload = guide(section)
        return {"section": section, "status": "waiting_for_manual_upload",
                "imported_count": 0, "failed_count": 0, "need_review_count": 0,
                "total_raw_records": 0, "total_cleaned_records": 0,
                "imported_datasets": [], "failed_datasets": [], "lineage_ids": [],
                "manifest_path": f"{section}/manifest.json",
                "used_for_training": False, "test_contamination_risk": False, "leakage_risk": False,
                "guide": guide_payload,
                "notes": ["未发现待导入文件/清单条目；请按 guide 人工下载后填写 import_manifest_input.json。"]}

    imported: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    lineage_ids: list[str] = []
    total_raw = total_cleaned = 0

    for e in entries:
        name = e.get("dataset_name") or e.get("document_title") or Path(str(e.get("file_path", "file"))).name
        missing = [k for k in cfg["required"] if not str(e.get(k, "")).strip()]
        if missing:
            failed.append({"name": name, "failed_reason": f"缺少必填字段：{missing}"})
            continue
        if cfg.get("require_open_type") and str(e.get("open_type", "")).strip() != cfg["require_open_type"]:
            failed.append({"name": name, "failed_reason": f"open_type 必须为『{cfg['require_open_type']}』，"
                                                          "有条件开放/需申请不导入。"})
            continue
        fpath = _resolve_path(str(e["file_path"]))
        if not fpath.exists():
            # 兜底：尝试 manual_uploads 下同名文件
            alt = _manual_dir(section) / Path(str(e["file_path"])).name
            if alt.exists():
                fpath = alt
            else:
                failed.append({"name": name, "failed_reason": f"文件不存在：{e['file_path']}"})
                continue
        fmt = str(e.get("file_format", fpath.suffix.lstrip("."))).lower()
        try:
            parsed = _parse_file(fpath, fmt)
        except Exception as exc:  # noqa: BLE001
            failed.append({"name": name, "failed_reason": f"解析失败({fmt})：{type(exc).__name__}: {exc}"})
            continue

        # 授权房价训练许可把关
        license_status = str(e.get("license_status", cfg["default_license"]))
        can_train = bool(e.get("used_for_training", False))
        if section == "authorized_property":
            if license_status != "authorized":
                can_train = False
            elif not str(e.get("authorization_proof", "")).strip():
                can_train = False
        else:
            can_train = False  # 非授权房价分区一律不进训练

        proc_dir = _sec_dir(section) / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in name if c.isalnum() or "\u4e00" <= c <= "\u9fff")[:40] or "ds"
        proc_path = proc_dir / f"{safe}_{abs(hash(str(fpath))) % 10**8}.json"
        meta_out = {k: e.get(k) for k in (cfg["required"] + cfg.get("extra_meta", []))}
        with proc_path.open("w", encoding="utf-8") as f:
            json.dump({"name": name, "file_format": fmt, **parsed, "meta": meta_out,
                       "imported_at": _utcnow_iso()}, f, ensure_ascii=False, indent=2)

        lid = data_lineage_service.record_collection_lineage(
            source_id=section, source_name=name, source_type=cfg["source_type"],
            raw_count=parsed.get("raw_count", 0), cleaned_count=parsed.get("cleaned_count", 0),
            license_status=license_status, compliance_status="pass",
            used_for_feature_engineering=bool(e.get("used_for_feature_engineering", True)),
            used_for_report=bool(e.get("used_for_report", True)),
            used_for_training=can_train, file_path=str(proc_path.relative_to(settings.data_dir.parent)))
        lineage_ids.append(lid)
        total_raw += int(parsed.get("raw_count", 0))
        total_cleaned += int(parsed.get("cleaned_count", 0))
        imported.append({
            "name": name, "file_format": fmt, "raw_count": parsed.get("raw_count"),
            "parsed_count": parsed.get("parsed_count"), "cleaned_count": parsed.get("cleaned_count"),
            "field_schema": parsed.get("field_schema", []), "license_status": license_status,
            "used_for_training": can_train, "used_for_feature_engineering": bool(e.get("used_for_feature_engineering", True)),
            "used_for_report": bool(e.get("used_for_report", True)),
            "used_for_rag": cfg.get("used_for_rag", False),
            "source_url": e.get("source_url") or e.get("source_url_or_contract"),
            "lineage_id": lid, "processed_path": str(proc_path.relative_to(settings.data_dir.parent)),
        })

    manifest_path = _update_section_manifest(section, imported, failed, lineage_ids,
                                              total_raw, total_cleaned, cfg)
    return {
        "section": section, "status": "ok" if imported else "no_valid_import",
        "imported_count": len(imported), "failed_count": len(failed),
        "total_raw_records": total_raw, "total_cleaned_records": total_cleaned,
        "imported_datasets": imported, "failed_datasets": failed,
        "manifest_path": manifest_path, "lineage_ids": lineage_ids,
        "used_for_training": any(d["used_for_training"] for d in imported),
        "test_contamination_risk": False, "leakage_risk": False,
        "notes": [cfg["guide_notes"],
                  "外部数据不混入 competition_test；非授权数据 used_for_training=false。"],
    }


def _update_section_manifest(section: str, imported: list[dict[str, Any]],
                             failed: list[dict[str, Any]], lineage_ids: list[str],
                             total_raw: int, total_cleaned: int, cfg: dict[str, Any]) -> str:
    sec = _sec_dir(section)
    sec.mkdir(parents=True, exist_ok=True)
    manifest = sec / "manifest.json"
    data: dict[str, Any] = {}
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    datasets = (data.get("datasets") or []) + imported
    payload = {
        "source_id": section, "source_type": cfg["source_type"], "provider": cfg["provider"],
        "record_count": int(data.get("record_count", 0)) + total_cleaned,
        "imported_count": int(data.get("imported_count", 0)) + len(imported),
        "failed_count": len(failed),
        "datasets": datasets[-300:], "failed": failed[-100:],
        "lineage_ids": (data.get("lineage_ids") or []) + lineage_ids,
        "license_status": cfg["default_license"], "compliance_status": "pass",
        "used_for_feature_engineering": True, "used_for_report": True,
        "used_for_training": any(d.get("used_for_training") for d in datasets),
        "used_for_eval": False, "import_method": "manual_upload",
        "collection_time": _utcnow_iso(), "is_template": False,
    }
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return f"{section}/manifest.json"
