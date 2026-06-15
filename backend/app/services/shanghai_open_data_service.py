"""上海公共数据开放平台『无条件开放』真实公开下载器（第10B 正式增强）。

设计原则（红线）：
- 只允许下载页面明确标注"无条件开放"且提供公开直链/公开 API 的数据集；
- 优先 CSV，其次 JSON，其次 XLSX；同一数据集只下载一个优先格式；
- 严禁绕登录 / 绕验证码 / 绕申请流程 / 抓隐藏接口 / 高频扫全站 / 伪造下载成功或记录数；
- 若门户启用反爬挑战（HTTP 412 + 动态 JS 校验等），按红线**不绕过**，
  仅记录 ``blocked_by_anti_crawler`` 并登记 need_manual_apply，绝不执行其反爬 JS、绝不伪造数据；
- 外部数据仅用于 feature_engineering / report，不进入监督训练，不混入 competition_test。

注：本平台经探测对脚本访问返回 HTTP 412 + 反爬 JS bootstrap，属"需执行反爬脚本/登录"的受控访问，
按红线不得绕过，因此真实自动下载数为 0 属合规预期（非平台无数据、非伪造）。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services import data_lineage_service

logger = logging.getLogger("cityrenew.shanghai_open_data")

PROVIDER = "上海市公共数据开放平台"
PORTAL_BASE = "https://data.sh.gov.cn"
# 人工检索页（仅用于真实探测；不构造/抓取隐藏 XHR 接口）
SEARCH_URL_TEMPLATES = (
    "https://data.sh.gov.cn/datasetList?keyword={kw}",
    "https://data.sh.gov.cn/search?keyword={kw}",
)
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_TIMEOUT = 15
_SUPPORTED_FORMATS = ("csv", "json", "xlsx", "xml", "rdf")
_ANTI_BOT_HINTS = ("/1wvEDVw16TVN/", "请开启", "javascript is required", "challenge",
                   "window.location", "document.cookie", "security check", "访问出错")
_UNCONDITIONAL_HINTS = ("无条件开放",)
_CONDITIONAL_HINTS = ("有条件开放", "申请", "登录", "实名", "审批", "授权")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sec_dir() -> Path:
    return settings.data_dir / "external" / "shanghai_open_data"


def _manifest_path() -> Path:
    return _sec_dir() / "manifest.json"


def _requests():
    try:
        import requests  # noqa: PLC0415
        return requests
    except Exception:  # noqa: BLE001
        return None


def _looks_anti_bot(status: int, text: str) -> bool:
    if status in (403, 412, 429, 503):
        return True
    low = (text or "")[:4000].lower()
    return any(h.lower() in low for h in _ANTI_BOT_HINTS)


# --------------------------------------------------------------------------- #
# 1) 搜索（真实请求；遇反爬不绕过）
# --------------------------------------------------------------------------- #
def search_datasets(keyword: str, max_pages: int = 2) -> dict[str, Any]:
    """真实访问门户检索页，解析静态 HTML 中的数据集卡片/详情链接/格式按钮。

    返回 {status, datasets, blocked, failed_reason, attempts}。遇反爬挑战仅记录，不绕过。
    """
    requests = _requests()
    if requests is None:
        return {"status": "degraded", "datasets": [], "blocked": False,
                "failed_reason": "requests 不可用，未请求、未伪造数据。", "attempts": []}

    attempts: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    blocked = False
    for tmpl in SEARCH_URL_TEMPLATES:
        url = tmpl.format(kw=keyword)
        try:
            resp = requests.get(url, headers={"User-Agent": _UA, "Accept": "text/html"},
                                timeout=_TIMEOUT)
            status = resp.status_code
            text = resp.text or ""
        except Exception as exc:  # noqa: BLE001
            attempts.append({"url": url, "http_code": None, "error": type(exc).__name__})
            continue
        attempts.append({"url": url, "http_code": status, "size": len(text)})
        if _looks_anti_bot(status, text):
            blocked = True
            continue
        # 解析静态 HTML 中的详情链接（仅当门户返回真实 HTML 内容时）
        for m in re.finditer(r'href="(/?dataset[^"#?]+|/?detail[^"#?]+)"[^>]*>([^<]{2,80})', text):
            href, title = m.group(1), m.group(2).strip()
            detail = href if href.startswith("http") else f"{PORTAL_BASE}/{href.lstrip('/')}"
            datasets.append({"dataset_name": title, "detail_url": detail,
                             "department": None, "open_type": None,
                             "update_time": None, "formats": []})
        if datasets:
            break

    failed_reason = None
    if blocked and not datasets:
        failed_reason = (f"门户对脚本访问返回反爬挑战（HTTP {[a.get('http_code') for a in attempts]}），"
                         "按红线不绕过反爬/不执行其校验 JS；未取得数据集列表，未伪造数据。")
    return {"status": "blocked_by_anti_crawler" if (blocked and not datasets) else "ok",
            "keyword": keyword, "datasets": datasets, "blocked": blocked,
            "failed_reason": failed_reason, "attempts": attempts}


# --------------------------------------------------------------------------- #
# 2) 详情解析（真实请求；判断开放类型 + 解析格式直链）
# --------------------------------------------------------------------------- #
def parse_dataset_detail(dataset_url: str) -> dict[str, Any]:
    requests = _requests()
    if requests is None:
        return {"downloadable": False, "open_type": None, "formats": {},
                "failed_reason": "requests 不可用，未请求、未伪造数据。"}
    try:
        resp = requests.get(dataset_url, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
        status, text = resp.status_code, (resp.text or "")
    except Exception as exc:  # noqa: BLE001
        return {"downloadable": False, "open_type": None, "formats": {},
                "failed_reason": f"详情页请求失败：{type(exc).__name__}"}
    if _looks_anti_bot(status, text):
        return {"downloadable": False, "open_type": "unknown", "formats": {}, "blocked": True,
                "failed_reason": f"详情页反爬挑战（HTTP {status}），按红线不绕过，未解析、未伪造。"}

    open_type = None
    if any(h in text for h in _UNCONDITIONAL_HINTS):
        open_type = "unconditional_open"
    elif any(h in text for h in _CONDITIONAL_HINTS):
        open_type = "conditional_open"

    # 仅解析静态 HTML 中明确的公开直链（href 指向受支持格式文件）
    formats: dict[str, str] = {}
    for m in re.finditer(r'href="(https?://[^"\']+\.(csv|json|xlsx|xml|rdf)(?:\?[^"\']*)?)"', text, re.I):
        fmt = m.group(2).lower()
        formats.setdefault(fmt, m.group(1))

    downloadable = bool(formats) and open_type == "unconditional_open"
    failed_reason = None
    if not downloadable:
        if not formats:
            failed_reason = "详情页静态 HTML 未发现公开格式直链（格式按钮经动态 JS/受控接口提供，按红线不抓取）。"
        elif open_type != "unconditional_open":
            failed_reason = f"开放类型={open_type}，非『无条件开放』，按红线不下载。"
    return {"downloadable": downloadable, "open_type": open_type, "formats": formats,
            "failed_reason": failed_reason}


# --------------------------------------------------------------------------- #
# 3) 下载（仅无条件开放 + 公开直链）
# --------------------------------------------------------------------------- #
def _parse_records(fmt: str, raw_bytes: bytes) -> tuple[int, int, list[str]]:
    """返回 (parsed_count, cleaned_count, field_schema)。失败抛异常由调用方处理。"""
    if fmt == "json":
        obj = json.loads(raw_bytes.decode("utf-8", "ignore"))
        rows = obj if isinstance(obj, list) else obj.get("data") or obj.get("records") or []
        rows = rows if isinstance(rows, list) else []
        schema = sorted({k for r in rows[:50] if isinstance(r, dict) for k in r})
        return len(rows), len(rows), schema[:50]
    if fmt == "csv":
        import csv
        import io
        text = raw_bytes.decode("utf-8-sig", "ignore")
        reader = list(csv.reader(io.StringIO(text)))
        header = reader[0] if reader else []
        return max(0, len(reader) - 1), max(0, len(reader) - 1), header[:50]
    if fmt == "xlsx":
        try:
            import openpyxl  # noqa: PLC0415
            import io
            wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), read_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            header = [str(c) for c in rows[0]] if rows else []
            return max(0, len(rows) - 1), max(0, len(rows) - 1), header[:50]
        except Exception:  # noqa: BLE001
            return 0, 0, []
    return 0, 0, []


def download_dataset(dataset: dict[str, Any], detail: dict[str, Any],
                     preferred_formats: list[str]) -> dict[str, Any]:
    requests = _requests()
    if requests is None:
        return {"status": "degraded", "failed_reason": "requests 不可用，未下载、未伪造数据。"}
    formats = detail.get("formats", {})
    fmt = next((f for f in preferred_formats if f in formats), None)
    if not fmt:
        fmt = next((f for f in _SUPPORTED_FORMATS if f in formats), None)
    if not fmt:
        return {"status": "failed", "failed_reason": "无受支持的公开格式直链。"}
    download_url = formats[fmt]
    try:
        resp = requests.get(download_url, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
        if _looks_anti_bot(resp.status_code, resp.text[:2000] if hasattr(resp, "text") else ""):
            return {"status": "failed", "download_url": download_url, "file_format": fmt,
                    "failed_reason": f"下载链接返回反爬挑战（HTTP {resp.status_code}），不绕过。"}
        if resp.status_code != 200:
            return {"status": "failed", "download_url": download_url, "file_format": fmt,
                    "failed_reason": f"下载失败 HTTP {resp.status_code}。"}
        raw_bytes = resp.content
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "download_url": download_url, "file_format": fmt,
                "failed_reason": f"下载异常：{type(exc).__name__}"}

    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", str(dataset.get("dataset_name", "dataset")))[:40]
    fid = uuid.uuid4().hex[:8]
    raw_dir = _sec_dir() / "raw"
    proc_dir = _sec_dir() / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{safe}_{fid}.{fmt}"
    raw_path = raw_dir / file_name
    raw_path.write_bytes(raw_bytes)
    try:
        parsed, cleaned, schema = _parse_records(fmt, raw_bytes)
    except Exception as exc:  # noqa: BLE001
        parsed, cleaned, schema = 0, 0, []
        logger.warning("parse %s failed: %s", file_name, exc)
    proc_path = proc_dir / f"{safe}_{fid}.json"
    with proc_path.open("w", encoding="utf-8") as f:
        json.dump({"dataset_name": dataset.get("dataset_name"), "file_format": fmt,
                   "parsed_count": parsed, "cleaned_count": cleaned, "field_schema": schema,
                   "source_url": dataset.get("detail_url"), "download_url": download_url,
                   "downloaded_at": _utcnow_iso()}, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "file_format": fmt, "file_name": file_name,
            "download_url": download_url,
            "raw_path": f"shanghai_open_data/raw/{file_name}",
            "processed_path": f"shanghai_open_data/processed/{proc_path.name}",
            "raw_count": parsed, "parsed_count": parsed, "cleaned_count": cleaned,
            "field_schema": schema}


# --------------------------------------------------------------------------- #
# 4) 关键词批量发现 + 下载
# --------------------------------------------------------------------------- #
def collect_by_keywords(keywords: list[str], *, max_pages_per_keyword: int = 2,
                        max_datasets_per_keyword: int = 10, max_total_downloads: int = 80,
                        stop_after_success: int = 30,
                        preferred_formats: list[str] | None = None,
                        only_unconditional: bool = True) -> dict[str, Any]:
    preferred_formats = preferred_formats or ["csv", "json", "xlsx"]
    candidates: list[dict[str, Any]] = []
    downloaded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    need_manual: list[dict[str, Any]] = []
    lineage_ids: list[str] = []
    unconditional = conditional = downloadable = 0
    total_attempted = 0
    blocked_any = False
    total_raw = total_cleaned = 0

    for kw in keywords:
        if len(downloaded) >= stop_after_success or total_attempted >= max_total_downloads:
            break
        sr = search_datasets(kw, max_pages=max_pages_per_keyword)
        blocked_any = blocked_any or sr.get("blocked", False)
        ds_list = sr.get("datasets", [])[:max_datasets_per_keyword]
        if not ds_list:
            # 门户未返回可解析数据集（反爬/SPA）：登记关键词级 candidate，不伪造
            candidates.append({"keyword": kw, "status": sr.get("status"),
                               "failed_reason": sr.get("failed_reason")})
            continue
        for ds in ds_list:
            if len(downloaded) >= stop_after_success or total_attempted >= max_total_downloads:
                break
            candidates.append({"dataset_name": ds.get("dataset_name"), "detail_url": ds.get("detail_url")})
            detail = parse_dataset_detail(ds["detail_url"])
            ds.update({"open_type": detail.get("open_type")})
            if detail.get("open_type") == "unconditional_open":
                unconditional += 1
            elif detail.get("open_type") == "conditional_open":
                conditional += 1
                need_manual.append({"dataset_name": ds.get("dataset_name"),
                                    "detail_url": ds.get("detail_url"), "open_type": "conditional_open",
                                    "reason": "有条件开放，需登录/申请/审批，按红线不下载。"})
                continue
            if only_unconditional and detail.get("open_type") != "unconditional_open":
                need_manual.append({"dataset_name": ds.get("dataset_name"),
                                    "detail_url": ds.get("detail_url"),
                                    "open_type": detail.get("open_type"),
                                    "reason": detail.get("failed_reason") or "非无条件开放或无公开直链。"})
                continue
            if not detail.get("downloadable"):
                need_manual.append({"dataset_name": ds.get("dataset_name"),
                                    "detail_url": ds.get("detail_url"),
                                    "open_type": detail.get("open_type"),
                                    "reason": detail.get("failed_reason") or "无公开直链。"})
                continue
            downloadable += 1
            total_attempted += 1
            dl = download_dataset(ds, detail, preferred_formats)
            if dl.get("status") == "ok":
                lid = data_lineage_service.record_collection_lineage(
                    source_id="shanghai_open_data", source_name=ds.get("dataset_name", "上海公共数据"),
                    source_type="gov_open_data", raw_count=dl.get("raw_count", 0),
                    cleaned_count=dl.get("cleaned_count", 0),
                    license_status="open_unconditional", compliance_status="pass",
                    used_for_feature_engineering=True, used_for_report=True, used_for_training=False,
                    file_path=dl.get("processed_path"))
                lineage_ids.append(lid)
                dl["lineage_id"] = lid
                dl["dataset_name"] = ds.get("dataset_name")
                total_raw += int(dl.get("raw_count", 0))
                total_cleaned += int(dl.get("cleaned_count", 0))
                downloaded.append(dl)
            else:
                failed.append({"dataset_name": ds.get("dataset_name"),
                               "detail_url": ds.get("detail_url"),
                               "failed_reason": dl.get("failed_reason")})

    manifest_path = _write_manifest(downloaded, need_manual, failed, candidates,
                                    lineage_ids, blocked_any)
    catalog_paths = _export_catalog(downloaded, need_manual, failed, keywords)

    report = {
        "searched_keywords": keywords,
        "candidate_count": len(candidates),
        "unconditional_count": unconditional,
        "conditional_count": conditional,
        "downloadable_count": downloadable,
        "downloaded_count": len(downloaded),
        "failed_count": len(failed),
        "need_manual_apply_count": len(need_manual),
        "total_raw_records": total_raw,
        "total_cleaned_records": total_cleaned,
        "downloaded_datasets": downloaded,
        "failed_datasets": failed,
        "need_manual_apply_datasets": need_manual[:50],
        "manifest_path": manifest_path,
        "lineage_ids": lineage_ids,
        "blocked_by_anti_crawler": blocked_any,
        "blocked_by_waf": blocked_any and not downloaded,
        "can_auto_download": not blocked_any,
        "can_manual_import": True,
        "failed_reason": ("http_412_waf_or_js_challenge" if blocked_any and not downloaded else None),
        "manual_import_endpoint": "/api/external/shanghai-open-data/import-manual",
        "catalog_paths": catalog_paths,
        "notes": [
            "只下载『无条件开放』且有公开直链的数据集；有条件开放/需申请 → need_manual_apply。",
            "门户启用反爬挑战(HTTP 412/WAF/JS)时按红线不绕过，真实自动下载为 0 属合规预期；",
            "合规链路 B：浏览器人工下载后用 /shanghai-open-data/import-manual 导入。",
            "外部数据仅用于 feature_engineering / report，不进入监督训练、不混入 competition_test。",
        ],
    }
    return report


# --------------------------------------------------------------------------- #
# 合规链路 B：人工下载导入（不绕 WAF）
# --------------------------------------------------------------------------- #
def manual_download_guide() -> dict[str, Any]:
    from app.services import manual_import_service
    return manual_import_service.guide("shanghai_open_data")


def import_manual_files(entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from app.services import manual_import_service
    return manual_import_service.import_manual("shanghai_open_data", entries)


def _write_manifest(downloaded, need_manual, failed, candidates, lineage_ids, blocked) -> str:
    sec = _sec_dir()
    sec.mkdir(parents=True, exist_ok=True)
    datasets = []
    for d in downloaded:
        datasets.append({
            "source_id": "shanghai_open_data", "dataset_name": d.get("dataset_name"),
            "provider": PROVIDER, "open_type": "unconditional_open",
            "download_url": d.get("download_url"), "file_format": d.get("file_format"),
            "file_name": d.get("file_name"), "raw_path": d.get("raw_path"),
            "processed_path": d.get("processed_path"), "raw_count": d.get("raw_count"),
            "parsed_count": d.get("parsed_count"), "cleaned_count": d.get("cleaned_count"),
            "field_schema": d.get("field_schema"), "license_status": "open_unconditional",
            "compliance_status": "pass", "used_for_feature_engineering": True,
            "used_for_report": True, "used_for_training": False, "used_for_eval": False,
            "lineage_id": d.get("lineage_id"), "failed_reason": None,
            "download_time": _utcnow_iso(),
        })
    payload = {
        "source_id": "shanghai_open_data", "provider": PROVIDER, "source_url": PORTAL_BASE,
        "record_count": sum(int(d.get("cleaned_count", 0)) for d in downloaded),
        "downloaded_count": len(downloaded), "need_manual_apply_count": len(need_manual),
        "failed_count": len(failed), "candidate_count": len(candidates),
        "blocked_by_anti_crawler": blocked, "lineage_ids": lineage_ids,
        "datasets": datasets, "need_manual_apply": need_manual[:80], "failed": failed[:80],
        "collection_time": _utcnow_iso(), "is_template": False,
        "compliance_status": "pass",
        "failed_reason": (None if downloaded else
                          "门户反爬/受控访问，按红线未绕过；真实下载=0，需人工在门户检索导入。"),
    }
    with _manifest_path().open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return "shanghai_open_data/manifest.json"


def _export_catalog(downloaded, need_manual, failed, keywords) -> dict[str, str]:
    out_dir = settings.data_dir / "outputs" / "data_catalog"
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / "上海公共数据下载清单.json"
    with j.open("w", encoding="utf-8") as f:
        json.dump({"provider": PROVIDER, "keywords": keywords,
                   "downloaded": downloaded, "need_manual_apply": need_manual,
                   "failed": failed, "updated_at": _utcnow_iso()},
                  f, ensure_ascii=False, indent=2)
    md = ["# 上海公共数据下载清单（脱敏）", "",
          f"- 提供方：{PROVIDER}",
          f"- 关键词：{', '.join(keywords)}",
          f"- 成功下载：{len(downloaded)}　需人工申请：{len(need_manual)}　失败：{len(failed)}", ""]
    if downloaded:
        md += ["| 数据集 | 格式 | 记录数 | 处理路径 |", "| --- | --- | --- | --- |"]
        for d in downloaded:
            md.append(f"| {d.get('dataset_name')} | {d.get('file_format')} | "
                      f"{d.get('cleaned_count')} | {d.get('processed_path')} |")
    else:
        md += ["> 本次未取得『无条件开放 + 公开直链』数据集：门户对脚本访问启用反爬挑战，",
               "> 按红线不绕登录/验证码/动态页隐藏接口，需人工在门户检索并下载后导入。"]
    m = out_dir / "上海公共数据下载清单.md"
    m.write_text("\n".join(md), encoding="utf-8")
    return {"json": str(j.relative_to(settings.data_dir.parent)),
            "md": str(m.relative_to(settings.data_dir.parent))}
