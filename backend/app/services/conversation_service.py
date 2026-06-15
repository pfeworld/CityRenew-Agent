"""对话编排（AgentConversation）——第一阶段真链路版。

面向正式用户的多轮对话：跨轮累积"项目档案"（名称/位置/用地/年代/面积/诉求/目标），
当信息充足时把地址地理编码为坐标、为该会话创建**独立真实项目**、调用本地空间分析与
自训练模型（ModelInferenceService）产出可追溯结论，再由 DeepSeek 受约束成文。

红线：
- 事实数字仅来自本地分析与自训练模型结果（model_inference_service），DeepSeek 只组织语言；
- 缺位置/超数据覆盖/范围内无数据 → fail-closed，主动追问，绝不编造、绝不套固定 demo；
- 绝不使用固定 project_id=1；每个会话有自己的项目；
- 前台不暴露模型/坐标系/英文字段/知识库/案例文件名等内部信息。
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.schemas.project import ProjectCreate, ProjectUpdate
from app.services import case_learning_service as cl
from app.services import deepseek_service
from app.services import geocoding_service as geo
from app.services import model_inference_service as mi
from app.services import project_service
from app.services import report_builder_service as rb
from app.services import report_quality_v2_service as q
from app.services import report_word_service as rw

logger = logging.getLogger("cityrenew.agent.conversation")

# 内存会话存储（单进程；重启清空，不落涉密内容到磁盘）。
_CONVERSATIONS: dict[str, dict[str, Any]] = {}

# 历史会话元数据持久化（仅标题/阶段/置顶/归档等元信息，不落对话原文，避免涉密外泄）。
from app.config import settings as _settings  # noqa: E402

_META_DIR = _settings.data_dir / "agent"
# 支持以环境变量指向隔离存储（回归/测试用，避免污染正式 conversations.json）。
_META_PATH = Path(os.environ.get("CITYRENEW_AGENT_CONV_PATH") or (_META_DIR / "conversations.json"))


def _load_meta() -> dict[str, dict[str, Any]]:
    try:
        if _META_PATH.exists():
            return json.loads(_META_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save_meta() -> None:
    try:
        _META_DIR.mkdir(parents=True, exist_ok=True)
        _META_PATH.write_text(json.dumps(_META, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


_META: dict[str, dict[str, Any]] = _load_meta()
_DEFAULT_TITLES = {"新的对话", "新的城市更新咨询", "城市更新咨询", "您好", "你好", "hi", "hello", ""}
# 低质量标题（纯数字/标点/空白）不应作为自动标题。
_LOW_QUALITY_TITLE_RE = re.compile(r"^[\d\s\W]+$")

STAGE_INPUT = "待完善项目信息"
STAGE_ANALYZED = "已完成研判"
STAGE_GENERATED = "报告已生成"

# 前台禁止出现的内部措辞 → 业务化替换 / 删除
_SANITIZE_SUBS = [
    (r"DeepSeek", ""),
    (r"自研城市更新模型|自研模型", "专业分析引擎"),
    (r"知识库", "资料"),
    (r"train/val|final test|\bMAPE\b|model_run_id|model\.pkl", ""),
    (r"（?置信度[^，。；）\n]*[%％]?）?", ""),
    (r"GCJ-?02坐标系?|WGS-?84坐标系?|GCJ-?02|WGS-?84", ""),
    (r"pseudo-?project[^，。；\n]*", ""),
    (r"\bT[1-3]\b", ""),
]
_RAW_KEY_RE = re.compile(r"[“\"'\(（]?[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+[”\"'\)）]?")
_ALLOW_LATIN = {"poi", "pdf", "word", "cityrenew", "agent", "saas", "ai"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _strip_raw_keys(s: str) -> str:
    s = _RAW_KEY_RE.sub("", s)

    def _repl(m):
        tok = m.group(0)
        return "" if tok.lower() not in _ALLOW_LATIN and len(tok) >= 4 else tok

    return re.sub(r"[A-Za-z]{4,}", _repl, s)


def _sanitize(text: str) -> str:
    s = str(text or "")
    for pat, rep in _SANITIZE_SUBS:
        s = re.sub(pat, rep, s)
    s = s.replace("**", "").replace("*", "")
    s = re.sub(r"#{1,6}\s*", "", s)
    s = re.sub(r"^\s*[-•]\s*", "", s, flags=re.MULTILINE)
    s = _strip_raw_keys(s)
    s = re.sub(r"[（(]\s*[）)]", "", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# --------------------------------------------------------------------------- #
# 会话管理
# --------------------------------------------------------------------------- #
def create_conversation(project_id: int | None = None) -> dict[str, Any]:
    cid = uuid.uuid4().hex[:16]
    conv = {
        "conversation_id": cid,
        "project_id": None,           # 会话自己的真实项目，按需创建（不再固定 1）
        "title": "新的城市更新咨询",
        "created_at": _now(),
        "messages": [],
        "profile": {
            "name": None, "address": None, "district": None,
            "land_use": None, "build_year": None,
            "project_area": None, "building_area": None,
            "update_demand": None, "expected_direction": None,
            "attachments": [],        # [{filename, chars, summary}]
            "attachments_text": "",
        },
        "state": {
            "stage": STAGE_INPUT,
            "last_suggestions": [],
            "report_ready": False,
            "report_id": None,
            "case_style": None,
            "analysis_done": False,
            "model_run_id": None,
            "renewal_type": None,
            "last_status": None,
        },
    }
    # 根因修复：新建会话仅创建内存草稿，绝不立即写入 conversations.json。
    # 仅当出现首条有效消息 / 附件 / 报告（_is_valid_conv）时，由 _touch_meta 落盘。
    _CONVERSATIONS[cid] = conv
    return conv


def get_conversation(cid: str) -> dict[str, Any] | None:
    return _CONVERSATIONS.get(cid)


def _ensure(cid: str | None, project_id: int | None = None) -> dict[str, Any]:
    # cid 为空或无效 → 返回内存草稿（不持久化），等到有有效内容再落盘。
    if cid and cid in _CONVERSATIONS:
        return _CONVERSATIONS[cid]
    return create_conversation(project_id)


# --------------------------------------------------------------------------- #
# 历史会话元数据：自动标题 / 列表 / 重命名 / 删除 / 置顶 / 归档 / 分享 / 搜索
# --------------------------------------------------------------------------- #
def _auto_title(conv: dict) -> str:
    """自动标题：项目名 > 地址/地块 > 首条有效问题(16字) > 兜底『城市更新咨询』。

    报告生成后若已识别项目名，标题升级为『项目名 · 前策报告』。
    """
    prof = conv["profile"]
    name = prof.get("name")
    if conv.get("state", {}).get("report_ready") and name:
        return f"{str(name)[:20]} · 前策报告"
    if name:
        return str(name)[:24]
    if prof.get("address"):
        a = re.sub(r"^上海市?", "", str(prof["address"])).strip()
        base = (a or str(prof["address"]))[:20]
        if base and not _LOW_QUALITY_TITLE_RE.match(base):
            return base
    for m in conv.get("messages", []):
        if m.get("role") != "user":
            continue
        t = (m.get("text") or "").strip()
        if (t and len(t) >= 2 and not _GREETING.match(t)
                and t not in _DEFAULT_TITLES
                and not _LOW_QUALITY_TITLE_RE.match(t)):
            return t[:16]
    return "城市更新咨询"


def _user_msg_count(conv: dict) -> int:
    return sum(1 for m in conv.get("messages", [])
               if m.get("role") == "user" and (m.get("text") or "").strip())


def _is_valid_conv(conv: dict) -> bool:
    """是否为『有效会话』：有有效用户消息 / 有附件 / 已识别项目 / 已出报告。

    空白消息、纯空格、无任何内容的草稿一律视为无效，不予落盘。
    """
    if _user_msg_count(conv) > 0:
        return True
    prof = conv.get("profile", {})
    if prof.get("attachments"):
        return True
    if prof.get("name") or prof.get("address"):
        return True
    if conv.get("state", {}).get("report_ready"):
        return True
    return False


def _meta_entry(conv: dict) -> dict[str, Any]:
    cid = conv["conversation_id"]
    e = _META.get(cid)
    if e is None:
        e = {"id": cid, "title": conv.get("title") or "城市更新咨询",
             "stage": conv["state"]["stage"], "pinned": False, "archived": False,
             "renamed": False, "project_id": conv.get("project_id"),
             "message_count": 0, "has_attachment": False, "has_report": False,
             "created_at": conv.get("created_at"), "updated_at": conv.get("created_at")}
        _META[cid] = e
    return e


def _meta_visible(e: dict) -> bool:
    """历史列表可见性：仅展示有真实内容的会话（过滤空会话）。"""
    return bool(e.get("message_count") or e.get("has_attachment")
                or e.get("has_report") or e.get("project_id") or e.get("renamed"))


def _touch_meta(conv: dict) -> None:
    """有效会话才落盘并更新元信息；空会话直接跳过（根因修复）。"""
    if not _is_valid_conv(conv):
        return
    e = _meta_entry(conv)
    e["stage"] = conv["state"]["stage"]
    e["project_id"] = conv.get("project_id")
    e["updated_at"] = _now()
    e["message_count"] = _user_msg_count(conv)
    e["has_attachment"] = bool(conv["profile"].get("attachments"))
    e["has_report"] = bool(conv["state"].get("report_ready"))
    if not e.get("renamed"):
        title = _auto_title(conv)
        if title:
            e["title"] = title
            conv["title"] = title
    _save_meta()


def persist_if_valid(conv: dict) -> None:
    """供路由层（附件上传等）调用：仅在会话有效时落盘。"""
    _touch_meta(conv)


def _match_query(e: dict, conv: dict | None, q: str) -> bool:
    q = q.strip().lower()
    if not q:
        return True
    if q in (e.get("title") or "").lower():
        return True
    if conv:
        if q in (conv.get("profile", {}).get("name") or "").lower():
            return True
        if q in (conv.get("profile", {}).get("address") or "").lower():
            return True
        for m in conv.get("messages", []):
            if m.get("role") == "user" and q in (m.get("text") or "").lower():
                return True
    return False


def list_conversations(query: str | None = None, include_archived: bool = False) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for cid, e in _META.items():
        if not _meta_visible(e):  # 过滤空会话
            continue
        if e.get("archived") and not include_archived:
            continue
        if query and not _match_query(e, _CONVERSATIONS.get(cid), query):
            continue
        items.append({k: e.get(k) for k in
                      ("id", "title", "stage", "pinned", "archived", "project_id",
                       "message_count", "has_report", "created_at", "updated_at")})
    items.sort(key=lambda x: (bool(x.get("pinned")), x.get("updated_at") or ""), reverse=True)
    return items


def cleanup_empty_conversations() -> dict[str, int]:
    """一次性清理 conversations.json 中的空会话（无消息/无附件/无项目/无报告）。"""
    before = len(_META)
    remove = [cid for cid, e in _META.items() if not _meta_visible(e)]
    for cid in remove:
        _META.pop(cid, None)
        _CONVERSATIONS.pop(cid, None)
    if remove:
        _save_meta()
    after = len(_META)
    return {"before": before, "after": after, "removed": len(remove)}


def rename_conversation(cid: str, title: str) -> dict[str, Any] | None:
    e = _META.get(cid)
    if e is None:
        return None
    t = (title or "").strip()[:40]
    if t:
        e["title"] = t
        e["renamed"] = True
        e["updated_at"] = _now()
        if cid in _CONVERSATIONS:
            _CONVERSATIONS[cid]["title"] = t
        _save_meta()
    return e


def delete_conversation(cid: str) -> bool:
    existed = cid in _META or cid in _CONVERSATIONS
    _META.pop(cid, None)
    _CONVERSATIONS.pop(cid, None)
    _save_meta()
    return existed


def set_pinned(cid: str, pinned: bool) -> dict[str, Any] | None:
    e = _META.get(cid)
    if e is None:
        return None
    e["pinned"] = bool(pinned)
    e["updated_at"] = _now()
    _save_meta()
    return e


def set_archived(cid: str, archived: bool) -> dict[str, Any] | None:
    e = _META.get(cid)
    if e is None:
        return None
    e["archived"] = bool(archived)
    e["updated_at"] = _now()
    _save_meta()
    return e


def share_payload(cid: str) -> dict[str, Any] | None:
    e = _META.get(cid)
    conv = _CONVERSATIONS.get(cid)
    if e is None and conv is None:
        return None
    title = (e or {}).get("title") or (conv or {}).get("title") or "城市更新咨询"
    lines = [f"【CityRenew 城市更新前期策划咨询】{title}"]
    if conv:
        for m in conv.get("messages", [])[:6]:
            who = "我" if m.get("role") == "user" else "智能体"
            txt = _sanitize(m.get("text") or "")[:90]
            if txt:
                lines.append(f"{who}：{txt}")
    return {"id": cid, "title": title, "link": f"/agent?c={cid}",
            "summary": "\n".join(lines)}


# --------------------------------------------------------------------------- #
# 项目档案抽取（从自然语言里识别项目要素，跨轮累积）
# --------------------------------------------------------------------------- #
_LOCATION_HINT = re.compile(r"(上海|[\u4e00-\u9fa5]{2,3}区|路|号|镇|街道|弄|村|开发区|园区)")
_ADDRESS_PAT = re.compile(
    r"(上海市?[\u4e00-\u9fa5]{1,5}区[\u4e00-\u9fa5A-Za-z0-9]{0,18}"
    r"(?:路|街|道|镇|村|弄|号|大道|开发区|园区)?[0-9]{0,5}号?)"
)


def _first(*vals):
    for v in vals:
        if v:
            return v.strip()
    return None


def extract_profile(text: str) -> dict[str, Any]:
    """从一段用户输入中尽力抽取项目要素（只取明确表达，缺失留空）。"""
    t = (text or "").strip()
    out: dict[str, Any] = {}
    if not t:
        return out

    m = re.search(r"项目名称[:：]\s*([^\s，。,；\n]{2,30})", t)
    if not m:
        m = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]{2,20}(?:项目|地块|片区|园区|厂区|社区|小区))", t)
    if m:
        out["name"] = m.group(1).strip()

    m = re.search(r"(?:位于|地址[:：]?|坐落于|地处|在)\s*(上海[^\s，。,；\n]{2,40})", t)
    addr = m.group(1).strip() if m else None
    if not addr:
        m = _ADDRESS_PAT.search(t)
        addr = m.group(1).strip() if m else None
    if addr:
        out["address"] = addr

    m = re.search(r"用地性质[:：]?\s*([^\s，。,；\n]{2,20})", t)
    if m:
        out["land_use"] = m.group(1).strip()

    m = re.search(r"(?:建于|建成于|竣工于)?\s*(\d{4})\s*年(?:建成|建造|竣工|代)?", t)
    if m:
        yr = int(m.group(1))
        if 1900 <= yr <= 2025:
            out["build_year"] = yr

    m = re.search(r"用地面积[:：]?\s*(\d+(?:\.\d+)?)\s*(万)?\s*(?:平方米|㎡|平米|m2|亩)?", t)
    if m:
        val = float(m.group(1)) * (10000 if m.group(2) else 1)
        out["project_area"] = val
    m = re.search(r"建筑面积[:：]?\s*(\d+(?:\.\d+)?)\s*(万)?\s*(?:平方米|㎡|平米|m2)?", t)
    if m:
        out["building_area"] = float(m.group(1)) * (10000 if m.group(2) else 1)

    m = re.search(r"(?:现状|存在|问题|痛点|诉求|短板)[:：]?\s*(.{4,60})", t)
    if m:
        out["update_demand"] = m.group(1).strip(" 。.，,；;")
    m = re.search(r"(?:目标|希望|打算|期望|定位|改造方向|想做|计划)[:：]?\s*(.{2,60})", t)
    if m:
        out["expected_direction"] = m.group(1).strip(" 。.，,；;")

    return out


def _merge_profile(conv: dict, extracted: dict) -> None:
    prof = conv["profile"]
    for k, v in extracted.items():
        if v and not prof.get(k):
            prof[k] = v


def _looks_like_location(text: str) -> bool:
    return bool(_LOCATION_HINT.search(text or "")) and ("上海" in text or "区" in text)


def ingest_attachment(conv: dict, parsed: dict) -> dict[str, Any]:
    """把已解析的附件并入会话档案：抽取项目要素 + 登记附件元信息。

    parsed 来自 attachment_service.parse；只保留元信息与抽取结果，不回显全文到前台。
    """
    prof = conv["profile"]
    text = parsed.get("text") or ""
    extracted = extract_profile(text) if text else {}
    _merge_profile(conv, extracted)
    if text:
        prof["attachments_text"] = (prof.get("attachments_text", "") + "\n" + text)[:40000]
    prof["attachments"].append({
        "filename": parsed.get("filename"),
        "ext": parsed.get("ext"),
        "chars": parsed.get("chars", 0),
        "summary": parsed.get("summary", ""),
        "note": parsed.get("note", ""),
        "extracted_fields": sorted(extracted.keys()),
    })
    # 附件即有效内容 → 落盘历史会话（草稿转正）。
    _touch_meta(conv)
    return {
        "filename": parsed.get("filename"),
        "ok": parsed.get("ok", False),
        "chars": parsed.get("chars", 0),
        "extracted_fields": sorted(extracted.keys()),
        "note": parsed.get("note", ""),
        "profile_snapshot": {k: bool(prof.get(k)) for k in
                             ("name", "address", "land_use", "build_year",
                              "update_demand", "expected_direction")},
    }


# --------------------------------------------------------------------------- #
# 项目落库 + 推理
# --------------------------------------------------------------------------- #
def _ensure_project(db: Session, conv: dict) -> dict[str, Any]:
    """根据会话档案确保存在一个带合法坐标的真实项目。

    返回 {ok, status, message}。status: ready / need_location / geocode_failed / out_of_coverage
    """
    prof = conv["profile"]
    address = prof.get("address")
    if not address:
        return {"ok": False, "status": "need_location",
                "message": "我还不知道项目的具体位置。请告诉我项目所在的上海市具体地址或地块名称（例如：上海市徐汇区龙华XX路）。"}

    g = geo.geocode(address, city="上海")
    if not g["ok"]:
        if g.get("error") == "out_of_coverage":
            return {"ok": False, "status": "out_of_coverage",
                    "message": "这个位置超出了当前可分析的数据覆盖范围（目前支持上海部分区域）。请提供上海市范围内的项目地址。"}
        if g.get("error") == "geocoder_not_configured":
            return {"ok": False, "status": "geocode_failed",
                    "message": "暂时无法解析该地址的位置。你可以换一种更完整的写法（含区、路名），或直接提供项目坐标。"}
        return {"ok": False, "status": "geocode_failed",
                "message": "没能识别这个地址，请补充更完整的地址（包含区与路名/地块名），我再为你定位分析。"}

    if not prof.get("district"):
        prof["district"] = g.get("district")
    name = prof.get("name") or (g.get("formatted_address") or address)

    if conv.get("project_id"):
        project_service.update_project(db, conv["project_id"], ProjectUpdate(
            name=name, address=address, district=prof.get("district"),
            center_lng=g["lng"], center_lat=g["lat"],
            land_use=prof.get("land_use"), build_year=prof.get("build_year"),
            project_area=prof.get("project_area"), building_area=prof.get("building_area"),
            update_demand=prof.get("update_demand"), expected_direction=prof.get("expected_direction"),
        ))
    else:
        po = project_service.create_project(db, ProjectCreate(
            name=name, address=address, city="上海", district=prof.get("district"),
            center_lng=g["lng"], center_lat=g["lat"],
            land_use=prof.get("land_use"), build_year=prof.get("build_year"),
            project_area=prof.get("project_area"), building_area=prof.get("building_area"),
            update_demand=prof.get("update_demand"), expected_direction=prof.get("expected_direction"),
        ))
        conv["project_id"] = po.id
        conv["title"] = name[:24]
    return {"ok": True, "status": "ready", "message": ""}


def _run_analysis(db: Session, conv: dict) -> dict[str, Any]:
    """确保项目存在并运行真实推理；结果缓存到会话。"""
    ens = _ensure_project(db, conv)
    if not ens["ok"]:
        conv["state"]["last_status"] = ens["status"]
        return {"ok": False, "status": ens["status"], "message": ens["message"]}

    project = project_service.get_project(db, conv["project_id"])
    result = mi.run_inference(db, project)
    conv["state"]["last_status"] = result["status"]
    conv["state"]["model_run_id"] = result.get("model_run_id")

    if result["status"] != mi.STATUS_OK:
        msg = {
            mi.STATUS_OUT_OF_COVERAGE: "这个位置超出了当前数据覆盖范围（目前支持上海部分区域），暂时无法生成可靠分析。请确认或更换项目位置。",
            mi.STATUS_INSUFFICIENT_DATA: "项目坐标周边暂时没有足够的可用数据来支撑分析。请确认项目位置是否准确，或补充更具体的地块信息与资料。",
            mi.STATUS_MISSING_LOCATION: "项目还缺少可用于分析的具体位置，请补充项目地址。",
        }.get(result["status"], "暂时无法完成分析，请补充项目信息后重试。")
        return {"ok": False, "status": result["status"], "message": msg}

    conv["_analysis"] = result
    conv["state"]["analysis_done"] = True
    conv["state"]["renewal_type"] = result.get("renewal_type")
    if conv["state"]["stage"] == STAGE_INPUT:
        conv["state"]["stage"] = STAGE_ANALYZED
    return {"ok": True, "status": "ok", "analysis": result}


# --------------------------------------------------------------------------- #
# 成文（DeepSeek 受约束 + 确定性兜底）
# --------------------------------------------------------------------------- #
def _fmt_int(v) -> str | None:
    try:
        return f"{int(round(float(v)))}"
    except (TypeError, ValueError):
        return None


def _facts_block(ar: dict) -> list[str]:
    """从结构化 analysis_result 抽取可用于成文的事实（只读，不新增数字）。"""
    facts: list[str] = []
    poi = _ring(ar.get("location_poi_analysis"), "radiation")
    pop = _ring(ar.get("population_analysis"), "radiation")
    house = _ring(ar.get("housing_space_analysis"), "radiation")
    ind = _ring(ar.get("industry_analysis"), "radiation")
    if poi.get("total") is not None:
        facts.append(f"辐射范围内各类配套兴趣点约 {_fmt_int(poi['total'])} 个")
    if pop.get("residential") is not None:
        facts.append(f"常住居住人口约 {_fmt_int(pop['residential'])} 人")
    if house.get("avg_unit_price") is not None:
        facts.append(f"周边二手房均价约 {_fmt_int(house['avg_unit_price'])} 元/平方米")
    if ind.get("enterprise_count") is not None:
        facts.append(f"产业企业约 {_fmt_int(ind['enterprise_count'])} 家")
    return facts


def _ring(dim, name):
    for r in (dim or {}).get("rings") or []:
        if r.get("ring") == name:
            return r
    return {}


# --- 去模板化：相似度护栏（连续两个不同项目回答过于雷同则重写） ---------------- #
_RECENT_DIAGNOSES: list[dict[str, str]] = []
_SIM_THRESHOLD = 0.58


def _char_bigrams(s: str) -> set[str]:
    s = re.sub(r"[\s，。、；：！？,.;:!?「」（）()]+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}


def _similarity(a: str, b: str) -> float:
    A, B = _char_bigrams(a), _char_bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _most_similar_other(text: str, key: str) -> tuple[float, str]:
    best, best_t = 0.0, ""
    for rec in _RECENT_DIAGNOSES:
        if rec["key"] == key:
            continue
        s = _similarity(text, rec["text"])
        if s > best:
            best, best_t = s, rec["text"]
    return best, best_t


def _remember_diagnosis(key: str, text: str) -> None:
    _RECENT_DIAGNOSES.append({"key": key, "text": text})
    if len(_RECENT_DIAGNOSES) > 12:
        del _RECENT_DIAGNOSES[: len(_RECENT_DIAGNOSES) - 12]


def _user_text(conv: dict) -> str:
    prof = conv.get("profile", {})
    parts = [prof.get("update_demand"), prof.get("expected_direction"),
             prof.get("land_use"), prof.get("name")]
    msgs = [m["text"] for m in conv.get("messages", []) if m.get("role") == "user"][-4:]
    return " ".join([p for p in parts if p] + msgs)


def _diff_profile(ar: dict, conv: dict) -> dict[str, Any]:
    return cl.map_to_type_profile(ar.get("renewal_type"), _user_text(conv))


def _deterministic_diagnosis(ar: dict, conv: dict, dprof: dict, *, variant: int = 0) -> str:
    """差异化确定性研判：按项目类型侧重 + 用户现状/目标组织，不同类型/目标输出不同。"""
    name = ar.get("project_understanding", {}).get("name") or "该项目"
    ptype = dprof.get("canonical_type") or ar.get("renewal_type") or "待明确的城市更新类型"
    fscore = ar.get("comprehensive_score")
    level = ar.get("score_level") or "中等"
    facts = _facts_block(ar)
    prof = conv["profile"]
    demand = prof.get("update_demand")
    goal = prof.get("expected_direction")
    emphasis = list(dprof.get("emphasis") or [])
    goal_focus = list(dprof.get("goal_focus") or [])
    core_logic = dprof.get("core_logic")
    dp = ar.get("demand_potential_analysis", {})
    risks = [str(x) for x in (dp.get("key_risks") or [])][:2]

    # 轮换侧重顺序，避免不同项目结构雷同
    if variant and emphasis:
        emphasis = emphasis[variant % len(emphasis):] + emphasis[: variant % len(emphasis)]

    lines: list[str] = []
    # 开头围绕用户现状/目标，而非固定话术
    if demand or goal:
        seg = []
        if demand:
            seg.append(f"你提到的现状问题——{demand}")
        if goal:
            seg.append(f"更新目标——{goal}")
        lines.append(
            f"针对{name}，结合{('、'.join(seg))}，本项目可按「{ptype}」方向推进，"
            + (f"核心更新逻辑是{core_logic}。" if core_logic else "重点围绕你的诉求展开。")
        )
    else:
        lines.append(f"{name}初步识别为「{ptype}」类更新，"
                     + (f"核心更新逻辑是{core_logic}。" if core_logic else ""))

    if facts:
        tail = f"，综合研判评分约 {fscore} 分（{level}）。" if fscore is not None else "，整体具备前期策划的数据基础。"
        lines.append("从已掌握的数据看，" + "，".join(facts) + tail)

    if emphasis:
        focus = emphasis[:3] + goal_focus[:2]
        focus = list(dict.fromkeys(focus))
        lines.append(f"结合该类型项目的更新规律，建议优先聚焦：{('；'.join(focus))}。")
    if risks:
        lines.append("需要重点应对的短板：" + "；".join(risks) + "。")

    lines.append("如果需要，我可以据此生成完整前策报告，或先就配套、人口、房价、产业中的某一维度深入分析。")
    return "\n\n".join([ln for ln in lines if ln and ln.strip()])


def _compose_diagnosis(ar: dict, conv: dict) -> str:
    """差异化研判：围绕用户现状/目标 + 项目类型侧重成文，并做相似度护栏。"""
    dprof = _diff_profile(ar, conv)
    facts = _facts_block(ar)
    dp = ar.get("demand_potential_analysis", {})
    prof = conv["profile"]
    name = ar.get("project_understanding", {}).get("name") or "该项目"
    ptype = dprof.get("canonical_type") or ar.get("renewal_type")
    key = str(conv.get("project_id") or conv.get("conversation_id"))

    def _prompt(extra: str = "") -> str:
        return (
            "你是资深城市更新前期策划顾问。请用中文、专业而通俗的口吻，基于下面"
            "『系统已算出的结构化结果』和『用户诉求』撰写一段差异化的项目初步研判（约220-340字，2-3段）。\n"
            "严格要求：\n"
            "1. 只能使用下列给出的数字与结论，不得新增、修改或推算任何数字、价格、比例、排名；\n"
            "2. 必须紧扣用户给出的现状问题与更新目标来展开，并体现该项目类型特有的更新重点；\n"
            "3. 不要套用固定三段式，不要写成'该项目属于X型、评分X、建议补短板'式的通用话术；\n"
            "4. 不得编造数据来源；不得出现模型、坐标系、英文字段、置信度、知识库等技术词；\n"
            "5. 不使用任何markdown符号（不要 # * - 等）。\n" + extra + "\n"
            f"项目名称：{name}\n"
            f"项目类型：{ptype}\n"
            f"该类型更新核心逻辑：{dprof.get('core_logic') or '（结合民生与产业需求确定）'}\n"
            f"该类型应重点强调：{('、'.join(dprof.get('emphasis') or [])) or '（按通用更新逻辑）'}\n"
            f"用户现状问题：{prof.get('update_demand') or '（用户暂未明确）'}\n"
            f"用户更新目标：{prof.get('expected_direction') or '（用户暂未明确）'}\n"
            f"目标相关侧重：{('、'.join(dprof.get('goal_focus') or [])) or '（无特别侧重）'}\n"
            f"综合评分：{ar.get('comprehensive_score')}（{ar.get('score_level')}）\n"
            f"现状量化：{('；'.join(facts)) or '（暂缺）'}\n"
            f"主要机会：{('；'.join(str(x) for x in (dp.get('key_opportunities') or [])[:3])) or '（暂缺）'}\n"
            f"主要风险：{('；'.join(str(x) for x in (dp.get('key_risks') or [])[:3])) or '（暂缺）'}\n"
            "请据此成文，并在结尾用一句话提示可生成完整前策报告或继续深入某一维度。"
        )

    res = deepseek_service.generate(_prompt(), thinking=False, temperature=0.5, max_tokens=900)
    text = res["text"] if res.get("ok") and res.get("text") else _deterministic_diagnosis(ar, conv, dprof)

    # 相似度护栏：与最近其它项目回答过于雷同 → 重写一次（换侧重/提高发散度）
    sim, _ = _most_similar_other(text, key)
    if sim > _SIM_THRESHOLD:
        logger.info("研判与既有项目相似度过高(%.2f)，触发重写。", sim)
        if res.get("ok"):
            res2 = deepseek_service.generate(
                _prompt("额外要求：本项目与其它项目差异明显，请重组结构、突出本项目独有的更新重点，避免与其它项目雷同。"),
                thinking=False, temperature=0.8, max_tokens=900)
            if res2.get("ok") and res2.get("text"):
                text = res2["text"]
        new_sim, _ = _most_similar_other(text, key)
        if new_sim > _SIM_THRESHOLD:
            text = _deterministic_diagnosis(ar, conv, dprof, variant=len(_RECENT_DIAGNOSES) + 1)

    _remember_diagnosis(key, text)
    return text


# --------------------------------------------------------------------------- #
# 报告（第一阶段沿用 builder/render；基于会话真实项目，不再用固定项目）
# --------------------------------------------------------------------------- #
def _generate_report(db: Session, conv: dict) -> dict[str, Any]:
    project = project_service.get_project(db, conv["project_id"])
    if project is None:
        return {"ok": False, "msg": "未找到当前项目，请先补充项目信息。"}
    # fail-closed：必须已有有效分析结果与模型运行编号，才允许出报告
    bundle = conv.get("_analysis")
    if not bundle or bundle.get("status") != "ok" or not bundle.get("model_run_id") \
            or not bundle.get("analysis_result"):
        return {"ok": False, "msg": "尚未获得有效的本地分析结果，请先完成项目研判再生成报告。"}
    style = conv["state"].get("case_style")
    content = rb.build_report(db, project, case_style_key=style)
    try:
        rendered = rw.build_and_convert(content, bundle)
    except rw.ReportGateError as exc:
        return {"ok": False, "msg": str(exc)}
    quality = q.evaluate(content,
                         pdf_ok=bool(rendered.get("pdf_size")),
                         docx_ok=bool(rendered.get("docx_size")))
    conv["state"].update({"report_ready": True, "report_id": content["report_id"],
                          "stage": STAGE_GENERATED})
    return {"ok": True, "content": content, "rendered": rendered, "quality": quality}


def _report_payload(conv: dict) -> dict[str, Any] | None:
    st = conv["state"]
    if not st.get("report_ready") or not st.get("report_id"):
        return None
    rid = st["report_id"]
    return {
        "ready": True, "report_id": rid, "chapters": 9,
        "docx_url": f"/api/report/{rid}/download-docx",
        "pdf_url": f"/api/report/{rid}/download-pdf",
        "quality_url": f"/api/report/{rid}/quality",
    }


# --------------------------------------------------------------------------- #
# 意图识别
# --------------------------------------------------------------------------- #
_GREETING = re.compile(r"^(你好|您好|哈喽|嗨|hi|hello|在吗|在么|早上好|下午好|晚上好)[!！。.~\s]*$",
                       re.IGNORECASE)


def _detect_intent(text: str, state: dict) -> tuple[str, Any]:
    t = (text or "").strip()
    low = t.lower()

    if re.fullmatch(r"[1-9]", t):
        opts = state.get("last_suggestions") or []
        idx = int(t) - 1
        if idx < len(opts):
            return "option", opts[idx]
        return "ask_clarify", None

    if _GREETING.match(t):
        return "greeting", None
    if any(k in t for k in ["你能做什么", "能干什么", "怎么用", "使用说明", "帮助", "介绍一下", "你是谁"]):
        return "help", None

    if (("pdf" in low) and ("导出" in t or "下载" in t)) or low == "pdf":
        return "export_pdf", None
    if ("word" in low or "文档" in t) and ("导出" in t or "下载" in t):
        return "export_docx", None
    if any(k in t for k in ["生成报告", "生成完整报告", "完整报告", "出报告", "生成前策报告", "生成正式报告", "写报告", "生成策划报告"]):
        return "generate_report", None
    if any(k in t for k in ["预览报告", "查看报告", "报告进度", "报告状态"]):
        return "report_status", None
    if ("鲁商" in t or "1992" in t or "标杆" in t) and any(k in t for k in ["风格", "案例", "参考", "思路", "写", "生成"]):
        return "style_lushang", None
    if any(k in t for k in ["分析", "研判", "评估", "看看", "怎么样", "诊断", "初判", "潜力"]):
        return "analyze", None
    if t in ("继续", "接着", "下一步", "go on", "continue"):
        return "continue", None
    return "chat", None


def _suggest_collect() -> list[dict]:
    return [
        {"label": "做项目初步研判", "action": "send", "text": "请给出这个项目的初步研判"},
        {"label": "生成前策报告", "action": "generate_report"},
    ]


def _suggest_after_analysis() -> list[dict]:
    return [
        {"label": "生成前策报告", "action": "generate_report"},
        {"label": "周边配套深入分析", "action": "send", "text": "请深入分析项目的周边配套情况"},
    ]


def _suggest_after_report() -> list[dict]:
    return [
        {"label": "下载 Word 报告", "action": "export_docx"},
        {"label": "导出 PDF", "action": "export_pdf"},
        {"label": "继续优化报告", "action": "send", "text": "请优化报告的前策核心建议部分"},
    ]


def _soft_missing(conv: dict) -> list[str]:
    prof = conv["profile"]
    miss = []
    if not prof.get("update_demand"):
        miss.append("现状问题/更新诉求")
    if not prof.get("expected_direction"):
        miss.append("更新目标/期望方向")
    return miss


def _preliminary_reply(db: Session, conv: dict) -> tuple[str, list[dict]]:
    """仅有地址、缺少现状/目标时：做初步位置识别与资料追问，不直接完整研判。"""
    ens = _ensure_project(db, conv)
    if not ens["ok"]:
        return ens["message"], _suggest_collect()
    prof = conv["profile"]
    district = prof.get("district") or "上海"
    guess = cl.classify_type_from_text(_user_text(conv))
    name = prof.get("name") or "该项目"
    line1 = f"已初步定位到{district}一带（{name}）。"
    if guess:
        gp = cl.TYPE_PROFILES.get(guess, {})
        line1 += f"从你目前提供的信息看，这可能偏向「{guess}」类更新，通常需要关注{gp.get('core_logic', '')}。"
    else:
        line1 += "目前还不能确定它属于哪类更新（老旧仓库/老旧社区/商业街区/工业遗存/公共空间等）。"
    line2 = ("要给出有依据的研判，请再补充两点：①目前最突出的现状问题（如界面陈旧、配套不足、"
             "停车困难、活力衰退、空间老化等）；②本次更新最希望达到的目标。")
    line3 = "你也可以点下方加号上传项目资料、现状照片或表格，我会把附件内容一并纳入分析。"
    return "\n\n".join([line1, line2, line3]), _suggest_collect()


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def chat(db: Session, *, message: str, conversation_id: str | None = None,
         project_id: int | None = None) -> dict[str, Any]:
    conv = _ensure(conversation_id, project_id)
    state = conv["state"]

    # 空消息 / 纯空格：不落盘、不研判，提示用户补充（根因修复，避免空会话冒出）。
    if not (message or "").strip():
        reply = "请先输入项目地址、地块名称或你的问题，再发送给我。"
        suggestions = _suggest_collect()
        state["last_suggestions"] = suggestions
        return _resp(conv, reply, suggestions, _report_payload(conv))

    conv["messages"].append({"role": "user", "text": message, "at": _now()})

    # 跨轮累积项目档案
    _merge_profile(conv, extract_profile(message))
    if conv["title"] == "新的城市更新咨询" and conv["profile"].get("name"):
        conv["title"] = conv["profile"]["name"][:24]

    intent, payload = _detect_intent(message, state)
    if intent == "option" and isinstance(payload, dict):
        act = payload.get("action")
        if act == "send":
            intent, message = "analyze", payload.get("text", message)
        else:
            intent = act
    if intent == "continue":
        intent = "generate_report" if state.get("analysis_done") and not state.get("report_ready") else "analyze"

    reply = ""
    suggestions: list[dict] = []
    report = _report_payload(conv)

    if intent == "greeting":
        reply = ("你好，我是城市更新前期策划助手。把你的项目情况告诉我，我可以帮你完成"
                 "区位配套、人口客群、房价空间与产业经济的研判，并生成完整的前期策划报告。\n\n"
                 "可以先告诉我：项目在上海市的具体位置（区/路/地块名称），以及现状问题和更新目标。")
        suggestions = _suggest_collect()

    elif intent == "help":
        reply = ("我可以做这几件事：一是基于项目位置做周边配套、人口、房价与产业的量化研判；"
                 "二是判断项目适合的更新类型并给出差异化策略方向；三是生成完整的城市更新前期策划报告，"
                 "支持下载 Word 与导出 PDF。\n\n你只需要提供项目的上海具体地址、现状问题与更新目标，"
                 "也可以通过加号上传相关资料，我会一并纳入分析。")
        suggestions = _suggest_collect()

    elif intent in ("analyze", "chat", "ask_clarify"):
        # 有位置（或档案里已有地址）→ 视信息充分度决定：初步识别 or 完整研判
        has_location = bool(conv["profile"].get("address")) or _looks_like_location(message)
        has_detail = bool(conv["profile"].get("update_demand") or conv["profile"].get("expected_direction"))
        explicit_analyze = (intent == "analyze")
        if has_location and not has_detail and not explicit_analyze:
            # 仅给地址：只做初步位置识别 + 资料追问，不直接完整研判
            reply, suggestions = _preliminary_reply(db, conv)
        elif has_location:
            res = _run_analysis(db, conv)
            if res["ok"]:
                ar = res["analysis"]["analysis_result"]
                reply = _compose_diagnosis(ar, conv)
                if not has_detail:
                    miss = _soft_missing(conv)
                    if miss:
                        reply += f"\n\n如果再补充【{('、'.join(miss))}】，我可以把更新方向与策略判断得更贴合你的项目。"
                suggestions = _suggest_after_analysis()
            else:
                reply = res["message"]
                suggestions = _suggest_collect()
        else:
            # 信息不足 → fail-closed 主动追问
            if intent == "analyze":
                reply = ("好的。要做研判，我需要先知道项目的具体位置。请告诉我项目在上海市的"
                         "地址或地块名称（例如：上海市徐汇区龙华XX路），有现状问题和更新目标也一并告诉我。")
            else:
                reply = ("我可以帮你研判这个城市更新项目。请先告诉我项目在上海市的具体位置（区/路/地块名称），"
                         "并简单描述现状问题与更新目标。")
            suggestions = _suggest_collect()

    elif intent == "generate_report":
        # 信息不足必须 fail-closed，不能凭空出报告
        if not state.get("analysis_done"):
            res = _run_analysis(db, conv)
            if not res["ok"]:
                reply = ("要生成完整的前期策划报告，我需要先基于真实数据完成分析。" + res["message"])
                suggestions = _suggest_collect()
                state["last_suggestions"] = suggestions
                conv["messages"].append({"role": "assistant", "text": _sanitize(reply), "at": _now()})
                _touch_meta(conv)
                return _resp(conv, _sanitize(reply), suggestions, report)
        result = _generate_report(db, conv)
        if not result["ok"]:
            reply = result["msg"]
        else:
            reply = ("前策报告已生成。已根据当前项目资料生成城市更新前期策划报告。"
                     "你可以预览内容、复制正文，或下载 Word / 导出 PDF。")
            report = _report_payload(conv)
            suggestions = _suggest_after_report()

    elif intent in ("export_pdf", "export_docx"):
        if not state.get("report_ready"):
            reply = "目前还没有已生成的报告。请先生成前策报告，我再为你导出。"
            suggestions = [{"label": "生成前策报告", "action": "generate_report"}]
        else:
            rid = state["report_id"]
            fmt = "PDF" if intent == "export_pdf" else "Word"
            url = f"/api/report/{rid}/download-{'pdf' if intent == 'export_pdf' else 'docx'}"
            reply = f"{fmt} 报告已准备好，可直接下载。"
            report = _report_payload(conv)
            report["download_now"] = url
            suggestions = _suggest_after_report()

    elif intent == "report_status":
        if state.get("report_ready"):
            reply = "当前报告已生成完毕，结构完整，可下载 Word 或导出 PDF。"
            suggestions = _suggest_after_report()
        else:
            reply = "当前尚未生成报告。完成研判后即可生成完整的城市更新前期策划报告。"
            suggestions = [{"label": "生成前策报告", "action": "generate_report"}]

    elif intent == "style_lushang":
        state["case_style"] = "按照鲁商1992案例风格"
        if not state.get("analysis_done"):
            res = _run_analysis(db, conv)
            if not res["ok"]:
                reply = ("我可以按存量商业更新与历史建筑活化的思路来组织策略，但需要先完成基于真实数据的分析。"
                         + res["message"])
                suggestions = _suggest_collect()
                state["last_suggestions"] = suggestions
                conv["messages"].append({"role": "assistant", "text": _sanitize(reply), "at": _now()})
                _touch_meta(conv)
                return _resp(conv, _sanitize(reply), suggestions, report)
        ar = conv["_analysis"]["analysis_result"]
        reply = ("已按存量商业更新与历史建筑活化的思路为你组织策略方向，重点参考连廊串联、"
                 "空间复合利用、首层界面激活与场所记忆延续等更新手法。\n\n"
                 + _compose_diagnosis(ar, conv)
                 + "\n\n如需要，我可以据此直接生成完整前策报告，该思路会贯穿到案例参考与核心建议章节。")
        suggestions = _suggest_after_analysis()

    if not reply.strip():
        reply = "我已记录你的输入。可以告诉我项目的上海具体地址与更新目标，或选择下一步操作。"
        suggestions = suggestions or _suggest_collect()

    reply = _sanitize(reply)
    state["last_suggestions"] = suggestions
    conv["messages"].append({"role": "assistant", "text": reply, "at": _now()})
    _touch_meta(conv)
    return _resp(conv, reply, suggestions, report)


def _resp(conv: dict, reply: str, suggestions: list[dict], report) -> dict[str, Any]:
    return {
        "conversation_id": conv["conversation_id"],
        "project_id": conv["project_id"],
        "title": conv["title"],
        "reply": reply,
        "stage": conv["state"]["stage"],
        "suggestions": suggestions,
        "report": report,
        "turn": len([m for m in conv["messages"] if m["role"] == "user"]),
    }
