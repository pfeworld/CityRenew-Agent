"""全网公开数据源发现服务（第10B 实现）。

目标：不是乱爬数据，而是对候选数据源做"发现 → 登记 → 合规等级判断 → 可采集性评估"。
本阶段使用本地预置候选源库 + 关键词规则匹配，不联网搜索。若后续需要真正联网检索，
必须在单独阶段确认后再开启（本文件预留 ``online_search=False`` 开关，默认关闭）。

红线：
- 不联网、不采集；仅返回候选源的元数据评估（脱敏）。
- robots 禁止 / 需绕登录验证码反爬 / 商业授权不明 / 含隐私 → 判 Level 0，不可采。
- 商业数据库默认不采；授权不明只做候选不入训练。
- 发现日志落 ``external/registry/source_discovery_log.json``、候选落
  ``external/registry/candidate_sources.json``（均已 gitignore）。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger("cityrenew.web_discovery")


def _cand(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "source_name": "",
        "source_url": "",
        "provider": "",
        "source_type": "",
        "data_category": "",
        "access_method": "unknown",  # api / download / manual / webpage / unknown
        "license_detected": "unknown",
        "robots_policy_status": "unknown",
        "api_available": False,
        "requires_key": False,
        "estimated_update_frequency": "unknown",
        "expected_fields": [],
        "expected_record_count": "unknown",
        "collection_feasibility": "unknown",  # high / medium / low / blocked
        "compliance_risk": "needs_review",  # low / medium / high / blocked
        "collection_level": 1,
        "recommended_usage": [],
        "can_use_for_training": False,
        "can_use_for_feature_engineering": False,
        "can_use_for_report": False,
        "can_use_for_eval": False,
        "match_keywords": [],
        "notes": "",
    }
    base.update(kwargs)
    return base


# 本地预置候选源库（官方/开放/授权优先；商业站标 blocked）
CANDIDATE_LIBRARY: tuple[dict[str, Any], ...] = (
    _cand(
        source_name="上海市公共数据开放平台",
        source_url="https://data.sh.gov.cn/",
        provider="上海市大数据中心",
        source_type="gov_open_data",
        data_category="open_data",
        access_method="download",
        license_detected="gov_open",
        robots_policy_status="allowed",
        api_available=True,
        requires_key=False,
        estimated_update_frequency="periodic",
        expected_fields=["dataset_name", "field_schema", "update_time"],
        expected_record_count="varies",
        collection_feasibility="high",
        compliance_risk="low",
        collection_level=2,
        recommended_usage=["feature_engineering", "report", "training"],
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        can_use_for_training=True,
        match_keywords=["上海", "开放数据", "公共数据", "数据", "人口", "产业", "房价", "公共服务"],
        notes="官方开放平台，优先来源；手动下载或 API 导入。",
    ),
    _cand(
        source_name="国家统计局公开数据",
        source_url="https://data.stats.gov.cn/",
        provider="国家统计局",
        source_type="gov_statistics",
        data_category="statistics",
        access_method="download",
        license_detected="gov_open",
        robots_policy_status="allowed",
        api_available=True,
        requires_key=False,
        estimated_update_frequency="periodic",
        expected_fields=["indicator", "year", "region", "value"],
        expected_record_count="varies",
        collection_feasibility="high",
        compliance_risk="low",
        collection_level=2,
        recommended_usage=["feature_engineering", "report"],
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        match_keywords=["统计", "统计局", "人口", "产业", "房价", "经济", "消费", "宏观"],
        notes="宏观统计指标，用于报告佐证与宏观特征。",
    ),
    _cand(
        source_name="上海统计年鉴",
        source_url="https://tjj.sh.gov.cn/tjnj/",
        provider="上海市统计局",
        source_type="gov_statistics",
        data_category="statistics",
        access_method="download",
        license_detected="gov_open",
        robots_policy_status="allowed",
        api_available=False,
        requires_key=False,
        estimated_update_frequency="yearly",
        expected_fields=["indicator", "year", "region", "value"],
        expected_record_count="varies",
        collection_feasibility="medium",
        compliance_risk="low",
        collection_level=2,
        recommended_usage=["feature_engineering", "report"],
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        match_keywords=["统计年鉴", "年鉴", "上海", "人口", "产业", "房价", "经济"],
        notes="年鉴 Excel/PDF 手动下载导入。",
    ),
    _cand(
        source_name="高德地图开放平台 Web 服务 API",
        source_url="https://lbs.amap.com/api/webservice/summary",
        provider="高德开放平台",
        source_type="map_poi",
        data_category="poi",
        access_method="api",
        license_detected="commercial_api_terms",
        robots_policy_status="api_terms",
        api_available=True,
        requires_key=True,
        estimated_update_frequency="on_demand",
        expected_fields=["name", "type", "location", "address"],
        expected_record_count="quota_limited",
        collection_feasibility="high",
        compliance_risk="medium",
        collection_level=3,
        recommended_usage=["feature_engineering", "report"],
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        match_keywords=["高德", "poi", "地图", "兴趣点", "医院", "学校", "养老", "周边", "可达性"],
        notes="需 AMAP_KEY；按配额/频率合规采集，不绕配额、不声称全量。",
    ),
    _cand(
        source_name="OpenStreetMap Overpass API",
        source_url="https://overpass-api.de/api/interpreter",
        provider="OpenStreetMap 社区",
        source_type="map_poi",
        data_category="poi_road",
        access_method="api",
        license_detected="ODbL",
        robots_policy_status="allowed",
        api_available=True,
        requires_key=False,
        estimated_update_frequency="continuous",
        expected_fields=["amenity", "name", "lat", "lon"],
        expected_record_count="varies",
        collection_feasibility="high",
        compliance_risk="low",
        collection_level=3,
        recommended_usage=["feature_engineering", "report", "training"],
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        can_use_for_training=True,
        match_keywords=["osm", "openstreetmap", "poi", "路网", "地图", "可达性", "交通"],
        notes="ODbL 开放许可，需署名；遵守 Overpass 限流。",
    ),
    _cand(
        source_name="医疗卫生设施名录（开放数据）",
        source_url="https://data.sh.gov.cn/",
        provider="卫健主管部门 / 开放数据",
        source_type="public_service",
        data_category="healthcare",
        access_method="download",
        license_detected="gov_open",
        robots_policy_status="allowed",
        api_available=False,
        requires_key=False,
        estimated_update_frequency="periodic",
        expected_fields=["name", "level", "address", "location"],
        expected_record_count="varies",
        collection_feasibility="medium",
        compliance_risk="low",
        collection_level=2,
        recommended_usage=["feature_engineering", "report"],
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        match_keywords=["医院", "医疗", "卫生", "公共服务", "名录"],
        notes="医疗设施名录，用于医疗可达性/公共服务短板分析。",
    ),
    _cand(
        source_name="学校教育设施名录（开放数据）",
        source_url="https://data.sh.gov.cn/",
        provider="教育主管部门 / 开放数据",
        source_type="public_service",
        data_category="education",
        access_method="download",
        license_detected="gov_open",
        robots_policy_status="allowed",
        api_available=False,
        requires_key=False,
        estimated_update_frequency="periodic",
        expected_fields=["name", "type", "address", "location"],
        expected_record_count="varies",
        collection_feasibility="medium",
        compliance_risk="low",
        collection_level=2,
        recommended_usage=["feature_engineering", "report"],
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        match_keywords=["学校", "教育", "公共服务", "名录"],
        notes="学校名录，用于 15 分钟生活圈/教育可达性。",
    ),
    _cand(
        source_name="养老服务设施名录（开放数据）",
        source_url="https://data.sh.gov.cn/",
        provider="民政主管部门 / 开放数据",
        source_type="public_service",
        data_category="elderly",
        access_method="download",
        license_detected="gov_open",
        robots_policy_status="allowed",
        api_available=False,
        requires_key=False,
        estimated_update_frequency="periodic",
        expected_fields=["name", "type", "address", "location"],
        expected_record_count="varies",
        collection_feasibility="medium",
        compliance_risk="low",
        collection_level=2,
        recommended_usage=["feature_engineering", "report"],
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        match_keywords=["养老", "养老机构", "民政", "公共服务", "名录"],
        notes="养老机构名录，用于老龄配套分析。",
    ),
    _cand(
        source_name="地铁/公交站点开放数据",
        source_url="https://data.sh.gov.cn/",
        provider="交通主管部门 / 开放数据 / OSM",
        source_type="transport",
        data_category="transport",
        access_method="download",
        license_detected="gov_open",
        robots_policy_status="allowed",
        api_available=False,
        requires_key=False,
        estimated_update_frequency="periodic",
        expected_fields=["station_name", "line", "location"],
        expected_record_count="varies",
        collection_feasibility="medium",
        compliance_risk="low",
        collection_level=2,
        recommended_usage=["feature_engineering", "report"],
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        match_keywords=["地铁", "公交", "站点", "交通", "可达性", "轨交"],
        notes="站点位置，用于轨交/公交可达性特征。",
    ),
    _cand(
        source_name="上海城市更新规划与政策公示",
        source_url="https://ghzyj.sh.gov.cn/",
        provider="上海市规划资源局",
        source_type="planning_policy",
        data_category="policy",
        access_method="webpage",
        license_detected="gov_public",
        robots_policy_status="allowed",
        api_available=False,
        requires_key=False,
        estimated_update_frequency="periodic",
        expected_fields=["title", "publish_date", "content_summary"],
        expected_record_count="varies",
        collection_feasibility="low",
        compliance_risk="low",
        collection_level=2,
        recommended_usage=["report"],
        can_use_for_report=True,
        match_keywords=["规划", "政策", "城市更新", "控规", "公告", "土地", "出让", "公示"],
        notes="公开规划/政策资料，仅作报告政策依据，建议人工核对。",
    ),
    # ---- 商业风险源：发现但判 Level 0 / blocked，默认不可采、不可训练 ----
    _cand(
        source_name="链家 / 贝壳 / 安居客 / 房天下（商业房产网站）",
        source_url="https://sh.lianjia.com/",
        provider="商业房产网站",
        source_type="commercial_property_site",
        data_category="property",
        access_method="webpage",
        license_detected="proprietary",
        robots_policy_status="restricted",
        api_available=False,
        requires_key=False,
        estimated_update_frequency="unknown",
        expected_fields=[],
        expected_record_count="unknown",
        collection_feasibility="blocked",
        compliance_risk="blocked",
        collection_level=0,
        recommended_usage=[],
        match_keywords=["链家", "贝壳", "安居客", "房天下", "二手房", "房价", "房源"],
        notes="未授权商业房产网站，含个人信息风险，默认不采集、不入训练；除非明确授权+条款允许。",
    ),
    _cand(
        source_name="企查查 / 天眼查 / 启信宝（商业企业数据库）",
        source_url="https://www.qcc.com/",
        provider="商业企业数据库",
        source_type="commercial_enterprise_db",
        data_category="enterprise",
        access_method="webpage",
        license_detected="proprietary",
        robots_policy_status="restricted",
        api_available=False,
        requires_key=False,
        estimated_update_frequency="unknown",
        expected_fields=[],
        expected_record_count="unknown",
        collection_feasibility="blocked",
        compliance_risk="blocked",
        collection_level=0,
        recommended_usage=[],
        match_keywords=["企查查", "天眼查", "启信宝", "企业", "工商", "产业"],
        notes="未授权商业企业数据库，默认不采集、不入训练；除非授权或合规 API。",
    ),
)

_TOKEN_SPLIT = re.compile(r"[\s,，;；/|]+")


def _tokenize(keyword: str) -> list[str]:
    if not keyword:
        return []
    raw = [t.strip().lower() for t in _TOKEN_SPLIT.split(keyword) if t.strip()]
    return raw


def _match_score(cand: dict[str, Any], tokens: list[str]) -> tuple[int, list[str]]:
    """返回 (命中分, 命中关键词)。命中候选 match_keywords / 名称 / 类目 / provider。"""
    if not tokens:
        return 0, []
    haystay = " ".join(
        [str(cand.get(k, "")) for k in ("source_name", "source_type", "data_category", "provider")]
        + [str(x) for x in cand.get("match_keywords", [])]
    ).lower()
    kws = [str(x).lower() for x in cand.get("match_keywords", [])]
    hits: list[str] = []
    for t in tokens:
        if any(t in kw or kw in t for kw in kws) or t in haystay:
            hits.append(t)
    return len(hits), hits


def _external_dir() -> Path:
    return settings.data_dir / "external"


def _registry_dir() -> Path:
    return _external_dir() / "registry"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persist(keyword: str, candidates: list[dict[str, Any]]) -> None:
    """把候选与发现日志落到 external/registry/（已 gitignore）。"""
    try:
        reg_dir = _registry_dir()
        reg_dir.mkdir(parents=True, exist_ok=True)
        # candidate_sources.json（全量候选库快照）
        cand_path = reg_dir / "candidate_sources.json"
        with cand_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"updated_at": _utcnow_iso(), "candidates": list(CANDIDATE_LIBRARY)},
                f, ensure_ascii=False, indent=2,
            )
        # source_discovery_log.json（追加一次发现记录）
        log_path = reg_dir / "source_discovery_log.json"
        log: list[dict[str, Any]] = []
        if log_path.exists():
            try:
                with log_path.open("r", encoding="utf-8") as f:
                    log = json.load(f)
                if not isinstance(log, list):
                    log = []
            except Exception:  # noqa: BLE001
                log = []
        log.append({
            "keyword": keyword,
            "matched_count": len(candidates),
            "online_search": False,
            "created_at": _utcnow_iso(),
        })
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(log[-200:], f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("discovery persist failed: %s", exc)


def discover_sources(keyword: str = "", online_search: bool = False) -> dict[str, Any]:
    """根据关键词返回候选数据源清单（本地规则，不联网）。

    online_search 预留：默认 False；本阶段不开启联网检索（需单独阶段确认）。
    无关键词时返回全部候选（官方/开放优先排序）。
    """
    tokens = _tokenize(keyword)
    scored: list[tuple[int, dict[str, Any]]] = []
    for cand in CANDIDATE_LIBRARY:
        score, hits = _match_score(cand, tokens)
        if tokens and score == 0:
            continue
        item = dict(cand)
        item["match_keywords"] = hits if tokens else cand.get("match_keywords", [])
        scored.append((score, item))

    if not tokens:
        # 无关键词：全部候选，官方/开放优先（compliance_risk low 在前）
        scored = [(0, dict(c)) for c in CANDIDATE_LIBRARY]

    risk_order = {"low": 0, "medium": 1, "high": 2, "blocked": 3, "needs_review": 1}
    scored.sort(key=lambda x: (-x[0], risk_order.get(x[1].get("compliance_risk"), 2)))
    candidates = [item for _, item in scored]

    _persist(keyword, candidates)

    blocked = [c for c in candidates if c.get("collection_level") == 0]
    return {
        "keyword": keyword,
        "online_search": online_search,
        "count": len(candidates),
        "candidates": candidates,
        "blocked_count": len(blocked),
        "notes": [
            "本发现服务使用本地预置候选源库 + 关键词规则匹配，不联网搜索、不采集数据。",
            "官方/开放/授权数据源优先；商业房产/企业数据库判 Level 0（blocked），默认不可采、不可训练。",
            "如需真正联网检索数据源，必须在单独阶段确认后开启（当前 online_search 恒为 False）。",
            "候选仅作登记评估，授权不明者仅入候选、不进训练。",
        ],
    }
