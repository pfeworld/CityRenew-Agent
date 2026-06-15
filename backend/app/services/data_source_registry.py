"""数据源登记（第10B 实现）。

职责：定义合规外部数据源的登记字段契约、预置数据源、合规采集等级，并把登记表
落地到 ``backend/data/external/registry/data_source_registry.json``（已 gitignore）。

红线（与项目规则 / 方案第十节一致）：
- 外部数据必须记录 来源 / 授权 / 采集方式 / 用途 / 血缘。
- API Key 一律从 .env 读取（``api_key_env_name`` 仅记录变量名，绝不存 Key 本身）。
- 商业网站默认不采集：``compliance_status=risk_or_unavailable`` 且 ``can_use_for_training=false``。
- 无 Key / 无授权时采集接口返回 not_configured / unavailable，绝不伪造数据。
- 本服务只读 / 登记，不联网、不采集（采集在 external_data_collector_service）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger("cityrenew.data_source_registry")

# 数据源登记字段契约（写入 registry/data_source_registry.json）
SOURCE_SCHEMA: tuple[str, ...] = (
    "source_id",
    "source_name",
    "source_type",
    "provider",
    "official_url_or_api",
    "license_type",
    "collection_method",
    "api_required",
    "api_key_env_name",
    "allowed_usage",
    "forbidden_usage",
    "update_frequency",
    "coordinate_system",
    "privacy_level",
    "compliance_status",
    "collection_level",
    "can_use_for_training",
    "can_use_for_feature_engineering",
    "can_use_for_report",
    "can_use_for_eval",
    "notes",
)

# 合规采集等级（与方案第八节一致）
COLLECTION_LEVELS: dict[int, str] = {
    0: "不可采集（robots 禁止 / 需绕登录验证码反爬 / 商业授权不明 / 含隐私）",
    1: "人工登记（授权不明，仅候选，不入训练）",
    2: "手动下载导入（官方开放平台 Excel/CSV/JSON，用户手动下载上传）",
    3: "官方 API 接入（高德/百度/腾讯/政府开放数据 API，按 Key/配额/频率）",
    4: "授权数据接入（用户授权文件 / 商业采购 / 脱敏合作，可入训练并记 license）",
}

# 需要的 API Key 环境变量名（一律从 .env 读取，缺失返回 not_configured）
API_KEY_ENV_NAMES: dict[str, str] = {
    "amap": "AMAP_KEY",
    "baidu_map": "BAIDU_MAP_KEY",
    "tencent_map": "TENCENT_MAP_KEY",
}


def _src(**kwargs: Any) -> dict[str, Any]:
    """构造一个登记条目，按 SOURCE_SCHEMA 补齐默认值。"""
    base: dict[str, Any] = {
        "source_id": "",
        "source_name": "",
        "source_type": "",
        "provider": "",
        "official_url_or_api": "",
        "license_type": "unknown",
        "collection_method": "manual",
        "api_required": False,
        "api_key_env_name": None,
        "allowed_usage": [],
        "forbidden_usage": [],
        "update_frequency": "unknown",
        "coordinate_system": "unknown",
        "privacy_level": "none",
        "compliance_status": "needs_review",
        "collection_level": 1,
        "can_use_for_training": False,
        "can_use_for_feature_engineering": False,
        "can_use_for_report": False,
        "can_use_for_eval": False,
        "notes": "",
    }
    base.update(kwargs)
    return base


# --------------------------------------------------------------------------- #
# 预置数据源（代码内默认；registry 文件初始化时写入，可经 register-source 追加）
# --------------------------------------------------------------------------- #
PRESET_SOURCES: tuple[dict[str, Any], ...] = (
    # ---- 地图与 POI（官方 API，Level 3）----
    _src(
        source_id="amap_poi",
        source_name="高德地图 POI / 周边 / 行政区划",
        source_type="map_poi",
        provider="高德开放平台 AMap",
        official_url_or_api="https://lbs.amap.com/api/webservice/summary",
        license_type="commercial_api_terms",
        collection_method="official_api",
        api_required=True,
        api_key_env_name="AMAP_KEY",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["claim_full_dataset", "bypass_quota", "personal_privacy"],
        update_frequency="on_demand",
        coordinate_system="GCJ02",
        privacy_level="none",
        compliance_status="ok_with_key",
        collection_level=3,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="需 AMAP_KEY；按配额/频率合规采集，不绕配额、不声称全量、不换账号/IP。",
    ),
    _src(
        source_id="baidu_map_poi",
        source_name="百度地图 地点检索 / 逆地理",
        source_type="map_poi",
        provider="百度地图开放平台",
        official_url_or_api="https://lbsyun.baidu.com/index.php?title=webapi",
        license_type="commercial_api_terms",
        collection_method="official_api",
        api_required=True,
        api_key_env_name="BAIDU_MAP_KEY",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["claim_full_dataset", "bypass_quota", "personal_privacy"],
        update_frequency="on_demand",
        coordinate_system="BD09",
        privacy_level="none",
        compliance_status="planned",
        collection_level=3,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="第10B 预留：无 BAIDU_MAP_KEY 返回 not_configured；坐标系 BD09 需转换。",
    ),
    _src(
        source_id="tencent_map_poi",
        source_name="腾讯位置服务 地点搜索 / 路线",
        source_type="map_poi",
        provider="腾讯位置服务",
        official_url_or_api="https://lbs.qq.com/service/webService/webServiceGuide/webServiceOverview",
        license_type="commercial_api_terms",
        collection_method="official_api",
        api_required=True,
        api_key_env_name="TENCENT_MAP_KEY",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["claim_full_dataset", "bypass_quota", "personal_privacy"],
        update_frequency="on_demand",
        coordinate_system="GCJ02",
        privacy_level="none",
        compliance_status="planned",
        collection_level=3,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="第10B 预留：无 TENCENT_MAP_KEY 返回 not_configured。",
    ),
    _src(
        source_id="osm_overpass",
        source_name="OpenStreetMap Overpass POI / 路网",
        source_type="map_poi",
        provider="OpenStreetMap 社区",
        official_url_or_api="https://overpass-api.de/api/interpreter",
        license_type="ODbL",
        collection_method="open_api",
        api_required=False,
        api_key_env_name=None,
        allowed_usage=["feature_engineering", "report", "training"],
        forbidden_usage=["abuse_rate", "ignore_attribution"],
        update_frequency="continuous",
        coordinate_system="WGS84",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=3,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        can_use_for_training=True,
        notes="ODbL 开放许可，需署名；遵守 Overpass 限流，支持手动导入 extract。",
    ),
    # ---- 政府开放数据（官方下载/接口，Level 2/3）----
    _src(
        source_id="shanghai_open_data",
        source_name="上海市公共数据开放平台",
        source_type="gov_open_data",
        provider="上海市大数据中心",
        official_url_or_api="https://data.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        api_required=False,
        api_key_env_name=None,
        allowed_usage=["feature_engineering", "report", "training"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="unknown",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        can_use_for_training=True,
        notes="官方开放平台，手动下载导入为主；部分数据集有 API。",
    ),
    _src(
        source_id="stats_cn",
        source_name="国家统计局公开数据",
        source_type="gov_statistics",
        provider="国家统计局",
        official_url_or_api="https://data.stats.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        api_required=False,
        api_key_env_name=None,
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="n/a",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="宏观统计指标，手动下载或公开接口导入；主要用于报告佐证与宏观特征。",
    ),
    _src(
        source_id="shanghai_statistical_yearbook",
        source_name="上海统计年鉴",
        source_type="gov_statistics",
        provider="上海市统计局",
        official_url_or_api="https://tjj.sh.gov.cn/tjnj/",
        license_type="gov_open",
        collection_method="manual_download",
        api_required=False,
        api_key_env_name=None,
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="yearly",
        coordinate_system="n/a",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="年鉴 Excel/PDF 手动下载导入；人口/经济/产业/消费宏观指标。",
    ),
    _src(
        source_id="shanghai_planning_policy",
        source_name="上海城市更新规划与政策公示",
        source_type="planning_policy",
        provider="上海市规划资源局 / 住建委等",
        official_url_or_api="https://ghzyj.sh.gov.cn/",
        license_type="gov_public",
        collection_method="manual_download",
        api_required=False,
        api_key_env_name=None,
        allowed_usage=["report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="n/a",
        privacy_level="none",
        compliance_status="public",
        collection_level=2,
        can_use_for_report=True,
        notes="控规/总规/专项规划/更新公告等公开资料，仅作报告政策依据。",
    ),
    # ---- 公共服务（主管部门名录，Level 2）----
    _src(
        source_id="education_public_service",
        source_name="教育公共服务设施名录（学校）",
        source_type="public_service",
        provider="教育主管部门 / 开放数据",
        official_url_or_api="https://data.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="unknown",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="学校名录，用于 15 分钟生活圈/公共服务覆盖特征。",
    ),
    _src(
        source_id="healthcare_public_service",
        source_name="医疗卫生设施名录（医院/社区卫生）",
        source_type="public_service",
        provider="卫健主管部门 / 开放数据",
        official_url_or_api="https://data.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="unknown",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="医疗设施名录，用于医疗可达性/公共服务短板分析。",
    ),
    _src(
        source_id="elderly_public_service",
        source_name="养老服务设施名录",
        source_type="public_service",
        provider="民政主管部门 / 开放数据",
        official_url_or_api="https://data.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="unknown",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="养老机构/服务设施名录，用于老龄配套分析。",
    ),
    _src(
        source_id="sports_public_service",
        source_name="体育健身设施名录",
        source_type="public_service",
        provider="体育主管部门 / 开放数据",
        official_url_or_api="https://data.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="unknown",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="体育场馆/健身设施名录，用于公共服务配套分析。",
    ),
    _src(
        source_id="culture_public_service",
        source_name="文化设施名录（文化馆/图书馆/博物馆）",
        source_type="public_service",
        provider="文旅主管部门 / 开放数据",
        official_url_or_api="https://data.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="unknown",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="文化设施名录，用于文化活力/公共服务分析。",
    ),
    # ---- 交通（开放数据，Level 2/3）----
    _src(
        source_id="metro_station_open_data",
        source_name="地铁站点开放数据",
        source_type="transport",
        provider="申通地铁 / 开放数据 / OSM",
        official_url_or_api="https://data.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="unknown",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="地铁站点位置，用于轨交可达性特征。",
    ),
    _src(
        source_id="bus_station_open_data",
        source_name="公交站点开放数据",
        source_type="transport",
        provider="交通主管部门 / 开放数据 / OSM",
        official_url_or_api="https://data.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="unknown",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="公交站点位置，用于公交便利度特征。",
    ),
    _src(
        source_id="road_network_open_data",
        source_name="路网开放数据",
        source_type="transport",
        provider="OSM / 开放数据",
        official_url_or_api="https://www.openstreetmap.org/",
        license_type="ODbL",
        collection_method="open_api",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["ignore_attribution"],
        update_frequency="continuous",
        coordinate_system="WGS84",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=3,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="路网用于通勤/可达性特征，ODbL 需署名。",
    ),
    # ---- 房价与产业（授权/政府/开放，Level 2/4）----
    _src(
        source_id="authorized_property_upload",
        source_name="用户授权房价数据上传",
        source_type="property",
        provider="用户授权 / 脱敏合作",
        official_url_or_api="local_upload",
        license_type="user_authorized",
        collection_method="user_upload",
        allowed_usage=["feature_engineering", "report", "training"],
        forbidden_usage=["scrape_commercial_site", "personal_privacy"],
        update_frequency="on_demand",
        coordinate_system="unknown",
        privacy_level="needs_desensitize",
        compliance_status="authorized_on_upload",
        collection_level=4,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        can_use_for_training=True,
        notes="仅用户上传授权文件；须填 data_owner/license/allowed_usage 并脱敏，授权明确方可入训练。",
    ),
    _src(
        source_id="government_property_statistics",
        source_name="政府房价/成交统计",
        source_type="property",
        provider="统计局 / 住建/房管部门",
        official_url_or_api="https://tjj.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="n/a",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="官方房价指数/成交统计，用于市场判断与价格梯度佐证。",
    ),
    _src(
        source_id="industrial_park_public_list",
        source_name="产业园区公开名录",
        source_type="industry",
        provider="发改 / 经信 / 招商公开名录",
        official_url_or_api="https://data.sh.gov.cn/",
        license_type="gov_open",
        collection_method="manual_download",
        allowed_usage=["feature_engineering", "report"],
        forbidden_usage=["personal_privacy"],
        update_frequency="periodic",
        coordinate_system="unknown",
        privacy_level="none",
        compliance_status="open_licensed",
        collection_level=2,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        notes="园区/招商公开名录，用于产业集聚与主导产业分析。",
    ),
    _src(
        source_id="authorized_enterprise_data",
        source_name="授权企业/产业数据",
        source_type="industry",
        provider="用户授权 / 商业采购 / 开放数据集",
        official_url_or_api="local_upload",
        license_type="user_authorized",
        collection_method="user_upload",
        allowed_usage=["feature_engineering", "report", "training"],
        forbidden_usage=["scrape_commercial_db", "personal_privacy"],
        update_frequency="on_demand",
        coordinate_system="unknown",
        privacy_level="needs_desensitize",
        compliance_status="authorized_on_upload",
        collection_level=4,
        can_use_for_feature_engineering=True,
        can_use_for_report=True,
        can_use_for_training=True,
        notes="优先政府/统计/开放数据；企查查/天眼查/启信宝默认不抓，授权或合规 API 方可。",
    ),
    # ---- 商业风险数据源（默认不可采集 / 不可训练）----
    _src(
        source_id="lianjia",
        source_name="链家",
        source_type="commercial_property_site",
        provider="链家（商业网站）",
        official_url_or_api="https://sh.lianjia.com/",
        license_type="proprietary",
        collection_method="not_collect",
        allowed_usage=[],
        forbidden_usage=["scrape", "training", "bypass_anti_crawler"],
        update_frequency="unknown",
        coordinate_system="unknown",
        privacy_level="contains_personal_info_risk",
        compliance_status="risk_or_unavailable",
        collection_level=0,
        notes="未授权商业房产网站，默认不采集、不入训练；除非明确授权+条款允许+频率合规。",
    ),
    _src(
        source_id="beike",
        source_name="贝壳找房",
        source_type="commercial_property_site",
        provider="贝壳（商业网站）",
        official_url_or_api="https://sh.ke.com/",
        license_type="proprietary",
        collection_method="not_collect",
        allowed_usage=[],
        forbidden_usage=["scrape", "training", "bypass_anti_crawler"],
        update_frequency="unknown",
        coordinate_system="unknown",
        privacy_level="contains_personal_info_risk",
        compliance_status="risk_or_unavailable",
        collection_level=0,
        notes="未授权商业房产网站，默认不采集、不入训练。",
    ),
    _src(
        source_id="anjuke",
        source_name="安居客",
        source_type="commercial_property_site",
        provider="安居客（商业网站）",
        official_url_or_api="https://shanghai.anjuke.com/",
        license_type="proprietary",
        collection_method="not_collect",
        allowed_usage=[],
        forbidden_usage=["scrape", "training", "bypass_anti_crawler"],
        update_frequency="unknown",
        coordinate_system="unknown",
        privacy_level="contains_personal_info_risk",
        compliance_status="risk_or_unavailable",
        collection_level=0,
        notes="未授权商业房产网站，默认不采集、不入训练。",
    ),
    _src(
        source_id="fangtianxia",
        source_name="房天下",
        source_type="commercial_property_site",
        provider="房天下（商业网站）",
        official_url_or_api="https://sh.fang.com/",
        license_type="proprietary",
        collection_method="not_collect",
        allowed_usage=[],
        forbidden_usage=["scrape", "training", "bypass_anti_crawler"],
        update_frequency="unknown",
        coordinate_system="unknown",
        privacy_level="contains_personal_info_risk",
        compliance_status="risk_or_unavailable",
        collection_level=0,
        notes="未授权商业房产网站，默认不采集、不入训练。",
    ),
    _src(
        source_id="qichacha",
        source_name="企查查",
        source_type="commercial_enterprise_db",
        provider="企查查（商业数据库）",
        official_url_or_api="https://www.qcc.com/",
        license_type="proprietary",
        collection_method="not_collect",
        allowed_usage=[],
        forbidden_usage=["scrape", "training", "bypass_anti_crawler"],
        update_frequency="unknown",
        coordinate_system="n/a",
        privacy_level="contains_personal_info_risk",
        compliance_status="risk_or_unavailable",
        collection_level=0,
        notes="未授权商业企业数据库，默认不采集、不入训练；除非授权或合规 API。",
    ),
    _src(
        source_id="tianyancha",
        source_name="天眼查",
        source_type="commercial_enterprise_db",
        provider="天眼查（商业数据库）",
        official_url_or_api="https://www.tianyancha.com/",
        license_type="proprietary",
        collection_method="not_collect",
        allowed_usage=[],
        forbidden_usage=["scrape", "training", "bypass_anti_crawler"],
        update_frequency="unknown",
        coordinate_system="n/a",
        privacy_level="contains_personal_info_risk",
        compliance_status="risk_or_unavailable",
        collection_level=0,
        notes="未授权商业企业数据库，默认不采集、不入训练。",
    ),
)


# --------------------------------------------------------------------------- #
# registry 文件读写（external/registry/ 已 gitignore）
# --------------------------------------------------------------------------- #
def _external_dir() -> Path:
    return settings.data_dir / "external"


def _registry_dir() -> Path:
    return _external_dir() / "registry"


def registry_path() -> Path:
    return _registry_dir() / "data_source_registry.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(item: dict[str, Any]) -> dict[str, Any]:
    """按 SOURCE_SCHEMA 规整一个登记条目（补缺省 / 丢弃未知键）。"""
    out = _src()
    for k in SOURCE_SCHEMA:
        if k in item:
            out[k] = item[k]
    return out


def ensure_registry_initialized() -> Path:
    """若 registry 文件不存在，则用预置数据源初始化（幂等，不覆盖已有）。"""
    path = registry_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": list(SOURCE_SCHEMA),
        "collection_levels": COLLECTION_LEVELS,
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
        "sources": [_normalize(s) for s in PRESET_SOURCES],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("registry initialized with %s preset sources", len(PRESET_SOURCES))
    return path


def _load_registry() -> dict[str, Any]:
    path = registry_path()
    if not path.exists():
        return {
            "schema": list(SOURCE_SCHEMA),
            "collection_levels": COLLECTION_LEVELS,
            "sources": [_normalize(s) for s in PRESET_SOURCES],
        }
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("sources"), list):
            return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry load failed, fallback to presets: %s", exc)
    return {
        "schema": list(SOURCE_SCHEMA),
        "collection_levels": COLLECTION_LEVELS,
        "sources": [_normalize(s) for s in PRESET_SOURCES],
    }


def list_sources() -> list[dict[str, Any]]:
    """列出全部登记数据源（已规整字段；不含任何 Key 本身）。"""
    ensure_registry_initialized()
    return [_normalize(s) for s in _load_registry().get("sources", [])]


def get_source(source_id: str) -> dict[str, Any] | None:
    for s in list_sources():
        if s.get("source_id") == source_id:
            return s
    return None


def register_source(item: dict[str, Any]) -> dict[str, Any]:
    """登记 / 更新一个数据源（按 source_id 去重，写回 registry 文件）。

    安全：剔除 api_key 等敏感键，只保留 SOURCE_SCHEMA 字段；商业风险源即便被登记，
    其 can_use_for_training 由调用方/审计据 compliance_status 把关，不在此放宽。
    """
    ensure_registry_initialized()
    reg = _load_registry()
    normalized = _normalize(item)
    if not normalized.get("source_id"):
        raise ValueError("source_id 不能为空")
    sources = reg.get("sources", [])
    replaced = False
    for i, s in enumerate(sources):
        if s.get("source_id") == normalized["source_id"]:
            sources[i] = normalized
            replaced = True
            break
    if not replaced:
        sources.append(normalized)
    reg["sources"] = sources
    reg["updated_at"] = _utcnow_iso()
    reg.setdefault("schema", list(SOURCE_SCHEMA))
    reg.setdefault("collection_levels", COLLECTION_LEVELS)
    path = registry_path()
    with path.open("w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    logger.info("source registered/updated: %s (replaced=%s)", normalized["source_id"], replaced)
    return {"registered": True, "replaced": replaced, "source_id": normalized["source_id"],
            "source": normalized}


def compliance_risk_summary() -> dict[str, Any]:
    """合规风险汇总：按 compliance_status / collection_level 分组（脱敏，仅元数据）。"""
    sources = list_sources()
    risk = [s for s in sources if s.get("compliance_status") == "risk_or_unavailable"]
    needs_review = [s for s in sources if s.get("compliance_status") in ("needs_review", "planned")]
    trainable = [s for s in sources if s.get("can_use_for_training")]
    by_level: dict[str, int] = {}
    for s in sources:
        lvl = str(s.get("collection_level"))
        by_level[lvl] = by_level.get(lvl, 0) + 1

    def _brief(s: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_id": s["source_id"],
            "source_name": s["source_name"],
            "provider": s["provider"],
            "compliance_status": s["compliance_status"],
            "collection_level": s["collection_level"],
            "can_use_for_training": s["can_use_for_training"],
            "notes": s["notes"],
        }

    return {
        "total_sources": len(sources),
        "collection_levels": COLLECTION_LEVELS,
        "by_collection_level": by_level,
        "risk_or_unavailable": [_brief(s) for s in risk],
        "needs_review_or_planned": [_brief(s) for s in needs_review],
        "trainable_sources": [s["source_id"] for s in trainable],
        "risk_count": len(risk),
        "trainable_count": len(trainable),
    }
