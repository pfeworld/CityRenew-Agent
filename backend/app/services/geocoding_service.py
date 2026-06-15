"""地址 → 经纬度 地理编码（接入主链路，让用户输入真正进入空间分析）。

职责：把用户输入/附件里的项目地址解析为中心点经纬度（高德 GCJ02），并按上海合法
范围校验。仅用于把"用户输入"转成"可分析的坐标"，不读取语料原文、不外发敏感数据。

红线：
- AMAP key 仅从 .env 读取，绝不写死、绝不返回前端、绝不打印。
- 解析失败 / 超出上海覆盖范围 → 返回结构化的不可用状态，由上层 fail-closed，绝不编造坐标。
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from app.config import settings
from app.utils import geo_utils

logger = logging.getLogger("cityrenew.geocoding")

_AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"


def is_configured() -> bool:
    return bool((settings.amap_key or "").strip())


def geocode(address: str, city: str = "上海") -> dict[str, Any]:
    """地址 → 中心点经纬度。

    返回统一结构：
      {ok, lng, lat, formatted_address, district, level, error}
    任何异常都被捕获为 ok=False（error 为内部码），绝不抛出、绝不泄露 key。
    """
    addr = (address or "").strip()
    if not addr:
        return {"ok": False, "error": "empty_address"}
    key = (settings.amap_key or "").strip()
    if not key:
        return {"ok": False, "error": "geocoder_not_configured"}

    params = {"address": addr, "city": city or "上海", "key": key}
    try:
        resp = requests.get(_AMAP_GEOCODE_URL, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning("geocode http 非 200：status=%s", resp.status_code)
            return {"ok": False, "error": f"http_{resp.status_code}"}
        data = resp.json()
        if str(data.get("status")) != "1":
            return {"ok": False, "error": "amap_failed"}
        geocodes = data.get("geocodes") or []
        if not geocodes:
            return {"ok": False, "error": "not_found"}
        loc = (geocodes[0].get("location") or "").split(",")
        if len(loc) != 2:
            return {"ok": False, "error": "no_location"}
        lng, lat = float(loc[0]), float(loc[1])
        # 上海合法范围校验（覆盖范围之外无法做真实分析 → fail-closed）
        checked = geo_utils.validate_center(lng, lat)
        if not checked.is_usable:
            return {"ok": False, "error": "out_of_coverage",
                    "lng": lng, "lat": lat,
                    "note": checked.note or "超出当前数据覆盖范围（仅支持上海）"}

        def _txt(v: Any) -> str | None:
            return v if isinstance(v, str) and v else None

        return {
            "ok": True,
            "lng": checked.lng,
            "lat": checked.lat,
            "formatted_address": _txt(geocodes[0].get("formatted_address")),
            "district": _txt(geocodes[0].get("district")),
            "level": _txt(geocodes[0].get("level")),
            "error": "",
        }
    except requests.Timeout:
        logger.warning("geocode 超时")
        return {"ok": False, "error": "timeout"}
    except requests.RequestException as exc:
        logger.warning("geocode 网络异常：%s", type(exc).__name__)
        return {"ok": False, "error": "network_error"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("geocode 未知异常：%s", type(exc).__name__)
        return {"ok": False, "error": "unknown_error"}
