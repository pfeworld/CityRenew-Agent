"""把已抓取的高德开放 POI（全市覆盖，GCJ02）补充入库到 PoiPoint。

背景：比赛官方四源语料仅覆盖徐汇区一小块，导致其它行政区 fail-closed。
本脚本将 backend/data/external/amap/processed 下已抓取的高德 POI 合并去重后
写入 PoiPoint，使「区位/POI」维度覆盖全上海；房价/人口/产业仍为官方数据，
其它区对应维度据实输出"暂无数据"（不编造）。

合规与口径：
- 这些 POI 来源为「高德开放POI」，非比赛专用数据库，source_file 明确标记，可回溯。
- 坐标为 GCJ02，与请求时高德 geocode 中心点同坐标系，圈层距离一致。
- split 一律置 train：始终可用于分析，且不进入 test，避免污染测试集评估。
- 名称为脱敏哈希（name_hash），不写入 name（红线：不外泄可识别原文）。
- 幂等：按 source_file 标记先清理旧的高德 POI，再重写，可重复执行。

用法：
    backend/.venv/bin/python -m scripts.ingest_amap_poi
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import PROJECT_ROOT
from app.database import SessionLocal
from app.models import IndustryPoint, PoiPoint
from app.utils import geo_utils

AMAP_SOURCE_FILE = "外部数据/高德开放POI(GCJ02)"
AMAP_INDUSTRY_SOURCE_FILE = "外部数据/高德开放企业POI(GCJ02)"
PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "external" / "amap" / "processed"


def _load_records() -> list[dict]:
    """合并 processed 下所有 large_scale_*.json，按 poi_id 去重。"""
    merged: dict[str, dict] = {}
    files = sorted(PROCESSED_DIR.glob("large_scale_*.json"))
    for fp in files:
        if fp.name == "large_scale_store.json":
            continue
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {fp.name}: {exc}")
            continue
        recs = obj.get("records") if isinstance(obj, dict) else obj
        if not isinstance(recs, list):
            continue
        for r in recs:
            if not isinstance(r, dict):
                continue
            pid = r.get("poi_id") or r.get("source_id")
            if not pid:
                continue
            merged[str(pid)] = r
        print(f"  读取 {fp.name}: {len(recs)} 条")
    return list(merged.values())


def _coords(rec: dict):
    loc = rec.get("location_gcj02") or rec.get("location")
    if not loc or "," not in str(loc):
        return None
    a, b = str(loc).split(",")[:2]
    try:
        return [float(a), float(b)]
    except ValueError:
        return None


def main() -> None:
    db = SessionLocal()
    try:
        records = _load_records()
        print(f"合并去重后高德 POI：{len(records)} 条")

        removed = (
            db.query(PoiPoint)
            .filter(PoiPoint.source_file == AMAP_SOURCE_FILE)
            .delete(synchronize_session=False)
        )
        db.commit()
        print(f"清理旧高德 POI：{removed} 条")

        written = 0
        skipped = 0
        for rec in records:
            pt = geo_utils.parse_point(_coords(rec))
            if not pt.is_usable:
                skipped += 1
                continue
            category = rec.get("type") or rec.get("category_name")
            pid = rec.get("poi_id") or rec.get("source_id")
            db.add(
                PoiPoint(
                    source_id=f"amap_{pid}",
                    name=None,  # name_hash 不可读，红线下不写入
                    category_name=category,
                    district_name=rec.get("district"),
                    address=None,
                    lng=pt.lng,
                    lat=pt.lat,
                    coord_status=pt.status,
                    split="train",
                    source_file=AMAP_SOURCE_FILE,
                    raw_json=json.dumps(
                        {
                            "poi_id": pid,
                            "type": category,
                            "district": rec.get("district"),
                            "location_gcj02": rec.get("location_gcj02"),
                            "source_id": "amap_poi",
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            written += 1
            if written % 5000 == 0:
                db.commit()
        db.commit()
        print(f"写入高德 POI：{written} 条；坐标不可用跳过：{skipped} 条")

        total = db.query(PoiPoint).count()
        print(f"PoiPoint 现总量：{total} 条")

        # ---- 产业维度：把「公司企业」类 POI 补充入库到 IndustryPoint（全市覆盖）----
        ind_removed = (
            db.query(IndustryPoint)
            .filter(IndustryPoint.source_file == AMAP_INDUSTRY_SOURCE_FILE)
            .delete(synchronize_session=False)
        )
        db.commit()
        print(f"清理旧高德企业 POI：{ind_removed} 条")
        ind_written = 0
        for rec in records:
            category = rec.get("type") or rec.get("category_name") or ""
            if not str(category).startswith("公司企业"):
                continue
            pt = geo_utils.parse_point(_coords(rec))
            if not pt.is_usable:
                continue
            pid = rec.get("poi_id") or rec.get("source_id")
            db.add(
                IndustryPoint(
                    source_id=f"amap_ind_{pid}",
                    name=None,
                    category_name=category,
                    district_name=rec.get("district"),
                    address=None,
                    lng=pt.lng,
                    lat=pt.lat,
                    coord_status=pt.status,
                    split="train",
                    source_file=AMAP_INDUSTRY_SOURCE_FILE,
                    raw_json=json.dumps(
                        {"poi_id": pid, "type": category, "district": rec.get("district"),
                         "location_gcj02": rec.get("location_gcj02"), "source_id": "amap_poi"},
                        ensure_ascii=False,
                    ),
                )
            )
            ind_written += 1
            if ind_written % 5000 == 0:
                db.commit()
        db.commit()
        print(f"写入高德企业 POI 到 IndustryPoint：{ind_written} 条")
        print(f"IndustryPoint 现总量：{db.query(IndustryPoint).count()} 条")
    finally:
        db.close()


if __name__ == "__main__":
    main()
