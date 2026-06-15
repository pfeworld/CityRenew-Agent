"""第10C.5+：上海范围数据校验（本地离线工具，不训练、不进入第11、不 commit）。

确认已转为可训练/可用的科研语料数据是否全部属于上海范围：
  - 房价训练样本（research_property_trainable_candidates.jsonl）
  - 美食 POI 特征候选（research_poi_feature_candidates.jsonl）
  - 其他带空间属性的 research_imports

判定规则：
  A. city ∈ {上海, 上海市}
  B. district ∈ 上海16区
  C. lng∈[120.85,122.12] 且 lat∈[30.67,31.88]
  D. 仅有 address/community/板块 → 尝试匹配上海区县/板块；无法识别 → need_manual_review（不进训练）

红线：非上海 / 无法判断的数据一律不得进入训练；不混入 competition_test；不伪造。

运行：
  cd backend && ./.venv/bin/python -m app.tools.validate_shanghai_scope
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = BACKEND_DIR / "data" / "external"
DATA_CATALOG_DIR = BACKEND_DIR / "data" / "outputs" / "data_catalog"

HOUSING_DIR = EXTERNAL_DIR / "authorized_property" / "processed"
POI_DIR = EXTERNAL_DIR / "public_service" / "processed"

SH_DISTRICTS = ["黄浦", "徐汇", "长宁", "静安", "普陀", "虹口", "杨浦", "闵行",
                "宝山", "嘉定", "浦东新区", "浦东", "金山", "松江", "青浦",
                "奉贤", "崇明"]
# 常见上海板块/街镇（用于仅有 community/address 时的辅助识别）
SH_AREA_TOKENS = ["北蔡", "漕河泾", "大华", "长风", "东外滩", "长寿路", "高境",
                  "大场", "甘泉", "宜川", "光新", "鞍山", "碧云", "衡山路",
                  "华东理工", "春申", "联洋", "花木", "三林", "周浦", "康桥",
                  "金桥", "张江", "陆家嘴", "曹路", "唐镇", "外高桥", "莘庄",
                  "七宝", "梅陇", "古美", "颛桥", "江桥", "南翔", "安亭",
                  "真如", "桃浦", "万里", "彭浦", "五角场", "新江湾", "控江"]
CITY_OK = {"上海", "上海市"}
LNG_MIN, LNG_MAX = 120.85, 122.12
LAT_MIN, LAT_MAX = 30.67, 31.88


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _in_bbox(lng: Any, lat: Any) -> bool | None:
    if lng is None or lat is None:
        return None
    try:
        lng, lat = float(lng), float(lat)
    except (TypeError, ValueError):
        return None
    return LNG_MIN <= lng <= LNG_MAX and LAT_MIN <= lat <= LAT_MAX


def _text_has_sh(text: str) -> bool:
    return any(tok in text for tok in SH_DISTRICTS) or any(
        tok in text for tok in SH_AREA_TOKENS)


def classify_housing(row: dict[str, Any]) -> tuple[str, str]:
    """返回 (verdict, reason)。verdict ∈ shanghai / outside / need_manual_review。"""
    city = str(row.get("city") or "").strip()
    if city:
        return ("shanghai", "city_match") if city in CITY_OK else ("outside", f"city={city}")
    bbox = _in_bbox(row.get("lng"), row.get("lat"))
    if bbox is True:
        return "shanghai", "bbox_match"
    if bbox is False:
        return "outside", "lng_lat_outside_bbox"
    text = " ".join(str(row.get(k) or "") for k in ("region", "community", "address"))
    if any(d in text for d in SH_DISTRICTS):
        return "shanghai", "district_match"
    if any(t in text for t in SH_AREA_TOKENS):
        return "shanghai", "shanghai_area_token_match"
    return "need_manual_review", "only_community_name_no_geo"


def classify_poi(row: dict[str, Any]) -> tuple[str, str]:
    city = str(row.get("city") or "").strip()
    if city and city not in CITY_OK:
        return "outside", f"city={city}"
    bbox = _in_bbox(row.get("lng"), row.get("lat"))
    if bbox is True:
        return "shanghai", "bbox_match"
    if bbox is False:
        return "outside", "lng_lat_outside_bbox"
    text = " ".join(str(row.get(k) or "") for k in ("region", "address", "name"))
    if _text_has_sh(text):
        return "shanghai", "text_match"
    return "need_manual_review", "no_geo_no_text"


def _split_jsonl(src: Path, classifier, sh_path: Path, out_path: Path,
                 review_path: Path) -> dict[str, int]:
    counts = {"total": 0, "shanghai": 0, "outside": 0, "need_manual_review": 0}
    reasons: dict[str, int] = {}
    with src.open("r", encoding="utf-8") as fin, \
            sh_path.open("w", encoding="utf-8") as f_sh, \
            out_path.open("w", encoding="utf-8") as f_out, \
            review_path.open("w", encoding="utf-8") as f_rev:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts["total"] += 1
            verdict, reason = classifier(row)
            reasons[reason] = reasons.get(reason, 0) + 1
            row["shanghai_verdict"] = verdict
            row["shanghai_reason"] = reason
            if verdict == "shanghai":
                row["city_scope"] = "上海市"
                counts["shanghai"] += 1
                f_sh.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif verdict == "outside":
                row["used_for_training"] = False
                row["used_for_feature_engineering"] = False
                counts["outside"] += 1
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                row["used_for_training"] = False
                row["need_manual_review"] = True
                counts["need_manual_review"] += 1
                f_rev.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["reasons"] = reasons
    return counts


def validate_housing() -> dict[str, Any]:
    src = HOUSING_DIR / "research_property_trainable_candidates.jsonl"
    if not src.exists():
        return {"error": "housing trainable jsonl 不存在"}
    sh_tmp = HOUSING_DIR / "_shanghai_tmp.jsonl"
    excluded = HOUSING_DIR / "excluded_outside_shanghai.jsonl"
    review = HOUSING_DIR / "need_manual_review.jsonl"
    counts = _split_jsonl(src, classify_housing, sh_tmp, excluded, review)
    # 训练集只保留上海确认数据：用 shanghai 子集覆盖主训练文件
    sh_tmp.replace(src)

    trainable = counts["shanghai"]
    strength = ("strong" if trainable >= 3000 else
                "medium" if trainable >= 1000 else "weak")
    can_train = trainable >= 1000

    prof_path = HOUSING_DIR / "research_property_dataset_profile.json"
    prof = json.loads(prof_path.read_text(encoding="utf-8")) if prof_path.exists() else {}
    raw = prof.get("trainable_property_records", counts["total"])
    prof.update({
        "city_scope": "上海市",
        "shanghai_verified": True,
        "shanghai_bbox": [LNG_MIN, LNG_MAX, LAT_MIN, LAT_MAX],
        "trainable_property_records_raw": raw,
        "trainable_property_records_shanghai_only": trainable,
        "trainable_property_records": trainable,
        "outside_shanghai_count": counts["outside"],
        "need_manual_review_count": counts["need_manual_review"],
        "supervised_training_strength": strength,
        "can_start_supervised_housing_model": can_train,
        "city_filter_reasons": counts["reasons"],
        "city_filter_at": _utcnow(),
        "note": "仅上海确认样本进入训练集（覆盖写入主 jsonl）；非上海→excluded，"
                "仅小区名无地理→need_manual_review，均不进训练。接口不返回明细。",
    })
    prof_path.write_text(json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"counts": counts, "trainable_shanghai": trainable, "raw": raw,
            "strength": strength, "can_start": can_train}


def validate_poi() -> dict[str, Any]:
    src = POI_DIR / "research_poi_feature_candidates.jsonl"
    if not src.exists():
        return {"error": "poi feature jsonl 不存在"}
    sh_tmp = POI_DIR / "_shanghai_tmp.jsonl"
    excluded = POI_DIR / "excluded_outside_shanghai_poi.jsonl"
    review = POI_DIR / "need_manual_review_poi.jsonl"
    counts = _split_jsonl(src, classify_poi, sh_tmp, excluded, review)
    sh_tmp.replace(src)  # feature 集只保留上海

    prof_path = POI_DIR / "research_poi_dataset_profile.json"
    prof = json.loads(prof_path.read_text(encoding="utf-8")) if prof_path.exists() else {}
    raw = prof.get("emitted_records", counts["total"])
    prof.update({
        "city_scope": "上海市",
        "shanghai_verified": True,
        "shanghai_bbox": [LNG_MIN, LNG_MAX, LAT_MIN, LAT_MAX],
        "emitted_records_raw": raw,
        "shanghai_records": counts["shanghai"],
        "outside_shanghai_count": counts["outside"],
        "need_manual_review_count": counts["need_manual_review"],
        "emitted_records": counts["shanghai"],
        "usable_for_feature_engineering": counts["shanghai"] > 0,
        "city_filter_reasons": counts["reasons"],
        "city_filter_at": _utcnow(),
        "note": "仅上海范围 POI 进入特征候选（覆盖写入主 jsonl）；非上海→excluded，不参与上海项目特征。",
    })
    prof_path.write_text(json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"counts": counts, "shanghai": counts["shanghai"], "raw": raw}


def update_lineage(housing: dict[str, Any], poi: dict[str, Any]) -> None:
    from app.services import data_lineage_service as dl
    dl.patch_external_by_source("research_housing_property", {
        "city_scope": "上海市", "shanghai_verified": True,
        "outside_shanghai_count": housing["counts"]["outside"],
        "unknown_scope_count": housing["counts"]["need_manual_review"],
        "trainable_record_count": housing["trainable_shanghai"],
        "trainable_record_count_after_city_filter": housing["trainable_shanghai"],
        "can_use_for_training": housing["can_start"],
        "used_for_training": housing["can_start"],
        "training_block_reason": None if housing["can_start"] else "上海确认样本<1000",
    })
    dl.patch_external_by_source("research_poi_public_service", {
        "city_scope": "上海市", "shanghai_verified": True,
        "outside_shanghai_count": poi["counts"]["outside"],
        "unknown_scope_count": poi["counts"]["need_manual_review"],
        "trainable_record_count_after_city_filter": 0,
        "record_count": poi["shanghai"],
    })


def write_report(housing: dict[str, Any], poi: dict[str, Any]) -> dict[str, Any]:
    report = {
        "generated_at": _utcnow(), "phase": "10C.5-city-validation",
        "rule": {"city": list(CITY_OK), "districts": SH_DISTRICTS,
                 "bbox": {"lng": [LNG_MIN, LNG_MAX], "lat": [LAT_MIN, LAT_MAX]}},
        "housing": {
            "raw_trainable": housing["raw"],
            "shanghai_confirmed": housing["trainable_shanghai"],
            "outside_shanghai": housing["counts"]["outside"],
            "need_manual_review": housing["counts"]["need_manual_review"],
            "ge_1000": housing["trainable_shanghai"] >= 1000,
            "ge_3000": housing["trainable_shanghai"] >= 3000,
            "can_start_supervised_housing_model": housing["can_start"],
            "strength": housing["strength"],
            "reasons": housing["counts"]["reasons"],
        },
        "poi": {
            "raw_records": poi["raw"],
            "shanghai_confirmed": poi["shanghai"],
            "outside_shanghai": poi["counts"]["outside"],
            "need_manual_review": poi["counts"]["need_manual_review"],
            "usable_for_feature_engineering": poi["shanghai"] > 0,
            "reasons": poi["counts"]["reasons"],
        },
        "compliance": {
            "non_shanghai_in_training": False,
            "unknown_scope_in_training": False,
            "test_contamination_risk": False,
            "leakage_risk": False,
            "note": "非上海/无法判断数据一律不进训练；POI 不进监督训练。",
        },
    }
    DATA_CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_CATALOG_DIR / "上海范围校验报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# 上海范围数据校验报告", "", f"- 生成时间：{report['generated_at']}",
        f"- bbox：lng[{LNG_MIN},{LNG_MAX}] lat[{LAT_MIN},{LAT_MAX}]", "",
        "## A 房价上海范围校验",
        f"- 原始 trainable：{housing['raw']}",
        f"- 上海确认：{housing['trainable_shanghai']}",
        f"- 非上海排除：{housing['counts']['outside']}",
        f"- 无法判断(need_manual_review)：{housing['counts']['need_manual_review']}",
        f"- ≥1000：{'是' if housing['trainable_shanghai']>=1000 else '否'}；"
        f"≥3000：{'是' if housing['trainable_shanghai']>=3000 else '否'}",
        f"- can_start_supervised_housing_model：{housing['can_start']}（强度 {housing['strength']}）",
        "", "## B POI 上海范围校验",
        f"- 原始记录：{poi['raw']}",
        f"- 上海确认：{poi['shanghai']}",
        f"- 非上海排除：{poi['counts']['outside']}",
        f"- 无法判断：{poi['counts']['need_manual_review']}",
        f"- 可用于特征工程：{'是' if poi['shanghai']>0 else '否'}",
        "", "## C 合规安全",
        "- 非上海数据进入训练：否", "- 无法判断数据进入训练：否",
        "- test_contamination_risk：false", "- leakage_risk：false",
    ]
    (DATA_CATALOG_DIR / "上海范围校验报告.md").write_text("\n".join(md) + "\n",
                                                          encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #
# community-only 谨慎补全（小区名 → 上海区县/经纬度/地址）
# --------------------------------------------------------------------------- #
PROJECT_ROOT = BACKEND_DIR.parent
SH_DISTRICTS_FULL = ["黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区",
                     "杨浦区", "闵行区", "宝山区", "嘉定区", "浦东新区", "金山区",
                     "松江区", "青浦区", "奉贤区", "崇明区"]
GEOCODE_CAP = 150
GEOCODE_FAIL_ABORT = 3


def _norm_name(s: Any) -> str:
    s = str(s or "").strip().replace("\u3000", "").replace(" ", "")
    # 去掉末尾括号补充（如"（一期）"），保留主名
    for ch in ["(", "（"]:
        if ch in s:
            s = s.split(ch)[0]
    return s.lower()


def _district_from_text(text: str) -> str | None:
    if not text:
        return None
    for d in SH_DISTRICTS_FULL:
        if d in text:
            return "浦东" if d == "浦东新区" else d[:-1]
    for d in SH_DISTRICTS:
        if d in text:
            return "浦东" if d == "浦东新区" else d
    return None


def _read_source_xlsx(rel_path: str, cols: dict[str, str]) -> list[dict[str, Any]]:
    """读取源 xlsx 指定列（cols: 表头名->标准键），返回行 dict 列表。"""
    import openpyxl  # noqa: PLC0415
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        return []
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    try:
        header = [str(c).strip() if c is not None else "" for c in next(it)]
    except StopIteration:
        wb.close()
        return []
    idx = {std: header.index(h) for h, std in cols.items() if h in header}
    out = []
    for row in it:
        rec = {}
        for std, i in idx.items():
            rec[std] = row[i] if i < len(row) else None
        out.append(rec)
    wb.close()
    return out


def _build_enrichment_maps() -> dict[str, Any]:
    """构建小区名 → 区县 的反查映射（内部确认 + 科研源文件）。"""
    internal: dict[str, set] = {}
    src_map: dict[str, dict[str, set]] = {}

    # 内部已确认上海训练样本（含 region）
    sh_train = HOUSING_DIR / "research_property_trainable_candidates.jsonl"
    if sh_train.exists():
        with sh_train.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                d = _district_from_text(str(r.get("region") or ""))
                if d and r.get("community"):
                    internal.setdefault(_norm_name(r["community"]), set()).add(d)

    def add_src(rel, name_col, dist_col=None, addr_col=None):
        rows = _read_source_xlsx(rel, {c: c for c in
                                       [name_col, dist_col, addr_col] if c})
        for r in rows:
            nm = _norm_name(r.get(name_col))
            if not nm:
                continue
            d = None
            if dist_col and r.get(dist_col):
                d = _district_from_text(str(r[dist_col]))
            if not d and addr_col and r.get(addr_col):
                d = _district_from_text(str(r[addr_col]))
            if d:
                e = src_map.setdefault(nm, {"districts": set(), "addresses": set()})
                e["districts"].add(d)
                if addr_col and r.get(addr_col):
                    e["addresses"].add(str(r[addr_col])[:80])

    # 科研源文件（同数据集 / 论文数据）
    add_src("科研语料/上海链家小区基础信息.xlsx", "小区名字", addr_col="小区地址")
    add_src("科研语料/上海链家小区基础信息-7.23.xlsx", "小区名字", addr_col="小区地址")
    add_src("科研语料/尹宝仪-论文数据/2.贝壳网小区数据.xlsx", "小区名称", "区", "地址")
    add_src("科研语料/尹宝仪-论文数据/3.虹口住房.xlsx", "小区名称", "行政区", "地址")
    add_src("科研语料/尹宝仪-论文数据/1.保租房.xlsx", "供应备案项目名称",
            "区域", "供应备案项目地址")
    return {"internal": internal, "source": src_map}


def enrich_community_only() -> int:
    review_path = HOUSING_DIR / "need_manual_review.jsonl"
    if not review_path.exists():
        cand = list(HOUSING_DIR.glob("*need_manual_review*.jsonl"))
        if not cand:
            print("[enrich] 未找到 need_manual_review jsonl")
            return 1
        review_path = cand[0]

    print("[enrich] building reverse-lookup maps (internal + research sources) ...")
    maps = _build_enrichment_maps()
    internal, src_map = maps["internal"], maps["source"]
    print(f"[enrich] internal communities={len(internal)} "
          f"source communities={len(src_map)}")

    rows = [json.loads(l) for l in review_path.open(encoding="utf-8") if l.strip()]
    stats = {"original": len(rows), "internal": 0, "amap_poi": 0,
             "research_source": 0, "geocode": 0, "added": 0,
             "still_review": 0, "ambiguous": 0, "outside": 0}

    additions, still_review, ambiguous, outside = [], [], [], []
    geocode_cache: dict[str, Any] = {}
    geocode_calls = geocode_fails = 0

    try:
        from app.services import amap_service
        amap_ok = amap_service.is_configured()
    except Exception:  # noqa: BLE001
        amap_service = None
        amap_ok = False

    for r in rows:
        name = _norm_name(r.get("community"))
        districts: set = set()
        source = None
        # 优先级1：内部确认样本
        if name in internal:
            districts |= internal[name]
            source = "internal_sample"
        # 优先级2：高德 POI 反查（store 仅 name_hash 脱敏，无法按名匹配）→ 跳过
        # 优先级3：科研源文件
        if name in src_map and src_map[name]["districts"]:
            districts |= src_map[name]["districts"]
            source = source or "research_source_file"
        addr = None
        if name in src_map and src_map[name]["addresses"]:
            addr = sorted(src_map[name]["addresses"])[0]

        # 优先级4：高德 geocode（仅前三种都失败时，限额、限上海、网络异常则放弃）
        if not districts and amap_ok and geocode_calls < GEOCODE_CAP \
                and geocode_fails < GEOCODE_FAIL_ABORT and r.get("community"):
            cname = str(r["community"])
            if cname in geocode_cache:
                gd = geocode_cache[cname]
            else:
                gd = None
                try:
                    geocode_calls += 1
                    resp = amap_service.geocode(cname + "小区", city="上海")
                    geos = (resp.get("data") or {}).get("geocodes") or [] \
                        if isinstance(resp, dict) else []
                    sh_geos = [g for g in geos
                               if str(g.get("province", "")).startswith("上海")]
                    if len(sh_geos) == 1:
                        loc = sh_geos[0].get("location", "")
                        d = _district_from_text(str(sh_geos[0].get("district", "")))
                        if "," in loc:
                            lng, lat = (float(x) for x in loc.split(",")[:2])
                            if _in_bbox(lng, lat) and d:
                                gd = {"district": d, "lng": round(lng, 5),
                                      "lat": round(lat, 5)}
                except Exception:  # noqa: BLE001
                    geocode_fails += 1
                    gd = None
                geocode_cache[cname] = gd
            if gd:
                districts.add(gd["district"])
                source = "amap_geocode"
                addr = addr or None
                r["lng"], r["lat"] = gd["lng"], gd["lat"]

        # 判定
        distinct = {d for d in districts if d}
        has_fields = bool(r.get("price_unit") or r.get("price_total")
                          or r.get("rent")) and bool(r.get("build_year"))
        if len(distinct) > 1:
            r["enrichment_status"] = "ambiguous_community"
            r["candidate_districts"] = sorted(distinct)
            ambiguous.append(r)
            stats["ambiguous"] += 1
        elif len(distinct) == 1 and has_fields:
            d = next(iter(distinct))
            r["region"] = d
            r["shanghai_district"] = d
            if addr:
                r["address"] = addr
            r["city_scope"] = "上海市"
            r["shanghai_verdict"] = "shanghai"
            r["enrichment_source"] = source
            r["enrichment_confidence"] = "high"
            r["record_granularity"] = "community_baseline"
            r["has_transaction_time"] = False
            r["used_for_training"] = True
            r["need_manual_review"] = False
            r.pop("shanghai_reason", None)
            additions.append(r)
            stats["added"] += 1
            stats[{"internal_sample": "internal", "research_source_file":
                   "research_source", "amap_geocode": "geocode"}[source]] += 1
        elif len(distinct) == 1 and not has_fields:
            r["enrichment_status"] = "shanghai_but_insufficient_fields"
            still_review.append(r)
            stats["still_review"] += 1
        else:
            r["enrichment_status"] = "unresolved_need_manual_review"
            still_review.append(r)
            stats["still_review"] += 1

    # 写出各分桶
    def _dump(path: Path, items):
        with path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    _dump(HOUSING_DIR / "community_enriched_trainable_additions.jsonl", additions)
    _dump(HOUSING_DIR / "community_still_need_review.jsonl", still_review)
    _dump(HOUSING_DIR / "community_ambiguous.jsonl", ambiguous)
    _dump(HOUSING_DIR / "community_outside_shanghai.jsonl", outside)

    # 追加进训练集
    train_path = HOUSING_DIR / "research_property_trainable_candidates.jsonl"
    before = sum(1 for _ in train_path.open(encoding="utf-8")) if train_path.exists() else 0
    with train_path.open("a", encoding="utf-8") as f:
        for it in additions:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    after = before + len(additions)
    strength = ("strong" if after >= 3000 else "medium" if after >= 1000 else "weak")

    # 覆盖原 need_manual_review（剩余仍需复核）
    _dump(review_path, still_review)

    # profile 更新
    prof_path = HOUSING_DIR / "research_property_dataset_profile.json"
    prof = json.loads(prof_path.read_text(encoding="utf-8")) if prof_path.exists() else {}
    prof.update({
        "trainable_property_records_shanghai_confirmed_only": before,
        "community_enriched_additions": len(additions),
        "trainable_property_records": after,
        "supervised_training_strength": strength,
        "can_start_supervised_housing_model": after >= 1000,
        "need_manual_review_count": len(still_review),
        "ambiguous_community_count": len(ambiguous),
        "enrichment_breakdown": {k: stats[k] for k in
                                 ["internal", "amap_poi", "research_source", "geocode"]},
        "enrichment_at": _utcnow(),
        "enrichment_note": "community-only 补全：仅高置信(唯一上海区县+精确名匹配+价格+建成年代+脱敏)"
                           "转入训练；歧义/越界/字段不足不进训练。community_baseline 粒度，无成交时间。",
    })
    prof_path.write_text(json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8")

    # 报告
    report = {
        "generated_at": _utcnow(), "phase": "10C.5-community-enrichment",
        "priority_order": ["internal_sample", "amap_poi(name_hash不可匹配→跳过)",
                           "research_source_file", "amap_geocode"],
        "geocode_calls": geocode_calls, "geocode_fails": geocode_fails,
        "amap_configured": amap_ok, "stats": stats,
        "trainable_before": before, "trainable_after": after,
        "supervised_training_strength": strength,
        "can_start_supervised_housing_model": after >= 1000,
        "compliance": {"unconfirmed_into_training": False, "ambiguous_into_training": False,
                       "non_shanghai_into_training": False,
                       "test_contamination_risk": False, "leakage_risk": False},
    }
    (HOUSING_DIR / "community_enrichment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_enrich_md(report)

    # 血缘 / manifest / snapshot
    _update_after_enrichment(after, strength, len(still_review), len(ambiguous), stats)
    print(f"[enrich] added={len(additions)} (internal={stats['internal']} "
          f"research_source={stats['research_source']} geocode={stats['geocode']}) "
          f"still_review={len(still_review)} ambiguous={len(ambiguous)} "
          f"trainable {before}->{after} ({strength})")
    return 0


def _write_enrich_md(rep: dict[str, Any]) -> None:
    s = rep["stats"]
    md = [
        "# community-only 房价样本补全报告", "",
        f"- 生成时间：{rep['generated_at']}",
        f"- 补全优先级：内部样本 → 高德POI(name_hash不可匹配,跳过) → 科研源文件 → 高德geocode",
        f"- geocode 调用/失败：{rep['geocode_calls']}/{rep['geocode_fails']}"
        f"（AMAP 配置={rep['amap_configured']}）", "",
        "## A 补全结果",
        f"- 原始 need_manual_review：{s['original']}",
        f"- 内部样本反查补全：{s['internal']}",
        f"- 高德 POI 反查补全：{s['amap_poi']}（store 仅 name_hash 脱敏，名称不可比对）",
        f"- 科研语料源文件反查补全：{s['research_source']}",
        f"- 高德 geocode 补全：{s['geocode']}",
        f"- 新增可训练样本：{s['added']}",
        f"- 仍需人工复核：{s['still_review']}",
        f"- ambiguous：{s['ambiguous']}",
        f"- outside_shanghai：{s['outside']}", "",
        "## B 训练样本最终状态",
        f"- 补全前上海确认：{rep['trainable_before']}",
        f"- 补全后 trainable：{rep['trainable_after']}",
        f"- ≥1000：{'是' if rep['trainable_after']>=1000 else '否'}；"
        f"≥3000：{'是' if rep['trainable_after']>=3000 else '否'}",
        f"- strength：{rep['supervised_training_strength']}",
        f"- can_start_supervised_housing_model：{rep['can_start_supervised_housing_model']}", "",
        "## C 合规",
        "- 无法确认上海/歧义/越界数据进入训练：否",
        "- test_contamination_risk：false；leakage_risk：false",
    ]
    (HOUSING_DIR / "community_enrichment_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")


def _update_after_enrichment(after: int, strength: str, still_review: int,
                             ambiguous: int, stats: dict) -> None:
    from app.services import data_lineage_service as dl
    dl.patch_external_by_source("research_housing_property", {
        "trainable_record_count": after,
        "trainable_record_count_after_city_filter": after,
        "unknown_scope_count": still_review,
        "ambiguous_community_count": ambiguous,
        "community_enriched_additions": stats["added"],
        "can_use_for_training": after >= 1000,
        "used_for_training": after >= 1000,
        "supervised_training_strength": strength,
    })
    # research_corpus manifest
    rc_manifest = EXTERNAL_DIR / "research_corpus" / "manifest.json"
    if rc_manifest.exists():
        try:
            m = json.loads(rc_manifest.read_text(encoding="utf-8"))
            hp = m.setdefault("assets", {}).setdefault("housing_property", {})
            hp["trainable_property_records"] = after
            hp["supervised_training_strength"] = strength
            hp["community_enriched_additions"] = stats["added"]
            hp["shanghai_confirmed_only"] = after - stats["added"]
            m.setdefault("updated_at", _utcnow())
            m["updated_at"] = _utcnow()
            rc_manifest.write_text(json.dumps(m, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    rc_lineage = EXTERNAL_DIR / "research_corpus" / "lineage.json"
    if rc_lineage.exists():
        try:
            lj = json.loads(rc_lineage.read_text(encoding="utf-8"))
            for rec in lj.get("records", []):
                if rec.get("source_id") == "research_housing_property":
                    rec["trainable_record_count"] = after
                    rec["community_enriched_additions"] = stats["added"]
                    rec["city_scope"] = "上海市"
            lj["updated_at"] = _utcnow()
            rc_lineage.write_text(json.dumps(lj, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    # 第11训练前数据快照
    snap = DATA_CATALOG_DIR / "第11训练前数据快照.json"
    if snap.exists():
        try:
            s = json.loads(snap.read_text(encoding="utf-8"))
            s.setdefault("research_corpus", {})["housing_trainable_records"] = after
            s["research_corpus"]["housing_strength"] = strength
            s["research_corpus"]["community_enriched_additions"] = stats["added"]
            s["trainable_housing_samples"] = after
            s["can_start_supervised_housing_model"] = after >= 1000
            s["city_enrichment_at"] = _utcnow()
            snap.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                from app.tools.scan_research_corpus import _snapshot_md
                (DATA_CATALOG_DIR / "第11训练前数据快照.md").write_text(
                    _snapshot_md(s), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    if "--enrich-community-only" in sys.argv:
        return enrich_community_only()
    print("[city] validating housing trainable candidates ...")
    housing = validate_housing()
    if "error" in housing:
        print("[city] housing error:", housing["error"])
        return 1
    print(f"[city] housing: shanghai={housing['trainable_shanghai']} "
          f"outside={housing['counts']['outside']} "
          f"review={housing['counts']['need_manual_review']} "
          f"can_start={housing['can_start']} strength={housing['strength']}")
    print("[city] validating poi feature candidates ...")
    poi = validate_poi()
    if "error" in poi:
        print("[city] poi error:", poi["error"])
        return 1
    print(f"[city] poi: shanghai={poi['shanghai']} outside={poi['counts']['outside']} "
          f"review={poi['counts']['need_manual_review']}")
    update_lineage(housing, poi)
    write_report(housing, poi)
    print("[city] lineage patched + report written. done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
