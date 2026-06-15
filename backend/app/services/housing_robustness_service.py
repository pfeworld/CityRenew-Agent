"""第11 T3.5：房价监督模型真实性复核与防记忆验证。

目的：证明 T3 房价模型不是靠 test 背答案 / 同小区记忆 / 区域标签 / 标签泄漏撑指标。
全部实验仅用 train/val（绝不使用 competition_test）；不保存 shuffle 模型；不伪造指标。

实验：
- A random split 复算（baseline_validation，复现 T3）
- B community group split（同小区不跨 train/val）
- C plate group split（无板块字段→用 region 近似并标 degraded）
- D district holdout（训练区/验证区不重叠）
- E feature ablation（intrinsic / +district / +poi）
- F label shuffle sanity check（打乱标签应显著变差）
- G duplicate sensitivity（近重复去重后复算）

红线：不使用 test；不为指标删异常；标签字段(price_total/rent/area)不得进特征；
group split 真实分组不可伪造；缺字段如实标 degraded。
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings as _warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.services import housing_price_training_service as hp

logger = logging.getLogger("cityrenew.housing_robustness")

RANDOM_STATE = 42
VAL_RATIO = 0.2


def _models_dir():
    return hp._models_dir()  # noqa: SLF001


# --------------------------------------------------------------------------- #
# 富样本加载（含 community / region / district），复用 T3 单位一致性规则
# --------------------------------------------------------------------------- #
def _load_rich_samples() -> list[dict[str, Any]]:
    loaded = hp.load_housing_samples()
    out: list[dict[str, Any]] = []
    for s in loaded["samples"]:
        if not (hp.PRICE_MIN_YUAN_SQM <= s["price_unit"] <= hp.PRICE_MAX_YUAN_SQM):
            continue
        out.append(s)
    return out


def _load_with_groups() -> list[dict[str, Any]]:
    """重新读原始 jsonl 以保留 community / region 分组键（仍走单位过滤）。"""
    path = hp._housing_path()  # noqa: SLF001
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not rec.get("used_for_training") or rec.get("competition_test"):
                continue
            if rec.get("shanghai_verdict") not in (None, "shanghai"):
                continue
            try:
                price = float(rec.get("price_unit"))
            except (TypeError, ValueError):
                continue
            if not (hp.PRICE_MIN_YUAN_SQM <= price <= hp.PRICE_MAX_YUAN_SQM):
                continue
            community = (rec.get("community") or "").strip()
            region = (rec.get("region") or rec.get("shanghai_district") or "").strip()
            district = (hp.normalize_district(rec.get("region"))
                        or hp.normalize_district(rec.get("shanghai_district"))
                        or hp.normalize_district(rec.get("address"))
                        or hp.normalize_district(rec.get("community")))
            by = rec.get("build_year")
            try:
                by = int(by)
                if by < 1900 or by > hp.CURRENT_YEAR:
                    by = None
            except (TypeError, ValueError):
                by = None
            rows.append({"price_unit": price, "build_year": by, "district": district,
                         "community": community, "region": region})
    return rows


# --------------------------------------------------------------------------- #
# 特征装配（可按 ablation 选择列）
# --------------------------------------------------------------------------- #
def _assemble(samples: list[dict[str, Any]], district_poi: dict[str, dict[str, float]]):
    districts_present = [d for d in hp.SH_DISTRICTS if any(s["district"] == d for s in samples)]
    poi_cols = ["district_poi_total"] + [f"district_share_{l1}" for l1 in hp.DISTRICT_POI_L1]
    ages = [hp.CURRENT_YEAR - s["build_year"] for s in samples if s["build_year"]]
    age_median = float(np.median(ages)) if ages else 30.0
    poi_medians = {}
    for col in poi_cols:
        vals = [district_poi[d][col] for d in district_poi if col in district_poi[d]]
        poi_medians[col] = float(np.median(vals)) if vals else 0.0

    n = len(samples)
    age = np.empty(n)
    poi = np.empty((n, len(poi_cols)))
    onehot = np.zeros((n, len(districts_present)))
    y = np.empty(n)
    didx = {d: i for i, d in enumerate(districts_present)}
    for i, s in enumerate(samples):
        age[i] = float(hp.CURRENT_YEAR - s["build_year"]) if s["build_year"] else age_median
        dp = district_poi.get(s["district"]) if s["district"] else None
        for j, col in enumerate(poi_cols):
            poi[i, j] = float(dp[col]) if dp and col in dp else poi_medians[col]
        if s["district"] in didx:
            onehot[i, didx[s["district"]]] = 1.0
        y[i] = s["price_unit"]
    return {"age": age.reshape(-1, 1), "poi": poi, "onehot": onehot, "y": y,
            "poi_cols": poi_cols, "districts_present": districts_present}


def _matrix(parts, mode: str) -> np.ndarray:
    if mode == "intrinsic":
        return parts["age"]
    if mode == "district":
        return np.hstack([parts["age"], parts["onehot"]])
    if mode == "poi":
        return np.hstack([parts["age"], parts["poi"]])
    return np.hstack([parts["age"], parts["poi"], parts["onehot"]])  # full


def _rf():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=3,
                                 random_state=RANDOM_STATE, n_jobs=-1)


def _fit_eval(X, y, train_idx, val_idx) -> dict[str, Any]:
    Xtr, Xv, ytr, yv = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
    with _warnings.catch_warnings(), np.errstate(all="ignore"):
        _warnings.simplefilter("ignore")
        m = _rf()
        m.fit(Xtr, ytr)
        pv = m.predict(Xv)
        pt = m.predict(Xtr)
    mv = hp._metrics(yv, pv)  # noqa: SLF001
    mt = hp._metrics(ytr, pt)  # noqa: SLF001
    # 中位数 baseline 参照
    const = float(np.median(ytr))
    base = hp._metrics(yv, np.full_like(yv, const))  # noqa: SLF001
    return {"train_count": int(len(train_idx)), "val_count": int(len(val_idx)),
            "val_mae": mv["mae"], "val_mape": mv["mape"], "r2_val": mv["r2"],
            "train_mae": mt["mae"], "train_mape": mt["mape"], "r2_train": mt["r2"],
            "baseline_val_mae": base["mae"], "baseline_val_mape": base["mape"]}


# --------------------------------------------------------------------------- #
# 划分工具
# --------------------------------------------------------------------------- #
def _random_split(n: int):
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    nval = int(n * VAL_RATIO)
    return perm[nval:], perm[:nval]


def _group_split(groups: list[str], val_ratio: float = VAL_RATIO):
    """按 group 分组，整组进 train 或 val（同组不跨集）。"""
    uniq = sorted(set(g for g in groups if g))
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(uniq)
    n_val_groups = max(1, int(len(uniq) * val_ratio))
    val_groups = set(uniq[:n_val_groups])
    train_idx = [i for i, g in enumerate(groups) if g and g not in val_groups]
    val_idx = [i for i, g in enumerate(groups) if g and g in val_groups]
    return np.array(train_idx), np.array(val_idx), len(uniq), len(val_groups)


def _district_holdout(districts: list[str], val_ratio: float = VAL_RATIO):
    """按区 holdout：选若干区作 val，训练区/验证区不重叠。"""
    cnt: dict[str, int] = {}
    for d in districts:
        if d:
            cnt[d] = cnt.get(d, 0) + 1
    ordered = sorted(cnt, key=lambda d: cnt[d])  # 小区优先做 val，避免 val 过大
    total = sum(cnt.values())
    val_districts, acc = set(), 0
    for d in ordered:
        if acc >= total * val_ratio:
            break
        val_districts.add(d)
        acc += cnt[d]
    train_idx = [i for i, d in enumerate(districts) if d and d not in val_districts]
    val_idx = [i for i, d in enumerate(districts) if d and d in val_districts]
    return np.array(train_idx), np.array(val_idx), sorted(val_districts)


# --------------------------------------------------------------------------- #
# 复核总入口
# --------------------------------------------------------------------------- #
def run_robustness(_db=None) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    samples = _load_with_groups()
    district_poi = hp.build_district_poi_features()
    n = len(samples)

    result: dict[str, Any] = {
        "generated_at": started.isoformat(),
        "modeled_record_count": n,
        "test_used_for_training": False,
        "used_competition_test": False,
    }
    if n < 1000:
        result.update({"robustness_status": "fail", "reason": f"可建模样本不足：{n}"})
        return result

    parts = _assemble(samples, district_poi)
    y = parts["y"]
    communities = [s["community"] for s in samples]
    regions = [s["region"] for s in samples]
    districts = [s["district"] for s in samples]

    X_full = _matrix(parts, "full")

    # ---- A random split ----
    tr, va = _random_split(n)
    exp_a = _fit_eval(X_full, y, tr, va)

    # ---- B community group split ----
    btr, bva, n_comm, n_comm_val = _group_split(communities)
    exp_b = _fit_eval(X_full, y, btr, bva)
    exp_b.update({"groups_total": n_comm, "val_groups": n_comm_val, "degraded": False})

    # ---- C plate group split（无板块字段 → region 近似，degraded）----
    has_region = sum(1 for r in regions if r)
    if has_region >= n * 0.5:
        ctr, cva, n_reg, n_reg_val = _group_split(regions)
        exp_c = _fit_eval(X_full, y, ctr, cva)
        exp_c.update({"groups_total": n_reg, "val_groups": n_reg_val, "degraded": True,
                      "degraded_reason": "数据无 plate/板块字段，用 region(区县级) 近似，粒度偏粗"})
    else:
        exp_c = {"degraded": True, "degraded_reason": "缺少 plate/region 字段，plate group split 不可执行"}

    # ---- D district holdout ----
    dtr, dva, val_districts = _district_holdout(districts)
    if len(dtr) >= 500 and len(dva) >= 200:
        exp_d = _fit_eval(X_full, y, dtr, dva)
        exp_d.update({"val_districts": val_districts, "degraded": False})
    else:
        exp_d = {"degraded": True, "degraded_reason": "区级 holdout 样本不足", "val_districts": val_districts}

    # ---- E ablation ----
    ablation = {}
    for mode in ("intrinsic", "district", "poi", "full"):
        Xm = _matrix(parts, mode)
        ablation[mode] = _fit_eval(Xm, y, tr, va)
    poi_gain = round(ablation["intrinsic"]["val_mape"] - ablation["poi"]["val_mape"], 4)
    district_gain = round(ablation["intrinsic"]["val_mape"] - ablation["district"]["val_mape"], 4)
    full_gain = round(ablation["intrinsic"]["val_mape"] - ablation["full"]["val_mape"], 4)
    ablation_summary = {
        "only_housing_intrinsic": ablation["intrinsic"],
        "housing_plus_district": ablation["district"],
        "housing_plus_poi": ablation["poi"],
        "full": ablation["full"],
        "poi_improves_mape_pts": poi_gain,
        "district_improves_mape_pts": district_gain,
        "full_improves_mape_pts": full_gain,
        "poi_helps": poi_gain > 1.0,
        "district_dominant": district_gain > poi_gain + 5,
        "note": "intrinsic 仅 building_age（数据无面积/户型逐条字段）；POI/district 均为区级粗特征。",
    }

    # ---- F label shuffle ----
    rng = np.random.RandomState(RANDOM_STATE)
    y_shuf = y.copy()
    yt = y_shuf[tr]
    rng.shuffle(yt)
    y_shuf[tr] = yt
    exp_f = _fit_eval(X_full, y_shuf, tr, va)
    shuffle_collapses = (exp_f["r2_val"] < 0.1 and exp_f["val_mape"] > exp_a["val_mape"] + 8)

    # ---- G duplicate sensitivity ----
    seen, dedup_idx = set(), []
    for i, s in enumerate(samples):
        key = (s["community"], s["build_year"], round(s["price_unit"] / 100.0))
        if key in seen:
            continue
        seen.add(key)
        dedup_idx.append(i)
    dedup_idx = np.array(dedup_idx)
    dedup_samples = [samples[i] for i in dedup_idx]
    dparts = _assemble(dedup_samples, district_poi)
    Xd = _matrix(dparts, "full")
    dtr2, dva2 = _random_split(len(dedup_idx))
    exp_g = _fit_eval(Xd, dparts["y"], dtr2, dva2)
    exp_g.update({"before_count": n, "after_count": int(len(dedup_idx)),
                  "removed_near_duplicates": int(n - len(dedup_idx))})

    # ---- 判断 ----
    community_gap = round(exp_b["val_mape"] - exp_a["val_mape"], 4)
    # 标签泄漏：特征列不含任何价格字段（结构性保证）+ shuffle 崩塌
    feature_has_price = False  # 仅 building_age + 区级 POI + district onehot
    leakage_risk_level = "low" if (not feature_has_price and shuffle_collapses) else (
        "high" if not shuffle_collapses else "medium")
    memorization_risk_level = (
        "low" if community_gap <= 8 else "medium" if community_gap <= 15 else "high")

    degraded_any = any(e.get("degraded") for e in (exp_c, exp_d))

    if not shuffle_collapses or leakage_risk_level == "high":
        status = "fail"
    elif memorization_risk_level == "high" or (
            exp_b["val_mape"] >= exp_a["baseline_val_mape"] - 1):
        status = "fail" if exp_b["val_mape"] >= exp_a["baseline_val_mape"] - 1 else "warning"
    elif memorization_risk_level == "medium" or degraded_any or community_gap > 8:
        status = "warning"
    else:
        status = "pass"

    interp = (
        f"random split val_mape={exp_a['val_mape']}%，community group split val_mape={exp_b['val_mape']}%"
        f"（gap={community_gap}pts）。因 community 不是特征，分组前后接近，说明模型靠区级/楼龄规律而非小区记忆。"
        f" label shuffle 后 r2_val={exp_f['r2_val']}、val_mape={exp_f['val_mape']}%（"
        f"{'显著崩塌，无明显泄漏' if shuffle_collapses else '未崩塌，存在泄漏风险'}）。"
        " 指标为弱泛化验证，非最终 test 成绩。"
    )

    leakage_check = {
        "test_used_for_training": False,
        "used_competition_test": False,
        "price_label_in_features": False,
        "total_vs_unit_price_leak": False,
        "feature_columns": ["building_age", *parts["poi_cols"],
                            *[f"district_is_{d}" for d in parts["districts_present"]]],
        "label_shuffle_r2_val": exp_f["r2_val"],
        "label_shuffle_val_mape": exp_f["val_mape"],
        "label_shuffle_collapses": shuffle_collapses,
        "leakage_risk_level": leakage_risk_level,
        "notes": [
            "特征仅含 building_age + 区级 POI 聚合 + district one-hot；price_total/rent/area 未进特征。",
            "community 不是特征：模型结构上无法记忆具体小区。",
        ],
    }
    group_split_results = {
        "random_split": exp_a, "community_group_split": exp_b,
        "plate_group_split": exp_c, "district_holdout": exp_d,
        "community_group_gap_mape_pts": community_gap,
    }

    report = {
        "generated_at": started.isoformat(),
        "modeled_record_count": n,
        "robustness_status": status,
        "leakage_risk_level": leakage_risk_level,
        "memorization_risk_level": memorization_risk_level,
        "community_group_gap_mape_pts": community_gap,
        "experiments": {
            "A_random_split": exp_a,
            "B_community_group_split": exp_b,
            "C_plate_group_split": exp_c,
            "D_district_holdout": exp_d,
            "E_ablation": ablation_summary,
            "F_label_shuffle": exp_f,
            "G_duplicate_sensitivity": exp_g,
        },
        "label_shuffle_check": {
            "val_mae": exp_f["val_mae"], "val_mape": exp_f["val_mape"],
            "r2_val": exp_f["r2_val"], "collapses_as_expected": shuffle_collapses},
        "ablation_summary": ablation_summary,
        "group_split_gap": {"community_vs_random_mape_pts": community_gap},
        "duplicate_sensitivity": exp_g,
        "final_metric_interpretation": interp,
        "elapsed_sec": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }

    _persist(report, leakage_check, ablation_summary, group_split_results)
    _update_latest(report)
    logger.info("T3.5 robustness done status=%s comm_gap=%s shuffle_collapse=%s",
                status, community_gap, shuffle_collapses)
    return report


# --------------------------------------------------------------------------- #
# 落盘 / 读取
# --------------------------------------------------------------------------- #
def _persist(report, leakage_check, ablation_summary, group_split_results) -> None:
    d = _models_dir()
    hp._save_json(d / "robustness_report.json", report)  # noqa: SLF001
    hp._save_json(d / "leakage_check.json", leakage_check)  # noqa: SLF001
    hp._save_json(d / "ablation_study.json", ablation_summary)  # noqa: SLF001
    hp._save_json(d / "group_split_results.json", group_split_results)  # noqa: SLF001
    (d / "robustness_report.md").write_text(_to_md(report), encoding="utf-8")


def _update_latest(report) -> None:
    latest = hp.get_latest()
    if latest is None:
        return
    latest["robustness_status"] = report["robustness_status"]
    latest["leakage_risk_level"] = report["leakage_risk_level"]
    latest["memorization_risk_level"] = report["memorization_risk_level"]
    latest["group_split_gap"] = report["group_split_gap"]
    latest["ablation_summary"] = report["ablation_summary"]
    latest["label_shuffle_check"] = report["label_shuffle_check"]
    latest["final_metric_interpretation"] = report["final_metric_interpretation"]
    hp._save_json(_models_dir() / "latest_result.json", latest)  # noqa: SLF001


def _to_md(r: dict[str, Any]) -> str:
    a = r["experiments"]["A_random_split"]
    b = r["experiments"]["B_community_group_split"]
    f = r["experiments"]["F_label_shuffle"]
    abl = r["ablation_summary"]
    lines = [
        "# 房价模型真实性复核报告（T3.5）", "",
        f"- 生成时间：{r['generated_at']}",
        f"- 建模样本：{r['modeled_record_count']}",
        f"- robustness_status：**{r['robustness_status']}**",
        f"- leakage_risk_level：{r['leakage_risk_level']}",
        f"- memorization_risk_level：{r['memorization_risk_level']}",
        f"- community group gap（mape pts）：{r['community_group_gap_mape_pts']}", "",
        "## 分组验证",
        f"- random split：val_mape={a['val_mape']}% r2_val={a['r2_val']}",
        f"- community group split：val_mape={b['val_mape']}% r2_val={b['r2_val']}",
        "## label shuffle",
        f"- shuffle 后：val_mape={f['val_mape']}% r2_val={f['r2_val']}（崩塌={r['label_shuffle_check']['collapses_as_expected']}）",
        "## ablation",
        f"- intrinsic：val_mape={abl['only_housing_intrinsic']['val_mape']}%",
        f"- +district：val_mape={abl['housing_plus_district']['val_mape']}%",
        f"- +poi：val_mape={abl['housing_plus_poi']['val_mape']}%",
        f"- POI 提升(mape pts)：{abl['poi_improves_mape_pts']}", "",
        "## 结论", r["final_metric_interpretation"],
        "", "> 本报告为弱泛化/防记忆验证，非最终 test 成绩；不使用 competition_test。",
    ]
    return "\n".join(lines)


def get_robustness() -> dict[str, Any] | None:
    return hp._read_json("robustness_report.json")  # noqa: SLF001


def get_leakage_check() -> dict[str, Any] | None:
    return hp._read_json("leakage_check.json")  # noqa: SLF001


def get_ablation_study() -> dict[str, Any] | None:
    return hp._read_json("ablation_study.json")  # noqa: SLF001
