"""第11 T3：房价单价监督回归训练服务。

链路（全部 train/val，绝不使用 test）：
1. 强制经过 training_guard_service.assert_training_allowed（fail 抛 TrainingGuardError）。
2. 读取授权脱敏上海确认房价样本（research_property_trainable_candidates.jsonl，18215 条）。
3. 装配特征：房价内在特征（建成年代→楼龄）+ 按行政区聚合的高德 POI 特征（区级 join）。
   说明：可训练房价样本多数无经纬度，按红线**不编造坐标**，POI 特征以"行政区聚合"口径 join，
   逐条圈层特征待坐标补全后再做（记入 warning，不伪造）。
4. 训练多模型（baseline / ridge / elasticnet / random_forest / gradient_boosting /
   hist_gradient_boosting；xgboost/lightgbm 不可用自动跳过并标 partial_degraded）。
5. 仅用 val 选 best_model；输出 train/val 指标、overfit_gap、feature_importance、model_card、
   training_log、model_comparison、data_usage_audit、data_lineage_ids。
6. 模型与产物落盘到 backend/data/models/housing_price/（已 gitignore）。

红线：不写假指标；样本不足/库缺失标 degraded + 原因；test_used_for_training 恒 false；
POI 仅作特征不作标签；不删数据（异常价 winsorize 并记录 outlier_count/clip 策略）。
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.services import feature_engineering_service as fe
from app.services import poi_feature_service
from app.services import training_guard_service

logger = logging.getLogger("cityrenew.housing_train")

TRAINING_TASK = "housing_price_regression"
TRAINING_SOURCE_ID = "research_housing_property"
CURRENT_YEAR = datetime.now(timezone.utc).year
RANDOM_STATE = 42
VAL_RATIO = 0.2
LABEL_CLIP_LOW_Q = 0.01
LABEL_CLIP_HIGH_Q = 0.99

# 标签单位一致性带（元/㎡）：上海住宅挂牌均价合理区间。
# 授权样本中存在大量 <5000 的记录（p25≈123），与 元/㎡ 口径明显不同（疑似不同单位/口径），
# 两簇间在 ~200 与 ~8000 之间有干净缺口。按红线既不混训也不为指标乱删，
# 用本规则剔除"非 元/㎡ 口径"记录并如实记录数量与原因（不重构、不臆测原值）。
PRICE_MIN_YUAN_SQM = 5000.0
PRICE_MAX_YUAN_SQM = 300000.0
UNIT_CONSISTENCY_RULE = (
    f"仅保留 price_unit ∈ [{PRICE_MIN_YUAN_SQM:.0f}, {PRICE_MAX_YUAN_SQM:.0f}] 元/㎡ 的记录；"
    "区间外为非元/㎡口径（无面积字段无法换算，按红线剔除并记录，不臆测原值）。"
)

# 区级 POI 聚合一级类（用于房价特征 join）
DISTRICT_POI_L1 = (
    poi_feature_service.L1_PUBLIC_SERVICE, poi_feature_service.L1_COMMERCIAL,
    poi_feature_service.L1_TRANSPORT, poi_feature_service.L1_CULTURE_SPORTS,
    poi_feature_service.L1_INDUSTRY_OFFICE, poi_feature_service.L1_GREEN_SPACE,
    poi_feature_service.L1_GOVERNMENT, poi_feature_service.L1_RESIDENTIAL,
)

SH_DISTRICTS = (
    "黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区", "杨浦区",
    "浦东新区", "闵行区", "宝山区", "嘉定区", "金山区", "松江区", "青浦区",
    "奉贤区", "崇明区",
)


def _models_dir():
    d = settings.data_dir / "models" / "housing_price"
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalize_district(text: str | None) -> str | None:
    """从任意文本中识别标准上海行政区名（黄浦区/浦东新区/崇明区...）。"""
    if not text:
        return None
    if "浦东" in text:
        return "浦东新区"
    if "崇明" in text:
        return "崇明区"
    for d in SH_DISTRICTS:
        if d in text or d[:-1] in text:  # 含"徐汇"或"徐汇区"
            return d
    return None


def _housing_path():
    return (settings.data_dir / "external" / "authorized_property" / "processed"
            / "research_property_trainable_candidates.jsonl")


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_housing_samples() -> dict[str, Any]:
    """加载授权脱敏上海确认可训练房价样本（仅 used_for_training 且非 test）。"""
    path = _housing_path()
    samples: list[dict[str, Any]] = []
    skipped = 0
    if not path.exists():
        return {"samples": [], "skipped": 0, "available": False}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                skipped += 1
                continue
            # 红线：仅可训练 + 非 competition_test + 上海确认 + 有有效标签
            if not rec.get("used_for_training"):
                skipped += 1
                continue
            if rec.get("competition_test"):
                skipped += 1
                continue
            if rec.get("shanghai_verdict") not in (None, "shanghai"):
                skipped += 1
                continue
            price = rec.get("price_unit")
            try:
                price = float(price)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not math.isfinite(price) or price <= 0:
                skipped += 1
                continue
            district = (normalize_district(rec.get("region"))
                        or normalize_district(rec.get("shanghai_district"))
                        or normalize_district(rec.get("address"))
                        or normalize_district(rec.get("community")))
            by = rec.get("build_year")
            try:
                by = int(by)
                if by < 1900 or by > CURRENT_YEAR:
                    by = None
            except (TypeError, ValueError):
                by = None
            samples.append({"price_unit": price, "build_year": by, "district": district})
    return {"samples": samples, "skipped": skipped, "available": True}


@lru_cache(maxsize=1)
def district_price_baseline() -> dict[str, dict[str, float]]:
    """各行政区授权脱敏成交样本的真实房价基线（中位/分位/样本量）。

    仅采用 price_unit ∈ [PRICE_MIN_YUAN_SQM, PRICE_MAX_YUAN_SQM] 的口径一致样本，
    供运行时「外区无本地落圈成交样本」时作为价格基线使用——真实数据、可回溯、
    非编造；与正式房价模型同源（同一批授权脱敏上海确认样本）。
    """
    data = load_housing_samples()
    buckets: dict[str, list[float]] = defaultdict(list)
    for s in data.get("samples", []):
        d = s.get("district")
        p = s.get("price_unit")
        if not d or p is None:
            continue
        if p < PRICE_MIN_YUAN_SQM or p > PRICE_MAX_YUAN_SQM:
            continue
        buckets[d].append(float(p))
    out: dict[str, dict[str, float]] = {}
    for d, ps in buckets.items():
        if len(ps) < 20:  # 样本过少不作区基线，避免不稳健
            continue
        arr = np.array(ps, dtype=float)
        out[d] = {
            "median": round(float(np.median(arr)), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "count": int(len(ps)),
        }
    return out


@lru_cache(maxsize=1)
def district_build_year_baseline() -> dict[str, dict[str, float]]:
    """各行政区授权脱敏样本的建成年代/楼龄真实基线（中位/分位/样本量）。

    供运行时「圈层内无本地建成年代样本」时作为楼龄基线使用——真实、可回溯、非编造。
    """
    data = load_housing_samples()
    buckets: dict[str, list[int]] = defaultdict(list)
    for s in data.get("samples", []):
        d = s.get("district")
        by = s.get("build_year")
        if not d or not by:
            continue
        buckets[d].append(int(by))
    out: dict[str, dict[str, float]] = {}
    for d, ys in buckets.items():
        if len(ys) < 20:
            continue
        arr = np.array(ys, dtype=float)
        median_year = float(np.median(arr))
        out[d] = {
            "median_build_year": round(median_year, 0),
            "median_building_age": round(CURRENT_YEAR - median_year, 0),
            "p25_build_year": round(float(np.percentile(arr, 25)), 0),
            "p75_build_year": round(float(np.percentile(arr, 75)), 0),
            "count": int(len(ys)),
        }
    return out


def _rent_baseline_path():
    return (settings.data_dir / "external" / "authorized_property" / "processed"
            / "rent_baseline.json")


@lru_cache(maxsize=1)
def citywide_rent_baseline() -> dict[str, Any] | None:
    """全市租金参考基线（元/㎡/月）：来自科研授权链家租赁挂牌（去除小区信息，仅全市口径）。

    数据已去除小区信息、无法稳定定位到区，按红线只给"全市参考"，并明确标注口径；
    由一次性预处理脚本写入 rent_baseline.json（已 gitignore），运行时只读。
    """
    path = _rent_baseline_path()
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not obj.get("median_rent_unit"):
        return None
    return obj


def build_district_poi_features() -> dict[str, dict[str, float]]:
    """按行政区聚合高德去重 POI 的一级类计数/占比（区级特征，供房价 join）。"""
    per_district_l1: dict[str, Counter] = defaultdict(Counter)
    per_district_total: Counter = Counter()
    path = poi_feature_service.amap_dedup_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            d = normalize_district(rec.get("district"))
            if d is None:
                continue
            l1, _l2 = poi_feature_service.map_poi_category(
                rec.get("type"), matched_keywords=rec.get("matched_keywords"))
            per_district_l1[d][l1] += 1
            per_district_total[d] += 1

    out: dict[str, dict[str, float]] = {}
    for d, total in per_district_total.items():
        feats = {"district_poi_total": float(total)}
        for l1 in DISTRICT_POI_L1:
            feats[f"district_share_{l1}"] = round(per_district_l1[d][l1] / total, 6) if total else 0.0
        out[d] = feats
    return out


# --------------------------------------------------------------------------- #
# 特征矩阵
# --------------------------------------------------------------------------- #
def build_feature_matrix(
    samples: list[dict[str, Any]], district_poi: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """装配数值特征矩阵 X、标签 y、特征名与缺失/异常审计。"""
    districts_present = [d for d in SH_DISTRICTS if any(s["district"] == d for s in samples)]
    onehot_cols = [f"district_is_{d}" for d in districts_present]
    poi_cols = ["district_poi_total"] + [f"district_share_{l1}" for l1 in DISTRICT_POI_L1]

    feature_names = ["building_age"] + poi_cols + onehot_cols
    feature_groups_used = {
        "housing_intrinsic": ["building_age"],
        "district_poi_aggregate": poi_cols,
        "district_onehot": onehot_cols,
    }

    # 楼龄中位数（用于缺失填充）
    ages = [CURRENT_YEAR - s["build_year"] for s in samples if s["build_year"]]
    age_median = float(np.median(ages)) if ages else 30.0
    # 区级 POI 缺失填充：用全市中位数
    poi_medians: dict[str, float] = {}
    for col in poi_cols:
        vals = [district_poi[d][col] for d in district_poi if col in district_poi[d]]
        poi_medians[col] = float(np.median(vals)) if vals else 0.0

    missing_counts: dict[str, int] = {"building_age": 0, "district": 0}
    rows: list[list[float]] = []
    y: list[float] = []
    for s in samples:
        if s["build_year"]:
            age = float(CURRENT_YEAR - s["build_year"])
        else:
            age = age_median
            missing_counts["building_age"] += 1
        d = s["district"]
        dp = district_poi.get(d) if d else None
        if dp is None:
            missing_counts["district"] += 1
        row = [age]
        for col in poi_cols:
            row.append(float(dp[col]) if dp and col in dp else poi_medians[col])
        for dd in districts_present:
            row.append(1.0 if d == dd else 0.0)
        rows.append(row)
        y.append(s["price_unit"])

    X = np.asarray(rows, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # 异常价 winsorize（不删数据，仅夹断并记录）
    lo = float(np.quantile(y_arr, LABEL_CLIP_LOW_Q))
    hi = float(np.quantile(y_arr, LABEL_CLIP_HIGH_Q))
    outlier_count = int(np.sum((y_arr < lo) | (y_arr > hi)))
    y_clipped = np.clip(y_arr, lo, hi)

    missing_features = [k for k, v in missing_counts.items() if v > 0]
    return {
        "X": X, "y": y_clipped, "y_raw": y_arr,
        "feature_names": feature_names,
        "feature_groups_used": feature_groups_used,
        "missing_counts": missing_counts,
        "missing_features": missing_features,
        "outlier_count": outlier_count,
        "price_clip_strategy": f"winsorize_q[{LABEL_CLIP_LOW_Q},{LABEL_CLIP_HIGH_Q}]=[{lo:.0f},{hi:.0f}]",
        "districts_present": districts_present,
        "age_median": age_median,
    }


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.where(y_true == 0, np.nan, y_true)) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": round(mae, 2), "rmse": round(rmse, 2),
            "mape": round(mape, 4), "r2": round(r2, 4)}


# --------------------------------------------------------------------------- #
# 模型族
# --------------------------------------------------------------------------- #
def _build_estimators() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """返回 (可用估计器, 跳过的模型+原因)。"""
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    est: list[dict[str, Any]] = [
        {"name": "median_baseline", "kind": "baseline", "strategy": "median"},
        {"name": "mean_baseline", "kind": "baseline", "strategy": "mean"},
        {"name": "ridge", "kind": "sklearn",
         "model": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0, random_state=RANDOM_STATE))]),
         "scaled": True},
        {"name": "elasticnet", "kind": "sklearn",
         "model": Pipeline([("sc", StandardScaler()),
                            ("m", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000))]),
         "scaled": True},
        {"name": "random_forest", "kind": "sklearn",
         "model": RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=3,
                                        random_state=RANDOM_STATE, n_jobs=-1)},
        {"name": "gradient_boosting", "kind": "sklearn",
         "model": GradientBoostingRegressor(random_state=RANDOM_STATE)},
        {"name": "hist_gradient_boosting", "kind": "sklearn",
         "model": HistGradientBoostingRegressor(random_state=RANDOM_STATE)},
    ]
    skipped: list[dict[str, Any]] = []
    for name, mod in (("xgboost", "xgboost"), ("lightgbm", "lightgbm")):
        try:
            __import__(mod)
            # 已安装但本阶段不强依赖：若安装则加入
            if name == "xgboost":
                from xgboost import XGBRegressor
                est.append({"name": "xgboost", "kind": "sklearn",
                            "model": XGBRegressor(n_estimators=300, max_depth=6,
                                                  random_state=RANDOM_STATE, n_jobs=-1)})
            else:
                from lightgbm import LGBMRegressor
                est.append({"name": "lightgbm", "kind": "sklearn",
                            "model": LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)})
        except Exception as exc:  # noqa: BLE001
            skipped.append({"model": name, "reason": f"未安装/不可用：{type(exc).__name__}"})
    return est, skipped


def _feature_importance(spec: dict[str, Any], feature_names: list[str]) -> list[dict[str, Any]]:
    model = spec.get("model")
    if model is None:
        return []
    est = model.named_steps["m"] if spec.get("scaled") else model
    importances = None
    if hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        importances = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    if importances is None or importances.size != len(feature_names):
        return []
    total = float(importances.sum()) or 1.0
    pairs = sorted(zip(feature_names, importances), key=lambda kv: kv[1], reverse=True)
    return [{"feature": n, "importance": round(float(v), 6),
             "importance_norm": round(float(v) / total, 6)} for n, v in pairs]


# --------------------------------------------------------------------------- #
# 训练总入口
# --------------------------------------------------------------------------- #
def train(db: Session, req: dict[str, Any]) -> dict[str, Any]:
    """房价监督训练总入口（先 guard，再 fit；dry_run 仅 guard+数据审计不训练）。"""
    guard_req = {
        "training_task": TRAINING_TASK,
        "project_id": req.get("project_id", 1),
        "use_authorized_property": req.get("use_authorized_property", True),
        "use_poi_features": req.get("use_poi_features", True),
        "requested_splits": ["train", "val"],
        "dry_run": bool(req.get("dry_run", False)),
    }
    # 1) 强制护栏（fail 抛 TrainingGuardError）
    guard = training_guard_service.assert_training_allowed(db, guard_req)

    started = datetime.now(timezone.utc)
    training_log: list[str] = [f"guard passed: status={guard['status']} strength={guard.get('supervised_training_strength')}"]

    # 2) 加载授权房价样本
    loaded = load_housing_samples()
    all_samples = loaded["samples"]
    authorized_count = len(all_samples)
    # 单位一致性过滤（透明规则；剔除非元/㎡口径，不臆测原值）
    samples = [s for s in all_samples
               if PRICE_MIN_YUAN_SQM <= s["price_unit"] <= PRICE_MAX_YUAN_SQM]
    excluded_unit_inconsistent = authorized_count - len(samples)
    training_log.append(
        f"loaded authorized={authorized_count}（skipped={loaded['skipped']}）；"
        f"单位一致性过滤后 modelable={len(samples)}，剔除非元/㎡口径={excluded_unit_inconsistent}"
    )

    data_lineage_ids = list(dict.fromkeys(
        fe._poi_lineage_ids()  # noqa: SLF001
        + [str(x) for s in guard["data_usage_audit"]["sources"] for x in s.get("data_lineage_ids", [])]
        + ["lin:research_property_trainable_candidates.jsonl"]
    ))

    if req.get("dry_run"):
        return {
            "status": "dry_run", "trained": False, "guard": guard,
            "trainable_record_count": len(samples),
            "data_lineage_ids": data_lineage_ids,
            "reason": "dry_run=true：仅通过护栏与数据审计，未训练。",
        }

    if len(samples) < training_guard_service.MIN_TRAINABLE_RECORDS:
        return _degraded_result(
            guard, data_lineage_ids,
            reason=f"可训练样本 {len(samples)} < {training_guard_service.MIN_TRAINABLE_RECORDS}",
            trainable=len(samples),
        )

    # 3) 区级 POI 聚合 + 特征矩阵
    district_poi = build_district_poi_features()
    training_log.append(f"district POI aggregated: {len(district_poi)} 区")
    mat = build_feature_matrix(samples, district_poi)
    X, y = mat["X"], mat["y"]
    training_log.append(
        f"feature matrix: X={X.shape}, outliers(winsorized)={mat['outlier_count']}, "
        f"missing={mat['missing_counts']}"
    )

    # 4) train/val 划分（确定性，无 test）
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(len(y))
    n_val = int(len(y) * VAL_RATIO)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    Xtr, Xv, ytr, yv = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
    training_log.append(f"split: train={len(train_idx)} val={len(val_idx)} test=0")

    warnings: list[str] = []
    coord_warn = ("房价样本多数无经纬度：POI 特征按行政区聚合 join，逐条圈层特征待坐标补全（未编造坐标）。")
    warnings.append(coord_warn)
    if excluded_unit_inconsistent > 0:
        warnings.append(
            f"授权样本中 {excluded_unit_inconsistent}/{authorized_count} 条 price_unit 非元/㎡口径，"
            f"已按单位一致性规则剔除（{UNIT_CONSISTENCY_RULE}）；数据质量问题已记录，未臆测换算。"
        )
    if mat["missing_features"]:
        warnings.append(f"存在缺失特征已中位数填充：{mat['missing_features']}")

    # 5) 训练多模型
    estimators, skipped_models = _build_estimators()
    partial_degraded = len(skipped_models) > 0
    if partial_degraded:
        warnings.append(f"以下模型不可用已跳过：{[s['model'] for s in skipped_models]}（不伪造其指标）")

    import warnings as _warnings

    comparison: list[dict[str, Any]] = []
    fitted: dict[str, dict[str, Any]] = {}
    for spec in estimators:
        try:
            if spec["kind"] == "baseline":
                const = float(np.median(ytr)) if spec["strategy"] == "median" else float(np.mean(ytr))
                pred_tr = np.full_like(ytr, const)
                pred_v = np.full_like(yv, const)
                spec["_const"] = const
            else:
                with _warnings.catch_warnings(), np.errstate(all="ignore"):
                    _warnings.simplefilter("ignore")
                    spec["model"].fit(Xtr, ytr)
                    pred_tr = spec["model"].predict(Xtr)
                    pred_v = spec["model"].predict(Xv)
                if not (np.all(np.isfinite(pred_tr)) and np.all(np.isfinite(pred_v))):
                    raise ValueError("预测出现非有限值（数值不稳定）")
            mtr = _metrics(ytr, pred_tr)
            mv = _metrics(yv, pred_v)
            comparison.append({
                "model": spec["name"], "kind": spec["kind"],
                "train_mae": mtr["mae"], "val_mae": mv["mae"],
                "train_rmse": mtr["rmse"], "val_rmse": mv["rmse"],
                "train_mape": mtr["mape"], "val_mape": mv["mape"],
                "r2_train": mtr["r2"], "r2_val": mv["r2"],
                "overfit_gap_mae": round(mv["mae"] - mtr["mae"], 2),
                "overfit_gap_mape": round(mv["mape"] - mtr["mape"], 4),
            })
            fitted[spec["name"]] = {"spec": spec, "val_mae": mv["mae"]}
        except Exception as exc:  # noqa: BLE001
            skipped_models.append({"model": spec["name"], "reason": f"训练异常：{type(exc).__name__}: {exc}"})
            logger.warning("model %s failed: %s", spec["name"], exc)

    if not comparison:
        return _degraded_result(guard, data_lineage_ids, reason="所有模型训练失败", trainable=len(samples))

    # 6) 仅用 val 选 best
    comparison.sort(key=lambda c: c["val_mae"])
    best = comparison[0]
    best_spec = fitted[best["model"]]["spec"]
    training_log.append(f"best_model={best['model']} val_mae={best['val_mae']} val_mape={best['val_mape']}")

    # 7) feature_importance（best；若 best 为 baseline 取最优有重要性的模型）
    fi = _feature_importance(best_spec, mat["feature_names"])
    fi_source = best["model"]
    if not fi:
        for c in comparison:
            cand = fitted.get(c["model"], {}).get("spec")
            if cand and cand["kind"] == "sklearn":
                fi = _feature_importance(cand, mat["feature_names"])
                if fi:
                    fi_source = c["model"]
                    warnings.append(f"best_model={best['model']} 无重要性，feature_importance 取自 {fi_source}")
                    break

    # 8) 产物
    model_card = {
        "model_name": "house_price_regressor",
        "task": TRAINING_TASK,
        "target": "price_unit (元/㎡, 挂牌均价)",
        "best_model": best["model"],
        "feature_count": len(mat["feature_names"]),
        "feature_groups_used": mat["feature_groups_used"],
        "authorized_record_count": authorized_count,
        "modeled_record_count": len(samples),
        "excluded_unit_inconsistent": excluded_unit_inconsistent,
        "unit_consistency_rule": UNIT_CONSISTENCY_RULE,
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": 0,
        "test_used_for_training": False,
        "supervised_training_strength": guard.get("supervised_training_strength"),
        "price_clip_strategy": mat["price_clip_strategy"],
        "outlier_count": mat["outlier_count"],
        "degraded": False,
        "partial_degraded": partial_degraded,
        "skipped_models": skipped_models,
        "metrics": {k: best[k] for k in (
            "train_mae", "val_mae", "train_rmse", "val_rmse",
            "train_mape", "val_mape", "r2_train", "r2_val",
            "overfit_gap_mae", "overfit_gap_mape")},
        "coordinate_limitation": coord_warn,
        "created_at": started.isoformat(),
        "random_state": RANDOM_STATE,
    }
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    training_log.append(f"training finished in {elapsed:.1f}s")

    audit = dict(guard["data_usage_audit"])
    audit["train_count"] = int(len(train_idx))
    audit["val_count"] = int(len(val_idx))
    audit["test_count"] = 0
    audit["trainable_record_count"] = authorized_count
    audit["modeled_record_count"] = len(samples)
    audit["excluded_unit_inconsistent"] = excluded_unit_inconsistent
    audit["unit_consistency_rule"] = UNIT_CONSISTENCY_RULE
    audit["data_lineage_ids"] = data_lineage_ids

    artifacts = _persist_artifacts(
        best_spec, model_card, training_log, fi, comparison, audit, mat["feature_names"]
    )

    result = {
        "status": "success",
        "trained": True,
        "training_task": TRAINING_TASK,
        "guard_status": guard["status"],
        "best_model": best["model"],
        "model_comparison": comparison,
        "metrics": model_card["metrics"],
        "overfit_gap": {"mae": best["overfit_gap_mae"], "mape": best["overfit_gap_mape"]},
        "feature_importance": fi,
        "feature_importance_source": fi_source,
        "feature_groups_used": mat["feature_groups_used"],
        "missing_features": mat["missing_features"],
        "skipped_models": skipped_models,
        "degraded": False,
        "partial_degraded": partial_degraded,
        "trainable_record_count": authorized_count,
        "modeled_record_count": len(samples),
        "excluded_unit_inconsistent": excluded_unit_inconsistent,
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": 0,
        "test_used_for_training": False,
        "data_lineage_ids": data_lineage_ids,
        "data_usage_audit": audit,
        "model_card": model_card,
        "training_log": training_log,
        "warnings": warnings,
        "artifacts": artifacts,
        "created_at": started.isoformat(),
    }
    _save_json(_models_dir() / "latest_result.json", _strip_heavy(result))
    logger.info("T3 training done best=%s val_mae=%s partial_degraded=%s",
                best["model"], best["val_mae"], partial_degraded)
    return result


def _degraded_result(guard, lineage, *, reason: str, trainable: int) -> dict[str, Any]:
    return {
        "status": "degraded", "trained": False, "degraded": True,
        "training_task": TRAINING_TASK, "guard_status": guard["status"],
        "reason": reason, "trainable_record_count": trainable,
        "test_used_for_training": False, "data_lineage_ids": lineage,
        "warnings": [reason], "skipped_models": [],
    }


def _strip_heavy(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    return out


# --------------------------------------------------------------------------- #
# 产物落盘 / 读取
# --------------------------------------------------------------------------- #
def _save_json(path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _persist_artifacts(best_spec, model_card, training_log, fi, comparison, audit, feature_names):
    import joblib

    d = _models_dir()
    model_path = d / "model.pkl"
    try:
        if best_spec["kind"] == "baseline":
            joblib.dump({"baseline_const": best_spec.get("_const"),
                         "feature_names": feature_names}, model_path)
        else:
            joblib.dump({"model": best_spec["model"], "feature_names": feature_names}, model_path)
        model_saved = True
    except Exception as exc:  # noqa: BLE001
        logger.error("model save failed: %s", exc)
        model_saved = False

    _save_json(d / "model_card.json", model_card)
    _save_json(d / "training_log.json", {"log": training_log})
    _save_json(d / "feature_importance.json", {"feature_importance": fi})
    _save_json(d / "model_comparison.json", {"model_comparison": comparison})
    _save_json(d / "data_usage_audit.json", audit)
    return {
        "dir": str(d),
        "model_pkl": str(model_path) if model_saved else None,
        "model_saved": model_saved,
        "files": ["model.pkl", "model_card.json", "training_log.json",
                  "feature_importance.json", "model_comparison.json", "data_usage_audit.json"],
    }


def _read_json(name: str) -> dict[str, Any] | None:
    p = _models_dir() / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, TypeError):
        return None


def get_latest() -> dict[str, Any] | None:
    return _read_json("latest_result.json")


def get_audit() -> dict[str, Any] | None:
    return _read_json("data_usage_audit.json")


def get_feature_importance() -> dict[str, Any] | None:
    return _read_json("feature_importance.json")


def get_training_log() -> dict[str, Any] | None:
    return _read_json("training_log.json")


# --------------------------------------------------------------------------- #
# 质量门禁
# --------------------------------------------------------------------------- #
def training_quality(result: dict[str, Any] | None) -> dict[str, Any]:
    """T3 训练质量门禁：pass / warning / fail。"""
    if result is None:
        return {"training_quality_status": "fail", "fail": ["尚无训练结果"],
                "pass": [], "warning": [], "can_enter_t4_t5": False}

    passed: list[str] = []
    failed: list[str] = []
    warning: list[str] = []

    def hard(cond: bool, name: str) -> None:
        passed.append(name) if cond else failed.append(name)

    metrics = result.get("metrics", {}) or {}
    val_mae = metrics.get("val_mae")
    val_mape = metrics.get("val_mape")

    hard(result.get("guard_status") == "pass", "guard_status=pass")
    hard(int(result.get("trainable_record_count", 0)) >= training_guard_service.MIN_TRAINABLE_RECORDS,
         f"trainable_record_count>={training_guard_service.MIN_TRAINABLE_RECORDS}")
    hard(isinstance(val_mae, (int, float)) and val_mae > 0, "val_mae 有效")
    hard(isinstance(val_mape, (int, float)) and val_mape > 0, "val_mape 有效")
    hard(bool(result.get("best_model")), "best_model 不为空")
    hard(bool(result.get("feature_importance")), "feature_importance 不为空")
    hard(bool(result.get("model_card")), "model_card 存在")
    hard(bool(result.get("training_log")), "training_log 存在")
    hard(bool(result.get("data_usage_audit")), "data_usage_audit 存在")
    hard(result.get("test_used_for_training") is False, "test_used_for_training=false")
    hard(len(result.get("data_lineage_ids", [])) > 0, "data_lineage_ids 非空")
    hard(bool(result.get("artifacts", {}).get("model_saved")), "模型文件保存成功")

    if result.get("partial_degraded"):
        warning.append(f"xgboost/lightgbm 不可用（partial_degraded），已跳过：{[s['model'] for s in result.get('skipped_models', [])]}")
    og = result.get("overfit_gap", {}) or {}
    if isinstance(og.get("mape"), (int, float)) and og["mape"] > 5:
        warning.append(f"overfit_gap_mape 偏高：{og['mape']}")
    if result.get("missing_features"):
        warning.append(f"部分特征缺失（已填充）：{result['missing_features']}")
    for w in result.get("warnings", []):
        if "经纬度" in w or "坐标" in w:
            warning.append("逐条 POI 圈层特征待坐标补全（区级聚合替代）")
            break

    status = "fail" if failed else ("warning" if warning else "pass")
    return {
        "training_quality_status": status,
        "pass": passed, "warning": warning, "fail": failed,
        "can_enter_t4_t5": status in ("pass", "warning"),
        "best_model": result.get("best_model"),
        "val_mae": val_mae, "val_mape": val_mape,
        "recommended_next_action": (
            "修复 fail 项后重训" if failed else
            "可进入 T4 项目类型识别 / T5 评分校准；建议后续补样本坐标做逐条圈层特征"
        ),
    }
