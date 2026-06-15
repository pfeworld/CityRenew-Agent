"""房价基线模型（第5阶段）。

定位：为 H 维度提供"项目位置房价基线区间"，并产出可验证的模型指标。
原则：
- **只用 train 训练、val 验证**；test 严禁参与训练/调参/模型选择（红线）。
- 优先 sklearn GradientBoostingRegressor / RandomForestRegressor；
  样本不足、依赖缺失或异常时**自动降级**为分位数/中位数统计基线。
- 不追求复杂模型，保证可解释、可运行、可验证。
- 模型产物落 backend/data/processed/models/（已 gitignore），不提交、不外发。

异常值过滤（与 housing_analysis_service 对齐）：
- unit_price <= 0 删除；area <= 0 删除；year == 0 视为缺失（中位数填补）。
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import HousingRecord
from app.services import analysis_common as ac

logger = logging.getLogger("cityrenew.housing_model")

MIN_TRAIN_FOR_ML = 30  # 低于该训练样本量则降级为统计基线
FEATURE_KEYS = ("lng", "lat", "area", "year")

MODEL_GBR = "gradient_boosting"
MODEL_RF = "random_forest"
MODEL_MEDIAN = "median_baseline"


def _models_dir():
    d = settings.processed_dir / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_path():
    return _models_dir() / "housing_baseline.pkl"


def _metrics_path():
    return _models_dir() / "housing_baseline_metrics.json"


@dataclass
class ModelBundle:
    """模型束：可序列化，含模型句柄或统计基线 + 验证指标 + 残差分位。"""

    model_type: str
    train_count: int = 0
    val_count: int = 0
    val_mape: float | None = None
    val_mae: float | None = None
    degraded: bool = False
    note: str | None = None
    # 统计基线 / 特征填补
    median_unit_price: float | None = None
    median_year: float | None = None
    median_area: float | None = None
    # 残差分位（用于区间），相对误差
    residual_low: float | None = None  # 10% 分位相对残差
    residual_high: float | None = None  # 90% 分位相对残差
    # sklearn 模型（降级时为 None）
    sk_model: Any = field(default=None, repr=False)

    def to_metrics(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "val_mape": self.val_mape,
            "val_mae": self.val_mae,
            "degraded": self.degraded,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# 取数与清洗（仅 train/val）
# --------------------------------------------------------------------------- #
def _clean_rows(rows: list[HousingRecord]) -> list[dict[str, float | None]]:
    """异常值过滤 + 提取特征/标签；year==0 记为缺失（None）。"""
    cleaned: list[dict[str, float | None]] = []
    for r in rows:
        up = r.unit_price
        area = r.area
        if up is None or up <= 0:
            continue
        if area is None or area <= 0:
            continue
        if r.lng is None or r.lat is None:
            continue
        year = r.year if (r.year and r.year > 0) else None
        cleaned.append(
            {
                "lng": float(r.lng),
                "lat": float(r.lat),
                "area": float(area),
                "year": float(year) if year is not None else None,
                "unit_price": float(up),
            }
        )
    return cleaned


def _load_split(db: Session, split: str) -> list[dict[str, float | None]]:
    rows = db.query(HousingRecord).filter(HousingRecord.split == split).all()
    return _clean_rows(rows)


def _impute_year(samples: list[dict], median_year: float | None) -> None:
    for s in samples:
        if s["year"] is None:
            s["year"] = median_year


def _feature_matrix(samples: list[dict]) -> list[list[float]]:
    return [[s["lng"], s["lat"], s["area"], s["year"]] for s in samples]


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def _try_import_sklearn():
    try:
        from sklearn.ensemble import (  # noqa: F401
            GradientBoostingRegressor,
            RandomForestRegressor,
        )

        return True
    except Exception:  # pragma: no cover - 依赖缺失时降级
        return False


def _compute_val_metrics(y_true: list[float], y_pred: list[float]) -> tuple[float | None, float | None]:
    if not y_true:
        return None, None
    n = len(y_true)
    abs_err = [abs(float(p) - float(t)) for p, t in zip(y_pred, y_true)]
    mae = sum(abs_err) / n
    mape_terms = [abs(float(p) - float(t)) / float(t) for p, t in zip(y_pred, y_true) if t]
    mape = (sum(mape_terms) / len(mape_terms)) if mape_terms else None
    return (
        round(float(mape), 4) if mape is not None else None,
        round(float(mae), 2),
    )


def _residual_quantiles(y_true: list[float], y_pred: list[float]) -> tuple[float | None, float | None]:
    """验证集相对残差 (pred-true)/true 的 10%/90% 分位，用于基线区间。"""
    rels = sorted((float(p) - float(t)) / float(t) for p, t in zip(y_pred, y_true) if t)
    if not rels:
        return None, None
    lo = ac.percentile(rels, 0.1)
    hi = ac.percentile(rels, 0.9)
    return (float(lo) if lo is not None else None, float(hi) if hi is not None else None)


def train_baseline(db: Session, force_retrain: bool = False) -> ModelBundle:
    """训练（或加载已训练）房价基线模型。仅使用 train/val。"""
    if not force_retrain:
        cached = _load_bundle()
        if cached is not None:
            return cached

    train = _load_split(db, "train")
    val = _load_split(db, "val")

    median_year = ac.median([s["year"] for s in train if s["year"] is not None])
    median_area = ac.median([s["area"] for s in train])
    median_unit_price = ac.median([s["unit_price"] for s in train])
    _impute_year(train, median_year)
    _impute_year(val, median_year)

    bundle: ModelBundle

    if _try_import_sklearn() and len(train) >= MIN_TRAIN_FOR_ML:
        bundle = _train_ml(train, val, median_year, median_area, median_unit_price)
    else:
        reason = (
            "sklearn 不可用" if not _try_import_sklearn()
            else f"训练样本不足（{len(train)}<{MIN_TRAIN_FOR_ML}）"
        )
        bundle = _train_median(train, val, median_year, median_area, median_unit_price, reason)

    _save_bundle(bundle)
    logger.info(
        "housing baseline trained type=%s train=%s val=%s mape=%s degraded=%s",
        bundle.model_type, bundle.train_count, bundle.val_count,
        bundle.val_mape, bundle.degraded,
    )
    return bundle


def _train_ml(
    train: list[dict], val: list[dict],
    median_year: float | None, median_area: float | None, median_unit_price: float | None,
) -> ModelBundle:
    from sklearn.ensemble import GradientBoostingRegressor

    x_train = _feature_matrix(train)
    y_train = [s["unit_price"] for s in train]
    model = GradientBoostingRegressor(random_state=42)
    model.fit(x_train, y_train)

    val_mape = val_mae = None
    res_low = res_high = None
    if val:
        x_val = _feature_matrix(val)
        y_val = [s["unit_price"] for s in val]
        y_pred = list(model.predict(x_val))
        val_mape, val_mae = _compute_val_metrics(y_val, y_pred)
        res_low, res_high = _residual_quantiles(y_val, y_pred)

    return ModelBundle(
        model_type=MODEL_GBR,
        train_count=len(train),
        val_count=len(val),
        val_mape=val_mape,
        val_mae=val_mae,
        degraded=False,
        note="GradientBoostingRegressor（特征：lng/lat/area/year，仅 train 训练 / val 验证）",
        median_unit_price=median_unit_price,
        median_year=median_year,
        median_area=median_area,
        residual_low=res_low,
        residual_high=res_high,
        sk_model=model,
    )


def _train_median(
    train: list[dict], val: list[dict],
    median_year: float | None, median_area: float | None, median_unit_price: float | None,
    reason: str,
) -> ModelBundle:
    """降级统计基线：以 train 中位单价为预测，val 上计算指标与残差分位。"""
    val_mape = val_mae = None
    res_low = res_high = None
    if val and median_unit_price is not None:
        y_val = [s["unit_price"] for s in val]
        y_pred = [median_unit_price] * len(y_val)
        val_mape, val_mae = _compute_val_metrics(y_val, y_pred)
        res_low, res_high = _residual_quantiles(y_val, y_pred)
    return ModelBundle(
        model_type=MODEL_MEDIAN,
        train_count=len(train),
        val_count=len(val),
        val_mape=val_mape,
        val_mae=val_mae,
        degraded=True,
        note=f"降级为中位数统计基线（原因：{reason}）",
        median_unit_price=median_unit_price,
        median_year=median_year,
        median_area=median_area,
        residual_low=res_low,
        residual_high=res_high,
        sk_model=None,
    )


# --------------------------------------------------------------------------- #
# 预测 / 区间
# --------------------------------------------------------------------------- #
def predict_point(
    bundle: ModelBundle, lng: float | None, lat: float | None,
    area: float | None = None, year: float | None = None,
) -> float | None:
    """对单点预测基线单价。降级模型返回中位单价。"""
    if bundle.sk_model is None:
        return bundle.median_unit_price
    if lng is None or lat is None:
        return bundle.median_unit_price
    feat = [
        float(lng),
        float(lat),
        float(area) if area and area > 0 else (bundle.median_area or 0.0),
        float(year) if year and year > 0 else (bundle.median_year or 0.0),
    ]
    try:
        return float(bundle.sk_model.predict([feat])[0])
    except Exception:  # pragma: no cover
        return bundle.median_unit_price


def baseline_interval(
    bundle: ModelBundle, lng: float | None, lat: float | None,
    area: float | None = None, year: float | None = None,
    observed_median: float | None = None,
) -> dict[str, float | None]:
    """生成基线区间 [low, mid, high]。

    mid：模型预测；若可用 observed_median（落圈实测中位）则与之取均值以增稳。
    low/high：用 val 相对残差分位展开；无残差时用 ±12% 经验带。
    """
    pred = predict_point(bundle, lng, lat, area, year)
    if pred is None and observed_median is None:
        return {"low": None, "mid": None, "high": None}

    if pred is not None and observed_median is not None:
        mid = (pred + observed_median) / 2.0
    else:
        mid = pred if pred is not None else observed_median

    rl = bundle.residual_low if bundle.residual_low is not None else -0.12
    rh = bundle.residual_high if bundle.residual_high is not None else 0.12
    low = mid * (1 + rl)
    high = mid * (1 + rh)
    if low > high:
        low, high = high, low
    return {
        "low": round(low, 2),
        "mid": round(mid, 2),
        "high": round(high, 2),
    }


# --------------------------------------------------------------------------- #
# 序列化
# --------------------------------------------------------------------------- #
def _save_bundle(bundle: ModelBundle) -> None:
    try:
        with _model_path().open("wb") as f:
            pickle.dump(bundle, f)
        with _metrics_path().open("w", encoding="utf-8") as f:
            json.dump(bundle.to_metrics(), f, ensure_ascii=False, indent=2)
    except Exception as exc:  # pragma: no cover
        logger.warning("save housing model failed: %s", type(exc).__name__)


def load_metrics() -> dict[str, Any] | None:
    """只读加载已训练模型的指标（不触发训练、不读取 test）。"""
    path = _metrics_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # pragma: no cover
        return None


def _load_bundle() -> ModelBundle | None:
    path = _model_path()
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, ModelBundle) else None
    except Exception:  # pragma: no cover
        return None
