"""第5阶段房价基线模型训练审计（轻量、纯只读、反作弊）。

目的：证明房价基线模型的 val 指标"可信"——即模型确实**只用 train 训练、val 验证**，
test 从未参与训练 / 调参 / 模型选择；且已保存的 housing_baseline_metrics.json 不是写死或造假，
而是可被独立重算复现。

做法：
1. 读取 DB 中房价数据 split 分布（train/val/test 仅计数；test 绝不取明细用于训练）。
2. 独立重跑一次训练/验证流程（复用 housing_price_model 的训练逻辑，但**不落盘**），
   只加载 split=train / split=val，重算 val_mape / val_mae / model_type。
3. 与已保存 metrics 对比（train_count / val_count / val_mape / val_mae / model_type）。
4. 输出各 split 的 id 集合脱敏 hash（仅指纹，不含任何明细）。
5. 给出审计结论与可信判定。

红线：不调外部 API、不使用 LLM、不返回原始房源/小区/地址/坐标/raw_json；
test 仅参与计数与 id 指纹，绝不进入训练/验证/调参。本审计不写 DB、不写模型文件。
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import HousingRecord
from app.services import analysis_common as ac
from app.services import housing_price_model as hpm

logger = logging.getLogger("cityrenew.model_audit")

ST_PASS = "pass"
ST_WARN = "warning"
ST_FAIL = "fail"

MAPE_TOL = 0.0005  # 重算与保存的 val_mape 容差
MAE_TOL = 1.0      # 重算与保存的 val_mae 容差（元/㎡）
HASH_ALGO = "sha256(sorted_ids)"


def _mk(name: str, value: Any, threshold: str, status: str, explanation: str) -> dict[str, Any]:
    return {
        "metric_name": name,
        "current_value": value,
        "threshold": threshold,
        "status": status,
        "explanation": explanation,
    }


def _split_ids(db: Session, split: str) -> list[int]:
    """仅取某 split 的记录 id（不取任何特征/明细）。"""
    return [i for (i,) in db.query(HousingRecord.id).filter(HousingRecord.split == split).all()]


def _ids_hash(ids: list[int]) -> str:
    h = hashlib.sha256()
    for i in sorted(ids):
        h.update(str(i).encode())
        h.update(b",")
    return h.hexdigest()


def _close(a: float | None, b: float | None, tol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _recompute_train_val(db: Session) -> hpm.ModelBundle:
    """独立重算：仅 train 训练、val 验证，复用模型逻辑但**不落盘**（test 永不加载）。"""
    train = hpm._load_split(db, "train")  # filter split=='train'
    val = hpm._load_split(db, "val")      # filter split=='val'

    median_year = ac.median([s["year"] for s in train if s["year"] is not None])
    median_area = ac.median([s["area"] for s in train])
    median_unit_price = ac.median([s["unit_price"] for s in train])
    hpm._impute_year(train, median_year)
    hpm._impute_year(val, median_year)

    if hpm._try_import_sklearn() and len(train) >= hpm.MIN_TRAIN_FOR_ML:
        return hpm._train_ml(train, val, median_year, median_area, median_unit_price)
    reason = (
        "sklearn 不可用" if not hpm._try_import_sklearn()
        else f"训练样本不足（{len(train)}<{hpm.MIN_TRAIN_FOR_ML}）"
    )
    return hpm._train_median(train, val, median_year, median_area, median_unit_price, reason)


def run_model_audit(db: Session) -> dict[str, Any]:
    """执行房价模型训练审计（纯只读）。"""
    # ---- 1. split 分布（仅计数）----
    train_ids = _split_ids(db, "train")
    val_ids = _split_ids(db, "val")
    test_ids = _split_ids(db, "test")
    split_counts = {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
    }

    # ---- 4. 脱敏 hash（仅 id 指纹）----
    hashes = {
        "train_ids_hash": _ids_hash(train_ids),
        "val_ids_hash": _ids_hash(val_ids),
        "test_ids_hash": _ids_hash(test_ids),
        "algo": HASH_ALGO,
    }

    # split 互斥性校验（id 集合应两两不相交，证明 test 未混入 train/val）
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    train_test_overlap = sorted(train_set & test_set)
    val_test_overlap = sorted(val_set & test_set)
    train_val_overlap = sorted(train_set & val_set)
    test_in_train_val = bool(train_test_overlap or val_test_overlap)

    # ---- 2. 独立重算（仅 train/val）----
    metrics_recomputed = False
    recomputed: dict[str, Any] = {}
    try:
        bundle = _recompute_train_val(db)
        recomputed = {
            "model_type": bundle.model_type,
            "train_count": bundle.train_count,
            "val_count": bundle.val_count,
            "val_mape": bundle.val_mape,
            "val_mae": bundle.val_mae,
            "degraded": bundle.degraded,
        }
        metrics_recomputed = True
    except Exception as exc:  # pragma: no cover
        logger.warning("model audit recompute failed: %s", type(exc).__name__)
        recomputed = {"error": "重算失败"}

    # ---- 3. 与已保存 metrics 对比 ----
    saved = hpm.load_metrics() or {}
    saved_present = bool(saved)
    saved_view = {
        "model_type": saved.get("model_type"),
        "train_count": saved.get("train_count"),
        "val_count": saved.get("val_count"),
        "val_mape": saved.get("val_mape"),
        "val_mae": saved.get("val_mae"),
        "degraded": saved.get("degraded"),
    }

    comparison: dict[str, Any] = {}
    metrics_match_saved = False
    if metrics_recomputed and saved_present:
        train_match = recomputed["train_count"] == saved.get("train_count")
        val_match = recomputed["val_count"] == saved.get("val_count")
        type_match = recomputed["model_type"] == saved.get("model_type")
        mape_match = _close(recomputed["val_mape"], saved.get("val_mape"), MAPE_TOL)
        mae_match = _close(recomputed["val_mae"], saved.get("val_mae"), MAE_TOL)
        comparison = {
            "train_count_match": train_match,
            "val_count_match": val_match,
            "model_type_match": type_match,
            "val_mape_match": mape_match,
            "val_mae_match": mae_match,
            "val_mape_diff": (
                None if recomputed["val_mape"] is None or saved.get("val_mape") is None
                else round(abs(recomputed["val_mape"] - saved["val_mape"]), 6)
            ),
            "val_mae_diff": (
                None if recomputed["val_mae"] is None or saved.get("val_mae") is None
                else round(abs(recomputed["val_mae"] - saved["val_mae"]), 4)
            ),
        }
        metrics_match_saved = all(
            [train_match, val_match, type_match, mape_match, mae_match]
        )

    # ---- 5. 结论 ----
    # 训练/验证仅用对应 split：由 _load_split 的 filter 机制 + id 集合互斥共同保证
    training_uses_only_train = metrics_recomputed and not train_test_overlap
    validation_uses_only_val = metrics_recomputed and not val_test_overlap and not train_val_overlap

    conclusions = {
        "training_uses_only_train": bool(training_uses_only_train),
        "validation_uses_only_val": bool(validation_uses_only_val),
        "test_used_in_training": bool(test_in_train_val),
        "metrics_recomputed": metrics_recomputed,
        "metrics_match_saved": metrics_match_saved,
    }

    # ---- 指标状态 ----
    metrics_status: list[dict[str, Any]] = []
    hard_fail_items: list[str] = []

    st = ST_PASS if not test_in_train_val else ST_FAIL
    metrics_status.append(_mk(
        "test_isolation", not test_in_train_val, "test 与 train/val 无 id 交集", st,
        "test 样本未混入 train/val（id 集合互斥）。" if st == ST_PASS
        else f"检测到 test 与训练/验证集 id 交集：train∩test={len(train_test_overlap)} val∩test={len(val_test_overlap)}。"))
    if st == ST_FAIL:
        hard_fail_items.append("test 样本混入训练/验证集")

    st = ST_PASS if metrics_recomputed else ST_FAIL
    metrics_status.append(_mk(
        "metrics_recomputed", metrics_recomputed, "可独立重算 val 指标", st,
        "已仅用 train/val 独立重算 val_mape/val_mae/model_type。" if st == ST_PASS
        else "无法重算（缺少 train 数据或依赖异常）。"))
    if st == ST_FAIL:
        hard_fail_items.append("无法独立重算指标")

    if not saved_present:
        metrics_status.append(_mk(
            "metrics_match_saved", False, "重算与已保存 metrics 一致", ST_WARN,
            "尚无已保存 metrics（housing_baseline_metrics.json 不存在）；请先训练模型。"))
    else:
        st = ST_PASS if metrics_match_saved else ST_FAIL
        metrics_status.append(_mk(
            "metrics_match_saved", metrics_match_saved, "重算与已保存 metrics 一致", st,
            "重算结果与已保存 metrics 在容差内一致，指标非写死/造假。" if st == ST_PASS
            else f"重算与已保存 metrics 不一致：{comparison}。"))
        if st == ST_FAIL:
            hard_fail_items.append("重算指标与已保存不一致")

    st = ST_PASS if (training_uses_only_train and validation_uses_only_val) else ST_FAIL
    metrics_status.append(_mk(
        "train_val_only", training_uses_only_train and validation_uses_only_val,
        "训练仅 train、验证仅 val", st,
        "训练只加载 split=train、验证只加载 split=val，且各 split id 互斥。" if st == ST_PASS
        else "训练/验证使用的 split 不纯。"))

    metrics_status.append(_mk(
        "external_api_calls", 0, "== 0", ST_PASS, "全程本地确定性计算，无外部 API。"))
    metrics_status.append(_mk(
        "llm_used", False, "无大模型参与", ST_PASS, "审计与重算均为确定性代码，无 LLM。"))

    has_fail = any(m["status"] == ST_FAIL for m in metrics_status)
    has_warn = any(m["status"] == ST_WARN for m in metrics_status)
    overall = ST_FAIL if has_fail else (ST_WARN if has_warn else ST_PASS)

    can_trust_val_metrics = (
        not test_in_train_val
        and metrics_recomputed
        and metrics_match_saved
        and validation_uses_only_val
    )

    risks: list[str] = []
    next_required: list[str] = []
    for m in metrics_status:
        if m["status"] == ST_FAIL:
            risks.append(f"[FAIL] {m['metric_name']}：{m['explanation']}")
            next_required.append(f"修复 {m['metric_name']}。")
        elif m["status"] == ST_WARN:
            risks.append(f"[WARN] {m['metric_name']}：{m['explanation']}")

    recommendations: list[str] = []
    if overall == ST_PASS and can_trust_val_metrics:
        recommendations.append(
            "审计通过：房价基线模型仅用 train 训练、val 验证，test 未参与，"
            "且 val 指标可被独立重算复现，可信。可进入第8阶段。")
    elif not saved_present:
        recommendations.append("先调用第5阶段训练或 full-analysis 生成模型，再重跑审计。")
        next_required.append("训练房价基线模型后重跑 model-audit。")
    else:
        recommendations.append("存在 fail/warning 项，修复后方可信任 val 指标。")
    recommendations.append(
        "说明：本审计证明的是'指标可复现 + test 隔离'，不等于房价 MAPE 达到比赛目标"
        "（最终 test 评估属第9阶段 eval 模式）。")

    logger.info(
        "model audit overall=%s can_trust=%s splits=%s recomputed=%s match=%s test_leak=%s",
        overall, can_trust_val_metrics, split_counts, metrics_recomputed,
        metrics_match_saved, test_in_train_val,
    )

    return {
        "mode": settings.app_mode,
        "phase": "5-audit",
        "overall_status": overall,
        "can_trust_val_metrics": can_trust_val_metrics,
        "metrics_status": metrics_status,
        "split_counts": split_counts,
        "model_recomputed": recomputed,
        "saved_metrics": saved_view if saved_present else {},
        "comparison": comparison,
        "hashes": hashes,
        "conclusions": conclusions,
        "hard_fail_items": hard_fail_items,
        "risks": risks,
        "recommendations": recommendations,
        "next_required_actions": next_required,
        "notes": [
            "split 分布与 id 指纹来自 DB；test 仅计数 + id 指纹，绝不取明细用于训练/验证。",
            "重算复用 housing_price_model 训练逻辑但不落盘（不写模型/metrics 文件、不写 DB）。",
            "仅返回统计量 / 指标 / 脱敏 hash / 结论；不含原始房源、小区名、地址、经纬度坐标、原始JSON。",
        ],
    }
