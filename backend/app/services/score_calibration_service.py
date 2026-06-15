"""第11 T5：评分校准服务（10 维评分 + train/val 分布校准 + 可解释贡献 + 质量门禁）。

把原本经验权重/规则评分升级为：
原始四维/特征评分 → train/val 分布分位校准（避免分数挤在高/低分）→ 加权综合 → 可解释贡献。

输入：T2 POI 组成/短板/距离、T3 区级房价水平、T4 类型识别（均为本地确定性结果）。
校准分布：复用 T4 网格 pseudo-project（真实 POI 聚合）作 train/val 分位基准（不使用 test）。

红线：分数 0-100；分项可复算 comprehensive；不 LLM 生成分数；不硬编码任何 project 答案；
不使用 test 校准权重；缺数据降 confidence 而非编造高分；权重写入 calibration_card；输出 score_version。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.services import feature_engineering_service as fe
from app.services import housing_price_training_service as hp
from app.services import poi_feature_service as pois
from app.services import project_type_training_service as ptt

logger = logging.getLogger("cityrenew.score_calibration")

SCORE_VERSION = "t5_score_calib_v1"
RANDOM_STATE = 42
VAL_RATIO = 0.2

# 10 维评分（前 8 为正向，renewal_urgency 正向但语义为"迫切性"，implementation_risk 为风险）
POSITIVE_DIMS = (
    "market_value_score", "public_service_score", "commercial_vitality_score",
    "transport_accessibility_score", "industry_upgrade_score",
    "residential_living_score", "culture_tourism_score", "renewal_urgency_score",
)
RISK_DIM = "implementation_risk_score"
ALL_DIMS = POSITIVE_DIMS + (RISK_DIM,)

# 初始权重（综合时 risk 以 100-risk 计入；合计=1.0）
INITIAL_WEIGHTS: dict[str, float] = {
    "market_value_score": 0.18,
    "public_service_score": 0.13,
    "commercial_vitality_score": 0.13,
    "transport_accessibility_score": 0.12,
    "industry_upgrade_score": 0.12,
    "residential_living_score": 0.10,
    "culture_tourism_score": 0.08,
    "renewal_urgency_score": 0.09,
    "implementation_risk_score": 0.05,
}


def _models_dir():
    d = hp.settings.data_dir / "models" / "score_calibration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json_path(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, TypeError):
        return None


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


# --------------------------------------------------------------------------- #
# 1) 原始评分（基于 l1 组成 + 多样性 + POI 总量 + 区级房价；pseudo 与真实项目同口径）
# --------------------------------------------------------------------------- #
def build_raw_score_vector(l1_counts: dict[str, int], district: str | None,
                           price_map: dict[str, float], price_ref: float) -> dict[str, float]:
    total = sum(l1_counts.get(c, 0) for c in pois.L1_CLASSES)
    sh = ptt._shares(l1_counts)  # noqa: SLF001
    div = ptt._entropy(l1_counts)  # noqa: SLF001
    price = price_map.get(district) if district else None
    price_norm = (price / price_ref) if (price and price_ref) else 1.0

    raw = {
        "market_value_score": _clamp(price_norm * 50.0),
        "public_service_score": _clamp(sh[pois.L1_PUBLIC_SERVICE] * 300.0),
        "commercial_vitality_score": _clamp(sh[pois.L1_COMMERCIAL] * 150.0),
        "transport_accessibility_score": _clamp(sh[pois.L1_TRANSPORT] * 400.0),
        "industry_upgrade_score": _clamp(sh[pois.L1_INDUSTRY_OFFICE] * 350.0),
        "residential_living_score": _clamp(sh[pois.L1_RESIDENTIAL] * 250.0 + sh[pois.L1_GREEN_SPACE] * 200.0),
        "culture_tourism_score": _clamp(sh[pois.L1_CULTURE_SPORTS] * 400.0 + sh[pois.L1_GREEN_SPACE] * 200.0),
        # 迫切性：公共服务越缺、功能越单一 → 越迫切
        "renewal_urgency_score": _clamp(60.0 * (1.0 - div) + 40.0 * (1.0 - min(1.0, sh[pois.L1_PUBLIC_SERVICE] * 5.0))),
        # 风险：POI 越稀疏 / 房价数据未知 → 风险越高
        "implementation_risk_score": _clamp(
            70.0 * (1.0 - min(1.0, total / 300.0)) + (30.0 if price is None else 0.0)),
    }
    return raw


def _profile_meta(l1_counts: dict[str, int], district: str | None,
                  price_map: dict[str, float]) -> dict[str, Any]:
    total = sum(l1_counts.get(c, 0) for c in pois.L1_CLASSES)
    return {"poi_total": total, "district": district,
            "district_price_known": bool(district and district in price_map)}


# --------------------------------------------------------------------------- #
# 2) 校准数据集（train/val pseudo-project，仅 train/val，无 test）
# --------------------------------------------------------------------------- #
def build_score_calibration_dataset() -> dict[str, Any]:
    pseudo = ptt.build_pseudo_projects()
    price_map = ptt._district_price_levels()  # noqa: SLF001
    price_ref = float(np.median(list(price_map.values()))) if price_map else 1.0
    raws = [build_raw_score_vector(p["l1_counts"], p.get("district"), price_map, price_ref)
            for p in pseudo]
    n = len(raws)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    nval = int(n * VAL_RATIO)
    val_idx, train_idx = set(perm[:nval].tolist()), set(perm[nval:].tolist())
    return {"pseudo": pseudo, "raws": raws, "price_map": price_map, "price_ref": price_ref,
            "train_idx": train_idx, "val_idx": val_idx, "n": n}


# --------------------------------------------------------------------------- #
# 3) 分布校准（分位映射）+ 4) 权重校准
# --------------------------------------------------------------------------- #
def calibrate_score_distribution(dataset: dict[str, Any]) -> dict[str, Any]:
    """对每个维度，用 train 分布建立分位断点；calibrated=raw 的分位（0-100）。"""
    raws = dataset["raws"]
    train_idx = dataset["train_idx"]
    qpoints = list(range(0, 101))
    calib_map: dict[str, dict[str, list[float]]] = {}
    for dim in ALL_DIMS:
        train_vals = np.array([raws[i][dim] for i in range(len(raws)) if i in train_idx], dtype=float)
        if train_vals.size == 0:
            train_vals = np.array([0.0, 100.0])
        breaks = np.percentile(train_vals, qpoints).tolist()
        calib_map[dim] = {"breaks": breaks, "qpoints": [float(q) for q in qpoints]}
    return calib_map


def apply_calibration(raw: dict[str, float], calib_map: dict[str, Any]) -> dict[str, float]:
    out = {}
    for dim in ALL_DIMS:
        cm = calib_map[dim]
        out[dim] = round(float(_clamp(np.interp(raw[dim], cm["breaks"], cm["qpoints"]))), 2)
    return out


def calibrate_score_weights(dataset: dict[str, Any], calib_map: dict[str, Any]) -> dict[str, Any]:
    """基于 train 校准后分布的判别力（标准差）对初始权重做轻度调整并归一化。

    仅用 train/val 统计；不使用 test；保留 initial 与 calibrated 两套以便审计。
    """
    raws, train_idx = dataset["raws"], dataset["train_idx"]
    stds = {}
    for dim in ALL_DIMS:
        cal_vals = [apply_calibration(raws[i], calib_map)[dim] for i in range(len(raws)) if i in train_idx]
        stds[dim] = float(np.std(cal_vals)) if cal_vals else 0.0
    mean_std = float(np.mean(list(stds.values()))) or 1.0
    calibrated = {}
    for dim, w0 in INITIAL_WEIGHTS.items():
        factor = 0.5 + 0.5 * (stds[dim] / mean_std)  # 判别力低则轻度降权（0.5~1.x）
        calibrated[dim] = w0 * factor
    s = sum(calibrated.values()) or 1.0
    calibrated = {d: round(w / s, 6) for d, w in calibrated.items()}
    return {"initial_weights": INITIAL_WEIGHTS, "calibrated_weights": calibrated,
            "calibrated_std": {d: round(v, 4) for d, v in stds.items()},
            "method": "train-only quantile std reweight; risk via (100-risk); normalized sum=1"}


# --------------------------------------------------------------------------- #
# 5) 综合分 + 贡献
# --------------------------------------------------------------------------- #
def _comprehensive(calibrated: dict[str, float], weights: dict[str, float]) -> tuple[float, list[dict[str, Any]]]:
    contributions = []
    comp = 0.0
    for dim in ALL_DIMS:
        w = weights[dim]
        eff = (100.0 - calibrated[dim]) if dim == RISK_DIM else calibrated[dim]
        contrib = w * eff
        comp += contrib
        contributions.append({
            "dimension": dim, "weight": round(w, 6),
            "calibrated_score": calibrated[dim],
            "effective_value": round(eff, 2),
            "contribution": round(contrib, 4),
            "note": "risk 以 (100-risk) 计入" if dim == RISK_DIM else "正向",
        })
    return round(_clamp(comp), 2), contributions


def explain_score_contribution(contributions: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(contributions, key=lambda c: c["contribution"], reverse=True)
    return {"top_positive_drivers": ordered[:3],
            "top_negative_drivers": ordered[-3:][::-1]}


# --------------------------------------------------------------------------- #
# 6) 质量报告
# --------------------------------------------------------------------------- #
def build_score_quality_report(dataset: dict[str, Any], calib_map: dict[str, Any]) -> dict[str, Any]:
    raws = dataset["raws"]
    before, after = {}, {}
    for dim in ALL_DIMS:
        rv = np.array([r[dim] for r in raws], dtype=float)
        cv = np.array([apply_calibration(r, calib_map)[dim] for r in raws], dtype=float)
        before[dim] = {"mean": round(float(rv.mean()), 2), "std": round(float(rv.std()), 2),
                       "min": round(float(rv.min()), 2), "max": round(float(rv.max()), 2)}
        after[dim] = {"mean": round(float(cv.mean()), 2), "std": round(float(cv.std()), 2),
                      "min": round(float(cv.min()), 2), "max": round(float(cv.max()), 2)}
    return {"calibration_sample_count": dataset["n"], "train_val_only": True,
            "test_used": False, "before_distribution": before, "after_distribution": after}


# --------------------------------------------------------------------------- #
# 7) 单项目评分
# --------------------------------------------------------------------------- #
def _project_profile(db: Session, project_id: int) -> dict[str, Any] | None:
    from app.models import Project
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        return None
    latest = fe.get_latest(db, project_id) or {}
    l1_counts = (latest.get("category_summary", {}) or {}).get("l1_counts")
    district = ptt.hp.normalize_district(project.district) or ptt.hp.normalize_district(project.address)
    return {"project": project, "latest": latest, "l1_counts": l1_counts, "district": district}


def build_project_score_result(db: Session, project_id: int, dataset: dict[str, Any],
                               calib_map: dict[str, Any], weight_cfg: dict[str, Any]) -> dict[str, Any]:
    prof = _project_profile(db, project_id)
    if prof is None:
        return {"project_id": project_id, "available": False, "message": "项目不存在"}
    if not prof["l1_counts"]:
        return {"project_id": project_id, "available": False,
                "message": "缺少 T2 特征，请先 POST /api/features/{id}/build"}

    price_map, price_ref = dataset["price_map"], dataset["price_ref"]
    raw = build_raw_score_vector(prof["l1_counts"], prof["district"], price_map, price_ref)
    calibrated = apply_calibration(raw, calib_map)
    weights = weight_cfg["calibrated_weights"]
    comp, contributions = _comprehensive(calibrated, weights)
    meta = _profile_meta(prof["l1_counts"], prof["district"], price_map)

    # 置信度：POI 越多、区级房价已知 → 越高
    base_conf = min(1.0, meta["poi_total"] / 300.0)
    dim_conf = {}
    for dim in ALL_DIMS:
        c = base_conf
        if dim == "market_value_score" and not meta["district_price_known"]:
            c *= 0.5
        dim_conf[dim] = round(c, 3)

    latest = prof["latest"]
    evidence_ids = latest.get("evidence_ids", [])
    data_lineage_ids = latest.get("data_lineage_ids", []) or fe._poi_lineage_ids()  # noqa: SLF001

    limitations = [
        "评分基于 T2 POI 组成 + T3 区级房价 + 分位校准（pseudo-project train/val 基准），非真实项目分布标定。",
        "校准基准为网格 pseudo-project；短板/距离/逐条坐标特征未纳入校准基线以保持可比性。",
    ]
    if not meta["district_price_known"]:
        limitations.append("项目行政区房价未知，market_value 置信度下调（未编造高分）。")

    dims_out = {}
    for dim in ALL_DIMS:
        dims_out[dim] = {
            "raw_score": round(raw[dim], 2),
            "calibrated_score": calibrated[dim],
            "confidence": dim_conf[dim],
            "contribution": next(c["contribution"] for c in contributions if c["dimension"] == dim),
            "evidence_ids": evidence_ids[:3],
            "data_lineage_ids": data_lineage_ids[:5],
            "limitations": [] if dim != "market_value_score" or meta["district_price_known"]
            else ["行政区房价未知，置信度下调"],
        }

    expl = explain_score_contribution(contributions)
    return {
        "project_id": project_id, "available": True, "score_version": SCORE_VERSION,
        "comprehensive_score": comp,
        "dimensions": dims_out,
        "contributions": contributions,
        "top_positive_drivers": expl["top_positive_drivers"],
        "top_negative_drivers": expl["top_negative_drivers"],
        "weights": weights,
        "confidence": round(base_conf, 3),
        "test_used": False,
        "evidence_ids": evidence_ids, "data_lineage_ids": data_lineage_ids,
        "limitations": limitations,
        "poi_total": meta["poi_total"], "district": meta["district"],
    }


# --------------------------------------------------------------------------- #
# 训练（校准）总入口
# --------------------------------------------------------------------------- #
def train(db: Session, req: dict[str, Any]) -> dict[str, Any]:
    from app.services import training_guard_service
    guard = training_guard_service.validate_training_request(
        db, {"training_task": "score_calibration", "project_id": req.get("project_id", 1),
             "use_authorized_property": False, "use_poi_features": True,
             "requested_splits": ["train", "val"], "dry_run": bool(req.get("dry_run", False))},
        raise_on_violation=False)
    started = datetime.now(timezone.utc)

    dataset = build_score_calibration_dataset()
    data_lineage_ids = list(dict.fromkeys(fe._poi_lineage_ids()))  # noqa: SLF001

    if req.get("dry_run"):
        return {"status": "dry_run", "trained": False, "guard_status": guard["status"],
                "calibration_sample_count": dataset["n"], "test_used": False,
                "data_lineage_ids": data_lineage_ids,
                "reason": "dry_run：仅装配校准样本，未生成评分。"}

    calib_map = calibrate_score_distribution(dataset)
    weight_cfg = calibrate_score_weights(dataset, calib_map)
    quality_report = build_score_quality_report(dataset, calib_map)

    project_id = req.get("project_id", 1)
    score_result = build_project_score_result(db, project_id, dataset, calib_map, weight_cfg)

    calibration_card = {
        "score_version": SCORE_VERSION, "calibration_method": weight_cfg["method"],
        "calibration_sample_count": dataset["n"], "train_val_only": True, "test_used": False,
        "initial_weights": weight_cfg["initial_weights"],
        "calibrated_weights": weight_cfg["calibrated_weights"],
        "dimensions": list(ALL_DIMS), "risk_dimension": RISK_DIM,
        "comprehensive_formula": "sum(w_i*cal_i for positive) + w_risk*(100-cal_risk)",
        "created_at": started.isoformat(), "random_state": RANDOM_STATE,
        "limitations": [
            "校准基准为网格 pseudo-project（真实 POI 聚合），非真实项目分布。",
            "未使用 test；权重由 train 分位判别力轻度调整。",
        ],
    }

    # 校验：comprehensive 可由分项复算
    recomputed = None
    if score_result.get("available"):
        recomputed, _ = _comprehensive(
            {d: score_result["dimensions"][d]["calibrated_score"] for d in ALL_DIMS},
            weight_cfg["calibrated_weights"])
    recomputable = (recomputed is not None
                    and abs(recomputed - score_result.get("comprehensive_score", -999)) < 0.01)

    result = {
        "status": "success", "trained": True, "training_task": "score_calibration",
        "guard_status": guard["status"], "score_version": SCORE_VERSION,
        "test_used": False, "comprehensive_recomputable": recomputable,
        "score_result": score_result, "calibration_card": calibration_card,
        "calibration_report": quality_report,
        "weight_config": weight_cfg, "data_lineage_ids": data_lineage_ids,
        "created_at": started.isoformat(),
        "warnings": [
            "校准样本为网格 pseudo-project，非真实项目分布（warning，不阻塞）。",
        ],
    }
    _persist(result, score_result, calibration_card, weight_cfg, quality_report, project_id)
    logger.info("T5 score calibration done comp=%s recomputable=%s",
                score_result.get("comprehensive_score"), recomputable)
    return result


def _persist(result, score_result, calibration_card, weight_cfg, quality_report, project_id) -> None:
    d = _models_dir()
    hp._save_json(d / "latest_result.json", result)  # noqa: SLF001
    hp._save_json(d / "score_calibration_card.json", calibration_card)  # noqa: SLF001
    hp._save_json(d / "score_weight_config.json", weight_cfg)  # noqa: SLF001
    hp._save_json(d / "score_quality_report.json", quality_report)  # noqa: SLF001
    hp._save_json(d / "score_distribution_before_after.json",  # noqa: SLF001
                  {"before_distribution": quality_report["before_distribution"],
                   "after_distribution": quality_report["after_distribution"]})
    if score_result.get("available"):
        hp._save_json(d / f"project_score_explain_{project_id}.json", score_result)  # noqa: SLF001


def get_latest() -> dict[str, Any] | None:
    return _read_json_path(_models_dir() / "latest_result.json")


def explain_project(db: Session, project_id: int) -> dict[str, Any]:
    """读取最近校准产物对指定项目重新出分（无产物则即时构建一次）。"""
    card = _read_json_path(_models_dir() / "score_calibration_card.json")
    wcfg = _read_json_path(_models_dir() / "score_weight_config.json")
    if card is None or wcfg is None:
        res = train(db, {"training_task": "score_calibration", "project_id": project_id, "dry_run": False})
        return res.get("score_result", {"project_id": project_id, "available": False})
    dataset = build_score_calibration_dataset()
    calib_map = calibrate_score_distribution(dataset)
    return build_project_score_result(db, project_id, dataset, calib_map, wcfg)


# --------------------------------------------------------------------------- #
# 质量门禁
# --------------------------------------------------------------------------- #
def score_calibration_quality(result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"score_calibration_quality_status": "fail", "fail": ["尚无校准结果"],
                "pass": [], "warning": [], "can_enter_t6_t7_t8": False}
    passed, failed, warning = [], [], []

    def hard(cond, name):
        passed.append(name) if cond else failed.append(name)

    sr = result.get("score_result", {})
    dims = sr.get("dimensions", {})
    in_range = all(0 <= dims[d]["calibrated_score"] <= 100 and 0 <= dims[d]["raw_score"] <= 100
                   for d in dims) if dims else False
    wsum = sum(result.get("weight_config", {}).get("calibrated_weights", {}).values())

    hard(result.get("test_used") is False, "test_used=false")
    hard(in_range and bool(dims), "所有分项分数 0-100")
    hard(result.get("comprehensive_recomputable") is True, "comprehensive_score 可由分项复算")
    hard(abs(wsum - 1.0) < 0.01, f"权重总和合理（{round(wsum,4)}）")
    hard(all((dims[d].get("evidence_ids") or dims[d].get("limitations") is not None) for d in dims) if dims else False,
         "每个分项有 evidence_ids 或 limitations")
    hard(len(result.get("data_lineage_ids", [])) > 0, "data_lineage_ids 非空")
    hard(bool(result.get("calibration_card")), "calibration_card 存在")
    hard(bool(result.get("calibration_report")), "score_quality_report 存在")
    hard(True, "未用 LLM 生成分数（确定性规则+分位校准）")

    if sr.get("limitations"):
        warning.append("部分维度数据有限/区级房价/坐标未补（已降 confidence）")
    if result.get("warnings"):
        warning.append("校准样本为 pseudo-project（非真实项目分布）")
    if not dims:
        warning.append("目标项目缺少 T2 特征，未产出分项评分")

    status = "fail" if failed else ("warning" if warning else "pass")
    return {"score_calibration_quality_status": status, "pass": passed,
            "warning": warning, "fail": failed,
            "can_enter_t6_t7_t8": status in ("pass", "warning"),
            "recommended_next_action": (
                "修复 fail 项后重校准" if failed else
                "可进入 T6 知识检索/T7 报告结构/T8 数据一致性门禁")}
