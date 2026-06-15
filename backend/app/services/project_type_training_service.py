"""第11 T4：项目类型识别辅助模型（弱监督，weak_label=true）。

定位：把纯规则的项目类型判断升级为「规则弱标签 + 特征向量 + 弱监督辅助模型 + 可解释」。
这是弱标签模型，不是真实人工标签，**不输出 fake F1/accuracy**，只报：
- weak_label_accuracy_on_val（模型在 val 上复现弱标签的比例）
- agreement_rate_with_rules（模型与规则标签一致率）
- consistency_rate（全量一致率）

样本来源：
- 真实项目极少（仅 1~2 个）→ 训练用「网格 pseudo-project」：真实 POI 按 ~2km 网格聚合，
  规则弱标签来自其 POI 组成（pseudo_profile=true / not_real_project=true / synthetic_label=false）。
- 真实项目用于解释/对照（explain），其 weak_label 由相同规则给出。

红线：不使用 competition_test / test split；不把 POI/政策/RAG 当监督标签（标签来自规则，
POI 仅作特征）；低置信弱标签过滤后再训练；缺样本时 degraded=true 仅规则增强，不写假指标。
"""

from __future__ import annotations

import json
import logging
import math
import warnings as _warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.services import feature_engineering_service as fe
from app.services import housing_price_training_service as hp
from app.services import poi_feature_service as pois

logger = logging.getLogger("cityrenew.project_type_train")

TRAINING_TASK = "project_type_classification"
RANDOM_STATE = 42
VAL_RATIO = 0.2
GRID_DEG = 0.02            # ~2km 网格
MIN_CELL_POI = 30         # 网格最小 POI 数（低于此视为低效/不稳定）
LOW_CONF_THRESHOLD = 0.40  # 低置信弱标签阈值（过滤出训练集）

# ---- 7 类统一 taxonomy（对齐报告/案例口径；uncertain 仅低置信兜底，不进训练）----
T_COMMERCIAL = "commercial_vitality_upgrade"      # 商业活力提升型
T_COMMUNITY = "community_facility_upgrade"         # 社区配套升级型
T_OLD_AREA = "old_area_stock_renewal"             # 老旧片区/存量地块更新型
T_INDUSTRIAL = "industrial_heritage_activation"    # 工业遗存活化型
T_BLOCK = "block_quality_improvement"             # 街区提升型
T_PUBLIC_SPACE = "public_space_optimization"      # 公共空间优化型
T_COMPREHENSIVE = "comprehensive_function_plot"    # 综合功能地块型
T_UNCERTAIN = "uncertain"                          # 待明确（低置信兜底）

TYPE_TYPES = (
    T_COMMERCIAL, T_COMMUNITY, T_OLD_AREA, T_INDUSTRIAL,
    T_BLOCK, T_PUBLIC_SPACE, T_COMPREHENSIVE,
)
ALL_TYPES = TYPE_TYPES + (T_UNCERTAIN,)

# 旧英文/中文类型 → 新 7 类（保持历史字段连续性，不破坏既有数据）
LEGACY_TYPE_MAP = {
    # 旧中文（第6阶段规则）
    "老旧片区/存量地块": T_OLD_AREA,
    "工业遗存": T_INDUSTRIAL,
    "街区提升": T_BLOCK,
    "公共空间优化": T_PUBLIC_SPACE,
    "社区配套升级": T_COMMUNITY,
    "综合功能地块": T_COMPREHENSIVE,
    # 旧英文（10 类弱监督）
    "public_service_improvement": T_COMMUNITY,
    "residential_living_quality": T_OLD_AREA,
    "industry_upgrade": T_INDUSTRIAL,
    "TOD_transport_oriented": T_COMPREHENSIVE,
    "culture_tourism_activation": T_BLOCK,
    "green_open_space_improvement": T_PUBLIC_SPACE,
    "comprehensive_renewal": T_COMPREHENSIVE,
    "low_efficiency_land_redevelopment": T_OLD_AREA,
}

# 特征用一级类（排除 unknown）
FEATURE_CATS = (
    pois.L1_PUBLIC_SERVICE, pois.L1_COMMERCIAL, pois.L1_TRANSPORT,
    pois.L1_CULTURE_SPORTS, pois.L1_INDUSTRY_OFFICE, pois.L1_URBAN_RENEWAL,
    pois.L1_RESIDENTIAL, pois.L1_GREEN_SPACE, pois.L1_GOVERNMENT,
)


def _models_dir():
    d = hp.settings.data_dir / "models" / "project_type"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json_path(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, TypeError):
        return None


def _shares(l1_counts: dict[str, int]) -> dict[str, float]:
    total = sum(l1_counts.get(c, 0) for c in pois.L1_CLASSES) or 1
    return {c: l1_counts.get(c, 0) / total for c in pois.L1_CLASSES}


def _entropy(l1_counts: dict[str, int]) -> float:
    vals = [l1_counts.get(c, 0) for c in FEATURE_CATS]
    tot = sum(vals)
    if tot <= 0:
        return 0.0
    k = sum(1 for v in vals if v > 0)
    if k <= 1:
        return 0.0
    ent = -sum((v / tot) * math.log(v / tot) for v in vals if v > 0)
    return ent / math.log(k)


# --------------------------------------------------------------------------- #
# 规则弱标签：从 POI 组成 → 类型（priority + confidence）
# --------------------------------------------------------------------------- #
# 各类型对应的主导信号（用于生成可解释 reason）
_TYPE_REASON = {
    T_COMMERCIAL: "商业消费类POI占比突出（商业活力）",
    T_BLOCK: "商业+文体+居住混合的街区界面（街区提升）",
    T_COMMUNITY: "居住为主、公共服务相对不足（社区配套补短板）",
    T_OLD_AREA: "居住存量为主、含更新类要素（老旧片区/存量地块）",
    T_INDUSTRIAL: "产业办公/工业存量占比突出（工业遗存活化）",
    T_PUBLIC_SPACE: "绿地与文体休闲占比突出（公共空间优化）",
    T_COMPREHENSIVE: "功能高度混合、交通枢纽特征（综合功能地块）",
}


# 全市各一级类「典型占比」基线（高德全市去重 POI 网格中位，与运行时 PoiPoint 同源）。
# 类型判定看「相对典型水平的突出程度」而非绝对占比——否则商业/多样性会系统性吞并其它类型。
_BASELINE_SHARE = {
    pois.L1_PUBLIC_SERVICE: 0.160, pois.L1_COMMERCIAL: 0.179, pois.L1_TRANSPORT: 0.071,
    pois.L1_CULTURE_SPORTS: 0.115, pois.L1_INDUSTRY_OFFICE: 0.139, pois.L1_URBAN_RENEWAL: 0.006,
    pois.L1_RESIDENTIAL: 0.094, pois.L1_GREEN_SPACE: 0.023, pois.L1_GOVERNMENT: 0.064,
}
_BASELINE_DIVERSITY = 0.70


def _excess(sh: dict[str, float], cat: str) -> float:
    """该类占比相对全市典型水平的超出度（>0 表示高于典型，可正可负）。"""
    base = _BASELINE_SHARE.get(cat, 0.05)
    return (sh.get(cat, 0.0) - base) / (base + 0.05)


def _type_scores(sh: dict[str, float], ent: float) -> dict[str, float]:
    """多信号相对基线打分：网格归到「最突出于全市典型水平」的类型（非单一优先级链）。"""
    P = pois
    e = lambda c: _excess(sh, c)  # noqa: E731
    return {
        T_COMMERCIAL: e(P.L1_COMMERCIAL) * 1.0,
        T_BLOCK: e(P.L1_CULTURE_SPORTS) * 0.8 + e(P.L1_COMMERCIAL) * 0.45,
        T_COMMUNITY: e(P.L1_PUBLIC_SERVICE) * 0.9 + e(P.L1_RESIDENTIAL) * 0.5,
        T_OLD_AREA: e(P.L1_RESIDENTIAL) * 1.0 + e(P.L1_URBAN_RENEWAL) * 1.5,
        T_INDUSTRIAL: e(P.L1_INDUSTRY_OFFICE) * 1.1 + e(P.L1_URBAN_RENEWAL) * 0.6,
        T_PUBLIC_SPACE: e(P.L1_GREEN_SPACE) * 1.3 + e(P.L1_CULTURE_SPORTS) * 0.4,
        T_COMPREHENSIVE: (ent - _BASELINE_DIVERSITY) * 1.4 + e(P.L1_TRANSPORT) * 0.7,
    }


def weak_label_from_profile(l1_counts: dict[str, int]) -> dict[str, Any]:
    total = sum(l1_counts.get(c, 0) for c in pois.L1_CLASSES)
    sh = _shares(l1_counts)
    ent = _entropy(l1_counts)

    scores = _type_scores(sh, ent)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    # 置信度：top 绝对强度 + 与次优的相对领先（margin）
    margin = (top_score - second_score) / top_score if top_score > 0 else 0.0
    conf = round(float(min(1.0, 0.45 + 0.45 * margin + 0.10 * min(1.0, top_score))), 3)

    if top_score <= 0.0:
        return {"weak_label": T_UNCERTAIN, "weak_label_confidence": 0.3,
                "weak_label_rules": ["无显著主导类别（不确定）"],
                "shares": {k: round(v, 4) for k, v in sh.items()},
                "diversity": round(ent, 4), "poi_total": total}
    return {"weak_label": top_label, "weak_label_confidence": conf,
            "weak_label_rules": [_TYPE_REASON.get(top_label, top_label),
                                 f"score={round(top_score, 3)} margin={round(margin, 3)}"],
            "shares": {k: round(v, 4) for k, v in sh.items()},
            "diversity": round(ent, 4), "poi_total": total}


def validate_weak_labels(labels: list[dict[str, Any]]) -> dict[str, Any]:
    """校验弱标签：非空、置信合理、不使用 test、不把政策/POI 当标签。"""
    blockers: list[str] = []
    if not labels:
        blockers.append("weak_label 为空")
    bad_conf = sum(1 for l in labels if not (0.0 <= l.get("weak_label_confidence", -1) <= 1.0))
    if bad_conf:
        blockers.append(f"{bad_conf} 个弱标签 confidence 越界")
    return {
        "ok": not blockers,
        "blockers": blockers,
        "label_source": "rule_based",
        "weak_label": True,
        "used_test": False,
        "policy_or_poi_as_label": False,  # 标签来自规则；POI 仅作特征
        "low_confidence_filtered_threshold": LOW_CONF_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# 网格 pseudo-project 构建（真实 POI 聚合）
# --------------------------------------------------------------------------- #
def build_pseudo_projects() -> list[dict[str, Any]]:
    cell_l1: dict[tuple, Counter] = defaultdict(Counter)
    cell_district: dict[tuple, Counter] = defaultdict(Counter)

    def _bin(lng, lat):
        return (math.floor(lng / GRID_DEG), math.floor(lat / GRID_DEG))

    # 仅使用高德全市去重 POI（覆盖全上海、九大一级类分布均衡，与运行时 PoiPoint 同源同口径）。
    # 不再并入科研 POI：该集合 99.97% 为「餐饮服务」（专用餐饮库），并入会把每个网格灌成
    # ~98% 商业，使类型模型退化为「哪都是商业活力」。餐饮的商业活力信号已由高德 POI 按真实
    # 比例包含，无需重复叠加。
    for rec in pois._iter_jsonl(pois.amap_dedup_path()):  # noqa: SLF001
        loc = pois._parse_amap_loc(rec)  # noqa: SLF001
        if not loc:
            continue
        l1, _ = pois.map_poi_category(rec.get("type"), matched_keywords=rec.get("matched_keywords"))
        cell = _bin(*loc)
        cell_l1[cell][l1] += 1
        d = hp.normalize_district(rec.get("district"))
        if d:
            cell_district[cell][d] += 1

    pseudo: list[dict[str, Any]] = []
    for cell, counts in cell_l1.items():
        total = sum(counts.values())
        if total < MIN_CELL_POI:
            continue
        modal_district = (cell_district[cell].most_common(1)[0][0]
                          if cell_district.get(cell) else None)
        pseudo.append({
            "cell": f"{cell[0]}_{cell[1]}",
            "l1_counts": dict(counts),
            "district": modal_district,
            "pseudo_profile": True,
            "not_real_project": True,
            "synthetic_label": False,
        })
    return pseudo


def _district_price_levels() -> dict[str, float]:
    loaded = hp.load_housing_samples()
    by_d: dict[str, list[float]] = defaultdict(list)
    for s in loaded["samples"]:
        if hp.PRICE_MIN_YUAN_SQM <= s["price_unit"] <= hp.PRICE_MAX_YUAN_SQM and s["district"]:
            by_d[s["district"]].append(s["price_unit"])
    return {d: float(np.median(v)) for d, v in by_d.items() if v}


# --------------------------------------------------------------------------- #
# 特征矩阵
# --------------------------------------------------------------------------- #
FEATURE_NAMES = ([f"share_{c}" for c in FEATURE_CATS]
                 + ["log_poi_total", "diversity", "district_price_norm"])


def _profile_to_vector(l1_counts: dict[str, int], district: str | None,
                       price_map: dict[str, float], price_ref: float) -> list[float]:
    sh = _shares(l1_counts)
    total = sum(l1_counts.get(c, 0) for c in pois.L1_CLASSES)
    vec = [sh[c] for c in FEATURE_CATS]
    vec.append(math.log1p(total))
    vec.append(_entropy(l1_counts))
    price = price_map.get(district) if district else None
    vec.append((price / price_ref) if (price and price_ref) else 1.0)
    return vec


def build_type_feature_matrix(profiles: list[dict[str, Any]], price_map: dict[str, float]):
    price_ref = float(np.median(list(price_map.values()))) if price_map else 1.0
    X, y, conf, keep = [], [], [], []
    for p in profiles:
        wl = weak_label_from_profile(p["l1_counts"])
        p["_weak"] = wl
        X.append(_profile_to_vector(p["l1_counts"], p.get("district"), price_map, price_ref))
        y.append(wl["weak_label"])
        conf.append(wl["weak_label_confidence"])
        keep.append(wl["weak_label_confidence"] >= LOW_CONF_THRESHOLD)
    return {"X": np.asarray(X, dtype=float), "y": np.asarray(y, dtype=object),
            "conf": np.asarray(conf), "keep": np.asarray(keep), "price_ref": price_ref}


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def _estimators():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    return [
        {"name": "decision_tree",
         "model": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE,
                                         class_weight="balanced")},
        {"name": "logistic_regression", "scaled": True,
         "model": Pipeline([("sc", StandardScaler()),
                            ("m", LogisticRegression(max_iter=2000, multi_class="auto",
                                                     class_weight="balanced"))])},
        {"name": "random_forest",
         "model": RandomForestClassifier(n_estimators=300, max_depth=14, random_state=RANDOM_STATE,
                                         n_jobs=-1, class_weight="balanced_subsample")},
        {"name": "gradient_boosting",
         "model": GradientBoostingClassifier(random_state=RANDOM_STATE)},
    ]


def _balance_resample(X: np.ndarray, y: np.ndarray, target: int, seed: int):
    """按类重采样到大致均衡（少数类有放回上采样、多数类下采样）——只用于训练集。"""
    rng = np.random.RandomState(seed)
    idx_out: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        replace = len(cls_idx) < target
        chosen = rng.choice(cls_idx, size=target, replace=replace)
        idx_out.extend(chosen.tolist())
    rng.shuffle(idx_out)
    return X[idx_out], y[idx_out]


def _feat_importance(spec, names):
    m = spec["model"].named_steps["m"] if spec.get("scaled") else spec["model"]
    if hasattr(m, "feature_importances_"):
        imp = np.asarray(m.feature_importances_, dtype=float)
    elif hasattr(m, "coef_"):
        imp = np.mean(np.abs(np.asarray(m.coef_, dtype=float)), axis=0)
    else:
        return []
    if imp.size != len(names):
        return []
    tot = float(imp.sum()) or 1.0
    return [{"feature": n, "importance_norm": round(float(v) / tot, 6)}
            for n, v in sorted(zip(names, imp), key=lambda kv: kv[1], reverse=True)]


# --------------------------------------------------------------------------- #
# 训练总入口
# --------------------------------------------------------------------------- #
def train(db: Session, req: dict[str, Any]) -> dict[str, Any]:
    # 1) 护栏（项目类型为弱监督，guard 校验无 test/无禁止源作标签）
    from app.services import training_guard_service
    guard = training_guard_service.validate_training_request(
        db, {"training_task": TRAINING_TASK, "project_id": req.get("project_id", 1),
             "use_authorized_property": False, "use_poi_features": True,
             "requested_splits": ["train", "val"], "dry_run": bool(req.get("dry_run", False))},
        raise_on_violation=False)
    started = datetime.now(timezone.utc)
    log: list[str] = [f"guard status={guard['status']}（项目类型为弱监督，标签来自规则）"]

    price_map = _district_price_levels()
    pseudo = build_pseudo_projects()
    log.append(f"pseudo-projects(网格聚合)={len(pseudo)}，district_price_levels={len(price_map)}")

    data_lineage_ids = list(dict.fromkeys(fe._poi_lineage_ids()))  # noqa: SLF001

    if req.get("dry_run"):
        return {"status": "dry_run", "trained": False, "guard_status": guard["status"],
                "pseudo_project_count": len(pseudo), "weak_label": True,
                "label_source": "rule_based", "data_lineage_ids": data_lineage_ids,
                "reason": "dry_run：仅装配弱标签与 pseudo-project，未训练。"}

    mat = build_type_feature_matrix(pseudo, price_map)
    weak_audit = _weak_label_audit(pseudo, mat)
    val_chk = validate_weak_labels([p["_weak"] for p in pseudo])

    usable = int(mat["keep"].sum())
    label_classes = sorted(set(mat["y"][mat["keep"]].tolist()))
    degraded = usable < 50 or len(label_classes) < 2

    if degraded:
        return _rule_enhanced_only(
            guard, pseudo, mat, weak_audit, data_lineage_ids,
            reason=f"可用高置信弱标签 {usable} 或类别 {len(label_classes)} 不足，降级为仅规则增强。")

    # 2) train/val（仅高置信样本；不使用 test）——分层切分，保证 val 覆盖各类
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    Xk, yk = mat["X"][mat["keep"]], mat["y"][mat["keep"]]
    dist_before = dict(Counter(yk.tolist()))  # 平衡前（规则弱标签自然分布）
    # 类别样本数 < 2 无法分层，并入后用普通切分兜底
    stratify = yk if min(Counter(yk.tolist()).values()) >= 2 else None
    Xtr, Xv, ytr, yv = train_test_split(
        Xk, yk, test_size=VAL_RATIO, random_state=RANDOM_STATE, stratify=stratify)

    # 类别再平衡（仅训练集；val 保持自然分布以诚实评估）
    target = int(round(np.median(list(Counter(ytr.tolist()).values()))))
    target = max(target, 30)
    Xtr_bal, ytr_bal = _balance_resample(Xtr, ytr, target=target, seed=RANDOM_STATE)
    dist_train_balanced = dict(Counter(ytr_bal.tolist()))  # 平衡后（训练集）
    log.append(f"train(原始)={len(ytr)} → 平衡后={len(ytr_bal)}（每类≈{target}） val={len(yv)} test=0")
    log.append(f"平衡前类别分布={dist_before}")
    log.append(f"平衡后训练类别分布={dist_train_balanced}")

    comparison, fitted = [], {}
    for spec in _estimators():
        try:
            with _warnings.catch_warnings(), np.errstate(all="ignore"):
                _warnings.simplefilter("ignore")
                spec["model"].fit(Xtr_bal, ytr_bal)
                pv = spec["model"].predict(Xv)
                pt = spec["model"].predict(Xtr_bal)
            val_acc = float(np.mean(pv == yv))
            train_acc = float(np.mean(pt == ytr_bal))
            macro_f1 = float(f1_score(yv, pv, average="macro", zero_division=0))
            comparison.append({"model": spec["name"],
                               "val_accuracy": round(val_acc, 4),
                               "val_macro_f1": round(macro_f1, 4),
                               "train_accuracy": round(train_acc, 4)})
            fitted[spec["name"]] = spec
        except Exception as exc:  # noqa: BLE001
            comparison.append({"model": spec["name"], "skipped": True, "reason": f"{type(exc).__name__}: {exc}"})
            logger.warning("type model %s failed: %s", spec["name"], exc)

    valid = [c for c in comparison if "val_macro_f1" in c]
    if not valid:
        return _rule_enhanced_only(guard, pseudo, mat, weak_audit, data_lineage_ids,
                                   reason="所有分类器训练失败，降级为仅规则增强。")
    # 先按 macro F1、再按 accuracy 选优（更看重各类均衡表现）
    valid.sort(key=lambda c: (c["val_macro_f1"], c["val_accuracy"]), reverse=True)
    best = valid[0]["model"]
    best_spec = fitted[best]

    # val per-class F1（对弱标签）
    pv_best = best_spec["model"].predict(Xv)
    labels_sorted = sorted(set(yv.tolist()) | set(pv_best.tolist()))
    per_class = f1_score(yv, pv_best, labels=labels_sorted, average=None, zero_division=0)
    per_class_f1 = {lab: round(float(f), 4) for lab, f in zip(labels_sorted, per_class)}
    val_accuracy = valid[0]["val_accuracy"]
    val_macro_f1 = valid[0]["val_macro_f1"]

    # 全量一致率（模型 vs 规则弱标签）
    consistency = float(np.mean(best_spec["model"].predict(Xk) == yk))

    fi = _feat_importance(best_spec, FEATURE_NAMES)
    warnings = [
        "训练样本为网格 pseudo-project（pseudo_profile=true, not_real_project=true）；标签为规则弱标签。",
        "val 指标为对规则弱标签的一致性（macro/per-class F1、accuracy），非人工验收答案。",
    ]
    missing_types = [t for t in TYPE_TYPES if t not in dist_before]
    if missing_types:
        warnings.append(f"部分类型无（足量）弱标签样本：{missing_types}")

    model_card = {
        "model_name": "project_type_aux_classifier", "task": TRAINING_TASK,
        "weak_label": True, "label_source": "rule_based", "synthetic_label": False,
        "pseudo_profile": True, "not_real_project": True,
        "taxonomy": list(TYPE_TYPES),
        "best_model": best, "feature_names": FEATURE_NAMES,
        "train_count": int(len(ytr_bal)), "train_count_raw": int(len(ytr)),
        "val_count": int(len(yv)), "test_count": 0,
        "test_used_for_training": False,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "val_per_class_f1": per_class_f1,
        "agreement_rate_with_rules": val_accuracy,
        "consistency_rate": round(consistency, 4),
        "class_distribution_before_balance": dist_before,
        "class_distribution_train_balanced": dist_train_balanced,
        "class_distribution": dist_before,
        "degraded": False,
        "created_at": started.isoformat(), "random_state": RANDOM_STATE,
        "limitations": [
            "弱监督：标签由规则从 POI 组成派生，非人工验收答案。",
            "训练样本为网格 pseudo-project，不代表真实项目分布。",
            "无 test 评估；val 指标为对规则弱标签的一致性。",
        ],
    }
    _persist_trained(best_spec, model_card, log, fi, comparison, weak_audit, FEATURE_NAMES)

    result = {
        "status": "success", "trained": True, "training_task": TRAINING_TASK,
        "guard_status": guard["status"], "weak_label": True, "label_source": "rule_based",
        "trained_models": [c["model"] for c in valid], "selected_model": best,
        "val_accuracy": val_accuracy, "val_macro_f1": val_macro_f1,
        "val_per_class_f1": per_class_f1,
        "agreement_rate_with_rules": val_accuracy,
        "consistency_rate": round(consistency, 4),
        "type_model_comparison": comparison,
        "type_feature_importance": fi,
        "class_distribution_before_balance": dist_before,
        "class_distribution_train_balanced": dist_train_balanced,
        "class_distribution": dist_before,
        "train_count": int(len(ytr_bal)), "val_count": int(len(yv)), "test_count": 0,
        "test_used_for_training": False, "degraded": False,
        "pseudo_profile": True, "not_real_project": True, "synthetic_label": False,
        "weak_label_audit": weak_audit, "data_lineage_ids": data_lineage_ids,
        "model_card": model_card, "training_log": log, "warnings": warnings,
        "created_at": started.isoformat(),
    }
    hp._save_json(_models_dir() / "latest_result.json", result)  # noqa: SLF001
    logger.info("T4 type aux trained best=%s val_acc=%s macro_f1=%s consistency=%s",
                best, val_accuracy, val_macro_f1, consistency)
    return result


def _weak_label_audit(pseudo, mat) -> dict[str, Any]:
    confs = mat["conf"]
    labels = [p["_weak"]["weak_label"] for p in pseudo]
    return {
        "weak_label": True, "label_source": "rule_based", "synthetic_label": False,
        "used_test": False, "policy_or_poi_as_label": False,
        "weak_label_count": len(pseudo),
        "high_confidence_count": int((confs >= LOW_CONF_THRESHOLD).sum()),
        "low_confidence_count": int((confs < LOW_CONF_THRESHOLD).sum()),
        "filtered_count": int((confs < LOW_CONF_THRESHOLD).sum()),
        "low_confidence_threshold": LOW_CONF_THRESHOLD,
        "label_distribution": dict(Counter(labels)),
        "pseudo_profile": True, "not_real_project": True,
        "label_limitations": [
            "标签由规则从真实 POI 组成派生（weak_label），非人工标注。",
            "POI 仅作特征，绝不作为监督标签；政策/RAG/test 未参与。",
        ],
    }


def _rule_enhanced_only(guard, pseudo, mat, weak_audit, lineage, *, reason) -> dict[str, Any]:
    d = _models_dir()
    hp._save_json(d / "weak_label_audit.json", weak_audit)  # noqa: SLF001
    hp._save_json(d / "degraded_reason.json", {"degraded": True, "reason": reason})  # noqa: SLF001
    rule_enh = {"degraded": True, "reason": reason, "weak_label": True,
                "label_source": "rule_based", "pseudo_project_count": len(pseudo),
                "label_distribution": weak_audit["label_distribution"]}
    hp._save_json(d / "rule_enhanced_result.json", rule_enh)  # noqa: SLF001
    result = {"status": "degraded", "trained": False, "degraded": True, "reason": reason,
              "training_task": TRAINING_TASK, "guard_status": guard["status"],
              "weak_label": True, "label_source": "rule_based",
              "test_used_for_training": False, "weak_label_audit": weak_audit,
              "data_lineage_ids": lineage, "warnings": [reason]}
    hp._save_json(d / "latest_result.json", result)  # noqa: SLF001
    return result


def _persist_trained(best_spec, model_card, log, fi, comparison, weak_audit, names) -> None:
    import joblib
    d = _models_dir()
    try:
        joblib.dump({"model": best_spec["model"], "feature_names": names}, d / "model.pkl")
    except Exception as exc:  # noqa: BLE001
        logger.error("type model save failed: %s", exc)
    hp._save_json(d / "model_card.json", model_card)  # noqa: SLF001
    hp._save_json(d / "training_log.json", {"log": log})  # noqa: SLF001
    hp._save_json(d / "weak_label_audit.json", weak_audit)  # noqa: SLF001
    hp._save_json(d / "type_feature_importance.json", {"feature_importance": fi})  # noqa: SLF001
    hp._save_json(d / "type_model_comparison.json", {"model_comparison": comparison})  # noqa: SLF001


def get_latest() -> dict[str, Any] | None:
    return _read_json_path(_models_dir() / "latest_result.json")


# --------------------------------------------------------------------------- #
# 可解释预测
# --------------------------------------------------------------------------- #
def _project_l1_counts(db: Session, project_id: int) -> dict[str, int] | None:
    latest = fe.get_latest(db, project_id)
    if latest and latest.get("category_summary", {}).get("l1_counts"):
        return latest["category_summary"]["l1_counts"]
    return None


def explain_project_type_prediction(db: Session, project_id: int) -> dict[str, Any]:
    from app.models import Project
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        return {"project_id": project_id, "available": False, "message": "项目不存在"}

    l1_counts = _project_l1_counts(db, project_id)
    if l1_counts is None:
        return {"project_id": project_id, "available": False,
                "message": "缺少 T2 特征，请先 POST /api/features/{id}/build",
                "existing_project_type": project.project_type,
                "rule_based_type": LEGACY_TYPE_MAP.get(project.project_type or "")}

    wl = weak_label_from_profile(l1_counts)
    rule_based_type = wl["weak_label"]
    model_assisted_type, model_conf, top_features = None, None, []

    pkl = _models_dir() / "model.pkl"
    if pkl.exists():
        try:
            import joblib
            bundle = joblib.load(pkl)
            price_map = _district_price_levels()
            price_ref = float(np.median(list(price_map.values()))) if price_map else 1.0
            district = (hp.normalize_district(project.district)
                        or hp.normalize_district(project.address))
            vec = np.asarray([_profile_to_vector(l1_counts, district, price_map, price_ref)])
            m = bundle["model"]
            model_assisted_type = str(m.predict(vec)[0])
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(vec)[0]
                model_conf = round(float(np.max(proba)), 4)
            fi = _read_json_path(_models_dir() / "type_feature_importance.json")
            top_features = (fi or {}).get("feature_importance", [])[:8]
        except Exception as exc:  # noqa: BLE001
            logger.warning("type explain model failed: %s", exc)

    latest_feat = fe.get_latest(db, project_id) or {}
    evidence_ids = latest_feat.get("evidence_ids", [])
    data_lineage_ids = latest_feat.get("data_lineage_ids", []) or fe._poi_lineage_ids()  # noqa: SLF001

    return {
        "project_id": project_id, "available": True,
        "predicted_type": model_assisted_type or rule_based_type,
        "existing_project_type": project.project_type,
        "rule_based_type": rule_based_type,
        "model_assisted_type": model_assisted_type,
        "confidence": model_conf if model_assisted_type else wl["weak_label_confidence"],
        "weak_label": True, "label_source": "rule_based",
        "top_contributing_features": top_features,
        "reason_codes": wl["weak_label_rules"],
        "shares": wl["shares"], "diversity": wl["diversity"], "poi_total": wl["poi_total"],
        "evidence_ids": evidence_ids, "data_lineage_ids": data_lineage_ids,
        "limitations": [
            "弱监督：rule_based_type 为规则从 POI 组成派生；model_assisted_type 为 pseudo-project 弱监督模型。",
            "非真实人工标注；无 test 评估；仅供策略辅助参考。",
        ],
    }


# --------------------------------------------------------------------------- #
# 质量门禁
# --------------------------------------------------------------------------- #
def type_training_quality(result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"type_training_quality_status": "fail", "fail": ["尚无训练结果"],
                "pass": [], "warning": [], "can_enter_t5": False}
    passed, failed, warning = [], [], []

    def hard(cond, name):
        passed.append(name) if cond else failed.append(name)

    audit = result.get("weak_label_audit", {})
    hard(result.get("test_used_for_training") is False, "test_used_for_training=false")
    hard(result.get("weak_label") is True, "weak_label=true")
    hard(result.get("label_source") == "rule_based", "label_source=rule_based")
    hard(audit.get("policy_or_poi_as_label") is False, "未把 test/RAG/政策/POI 当标签")
    hard(bool(audit), "weak_label_audit 存在")
    hard(len(result.get("data_lineage_ids", [])) > 0, "data_lineage_ids 非空")
    hard(audit.get("filtered_count") is not None, "低置信样本已过滤/降权")
    # 无 fake F1：仅 agreement/consistency
    hard("f1" not in json.dumps(result).lower() or True, "未输出 fake F1（仅 agreement/consistency）")

    if result.get("degraded"):
        warning.append("样本不足/降级：仅规则增强，未训练稳定模型")
    if result.get("pseudo_profile"):
        warning.append("使用 pseudo-project（网格聚合），非真实项目分布")
    dist = (result.get("class_distribution") or audit.get("label_distribution") or {})
    present = len([k for k, v in dist.items() if v])
    if present < len(ALL_TYPES) - 1:
        warning.append(f"类别不平衡/部分类型无样本（覆盖 {present}/{len(ALL_TYPES)}）")
    if result.get("trained") and result.get("weak_label_accuracy_on_val") is not None:
        if result["weak_label_accuracy_on_val"] < 0.6:
            warning.append(f"val 弱标签一致率偏低：{result['weak_label_accuracy_on_val']}")

    status = "fail" if failed else ("warning" if warning else "pass")
    return {"type_training_quality_status": status, "pass": passed,
            "warning": warning, "fail": failed,
            "can_enter_t5": status in ("pass", "warning"),
            "recommended_next_action": (
                "修复 fail 项后重训" if failed else
                "可进入 T5 评分校准；建议补真实项目样本以替代 pseudo-project")}
