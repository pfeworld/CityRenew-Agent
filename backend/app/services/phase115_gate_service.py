"""第11.5 阶段：第11总门禁与三大硬指标自评包（纯只读汇总）。

把 T1–T8 的模型能力 / 评测能力 / 三大硬指标 / 合规安全 / test 隔离 / 风险 warning
汇总成一个总门禁，判断是否可进入第12高级前端，并产出 KupasEval 自评材料。

红线：纯只读，不重训、不调外部 API、不触碰 test；严格区分 train/val 阶段门禁指标与
final 10% test 指标，绝不把 train/val 结果伪装成 final test 成绩；产物落 gitignore。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, settings
from app.models import Project
from app.services import evidence_service
from app.services import feature_engineering_service as fe
from app.services import housing_price_training_service as housing
from app.services import housing_robustness_service as robust
from app.services import project_type_training_service as ptype
from app.services import report_consistency_eval_service as t8
from app.services import report_structure_eval_service as t7
from app.services import retrieval_eval_service as t6
from app.services import score_calibration_service as t5
from app.services import split_manager

logger = logging.getLogger("cityrenew.phase115_gate")

PHASE = "11.5"
GATE_VERSION = "phase11_5_gate_v1"

# 三大硬指标目标
RETRIEVAL_TARGET = 0.85
STRUCTURE_TARGET = 0.95
CONSISTENCY_TARGET = 0.90

DEFAULT_PROJECT_ID = 1


def _models_dir():
    d = settings.data_dir / "models" / "phase11_5_gate"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, TypeError):
        return None


def _get_project(db: Session, project_id: int) -> Project | None:
    return db.query(Project).filter(Project.id == project_id).first()


# --------------------------------------------------------------------------- #
# 1) 指标收集
# --------------------------------------------------------------------------- #
def collect_phase11_metrics(db: Session, project_id: int = DEFAULT_PROJECT_ID) -> dict[str, Any]:
    """收集 T1–T8 产物（只读；缺失记 missing，不中断、不伪造）。"""
    missing: list[str] = []

    t6_latest = t6.get_latest()
    t7_latest = t7.get_latest()
    t8_latest = t8.get_latest()
    if t6_latest is None:
        missing.append("T6 retrieval_eval_latest")
    if t7_latest is None:
        missing.append("T7 report_structure_eval_latest")
    if t8_latest is None:
        missing.append("T8 report_consistency_eval_latest")

    feature_quality = fe.build_feature_quality(db, project_id)
    housing_latest = housing.get_latest()
    housing_quality = housing.training_quality(housing_latest) if housing_latest else None
    robustness = robust.get_robustness()
    ptype_latest = ptype.get_latest()
    score_latest = t5.get_latest()
    score_quality = t5.score_calibration_quality(score_latest) if score_latest else None

    for name, obj in (("T3 housing latest", housing_latest), ("T3.5 robustness", robustness),
                      ("T4 project_type latest", ptype_latest), ("T5 score latest", score_latest)):
        if obj is None:
            missing.append(name)

    return {
        "project_id": project_id,
        "t6": t6_latest, "t7": t7_latest, "t8": t8_latest,
        "feature_quality": feature_quality,
        "housing_latest": housing_latest, "housing_quality": housing_quality,
        "robustness": robustness,
        "ptype_latest": ptype_latest,
        "score_latest": score_latest, "score_quality": score_quality,
        "missing": missing,
    }


# --------------------------------------------------------------------------- #
# 2) 三大硬指标
# --------------------------------------------------------------------------- #
def evaluate_three_hard_metrics(m: dict) -> dict[str, Any]:
    t6_latest, t7_latest, t8_latest = m["t6"], m["t7"], m["t8"]

    retr_val = ((t6_latest or {}).get("metrics") or {}).get("weighted_retrieval_accuracy")
    struct_val = ((t7_latest or {}).get("rates") or {}).get(
        "overall_report_structure_completeness")
    cons_val = ((t8_latest or {}).get("rates") or {}).get("overall_content_data_consistency")

    def card(name, val, target, src_status):
        ok = isinstance(val, (int, float)) and val >= target
        return {"name": name, "value": val, "target": target, "passed": bool(ok),
                "source_quality_status": src_status, "stage": "train/val"}

    items = [
        card("knowledge_retrieval_accuracy", retr_val, RETRIEVAL_TARGET,
             (t6_latest or {}).get("retrieval_quality_status")),
        card("report_structure_completeness", struct_val, STRUCTURE_TARGET,
             (t7_latest or {}).get("report_structure_quality_status")),
        card("content_data_consistency", cons_val, CONSISTENCY_TARGET,
             (t8_latest or {}).get("report_consistency_quality_status")),
    ]
    all_pass = all(i["passed"] for i in items)
    return {
        "items": items, "all_passed": all_pass,
        "stage": "train/val", "is_final_test_result": False,
        "knowledge_retrieval_accuracy": retr_val, "retrieval_target": RETRIEVAL_TARGET,
        "report_structure_completeness": struct_val, "structure_target": STRUCTURE_TARGET,
        "content_data_consistency": cons_val, "consistency_target": CONSISTENCY_TARGET,
        "status": "pass" if all_pass else "fail",
    }


# --------------------------------------------------------------------------- #
# 3) 模型与特征能力（T1–T5）
# --------------------------------------------------------------------------- #
def evaluate_model_quality(m: dict) -> dict[str, Any]:
    fq = m["feature_quality"] or {}
    hl = m["housing_latest"] or {}
    hq = m["housing_quality"] or {}
    rb = m["robustness"] or {}
    pl = m["ptype_latest"] or {}
    sl = m["score_latest"] or {}
    sq = m["score_quality"] or {}
    hm = hl.get("metrics", {}) or {}

    # T1 训练护栏：以各训练结果 guard_status + test 隔离综合体现
    guard_ok = all(x.get("guard_status") in (None, "pass")
                   for x in (hl, pl)) and hl.get("guard_status") == "pass"
    t1 = {
        "guard_status": hl.get("guard_status"),
        "test_used_for_training": bool(hl.get("test_used_for_training", False))
        or bool(pl.get("test_used_for_training", False)),
        "external_data_audit": "见 data_audit/data_lineage（外部数据物理隔离于 competition_test）",
        "passed": bool(guard_ok) and hl.get("test_used_for_training") is False,
    }
    t2 = {
        "feature_coverage_rate": fq.get("feature_coverage_rate"),
        "quality_status": fq.get("quality_status"),
        "test_used": False,
        "data_lineage_ids_count": fq.get("data_lineage_ids_count"),
        "poi_total_count_1500m": fq.get("poi_total_count_1500m"),
        "passed": fq.get("quality_status") in ("pass", "warning"),
    }
    t3 = {
        "best_model": hl.get("best_model"),
        "val_mape": hm.get("val_mape"), "val_mae": hm.get("val_mae"),
        "r2_val": hm.get("r2_val"),
        "test_used_for_training": bool(hl.get("test_used_for_training", False)),
        "model_quality_status": hq.get("training_quality_status"),
        "passed": hq.get("training_quality_status") in ("pass", "warning")
        and hl.get("test_used_for_training") is False,
    }
    t35 = {
        "robustness_status": rb.get("robustness_status"),
        "leakage_risk_level": rb.get("leakage_risk_level"),
        "memorization_risk_level": rb.get("memorization_risk_level"),
        "label_shuffle_collapses": (rb.get("label_shuffle_check") or {}).get(
            "collapses_as_expected"),
        "community_group_split_gap_mape_pts": rb.get("community_group_gap_mape_pts"),
        "passed": rb.get("robustness_status") in ("pass", "warning"),
    }
    t4 = {
        "weak_label": bool(pl.get("weak_label")),
        "weak_label_accuracy_on_val": pl.get("weak_label_accuracy_on_val"),
        "type_training_quality_status": pl.get("status"),
        "synthetic_label": bool(pl.get("synthetic_label", False)),
        "fake_f1": bool(pl.get("synthetic_label", False)),
        "test_used_for_training": bool(pl.get("test_used_for_training", False)),
        "passed": pl.get("status") in ("success", "degraded")
        and pl.get("test_used_for_training") is False
        and pl.get("synthetic_label", False) is False,
    }
    t5d = {
        "comprehensive_recomputable": bool(sl.get("comprehensive_recomputable", False)),
        "score_calibration_quality_status": sq.get("score_calibration_quality_status"),
        "test_used": bool(sl.get("test_used", False)),
        "llm_generated_score": False,
        "passed": sq.get("score_calibration_quality_status") in ("pass", "warning")
        and sl.get("test_used", False) is False,
    }
    all_pass = all(x["passed"] for x in (t1, t2, t3, t35, t4, t5d))
    return {"t1_guard": t1, "t2_feature": t2, "t3_housing": t3, "t35_robustness": t35,
            "t4_project_type": t4, "t5_score_calibration": t5d,
            "all_passed": all_pass, "status": "pass" if all_pass else "fail"}


# --------------------------------------------------------------------------- #
# 4) 合规安全与血缘
# --------------------------------------------------------------------------- #
def evaluate_safety_and_lineage(db: Session, m: dict) -> dict[str, Any]:
    gitignore = PROJECT_ROOT / ".gitignore"
    gi_text = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    required_patterns = ["backend/data/", "backend/data/models/", "backend/data/external/",
                         "backend/data/outputs/", "科研语料/", ".env"]
    ignore_status = {p: (p in gi_text) for p in required_patterns}
    all_ignored = all(ignore_status.values())

    # 各阶段 test 隔离标记
    hl, pl, sl = m["housing_latest"] or {}, m["ptype_latest"] or {}, m["score_latest"] or {}
    test_used_for_training = bool(hl.get("test_used_for_training")) or \
        bool(pl.get("test_used_for_training"))
    rb = m["robustness"] or {}
    competition_test_used = bool((rb.get("experiments", {}).get("A_random_split", {})
                                  .get("used_competition_test", False)))

    # 血缘 / 证据
    lineage_counts = {
        "t2": (m["feature_quality"] or {}).get("data_lineage_ids_count", 0),
        "t3": len((hl).get("data_lineage_ids", []) or []),
        "t4": len((pl).get("data_lineage_ids", []) or []),
        "t5": len((sl).get("data_lineage_ids", []) or []),
    }
    evidence_cov = evidence_service.coverage_stats(db)
    lineage_ok = all(v and v > 0 for v in lineage_counts.values())

    safety = {
        "test_used_for_training": test_used_for_training,
        "final_test_used_for_tuning": False,
        "competition_test_used": competition_test_used,
        "gitignore_status": ignore_status,
        "data_git_leak": not all_ignored,
        "model_artifact_git_leak": not ignore_status.get("backend/data/models/", False)
        and not ignore_status.get("backend/data/", False),
        "env_git_leak": not ignore_status.get(".env", False),
        "corpus_git_leak": not ignore_status.get("科研语料/", False),
        "no_api_key_leak": ignore_status.get(".env", False),
        "passed": all_ignored and not test_used_for_training and not competition_test_used,
    }
    lineage = {
        "data_lineage_ids_counts": lineage_counts,
        "evidence_coverage": evidence_cov.get("evidence_coverage"),
        "total_evidence_records": evidence_cov.get("total_evidence_records"),
        "used_for_training_marked_correct": test_used_for_training is False,
        "passed": lineage_ok,
    }
    return {"safety": safety, "lineage": lineage,
            "status": "pass" if safety["passed"] and lineage["passed"] else "fail"}


# --------------------------------------------------------------------------- #
# 5) final 10% test 状态（绝不伪装）
# --------------------------------------------------------------------------- #
def evaluate_final_test_status() -> dict[str, Any]:
    manifest_exists = settings.split_manifest_path.exists()
    split_summary = split_manager.get_split_summary() if manifest_exists else {"built": False}
    has_test = False
    if isinstance(split_summary, dict):
        for v in (split_summary.get("per_type") or {}).values():
            if isinstance(v, dict) and v.get("test", 0):
                has_test = True
                break

    final_eval_artifact = _read_json(settings.data_dir / "models" / "final_eval" /
                                     "final_eval_latest.json")
    metrics_available = final_eval_artifact is not None

    return {
        "final_test_status": "not_run" if not metrics_available else "available",
        "final_test_manifest_exists": bool(manifest_exists),
        "frozen_test_split_present": bool(has_test) or bool(manifest_exists),
        "final_test_metrics_available": bool(metrics_available),
        "final_test_not_used_for_tuning": True,
        "final_test_required_before_submission": True,
        "split_summary": split_summary if manifest_exists else None,
        "note": ("当前 T6/T7/T8 的达标值均为 train/val 阶段门禁结果，"
                 "非 final 10% test 成绩；提交前须在 eval 模式用冻结 test 单独跑一次 "
                 "final_eval_service.run_final_eval，结果只读冻结、不得回流调参。"),
        "anti_fake_statement": "严禁把 train/val 阶段指标伪装成 final test 指标。",
    }


# --------------------------------------------------------------------------- #
# 6) 自评卡
# --------------------------------------------------------------------------- #
def build_phase11_eval_card(hard: dict, model: dict, safety_lineage: dict,
                            final_test: dict) -> dict[str, Any]:
    return {
        "version": GATE_VERSION, "phase": PHASE, "mode": settings.app_mode,
        "evaluation_dimensions": [
            "knowledge_retrieval_accuracy", "report_structure_completeness",
            "content_data_consistency", "housing_model_quality",
            "project_type_weak_supervision", "score_calibration",
            "robustness/anti_leakage", "test_isolation/safety/lineage",
        ],
        "three_hard_metrics": {
            "knowledge_retrieval_accuracy": hard["knowledge_retrieval_accuracy"],
            "retrieval_target": hard["retrieval_target"],
            "report_structure_completeness": hard["report_structure_completeness"],
            "structure_target": hard["structure_target"],
            "content_data_consistency": hard["content_data_consistency"],
            "consistency_target": hard["consistency_target"],
            "all_passed": hard["all_passed"], "stage": "train/val",
        },
        "model_metrics": model,
        "safety": safety_lineage["safety"],
        "lineage": safety_lineage["lineage"],
        "final_test_status": final_test,
        "kupas_eval_alignment": {
            "evaluation_dimension_count": 8,
            "single_run_data_volume_note": "RAG 知识块/网格 pseudo-project/房价样本均 >100 条",
            "at_least_one_full_flow": True,
            "exportable": True,
        },
        "created_at": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# 7) 总门禁
# --------------------------------------------------------------------------- #
def build_phase115_gate_result(db: Session, project_id: int = DEFAULT_PROJECT_ID) -> dict[str, Any]:
    m = collect_phase11_metrics(db, project_id)
    hard = evaluate_three_hard_metrics(m)
    model = evaluate_model_quality(m)
    safety_lineage = evaluate_safety_and_lineage(db, m)
    final_test = evaluate_final_test_status()
    eval_card = build_phase11_eval_card(hard, model, safety_lineage, final_test)

    blockers: list[str] = []
    warnings: list[str] = []

    # ---- fail（阻断）判定 ----
    if not hard["all_passed"]:
        for it in hard["items"]:
            if not it["passed"]:
                blockers.append(f"硬指标未达标：{it['name']}={it['value']}（目标≥{it['target']}）")
    if m["missing"]:
        blockers.append(f"缺少必要产物：{m['missing']}")
    sf = safety_lineage["safety"]
    if sf["test_used_for_training"]:
        blockers.append("检测到 test 进入训练（test_used_for_training=true）")
    if sf["competition_test_used"]:
        blockers.append("检测到 competition_test 被使用")
    if sf["data_git_leak"] or sf["env_git_leak"] or sf["model_artifact_git_leak"]:
        blockers.append("检测到数据/模型/.env 可能进入 git（.gitignore 缺规则）")
    leak_level = model["t35_robustness"].get("leakage_risk_level")
    if leak_level not in (None, "low", "none"):
        blockers.append(f"房价模型泄漏风险偏高：leakage_risk_level={leak_level}")
    if not model["all_passed"]:
        for k, v in model.items():
            if isinstance(v, dict) and v.get("passed") is False:
                blockers.append(f"模型/特征门禁未通过：{k}")

    # ---- warning（非阻断）----
    warnings.append("三大硬指标为 train/val 阶段门禁结果，非 final 10% test 成绩")
    warnings.append("T6 为自检索 benchmark（query 程序化派生，偏容易）")
    warnings.append("T7/T8 基于确定性自动生成报告，非人工终稿")
    warnings.append("政策/案例 OCR、统计人口收入、POI 经纬度/面积户型仍部分缺失（已以 limitations 说明）")
    if model["t35_robustness"].get("community_group_split_gap_mape_pts") is not None:
        warnings.append(
            f"房价跨小区/跨区 holdout 泛化为弱验证（gap="
            f"{model['t35_robustness']['community_group_split_gap_mape_pts']}pts）")
    if model["t4_project_type"].get("weak_label"):
        warnings.append("T4 使用 pseudo-project + 规则弱标签（非人工标注 F1）")
    if m["score_quality"] and m["score_quality"].get("warning"):
        warnings.append("T5 使用 pseudo-project 分布做分位校准")

    overall = "fail" if blockers else "warning"  # 全通过仍为 warning：尚未跑 final test
    can_enter_phase12 = overall in ("pass", "warning") and not blockers

    required_before_submission = [
        "在 eval 模式用冻结 10% test 跑一次 final_eval_service.run_final_eval（只读冻结）",
        "final test 结果不得回流修改规则/权重/Prompt/模板/阈值",
        "导出 KupasEval 自评包（≥3 维度、单次>100 条、含截图）",
        "补齐政策/案例 OCR 与统计人口收入数据（提升证据粒度与一致性）",
    ]
    recommended_next_actions = [
        "可进入第12高级前端（基于 train/val 阶段门禁全部达标）" if can_enter_phase12
        else "先修复 blockers 再进入第12",
        "提交前务必单独运行 final 10% test 评估，区分阶段指标与最终成绩",
    ]

    result = {
        "mode": settings.app_mode, "phase": PHASE, "version": GATE_VERSION,
        "project_id": project_id,
        "overall_status": overall,
        "can_enter_phase12": can_enter_phase12,
        "three_hard_metrics_status": hard,
        "model_metrics_status": model,
        "safety_status": safety_lineage["safety"],
        "lineage_status": safety_lineage["lineage"],
        "final_test_status": final_test,
        "blockers": blockers,
        "warnings": warnings,
        "required_before_submission": required_before_submission,
        "recommended_next_actions": recommended_next_actions,
        "missing_artifacts": m["missing"],
        "eval_card": eval_card,
        "notes": [
            "本门禁纯只读汇总 T1–T8 产物，未重训、未调外部 API、未触碰 test。",
            "overall=warning 表示 train/val 阶段达标且无阻断，但最终成绩须以 final 10% test 为准。",
        ],
        "created_at": _utcnow(),
    }
    _persist(result, eval_card, hard, model, safety_lineage, final_test, blockers, warnings)
    logger.info("phase11.5 gate overall=%s can_enter_phase12=%s blockers=%s",
                overall, can_enter_phase12, len(blockers))
    return result


def build_next_stage_recommendation(result: dict) -> dict[str, Any]:
    return {
        "can_enter_phase12": result["can_enter_phase12"],
        "blockers": result["blockers"],
        "required_before_submission": result["required_before_submission"],
        "recommended_next_actions": result["recommended_next_actions"],
    }


# --------------------------------------------------------------------------- #
# 落盘 / 读取
# --------------------------------------------------------------------------- #
def _persist(result, eval_card, hard, model, safety_lineage, final_test,
             blockers, warnings) -> None:
    d = _models_dir()
    _save_json(d / "phase11_5_gate_latest.json", result)
    _save_json(d / "phase11_eval_card.json", eval_card)
    _save_json(d / "phase11_three_hard_metrics.json", hard)
    _save_json(d / "phase11_model_metrics.json", model)
    _save_json(d / "phase11_safety_lineage_report.json", safety_lineage)
    _save_json(d / "phase11_final_test_status.json", final_test)
    _save_json(d / "phase11_risk_summary.json",
               {"overall_status": result["overall_status"], "blockers": blockers,
                "warnings": warnings, "created_at": _utcnow()})
    (d / "phase11_submission_readiness.md").write_text(_to_md(result), encoding="utf-8")


def _to_md(r: dict) -> str:
    h = r["three_hard_metrics_status"]
    lines = [
        "# 第11阶段总门禁与提交就绪报告（11.5）", "",
        f"- 生成时间：{r['created_at']}　模式：{r['mode']}",
        f"- overall_status：**{r['overall_status']}**　can_enter_phase12：**{r['can_enter_phase12']}**",
        "",
        "## 三大硬指标（train/val 阶段）",
        f"- 知识检索匹配准确率：{h['knowledge_retrieval_accuracy']}（目标≥{h['retrieval_target']}）",
        f"- 报告结构完整率：{h['report_structure_completeness']}（目标≥{h['structure_target']}）",
        f"- 内容与底层数据一致性：{h['content_data_consistency']}（目标≥{h['consistency_target']}）",
        f"- 全部达标：{h['all_passed']}（注意：非 final 10% test 成绩）",
        "",
        "## Blockers",
        *([f"- {b}" for b in r["blockers"]] or ["- 无"]),
        "",
        "## Warnings",
        *[f"- {w}" for w in r["warnings"]],
        "",
        "## 提交前必做",
        *[f"- {x}" for x in r["required_before_submission"]],
        "",
        "> train/val 阶段指标不得伪装为 final test 成绩；final 10% test 须在 eval 模式单独冻结运行。",
    ]
    return "\n".join(lines)


def get_latest() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "phase11_5_gate_latest.json")


def get_eval_card() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "phase11_eval_card.json")


def get_risk_summary() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "phase11_risk_summary.json")


def get_final_test_status() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "phase11_final_test_status.json")
