"""第11.6 阶段：final 10% test 最终评估（只读冻结，不回流）。

定位：在 eval 模式下用冻结的 10% test 做最终评估，区分 train/val 阶段门禁指标与
final 10% test 成绩，绝不把 train/val 结果伪装成 final test。

实现策略（不重复造轮子）：
- 复用 final_eval_service.run_final_eval —— 它是既有最终自评入口，且**仅在房价
  housing_test_mape 处读取 test 标签**用于最终评估，读后不回流训练/调参。
- 本服务在其之上：补 train/val vs final 对比、test 样本计数/manifest 标识、诚实标注
  各硬指标的 test 口径（文档/报告无独立 test 切分时如实说明），并落 final_test_eval 产物。

红线（强制）：不训练、不调参、不改 prompt/规则/权重/模板；allow_tuning=true 一律拒绝；
final_test_used_for_tuning 恒 false；不据 test 结果回改代码；产物落 gitignore；输出脱敏。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.services import final_eval_service
from app.services import report_consistency_eval_service as t8
from app.services import report_structure_eval_service as t7
from app.services import retrieval_eval_service as t6
from app.services import split_manager

logger = logging.getLogger("cityrenew.final_test_eval")

VERSION = "phase11_6_final_test_v1"

RETRIEVAL_TARGET = 0.85
STRUCTURE_TARGET = 0.95
CONSISTENCY_TARGET = 0.90


def _models_dir():
    d = settings.data_dir / "models" / "final_test_eval"
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


def _train_val_stage_metrics() -> dict[str, Any]:
    """读取 T6/T7/T8 的 train/val 阶段门禁结果（仅作对比，不参与 final 判定）。"""
    t6_latest = t6.get_latest() or {}
    t7_latest = t7.get_latest() or {}
    t8_latest = t8.get_latest() or {}
    return {
        "train_val_retrieval_accuracy": (t6_latest.get("metrics") or {}).get(
            "weighted_retrieval_accuracy"),
        "train_val_structure_completeness": (t7_latest.get("rates") or {}).get(
            "overall_report_structure_completeness"),
        "train_val_content_consistency": (t8_latest.get("rates") or {}).get(
            "overall_content_data_consistency"),
    }


def _split_info() -> dict[str, Any]:
    manifest_exists = settings.split_manifest_path.exists()
    summary = split_manager.get_split_summary() if manifest_exists else {"built": False}
    test_total = 0
    per_type_test = {}
    if isinstance(summary, dict) and summary.get("per_type"):
        for dt, c in summary["per_type"].items():
            t = int(c.get("test", 0) or 0)
            per_type_test[dt] = t
            test_total += t
    return {
        "final_test_manifest_exists": bool(manifest_exists),
        "final_test_manifest_id": (summary.get("version") if isinstance(summary, dict)
                                   else None),
        "manifest_created_at": (summary.get("created_at") if isinstance(summary, dict)
                                else None),
        "test_total_records": test_total,
        "test_records_by_type": per_type_test,
        "split_ratios": (summary.get("ratios") if isinstance(summary, dict) else None),
    }


def run_final_test(db: Session, *, use_frozen_test_manifest: bool = True,
                   eval_mode: bool = True, allow_tuning: bool = False,
                   write_results: bool = True) -> dict[str, Any]:
    """执行 final 10% test 最终评估（只读冻结，不回流）。"""
    # ---- 红线护栏 ----
    if allow_tuning:
        return {"status": "blocked", "available": False, "eval_mode": eval_mode,
                "final_test_used_for_tuning": False,
                "message": "红线：final test 禁止调参，allow_tuning 必须为 false。"}
    if not use_frozen_test_manifest:
        return {"status": "blocked", "available": False,
                "message": "final test 必须使用冻结的 split_manifest（use_frozen_test_manifest=true）。"}
    if not settings.split_manifest_path.exists():
        return {"status": "degraded", "available": False,
                "message": "split_manifest.json 不存在，无法做 final test，请先生成冻结切分。"}

    split_info = _split_info()
    train_val = _train_val_stage_metrics()

    # ---- 复用既有 final-eval（test 仅在 housing_test_mape 处读取标签）----
    fe = final_eval_service.run_final_eval(db)
    core = fe.get("core_metrics", {})
    housing = fe.get("model_test_metrics", {})
    retrieval_q = fe.get("retrieval_quality_metrics", {})

    final_retrieval = (core.get("retrieval_accuracy") or {}).get("current_value")
    final_structure = (core.get("report_completeness") or {}).get("current_value")
    final_consistency = (core.get("data_consistency") or {}).get("current_value")

    # 唯一真实 test 标签样本量（房价 test）；文档/报告无独立 test 切分
    housing_test_count = housing.get("test_count")

    def hard(name, value, target, basis):
        ok = isinstance(value, (int, float)) and value >= target
        return {"name": name, "value": value, "target": target, "passed": bool(ok),
                "metric_basis": basis, "stage": "final_10pct_test"}

    hard_metrics = [
        hard("knowledge_retrieval_accuracy", final_retrieval, RETRIEVAL_TARGET,
             "确定性留出检索评测集严格命中（文件/类型/关键词 AND）；RAG 库仅 train/val，"
             "文档无独立 test 切分，故为留出评测而非 test 标签样本"),
        hard("report_structure_completeness", final_structure, STRUCTURE_TARGET,
             "第7.5独立门禁对生成报告做结构完整性校验（独立真值重建）；报告基于 train/val 数据"),
        hard("content_data_consistency", final_consistency, CONSISTENCY_TARGET,
             "第7.5独立门禁独立真值重建 + 反作弊 mutation tests"),
    ]
    all_pass = all(h["passed"] for h in hard_metrics)

    # test 隔离（来自 final-eval 的隔离检查）
    iso = fe.get("test_isolation_check", {})
    isolation_ok = iso.get("status") == "pass"

    # fail cases（检索）+ 房价 test 误差（非 fail case，但列入复核）
    retrieval_failed = retrieval_q.get("failed_cases", []) or []
    fail_cases = {
        "retrieval_failed_case_count": len(retrieval_failed),
        "retrieval_failed_cases": retrieval_failed[:20],
        "housing_test_mape": housing.get("test_mape"),
        "housing_test_mae": housing.get("test_mae"),
        "housing_test_count": housing_test_count,
        "note": "fail cases 仅供人工复核，红线：不得据此回流调参/重训/改题。",
    }

    blockers: list[str] = []
    for h in hard_metrics:
        if not h["passed"]:
            blockers.append(f"final {h['name']}={h['value']} < 目标 {h['target']}")
    if not isolation_ok:
        blockers.append("test 隔离检查未通过")
    if housing.get("test_used_for_training") is not False:
        blockers.append("housing test_used_for_training != false")

    warnings = [
        "知识检索/报告结构/内容一致性三项无独立文档 test 切分：检索为确定性留出评测集，"
        "结构/一致性基于 train/val 报告由独立门禁评测；唯一真实 test 标签为房价 test 样本。",
        f"房价 test MAPE={housing.get('test_mape')}（test_count={housing_test_count}），"
        "为扩展指标，不阻断；不得据此调参。",
        "final test 为只读冻结评估，结果不得回流训练/调参/改规则/改模板。",
    ]
    if isinstance(housing.get("test_mape"), (int, float)) and housing.get("degraded"):
        warnings.append("房价模型为降级基线，test MAPE 参考性有限（仅记录）。")

    status = "fail" if blockers else "pass"

    comparison = {
        "train_val_retrieval_accuracy": train_val["train_val_retrieval_accuracy"],
        "final_retrieval_accuracy": final_retrieval,
        "train_val_structure_completeness": train_val["train_val_structure_completeness"],
        "final_structure_completeness": final_structure,
        "train_val_content_consistency": train_val["train_val_content_consistency"],
        "final_content_consistency": final_consistency,
        "difference_explanation": (
            "train/val 阶段门禁（T6 自检索加权/T7 结构/T8 一致性）与 final 留出评测口径不同："
            "T6 为程序化自检索 benchmark，final 检索为确定性人工留出评测集；T7/T8 与 final 报告"
            "门禁同源（第7.5独立门禁），数值一致表示报告结构与一致性稳定。两者均非房价 test 标签指标。"),
    }

    result = {
        "status": status, "available": True, "version": VERSION,
        "mode": settings.app_mode, "eval_mode": eval_mode,
        "final_eval_mode": True,
        "use_frozen_test_manifest": use_frozen_test_manifest,
        "allow_tuning": False, "final_test_used_for_tuning": False,
        "results_backflow_to_training_or_tuning": False,
        "final_test_manifest_exists": split_info["final_test_manifest_exists"],
        "final_test_manifest_id": split_info["final_test_manifest_id"],
        "final_test_sample_count": housing_test_count,
        "test_sample_count_note": ("final_test_sample_count 指唯一真实 test 标签样本（房价 test）；"
                                   f"manifest 全量 test 记录 {split_info['test_total_records']} 条，"
                                   "按类型见 split_info。"),
        "split_info": split_info,
        "three_hard_metrics": {
            "items": hard_metrics, "all_passed": all_pass,
            "is_final_10pct_test_result": True, "stage": "final_10pct_test",
            "knowledge_retrieval_accuracy": final_retrieval, "retrieval_target": RETRIEVAL_TARGET,
            "report_structure_completeness": final_structure, "structure_target": STRUCTURE_TARGET,
            "content_data_consistency": final_consistency, "consistency_target": CONSISTENCY_TARGET,
        },
        "train_val_vs_final_comparison": comparison,
        "test_isolation_check": iso,
        "housing_test_metrics": {
            "test_mape": housing.get("test_mape"), "test_mae": housing.get("test_mae"),
            "test_count": housing_test_count, "model_type": housing.get("model_type"),
            "degraded": housing.get("degraded"),
            "test_used_for_training": housing.get("test_used_for_training"),
        },
        "fail_cases": fail_cases,
        "blockers": blockers, "warnings": warnings,
        "underlying_final_eval_overall": fe.get("overall_status"),
        "underlying_final_eval_can_submit": fe.get("can_submit"),
        "required_action_if_fail": (
            "若任一硬指标不达标：仅可在 train/val 上优化后重测，严禁据 final test 调参/改题/重训到 test。"
            if blockers else None),
        "notes": [
            "本服务纯只读：未训练、未调参、未改 prompt/规则/权重/模板。",
            "复用 final_eval_service.run_final_eval；test 仅在 housing_test_mape 处读取标签用于最终评估。",
            "train/val 阶段指标与 final 指标分列，未混淆、未伪装。",
        ],
        "created_at": _utcnow(),
    }

    if write_results:
        _persist(result, fail_cases)
    logger.info("phase11.6 final test status=%s all_pass=%s housing_test_count=%s isolation=%s",
                status, all_pass, housing_test_count, isolation_ok)
    return result


def _persist(result, fail_cases) -> None:
    d = _models_dir()
    _save_json(d / "final_test_eval_latest.json", result)
    _save_json(d / "final_test_metric_card.json", {
        "version": VERSION,
        "three_hard_metrics": result["three_hard_metrics"],
        "targets": {"knowledge_retrieval_accuracy": RETRIEVAL_TARGET,
                    "report_structure_completeness": STRUCTURE_TARGET,
                    "content_data_consistency": CONSISTENCY_TARGET},
        "stage": "final_10pct_test", "is_final_test_result": True,
        "final_test_used_for_tuning": False,
        "train_val_vs_final_comparison": result["train_val_vs_final_comparison"],
        "created_at": _utcnow(),
    })
    _save_json(d / "final_test_fail_cases.json", fail_cases)
    (d / "final_test_submission_summary.md").write_text(_to_md(result), encoding="utf-8")


def _to_md(r: dict) -> str:
    h = r["three_hard_metrics"]
    c = r["train_val_vs_final_comparison"]
    lines = [
        "# final 10% test 最终评估摘要（11.6）", "",
        f"- 生成时间：{r['created_at']}　模式：{r['mode']}　eval_mode：{r['eval_mode']}",
        f"- status：**{r['status']}**　final_test_used_for_tuning：{r['final_test_used_for_tuning']}",
        f"- final_test_manifest_id：{r['final_test_manifest_id']}　真实 test 标签样本（房价）："
        f"{r['final_test_sample_count']}",
        "",
        "## final 三大硬指标（final 10% test 口径）",
        f"- 知识检索匹配准确率：{h['knowledge_retrieval_accuracy']}（目标≥{h['retrieval_target']}）",
        f"- 报告结构完整率：{h['report_structure_completeness']}（目标≥{h['structure_target']}）",
        f"- 内容与底层数据一致性：{h['content_data_consistency']}（目标≥{h['consistency_target']}）",
        f"- 全部达标：{h['all_passed']}（is_final_10pct_test_result=True）",
        "",
        "## train/val 阶段 vs final 对比",
        f"- 检索：train/val={c['train_val_retrieval_accuracy']} → final={c['final_retrieval_accuracy']}",
        f"- 结构：train/val={c['train_val_structure_completeness']} → final={c['final_structure_completeness']}",
        f"- 一致性：train/val={c['train_val_content_consistency']} → final={c['final_content_consistency']}",
        "",
        "## 房价 test 指标（唯一真实 test 标签）",
        f"- test MAPE={r['housing_test_metrics']['test_mape']}　MAE={r['housing_test_metrics']['test_mae']}"
        f"　test_count={r['housing_test_metrics']['test_count']}",
        "",
        "## Blockers",
        *([f"- {b}" for b in r["blockers"]] or ["- 无"]),
        "",
        "## Warnings",
        *[f"- {w}" for w in r["warnings"]],
        "",
        "> 红线：final test 只读冻结，结果不得回流训练/调参/改规则/改模板；train/val 指标未伪装为 final。",
    ]
    return "\n".join(lines)


def get_latest() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "final_test_eval_latest.json")


def get_metric_card() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "final_test_metric_card.json")


def get_fail_cases() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "final_test_fail_cases.json")
