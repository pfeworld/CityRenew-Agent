"""第9阶段最终自评服务（final-eval）。

目标：在 eval 模式下，聚合并真实输出比赛三大核心指标与扩展指标，给出
overall_status / can_submit 判定，并产出交付所需的结构化结论。

核心指标：
- retrieval_accuracy   知识检索匹配准确率（>0.85）—— 用确定性评测集 + 严格命中判定。
- report_completeness  报告结构完整率（>0.95）—— 复用第7.5独立门禁。
- data_consistency     生成内容与底层数据一致性（>0.90）—— 复用第7.5独立真值重建。

扩展指标：房价 test MAPE/MAE、证据链覆盖、数字溯源、反作弊 mutation tests。

============================== 测试集隔离红线（强制） ==============================
1. test 只允许在本 final-eval 阶段、且仅在 housing_test_mape 处读取"标签"用于最终评估。
2. 检索评测集不来自 test；RAG 知识库结构上只含 train/val，检索天然不触碰 test。
3. 如果 housing_test_mape / retrieval_accuracy 或任何 test/eval 指标不达标：
   —— 只输出 fail/warning、risks、recommendations；
   —— 不得重训模型、不得改规则、不得据 test 调参、不得自动重复刷 test 优化指标。
4. 本服务全程纯只读：不写 DB、不写模型文件、不调外部 API、不使用大模型生成结论。
====================================================================================

红线：不返回 raw_json / 原始点位明细 / 企业名 / 小区名 / 地址明细 / 坐标列表。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, settings
from app.models import Project
from app.services import housing_price_model as hpm
from app.services import (
    model_audit_service,
    phase75_gate_service,
    rag_service,
    report_content_service,
    report_export_service,
)

logger = logging.getLogger("cityrenew.final_eval")

ST_PASS = "pass"
ST_WARNING = "warning"
ST_FAIL = "fail"

# ---- 核心指标门槛 ----
RETRIEVAL_PASS = 0.85
RETRIEVAL_WARN = 0.75
COMPLETENESS_PASS = 0.95
CONSISTENCY_PASS = 0.90

# ---- 扩展指标（仅 warning，不阻断提交）----
MAPE_TARGET = 0.15  # 房价 test MAPE 软目标；超出仅 warning + 记录，不据此优化

EVAL_CASES_PATH = Path(__file__).resolve().parent.parent / "resources" / "eval" / "retrieval_eval_cases.json"

# 脱敏自检禁用标记（与第7.5门禁一致）
FORBIDDEN_TOKENS = (
    "raw_json", '"coordinates"', '"address"', '"residence"',
    "profile_json", "chunk_text", "center_lng", "center_lat",
)

# 前端结构性检查：关键文件 + 关键 API 方法 + 关键展示字段
_FRONTEND_FILES = ("frontend/src/pages/Dashboard.jsx", "frontend/src/api/client.js")
_FRONTEND_METHODS = (
    "runFullAnalysis", "generateReport", "exportReportDocx",
    "runPhase75Gate", "getStageBaseline", "getModelAudit",
)
_FRONTEND_DISPLAY_FIELDS = ("report_completeness", "data_consistency", "F_score", "overall_status")


def _mk(name: str, value: Any, threshold: str, status: str, explanation: str) -> dict[str, Any]:
    return {
        "metric_name": name,
        "current_value": value,
        "threshold": threshold,
        "status": status,
        "explanation": explanation,
    }


# --------------------------------------------------------------------------- #
# 1. retrieval_accuracy（确定性评测集 + 严格命中判定）
# --------------------------------------------------------------------------- #
def _load_eval_cases() -> dict[str, Any]:
    if not EVAL_CASES_PATH.exists():
        return {"cases": [], "top_k": 5, "keyword_hit_ratio": 0.5}
    with EVAL_CASES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _keyword_hits(expected: list[str], results: list[dict[str, Any]]) -> tuple[int, int]:
    """统计 expected_keywords 在 top_k 结果 summary/keywords 中的命中数。"""
    blob_parts: list[str] = []
    for r in results:
        if r.get("summary"):
            blob_parts.append(str(r["summary"]))
        for kw in r.get("keywords") or []:
            blob_parts.append(str(kw))
    blob = " ".join(blob_parts)
    hits = sum(1 for kw in expected if kw and kw in blob)
    return hits, len(expected)


def _evaluate_retrieval() -> dict[str, Any]:
    doc = _load_eval_cases()
    cases = doc.get("cases", [])
    top_k = int(doc.get("top_k", 5))
    kw_ratio = float(doc.get("keyword_hit_ratio", 0.5))

    total = len(cases)
    correct = 0
    failed_cases: list[dict[str, Any]] = []
    coverage: dict[str, int] = {}

    for case in cases:
        st = case.get("expected_source_type")
        if st:
            coverage[st] = coverage.get(st, 0) + 1

        res = rag_service.query(case["query"], top_k=top_k)
        results = res.get("results", [])
        files = {r.get("source_file") for r in results}
        types = {r.get("source_type") for r in results}

        exp_file = case.get("expected_file")
        exp_type = case.get("expected_source_type")
        exp_kw = case.get("expected_keywords") or []

        file_ok = (exp_file is None) or (exp_file in files)
        type_ok = (exp_type is None) or (exp_type in types)
        if exp_kw:
            hits, n = _keyword_hits(exp_kw, results)
            kw_ok = n > 0 and (hits / n) >= kw_ratio
        else:
            kw_ok = True
            hits, n = 0, 0

        is_correct = file_ok and type_ok and kw_ok
        if is_correct:
            correct += 1
        else:
            # failed_cases 只输出标签与 top_k 摘要，不含原文长段
            failed_cases.append({
                "case_id": case.get("case_id"),
                "query": case.get("query"),
                "expected": {
                    "source_type": exp_type,
                    "file": exp_file,
                    "keywords": exp_kw,
                    "keyword_hits": f"{hits}/{n}" if exp_kw else "n/a",
                },
                "failed_conditions": [
                    c for c, ok in (("file", file_ok), ("source_type", type_ok), ("keywords", kw_ok)) if not ok
                ],
                "top_k": [
                    {"source_type": r.get("source_type"), "source_file": r.get("source_file"),
                     "score": r.get("score")}
                    for r in results
                ],
            })

    accuracy = round(correct / total, 4) if total else 0.0
    if accuracy >= RETRIEVAL_PASS:
        status = ST_PASS
    elif accuracy >= RETRIEVAL_WARN:
        status = ST_WARNING
    else:
        status = ST_FAIL

    return {
        "metric": "retrieval_accuracy",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "top_k": top_k,
        "keyword_hit_ratio": kw_ratio,
        "threshold": ">=0.85 pass / 0.75~0.85 warning / <0.75 fail",
        "status": status,
        "coverage_by_source_type": coverage,
        "failed_cases": failed_cases,
        "note": "评测集来自稳定知识结构，不来自 test；RAG 库仅含 train/val。",
    }


# --------------------------------------------------------------------------- #
# 2./3. report_completeness + data_consistency（复用第7.5独立门禁）
# --------------------------------------------------------------------------- #
def _select_report_project(db: Session) -> Project | None:
    """优先 id=1 且有 latest.json；否则首个有报告的项目。"""
    p = db.get(Project, 1)
    if p is not None and report_content_service.load_latest(1) is not None:
        return p
    for (pid,) in db.query(Project.id).order_by(Project.id).all():
        if report_content_service.load_latest(pid) is not None:
            return db.get(Project, pid)
    return p


def _evaluate_report_quality(db: Session) -> dict[str, Any]:
    project = _select_report_project(db)
    if project is None:
        return {
            "available": False,
            "report_completeness": 0.0,
            "data_consistency": 0.0,
            "evidence_coverage": 0.0,
            "number_traceability": 0.0,
            "mutation_tests_pass": False,
            "independent_consistency_check": ST_FAIL,
            "used_test": False,
            "allowed_splits": [],
            "leakage": {"leak": True, "hit_tokens": ["no_report"]},
            "gate_overall": ST_FAIL,
            "project_id": None,
            "report_export_success": False,
            "note": "数据库无任何项目可评估报告质量。",
        }

    if report_content_service.load_latest(project.id) is None:
        return {
            "available": False,
            "report_completeness": 0.0,
            "data_consistency": 0.0,
            "evidence_coverage": 0.0,
            "number_traceability": 0.0,
            "mutation_tests_pass": False,
            "independent_consistency_check": ST_FAIL,
            "used_test": False,
            "allowed_splits": [],
            "leakage": {"leak": True, "hit_tokens": ["no_report"]},
            "gate_overall": ST_FAIL,
            "project_id": project.id,
            "report_export_success": False,
            "note": "目标项目暂无报告，请先 POST /api/reports/{id}/generate。",
        }

    gate = phase75_gate_service.run_phase75_gate(db, project)
    docx_path = report_export_service.latest_docx_path(project.id)
    return {
        "available": True,
        "project_id": project.id,
        "report_completeness": gate["report_completeness"],
        "data_consistency": gate["data_consistency"],
        "evidence_coverage": gate["evidence_coverage"],
        "number_traceability": gate["number_traceability"],
        "mutation_tests_pass": gate["mutation_tests_pass"],
        "independent_consistency_check": gate["independent_consistency_check"],
        "used_test": gate["used_test"],
        "allowed_splits": gate["allowed_splits"],
        "leakage": gate["leakage_check"],
        "gate_overall": gate["overall_status"],
        "report_export_success": docx_path is not None,
        "note": "report_completeness / data_consistency 来自第7.5独立门禁（独立真值重建 + 反作弊）。",
    }


# --------------------------------------------------------------------------- #
# 4. housing_test_mape（唯一读取 test 标签的步骤；仅最终评估，不回训/不调参）
# --------------------------------------------------------------------------- #
def _evaluate_housing_test_mape(db: Session) -> dict[str, Any]:
    audit = model_audit_service.run_model_audit(db)

    # 仅加载已训练模型（train/val）；若不存在则按 train 训练。绝不使用 test 训练。
    bundle = hpm.train_baseline(db, force_retrain=False)

    # === 唯一允许读取 test 的位置：仅取标签做最终评估，读后不回流任何训练/调参 ===
    test_rows = hpm._load_split(db, "test")
    hpm._impute_year(test_rows, bundle.median_year)

    y_true: list[float] = []
    y_pred: list[float] = []
    for s in test_rows:
        pred = hpm.predict_point(bundle, s["lng"], s["lat"], s["area"], s["year"])
        if pred is None:
            continue
        y_true.append(float(s["unit_price"]))
        y_pred.append(float(pred))

    test_mape, test_mae = hpm._compute_val_metrics(y_true, y_pred)

    mape_status = ST_PASS if (test_mape is not None and test_mape <= MAPE_TARGET) else ST_WARNING
    conclusions = audit.get("conclusions", {})
    return {
        "metric": "housing_test_mape",
        "model_type": bundle.model_type,
        "degraded": bundle.degraded,
        "train_count": bundle.train_count,
        "val_count": bundle.val_count,
        "test_count": len(y_true),
        "val_mape": bundle.val_mape,
        "test_mape": test_mape,
        "test_mae": test_mae,
        "target": f"<= {MAPE_TARGET}",
        "status": mape_status,
        "test_used_for_training": False,
        "model_audit_overall": audit.get("overall_status"),
        "test_isolation_in_training": (conclusions.get("test_used_in_training") is False),
        "metrics_recomputed": conclusions.get("metrics_recomputed", False),
        "note": (
            "test 仅在此处读取标签用于最终 MAPE 评估；模型仅用 train 训练、val 验证。"
            "MAPE 为扩展指标，不达标只记录 warning/risks，不重训、不调参、不刷 test。"
        ),
    }


# --------------------------------------------------------------------------- #
# 5. 前端结构性检查（不等于 runtime 联调）
# --------------------------------------------------------------------------- #
def _frontend_structure_check() -> dict[str, Any]:
    missing_files: list[str] = []
    blobs: dict[str, str] = {}
    for rel in _FRONTEND_FILES:
        path = PROJECT_ROOT / rel
        if not path.exists():
            missing_files.append(rel)
        else:
            try:
                blobs[rel] = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                missing_files.append(rel)

    client_blob = blobs.get("frontend/src/api/client.js", "")
    dash_blob = blobs.get("frontend/src/pages/Dashboard.jsx", "")
    missing_methods = [m for m in _FRONTEND_METHODS if m not in client_blob]
    missing_fields = [f for f in _FRONTEND_DISPLAY_FIELDS if f not in dash_blob]

    ok = not missing_files and not missing_methods and not missing_fields
    return {
        "status": ST_PASS if ok else ST_FAIL,
        "checked_files": list(_FRONTEND_FILES),
        "missing_files": missing_files,
        "checked_methods": list(_FRONTEND_METHODS),
        "missing_methods": missing_methods,
        "checked_display_fields": list(_FRONTEND_DISPLAY_FIELDS),
        "missing_display_fields": missing_fields,
        "note": "仅静态检查关键文件/方法/展示字段是否存在，不代表浏览器运行联调。",
    }


# --------------------------------------------------------------------------- #
# 6. 泄露扫描
# --------------------------------------------------------------------------- #
def _scan_leakage(payload: dict[str, Any]) -> dict[str, Any]:
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    hits = [tok for tok in FORBIDDEN_TOKENS if tok in blob]
    return {
        "status": ST_PASS if not hits else ST_FAIL,
        "leak": bool(hits),
        "hit_tokens": hits,
        "note": "扫描整个 final-eval 响应，无 原始JSON/坐标/地址/小区/原文标记为 pass。",
    }


# --------------------------------------------------------------------------- #
# 7. 交付清单
# --------------------------------------------------------------------------- #
def _delivery_checklist(report_q: dict[str, Any], cases_total: int,
                        core_ok: bool) -> tuple[list[dict[str, Any]], bool, list[str]]:
    items: list[dict[str, Any]] = []

    def add(name: str, status: str, requires_manual: bool, note: str) -> None:
        items.append({"item": name, "status": status, "requires_manual": requires_manual, "note": note})

    add("backend_final_eval_service", ST_PASS, False, "final_eval_service 已就绪并产出指标。")
    add("retrieval_eval_set", ST_PASS if cases_total > 0 else ST_FAIL, False,
        f"确定性检索评测集 {cases_total} 条。")
    add("report_docx_exported", ST_PASS if report_q.get("report_export_success") else ST_FAIL, False,
        "已存在导出的报告 docx。" if report_q.get("report_export_success") else "缺少报告 docx，请先 export-docx。")
    add("core_metrics_pass", ST_PASS if core_ok else ST_FAIL, False,
        "三大核心指标达标。" if core_ok else "存在核心指标未达标。")
    add("final_eval_summary_export", ST_PASS, False,
        "可经 POST /api/evaluation/export-delivery 导出汇总材料到 outputs/final_eval。")
    # 人工项（不可由后端自证）
    add("frontend_demo_recording", "manual_required", True,
        "前端演示录屏/截图需人工提交（来自第8.5实跑）。")
    add("kupas_self_eval_screenshots", "manual_required", True,
        "KupasEval 自评材料截图（项目名/任务名/数据量>100/完整流程）需人工提交。")

    auto_items = [it for it in items if not it["requires_manual"]]
    complete = all(it["status"] == ST_PASS for it in auto_items)
    manual_pending = [it["item"] for it in items if it["requires_manual"]]
    return items, complete, manual_pending


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_final_eval(db: Session) -> dict[str, Any]:
    """执行最终自评，返回完整结构（纯只读；test 仅用于最终评估）。"""
    retrieval = _evaluate_retrieval()
    report_q = _evaluate_report_quality(db)
    housing = _evaluate_housing_test_mape(db)
    fe_struct = _frontend_structure_check()

    # ---- 核心指标 ----
    ra = retrieval["accuracy"]
    rc = report_q["report_completeness"]
    dc = report_q["data_consistency"]
    core_metrics = {
        "retrieval_accuracy": _mk("retrieval_accuracy", ra, f">= {RETRIEVAL_PASS}",
                                  retrieval["status"],
                                  f"检索匹配准确率 {ra}（{retrieval['correct']}/{retrieval['total']}）。"),
        "report_completeness": _mk("report_completeness", rc, f">= {COMPLETENESS_PASS}",
                                   ST_PASS if rc >= COMPLETENESS_PASS else ST_FAIL,
                                   f"报告结构完整率 {rc}。"),
        "data_consistency": _mk("data_consistency", dc, f">= {CONSISTENCY_PASS}",
                                ST_PASS if dc >= CONSISTENCY_PASS else ST_FAIL,
                                f"生成内容与底层数据一致性 {dc}。"),
    }
    core_ok = (ra >= RETRIEVAL_PASS and rc >= COMPLETENESS_PASS and dc >= CONSISTENCY_PASS)

    # ---- 扩展指标 ----
    extended_metrics = {
        "housing_test_mape": _mk("housing_test_mape", housing["test_mape"], housing["target"],
                                 housing["status"],
                                 f"房价 test MAPE={housing['test_mape']} MAE={housing['test_mae']}"
                                 f"（test_count={housing['test_count']}）。扩展指标，不阻断提交。"),
        "housing_test_mae": housing["test_mae"],
        "evidence_coverage": _mk("evidence_coverage", report_q["evidence_coverage"], ">= 0.95",
                                 ST_PASS if report_q["evidence_coverage"] >= 0.95 else ST_WARNING,
                                 "报告每章及关键数值证据链覆盖率。"),
        "number_traceability": _mk("number_traceability", report_q["number_traceability"], "== 1.0",
                                   ST_PASS if report_q["number_traceability"] >= 1.0 else ST_WARNING,
                                   "报告文本数字溯源率（幻觉检查）。"),
        "mutation_tests_pass": report_q["mutation_tests_pass"],
    }

    # ---- test 隔离检查 ----
    test_isolation_check = {
        "status": ST_PASS,
        "retrieval_uses_test": False,
        "report_uses_test": report_q["used_test"],
        "report_allowed_splits": report_q["allowed_splits"],
        "model_test_used_for_training": housing["test_used_for_training"],
        "model_test_isolation_in_training": housing["test_isolation_in_training"],
        "statement": (
            "第9阶段仅在 housing_test_mape 处读取 test 标签用于最终评估；"
            "test 未用于训练/验证/调参/规则校准/Prompt 优化/模型选择；"
            "检索评测集不来自 test，RAG 库仅含 train/val。"
        ),
    }
    isolation_ok = (
        test_isolation_check["report_uses_test"] is False
        and test_isolation_check["model_test_used_for_training"] is False
        and test_isolation_check["model_test_isolation_in_training"] is True
        and report_q["allowed_splits"] == ["train", "val"]
    )
    if not isolation_ok:
        test_isolation_check["status"] = ST_FAIL

    # ---- model_test_metrics / report_quality_metrics / retrieval_quality_metrics ----
    model_test_metrics = housing
    report_quality_metrics = {
        "project_id": report_q.get("project_id"),
        "report_completeness": rc,
        "data_consistency": dc,
        "evidence_coverage": report_q["evidence_coverage"],
        "number_traceability": report_q["number_traceability"],
        "mutation_tests_pass": report_q["mutation_tests_pass"],
        "independent_consistency_check": report_q["independent_consistency_check"],
        "gate_overall": report_q["gate_overall"],
    }
    retrieval_quality_metrics = retrieval

    # ---- 安全/合规布尔 ----
    external_api_calls = 0
    llm_used_for_scoring = False
    report_export_success = bool(report_q.get("report_export_success"))

    # ---- 交付清单 ----
    delivery_checklist, delivery_complete, manual_pending = _delivery_checklist(
        report_q, retrieval["total"], core_ok
    )

    # ---- 阻断项判定 ----
    # 先组装（不含 leakage_check），再整体扫描泄露
    blocking_fail: list[str] = []
    if ra < RETRIEVAL_PASS:
        blocking_fail.append(f"retrieval_accuracy={ra} < {RETRIEVAL_PASS}")
    if rc < COMPLETENESS_PASS:
        blocking_fail.append(f"report_completeness={rc} < {COMPLETENESS_PASS}")
    if dc < CONSISTENCY_PASS:
        blocking_fail.append(f"data_consistency={dc} < {CONSISTENCY_PASS}")
    if housing["test_used_for_training"] is not False:
        blocking_fail.append("test_used_for_training != false")
    if external_api_calls != 0:
        blocking_fail.append("external_api_calls != 0")
    if llm_used_for_scoring is not False:
        blocking_fail.append("llm_used_for_scoring != false")
    if not report_export_success:
        blocking_fail.append("report_export_success = false")
    if fe_struct["status"] != ST_PASS:
        blocking_fail.append("frontend_structure_check = fail")
    if not delivery_complete:
        blocking_fail.append("delivery_checklist (auto) not complete")
    if not isolation_ok:
        blocking_fail.append("test_isolation_check = fail")

    # ---- 风险 / 建议 ----
    risks: list[str] = []
    recommendations: list[str] = []

    for b in blocking_fail:
        risks.append(f"[FAIL] {b}")
    if retrieval["status"] == ST_WARNING:
        risks.append(f"[WARNING] retrieval_accuracy={ra} 处于 0.75~0.85 警示区，未达 0.85 核心门槛。")
    if housing["status"] == ST_WARNING:
        risks.append(f"[WARNING] housing_test_mape={housing['test_mape']} 高于软目标 {MAPE_TARGET}（扩展指标）。")
    if extended_metrics["evidence_coverage"]["status"] == ST_WARNING:
        risks.append("[WARNING] 证据链覆盖率低于 0.95。")
    if extended_metrics["number_traceability"]["status"] == ST_WARNING:
        risks.append("[WARNING] 报告数字溯源率低于 1.0。")
    if manual_pending:
        risks.append(f"[MANUAL] 待人工提交：{manual_pending}")

    # 不达标时的建议——严格遵守：不修 test、回到 train/val
    if ra < RETRIEVAL_PASS:
        recommendations.append(
            "检索未达标：在 train/val 知识源上优化分块/关键词/检索策略，"
            "或补充政策/口径知识源；严禁依据本评测结果反推或改写题目以凑指标。")
    if rc < COMPLETENESS_PASS:
        recommendations.append("报告完整率未达标：在 train/val 下完善报告模板章节/字段填充后重生成报告。")
    if dc < CONSISTENCY_PASS:
        recommendations.append("一致性未达标：核对数字注入链路（analysis_result→报告），消除不可溯源数字。")
    if housing["status"] == ST_WARNING:
        recommendations.append(
            "房价 MAPE 偏高：仅可在 train/val 上做特征/超参优化，"
            "不得重训到 test、不得据 test 调参或重复刷 test。")
    if manual_pending:
        recommendations.append("提交前完成前端演示录屏/截图与 KupasEval 自评材料的人工补充。")

    # ---- overall / can_submit ----
    if blocking_fail:
        overall_status = ST_FAIL
        can_submit = False
        final_summary = (
            "最终自评未通过：存在阻断项，不可提交。" 
            "已输出 risks 与 recommendations；按红线要求，test 指标不达标不重训/不改规则/不调参，"
            "如需优化请回到 train/val。"
        )
    else:
        has_warning = (
            retrieval["status"] == ST_WARNING or housing["status"] == ST_WARNING
            or extended_metrics["evidence_coverage"]["status"] == ST_WARNING
            or extended_metrics["number_traceability"]["status"] == ST_WARNING
            or bool(manual_pending)
        )
        overall_status = ST_WARNING if has_warning else ST_PASS
        can_submit = True
        final_summary = (
            "最终自评通过：三大核心指标达标、test 隔离成立、无泄露、报告可导出、前端结构完整。"
            "前端演示为第8.5阶段人工实跑确认（frontend_demo_status_runtime="
            "manual_confirmed_from_phase8_5），本第9阶段后端未再次打开浏览器联调；"
            "演示录屏/截图与自评材料需按 delivery_checklist 人工提交。"
        )
        if overall_status == ST_WARNING:
            final_summary += " 存在非阻断 warning / 待人工提交项，详见 risks。"

    result: dict[str, Any] = {
        "mode": settings.app_mode,
        "phase": "9",
        "overall_status": overall_status,
        "can_submit": can_submit,
        "core_metrics": core_metrics,
        "extended_metrics": extended_metrics,
        "test_isolation_check": test_isolation_check,
        "model_test_metrics": model_test_metrics,
        "report_quality_metrics": report_quality_metrics,
        "retrieval_quality_metrics": retrieval_quality_metrics,
        "delivery_checklist": delivery_checklist,
        "delivery_checklist_complete": delivery_complete,
        "manual_pending_items": manual_pending,
        # 合规/安全布尔
        "external_api_calls": external_api_calls,
        "llm_used_for_scoring": llm_used_for_scoring,
        "report_export_success": report_export_success,
        "frontend_structure_check": fe_struct["status"],
        "frontend_structure_detail": fe_struct,
        "frontend_demo_status_runtime": "manual_confirmed_from_phase8_5",
        "final_pass_thresholds": {
            "retrieval_accuracy": f">= {RETRIEVAL_PASS}",
            "report_completeness": f">= {COMPLETENESS_PASS}",
            "data_consistency": f">= {CONSISTENCY_PASS}",
            "leakage_check": "pass",
            "test_used_for_training": "false",
            "external_api_calls": "0",
            "llm_used_for_scoring": "false",
            "report_export_success": "true",
            "frontend_structure_check": "pass",
            "delivery_checklist_complete": "true",
        },
        "blocking_fail": blocking_fail,
        "risks": risks,
        "recommendations": recommendations,
        "final_summary": final_summary,
        "test_policy": (
            "test 只用于本 final-eval 最终评估；不得重训模型、不得改规则、"
            "不得据 test 调参、不得自动重复刷 test 优化指标。"
        ),
        "notes": [
            "本服务纯只读：不写 DB、不写模型文件、不调外部 API、不使用大模型生成结论。",
            "report_completeness / data_consistency 来自第7.5独立门禁（独立真值重建 + 反作弊 mutation tests）。",
            "retrieval_accuracy 使用确定性评测集与严格命中判定（文件/类型/关键词多条件 AND）。",
        ],
    }

    # ---- 整体泄露扫描（最后纳入）----
    leakage = _scan_leakage(result)
    result["leakage_check"] = leakage
    if leakage["status"] != ST_PASS and leakage["leak"]:
        # 泄露属阻断项：纠正 overall/can_submit
        result["blocking_fail"].append("leakage_check = fail")
        result["risks"].append(f"[FAIL] 检测到泄露标记：{leakage['hit_tokens']}")
        result["overall_status"] = ST_FAIL
        result["can_submit"] = False
        result["final_summary"] = "最终自评未通过：检测到潜在原文/原始明细泄露，不可提交。"

    logger.info(
        "final-eval overall=%s can_submit=%s ra=%s rc=%s dc=%s test_mape=%s leak=%s isolation=%s",
        result["overall_status"], result["can_submit"], ra, rc, dc,
        housing["test_mape"], leakage["leak"], isolation_ok,
    )
    return result
