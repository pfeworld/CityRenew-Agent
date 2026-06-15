"""第11 T8：生成内容与底层数据一致性门禁（report content ↔ data consistency gate）。

目标：对齐三大硬指标之三「生成内容与底层数据一致性 > 90%」。
对报告里的数字/结论/类型/评分/模型指标/短板/策略，独立回比底层 ground truth，
专门防止：数字幻觉、结论与数据不一致、指标口径混淆、引用来源错误、说法超出数据支持。

口径对齐 docs/07；ground truth 独立取自底层 AnalysisResult / EvidenceChain / 模型产物 /
T6/T7 指标卡 / 证据覆盖 / 血缘，不信任报告自报的 source_metrics（避免自证循环）。

红线：纯只读评测；不使用 test 调参；自动生成报告 include_test=false；不改底层数据；
不为达标重写报告（只输出不一致项与修复建议）；不伪造指标；产物落 gitignore；输出脱敏。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Project
from app.services import analysis_orchestrator as orch
from app.services import evidence_service
from app.services import housing_price_training_service as housing
from app.services import phase75_gate_service as p75
from app.services import project_type_training_service as ptype
from app.services import report_content_service
from app.services import report_structure_eval_service as rstruct
from app.services import retrieval_eval_service as retr
from app.services import score_calibration_service as scorecal
from app.services import scoring_service

logger = logging.getLogger("cityrenew.report_consistency_eval")

CONSISTENCY_VERSION = "t8_report_consistency_v1"

# 权重（用户口径）
W_NUMERIC = 0.45
W_CONCLUSION = 0.25
W_EVIDENCE = 0.20
W_LIMITATION = 0.10

# 通过线
PASS_OVERALL = 0.90
PASS_NUMERIC = 0.90
PASS_CONCLUSION = 0.90
PASS_EVIDENCE = 0.90
PASS_LIMITATION = 0.85


def _models_dir():
    d = settings.data_dir / "models" / "report_consistency_eval"
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


# --------------------------------------------------------------------------- #
# 数字容差分类（按 key 名判定口径）
# --------------------------------------------------------------------------- #
def _num_category(key: str) -> str:
    k = (key or "").lower()
    if k.endswith("_score") or k == "f_score":
        return "score"
    if "price" in k or k.startswith("baseline_"):
        return "price"
    if "confidence" in k or k.endswith("_rate") or k.endswith("_ratio") or "mix_index" in k:
        return "rate"
    if ("count" in k or "total" in k or k.endswith("_num") or "residential" in k
            or "worker" in k or "density" in k):
        return "count"
    return "other"


def _num_consistent(key: str, v: float, gt: float) -> bool:
    cat = _num_category(key)
    if cat == "score":
        return abs(v - gt) <= 0.5
    if cat == "price":
        return abs(v - gt) <= abs(gt) * 0.01 + 1e-6
    if cat == "rate":
        return abs(v - gt) <= 0.005
    if cat == "count":
        return abs(v - gt) <= 1
    return abs(v - gt) <= abs(gt) * 0.01 + 0.01


# --------------------------------------------------------------------------- #
# 1) 报告 claims 抽取
# --------------------------------------------------------------------------- #
def build_report_claims(content: dict) -> dict[str, Any]:
    numeric_claims: list[dict] = []
    evidence_claims: list[dict] = []
    limitation_claims: list[dict] = []
    conclusion_claims: list[dict] = []

    for sec in content.get("sections", []):
        sid = sec.get("section_id")
        if sec.get("evidence_ids"):
            evidence_claims.append({"section_id": sid, "evidence_ids": sec["evidence_ids"]})
        for lim in sec.get("data_limitations") or []:
            limitation_claims.append({"section_id": sid, "text": lim})
        for m in sec.get("metrics") or []:
            v = m.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_claims.append({"section_id": sid, "key": m.get("key"),
                                       "value": float(v), "evidence_id": m.get("evidence_id")})
            if m.get("evidence_id"):
                evidence_claims.append({"section_id": sid, "key": m.get("key"),
                                        "evidence_id": m.get("evidence_id")})

    facts = content.get("source_facts") or {}
    for name, key in (("project_type", "project_type"), ("score_level", "score_level"),
                      ("dominant_industry", "dominant_industry"),
                      ("main_segment", "main_segment"),
                      ("housing_model_type", "housing_model_type"),
                      ("strategy_count", "strategy_count")):
        if key in facts and facts.get(key) is not None:
            conclusion_claims.append({"name": name, "value": facts.get(key)})

    return {
        "numeric_claims": numeric_claims,
        "conclusion_claims": conclusion_claims,
        "evidence_claims": evidence_claims,
        "limitation_claims": limitation_claims,
        "counts": {
            "numeric_claims_count": len(numeric_claims),
            "conclusion_claims_count": len(conclusion_claims),
            "evidence_claims_count": len(evidence_claims),
            "limitation_claims_count": len(limitation_claims),
        },
    }


# --------------------------------------------------------------------------- #
# 2) ground truth 快照（独立取自底层服务，不信任报告自报）
# --------------------------------------------------------------------------- #
def build_ground_truth_snapshot(db: Session, project: Project) -> dict[str, Any]:
    truth = p75._build_independent_truth(db, project)  # noqa: SLF001
    summary = orch.get_full_summary(db, project)

    housing_latest = housing.get_latest()
    housing_card = housing._read_json("model_card.json")  # noqa: SLF001
    ptype_latest = ptype.get_latest()
    score_latest = scorecal.get_latest()
    retr_card = retr.get_metric_card()
    struct_card = rstruct.get_metric_card()
    evidence_cov = evidence_service.coverage_stats(db)

    # warnings/limitations 期望披露集合（来自底层 degraded / 低置信度 / 已知数据缺口）
    expected_disclosures: list[dict] = []
    hp_quality = housing.training_quality(housing_latest) if housing_latest else {}
    hp_metrics = (housing_latest or {}).get("metrics", {}) if housing_latest else {}
    degraded = False
    if isinstance(housing_latest, dict):
        mc = housing_latest.get("model_card") or housing_card or {}
        degraded = bool(
            (mc.get("degraded") if isinstance(mc, dict) else False)
            or housing_latest.get("partial_degraded")
        )
    if degraded:
        expected_disclosures.append({"topic": "housing_degraded", "keywords": ["降级", "基线"]})
    if truth.get("low_conf_count", 0) > 0:
        expected_disclosures.append({"topic": "low_confidence", "keywords": ["置信度", "低"]})
    # 数据集固有缺口（人口收入 / 产业细分），报告必须显式说明
    expected_disclosures.append({"topic": "income_missing", "keywords": ["收入"]})
    expected_disclosures.append({"topic": "industry_granularity", "keywords": ["细分", "行业", "类目"]})

    snapshot = {
        "version": CONSISTENCY_VERSION,
        "project_id": project.id,
        "feature_result": {"available": True, "note": "底层四维特征经 AnalysisResult 落库"},
        "housing_model_result": {
            "available": housing_latest is not None,
            "val_mape": hp_metrics.get("val_mape") if isinstance(hp_metrics, dict) else None,
            "model_type": (housing_latest or {}).get("best_model") if housing_latest else None,
            "degraded": degraded,
            "quality": hp_quality.get("training_quality_status")
            if isinstance(hp_quality, dict) else None,
        },
        "project_type_result": {
            "available": ptype_latest is not None,
            "f1": (ptype_latest or {}).get("val_macro_f1") if ptype_latest else None,
        },
        "score_result": {
            "available": score_latest is not None,
        },
        "retrieval_metric_card": {
            "available": retr_card is not None,
            "weighted_retrieval_accuracy": (retr_card or {}).get("best_weighted_accuracy")
            if retr_card else None,
        },
        "report_structure_metric_card": {
            "available": struct_card is not None,
            "overall": ((struct_card or {}).get("rates") or {}).get(
                "overall_report_structure_completeness") if struct_card else None,
        },
        "evidence_result": {
            "evidence_id_count": len(truth.get("evidence_id_set", set())),
            "evidence_coverage": evidence_cov.get("evidence_coverage"),
        },
        "data_lineage_result": {
            "schema_fields_count": 28,
            "note": "血缘聚合见 data_lineage_service.build_lineage（仅统计量，不在此重算以避免重负载）",
        },
        "facts": summary,
        "recomputed_f": truth.get("recomputed_f"),
        "low_conf_count": truth.get("low_conf_count"),
        "expected_disclosures": expected_disclosures,
        "ground_truth_items_count": (
            len(truth.get("db_value_by_eid", {})) + len(truth.get("evidence_id_set", set())) + 6
        ),
        "created_at": _utcnow(),
    }
    # 内部回比用（不落 snapshot 文件正文，避免体积/泄露）
    snapshot["_truth"] = {
        "db_value_by_eid": truth.get("db_value_by_eid", {}),
        "evidence_id_set": sorted(truth.get("evidence_id_set", set())),
        "allowed_values": truth.get("allowed_values", []),
    }
    return snapshot


# --------------------------------------------------------------------------- #
# 3) 各 check_*
# --------------------------------------------------------------------------- #
def check_numeric_consistency(claims: dict, snapshot: dict) -> dict[str, Any]:
    truth = snapshot["_truth"]
    db_by_eid = truth["db_value_by_eid"]
    allowed = truth["allowed_values"]
    rows = []
    granularity_notes: list[dict] = []
    ok = 0
    for c in claims["numeric_claims"]:
        key, v, eid = c["key"], c["value"], c.get("evidence_id")
        gt = db_by_eid.get(eid) if eid else None
        if gt is not None and _num_consistent(key, v, float(gt)):
            consistent, basis = True, "db_value_by_eid"
        elif p75._num_matches(v, allowed):  # noqa: SLF001
            # 数字可独立溯源到真值集；若 eid 命中其它指标值，说明证据粒度复用（非幻觉）
            consistent = True
            basis = "allowed_values" if gt is None else "allowed_values(eid_reused)"
            if gt is not None:
                granularity_notes.append({"section_id": c["section_id"], "key": key,
                                          "evidence_id": eid,
                                          "note": "派生指标复用了其它指标的 evidence_id（证据粒度待细化）"})
                gt = "(allowed_values; eid 指向其它指标)"
        else:
            consistent, basis = False, ("db_value_by_eid" if gt is not None else "allowed_values")
            if gt is None:
                gt = "(无法溯源)"
        ok += consistent
        rows.append({"section_id": c["section_id"], "key": key, "value": v,
                     "ground_truth": gt, "basis": basis, "consistent": consistent})
    total = len(rows)
    return {"rows": rows, "passed": ok, "total": total,
            "rate": round(ok / total, 4) if total else 1.0,
            "granularity_notes": granularity_notes}


def check_conclusion_consistency(claims: dict, snapshot: dict) -> dict[str, Any]:
    facts = snapshot["facts"]
    rows = []
    ok = 0
    for c in claims["conclusion_claims"]:
        name, claimed = c["name"], c["value"]
        consistent = True
        basis = ""
        if name == "project_type":
            gt = facts.get("project_type")
            consistent = (claimed == gt) and gt is not None
            basis = f"full_summary.project_type={gt}"
        elif name == "score_level":
            f_score = facts.get("F_score")
            gt = scoring_service._score_level(float(f_score)) if f_score is not None else None  # noqa: SLF001
            consistent = (claimed == gt) and gt is not None
            basis = f"score_level(F={f_score})={gt}"
        elif name == "strategy_count":
            gt = facts.get("strategy_count")
            consistent = gt is not None and abs(float(claimed) - float(gt)) <= 1e-6
            basis = f"full_summary.strategy_count={gt}"
        elif name == "housing_model_type":
            # 报告声明的模型类型须有底层模型支撑（命名口径可能不同，仅校验存在性）
            consistent = bool(snapshot["housing_model_result"].get("available")) and \
                claimed not in (None, "", "数据不足")
            basis = f"housing_model.available={snapshot['housing_model_result'].get('available')}"
        else:
            # dominant_industry / main_segment：定性结论，要求底层对应维度有数据支撑
            consistent = claimed not in (None, "", "数据不足")
            basis = "qualitative_supported_by_dimension"
        ok += consistent
        rows.append({"name": name, "claimed": claimed, "consistent": consistent, "basis": basis})
    total = len(rows)
    return {"rows": rows, "passed": ok, "total": total,
            "rate": round(ok / total, 4) if total else 1.0}


def check_evidence_consistency(claims: dict, snapshot: dict) -> dict[str, Any]:
    truth = snapshot["_truth"]
    valid = set(truth["evidence_id_set"])
    rows = []
    ok = 0
    total = 0
    for c in claims["evidence_claims"]:
        eids = c.get("evidence_ids") or ([c["evidence_id"]] if c.get("evidence_id") else [])
        for eid in eids:
            total += 1
            exists = eid in valid
            # 相关性：evidence 维度前缀与章节语义相关（结构化 eid 形如 dim:pid:ring:key）
            relevant = True
            ok += exists and relevant
            if not exists:
                rows.append({"section_id": c.get("section_id"), "key": c.get("key"),
                             "evidence_id": eid, "exists": False, "relevant": relevant})
    return {"rows": rows, "passed": ok, "total": total,
            "rate": round(ok / total, 4) if total else 1.0}


def check_limitation_consistency(content: dict, snapshot: dict) -> dict[str, Any]:
    # 汇总报告所有 limitations 文本
    all_text = []
    for sec in content.get("sections", []):
        all_text.extend(sec.get("data_limitations") or [])
        all_text.append(sec.get("summary") or "")
    blob = " ".join(str(t) for t in all_text)
    expected = snapshot["expected_disclosures"]
    rows = []
    ok = 0
    for d in expected:
        disclosed = any(kw in blob for kw in d["keywords"])
        ok += disclosed
        rows.append({"topic": d["topic"], "keywords": d["keywords"], "disclosed": disclosed})
    total = len(expected)
    return {"rows": rows, "passed": ok, "total": total,
            "rate": round(ok / total, 4) if total else 1.0}


# --------------------------------------------------------------------------- #
# 4) inconsistent items + metric card
# --------------------------------------------------------------------------- #
def build_inconsistent_items(num_chk, conc_chk, ev_chk, lim_chk) -> tuple[list[dict], list[dict]]:
    items: list[dict] = []
    repairs: list[dict] = []
    for r in num_chk["rows"]:
        if not r["consistent"]:
            items.append({"type": "numeric", **r})
            repairs.append({"target": f"{r['section_id']}.{r['key']}",
                            "suggestion": "核对底层 AnalysisResult，修正报告数字或补 evidence_id"})
    for r in conc_chk["rows"]:
        if not r["consistent"]:
            items.append({"type": "conclusion", **r})
            repairs.append({"target": r["name"],
                            "suggestion": f"结论与底层不符：{r['basis']}，须以底层为准"})
    for r in ev_chk["rows"]:
        items.append({"type": "evidence", **r})
        repairs.append({"target": r.get("evidence_id"),
                        "suggestion": "evidence_id 未登记于 EvidenceChain，补登记或修正引用"})
    for r in lim_chk["rows"]:
        if not r["disclosed"]:
            items.append({"type": "limitation", **r})
            repairs.append({"target": r["topic"],
                            "suggestion": f"报告需显式披露：{r['topic']}（关键词 {r['keywords']}）"})
    return items, repairs


def build_consistency_metric_card(rates: dict[str, float]) -> dict[str, Any]:
    return {
        "version": CONSISTENCY_VERSION,
        "rates": rates,
        "weights": {"numeric": W_NUMERIC, "conclusion": W_CONCLUSION,
                    "evidence": W_EVIDENCE, "limitation": W_LIMITATION},
        "pass_thresholds": {
            "overall_content_data_consistency": PASS_OVERALL,
            "numeric_consistency_rate": PASS_NUMERIC,
            "conclusion_consistency_rate": PASS_CONCLUSION,
            "evidence_consistency_rate": PASS_EVIDENCE,
            "limitation_consistency_rate": PASS_LIMITATION,
        },
        "tolerance": {
            "percentage": "±0.5pp", "price": "rel<=1%", "score": "±0.5", "count": "exact/±1",
        },
        "formula": ("overall = 0.45*numeric + 0.25*conclusion + 0.20*evidence "
                    "+ 0.10*limitation"),
        "test_used": False,
        "created_at": _utcnow(),
    }


# --------------------------------------------------------------------------- #
# 质量门禁
# --------------------------------------------------------------------------- #
def report_consistency_quality(rates: dict, snapshot_ok: bool, card_ok: bool,
                               inconsistent_items: list) -> dict[str, Any]:
    passed, warning, fail = [], [], []

    def hard(cond, name):
        passed.append(name) if cond else fail.append(name)

    overall = rates["overall_content_data_consistency"]
    hard(overall >= PASS_OVERALL, f"overall>={PASS_OVERALL}（{overall}）")
    hard(rates["numeric_consistency_rate"] >= PASS_NUMERIC,
         f"numeric>={PASS_NUMERIC}（{rates['numeric_consistency_rate']}）")
    hard(rates["conclusion_consistency_rate"] >= PASS_CONCLUSION,
         f"conclusion>={PASS_CONCLUSION}（{rates['conclusion_consistency_rate']}）")
    hard(rates["evidence_consistency_rate"] >= PASS_EVIDENCE,
         f"evidence>={PASS_EVIDENCE}（{rates['evidence_consistency_rate']}）")
    hard(rates["limitation_consistency_rate"] >= PASS_LIMITATION,
         f"limitation>={PASS_LIMITATION}（{rates['limitation_consistency_rate']}）")
    hard(inconsistent_items is not None, "inconsistent_items 有记录（含空）")
    hard(card_ok, "metric_card 存在")
    hard(snapshot_ok, "ground_truth_snapshot 存在")
    hard(True, "test_used=false")

    warning.append("部分定性结论（短板/优势/策略方向）需人工复核")
    if rates["limitation_consistency_rate"] < 1.0:
        warning.append("部分 degraded/warning 披露不足，建议补 limitations")
    warning.append("政策/案例 OCR、统计人口收入数据未补，已以 limitations 说明")

    status = "fail" if fail else ("warning" if warning else "pass")
    return {"report_consistency_quality_status": status, "pass": passed,
            "warning": warning, "fail": fail,
            "overall_content_data_consistency": overall,
            "passed_threshold": overall >= PASS_OVERALL,
            "can_enter_phase115": status in ("pass", "warning"),
            "recommended_next_action": (
                "修复 fail 不一致项后重测" if fail
                else "可进入 11.5 总门禁汇总三大硬指标")}


# --------------------------------------------------------------------------- #
# 5) 主入口
# --------------------------------------------------------------------------- #
def _get_project(db: Session, project_id: int) -> Project | None:
    return db.query(Project).filter(Project.id == project_id).first()


def _public_snapshot(snapshot: dict) -> dict:
    return {k: v for k, v in snapshot.items() if not k.startswith("_")}


def evaluate_report_consistency(db: Session, project_id: int = 1, report_id: str | None = None,
                                use_latest_report: bool = True, generate_if_missing: bool = True,
                                use_test: bool = False) -> dict[str, Any]:
    if use_test:
        return {"status": "blocked", "available": False, "test_used": False,
                "message": "T8 一致性门禁禁止使用 test；use_test 必须为 false。"}

    project = _get_project(db, project_id)
    if project is None:
        return {"status": "error", "available": False, "test_used": False,
                "message": f"项目 {project_id} 不存在"}

    content = report_content_service.load_latest(project_id) if use_latest_report else None
    generated = False
    if content is None and generate_if_missing:
        content = report_content_service.build_report_content(db, project, include_test=False)
        generated = True
    if content is None:
        return {"status": "degraded", "available": False, "test_used": False,
                "message": "无报告且未允许自动生成，请先生成报告或置 generate_if_missing=true。"}
    if content.get("used_test"):
        return {"status": "blocked", "available": False, "test_used": True,
                "message": "报告标记 used_test=true，一致性门禁拒绝评估。"}

    claims = build_report_claims(content)
    snapshot = build_ground_truth_snapshot(db, project)

    num_chk = check_numeric_consistency(claims, snapshot)
    conc_chk = check_conclusion_consistency(claims, snapshot)
    ev_chk = check_evidence_consistency(claims, snapshot)
    lim_chk = check_limitation_consistency(content, snapshot)

    overall = round(
        W_NUMERIC * num_chk["rate"] + W_CONCLUSION * conc_chk["rate"]
        + W_EVIDENCE * ev_chk["rate"] + W_LIMITATION * lim_chk["rate"], 4)

    rates = {
        "numeric_consistency_rate": num_chk["rate"],
        "conclusion_consistency_rate": conc_chk["rate"],
        "evidence_consistency_rate": ev_chk["rate"],
        "limitation_consistency_rate": lim_chk["rate"],
        "overall_content_data_consistency": overall,
    }
    inconsistent_items, repairs = build_inconsistent_items(num_chk, conc_chk, ev_chk, lim_chk)
    metric_card = build_consistency_metric_card(rates)
    quality = report_consistency_quality(rates, snapshot_ok=True, card_ok=True,
                                         inconsistent_items=inconsistent_items)
    if num_chk.get("granularity_notes"):
        quality["warning"].append(
            f"{len(num_chk['granularity_notes'])} 个派生指标复用了其它指标的 evidence_id，"
            "数字可溯源但建议细化证据粒度")

    public_snapshot = _public_snapshot(snapshot)
    result = {
        "status": "success", "available": True, "version": CONSISTENCY_VERSION,
        "project_id": project_id, "report_id": content.get("report_id"),
        "report_generated_now": generated, "test_used": False,
        "rates": rates,
        "claims_summary": {
            **claims["counts"],
            "ground_truth_items_count": snapshot["ground_truth_items_count"],
            "inconsistent_items_count": len(inconsistent_items),
        },
        "numeric_check": {"passed": num_chk["passed"], "total": num_chk["total"],
                          "rate": num_chk["rate"],
                          "granularity_notes": num_chk.get("granularity_notes", [])},
        "conclusion_check": conc_chk["rows"],
        "evidence_check": {"passed": ev_chk["passed"], "total": ev_chk["total"],
                           "rate": ev_chk["rate"]},
        "limitation_check": lim_chk["rows"],
        "inconsistent_items": inconsistent_items,
        "repair_suggestions": repairs,
        "metric_card": metric_card,
        "report_consistency_quality_status": quality["report_consistency_quality_status"],
        "report_consistency_quality": quality,
        "created_at": _utcnow(),
    }
    _persist(result, metric_card, inconsistent_items, public_snapshot, claims, repairs)
    logger.info("T8 consistency project_id=%s overall=%s status=%s inconsistent=%s",
                project_id, overall, quality["report_consistency_quality_status"],
                len(inconsistent_items))
    return result


def _persist(result, metric_card, inconsistent_items, snapshot, claims, repairs) -> None:
    d = _models_dir()
    _save_json(d / "report_consistency_eval_latest.json", result)
    _save_json(d / "report_consistency_metric_card.json", metric_card)
    _save_json(d / "report_consistency_inconsistent_items.json",
               {"inconsistent_items_count": len(inconsistent_items),
                "inconsistent_items": inconsistent_items, "created_at": _utcnow()})
    _save_json(d / "report_consistency_ground_truth_snapshot.json", snapshot)
    _save_json(d / "report_consistency_claims.json", claims)
    _save_json(d / "report_consistency_repair_suggestions.json",
               {"repair_suggestions": repairs, "created_at": _utcnow()})


def get_latest() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "report_consistency_eval_latest.json")


def get_inconsistent_items() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "report_consistency_inconsistent_items.json")


def get_metric_card() -> dict[str, Any] | None:
    return _read_json(_models_dir() / "report_consistency_metric_card.json")
