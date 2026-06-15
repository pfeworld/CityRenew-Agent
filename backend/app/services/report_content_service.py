"""报告内容生成（第7阶段，确定性模板 + 结构化数据填充，绝不使用大模型）。

职责：
- 基于第6阶段 run_full_analysis 的结构化结果，组装固定 9 章报告内容 JSON。
- 每章含 section_id / title / summary / key_findings / metrics / evidence_ids /
  data_limitations 七字段；缺数据章节亦保留，并写"数据不足/暂无法判断"。
- 同时产出 source_metrics（canonical 数字字典，供第7.5一致性回比）与 source_facts
  （类型/档位/策略数等非数值事实），全部来自 AnalysisResult / full analysis / Project。
- 报告内容落盘到 backend/data/outputs/reports/{project_id}/（已 gitignore）。

红线：
- 仅 train/val（include_test 恒 false）；不触碰 test；不调外部 API；不使用大模型。
- 报告所有数字来自前序确定性计算；每章带 evidence_id；缺数据显式标注，不编造。
- 不返回 raw_json / 原始点位 / 企业名 / 小区名 / 地址明细（仅脱敏统计量与类别名）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Project
from app.services import analysis_common as ac
from app.services import analysis_orchestrator as orch

logger = logging.getLogger("cityrenew.report.content")

REQUIRED_SECTIONS = 9


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _ring(rings: list[dict] | None, ring_name: str) -> dict:
    for r in rings or []:
        if r.get("ring") == ring_name:
            return r
    return {}


def _eid(dimension: str, project_id: int, ring: str | None, metric_key: str) -> str:
    """复算落库时使用的 evidence_id（与 analysis_common.record_metric 一致）。"""
    return ac.make_evidence_id(dimension, project_id, ring or "all", metric_key)


def _num(value: Any) -> str:
    """数字展示；None -> 数据不足。

    关键：保真展示，不用 {:g}（其 6 位有效数字会改写 91676.68→91676.7，破坏可溯源）。
    浮点按最多 4 位小数去尾零展示，与落库 round(.,2/3/4) 一致，确保数字回比可命中。
    """
    if value is None:
        return "数据不足"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _metric(
    key: str,
    label: str,
    value: Any,
    unit: str | None,
    evidence_id: str | None,
    source: dict[str, Any],
) -> dict[str, Any]:
    """构造一条章节指标并登记到 source_metrics（仅数值型登记回比）。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        source[key] = round(float(value), 4)
    return {
        "key": key,
        "label": label,
        "value": value,
        "unit": unit,
        "evidence_id": evidence_id,
    }


def _dedup(items: list[str]) -> list[str]:
    out: list[str] = []
    for it in items:
        if it and it not in out:
            out.append(it)
    return out


# --------------------------------------------------------------------------- #
# 章节组装
# --------------------------------------------------------------------------- #
def _sec_overview(project: Project, full: dict, src: dict, facts: dict) -> dict:
    pid = project.id
    ptype = full.get("project_type")
    f_score = full.get("F_score")
    level = full.get("score_level")
    facts["project_type"] = ptype
    facts["score_level"] = level

    type_eid = _eid("classification", pid, None, "project_type")
    f_eid = _eid("scoring", pid, None, "F_score")

    metrics = [
        _metric("F_score", "综合评分 F_score", f_score, "分", f_eid, src),
    ]
    summary = (
        f"项目「{project.name}」位于{project.city or '—'}{project.district or ''}，"
        f"识别类型为「{ptype or '数据不足'}」，综合评分 F_score={_num(f_score)}（{level or '数据不足'}）。"
        "本报告基于本地结构化数据确定性生成，数字均可溯源。"
    )
    key_findings = [
        f"项目类型：{ptype or '数据不足'}。",
        f"综合评分：{_num(f_score)}（{level or '数据不足'}）。",
        f"圈层口径：核心/近邻{project.nearby_buffer_m}m/辐射{project.radiation_buffer_m}m。",
    ]
    limitations = []
    if not project.land_use:
        limitations.append("项目用地性质字段缺失，类型识别置信度受限。")
    if not limitations:
        limitations.append("项目基础信息较完整，无重大数据局限。")
    return {
        "section_id": "S1",
        "title": "项目概况",
        "summary": summary,
        "key_findings": key_findings,
        "metrics": metrics,
        "evidence_ids": _dedup([f_eid, type_eid]),
        "data_limitations": limitations,
    }


def _sec_data_scope(project: Project, full: dict, src: dict, facts: dict) -> dict:
    pid = project.id
    four = full.get("four_dimension", {})
    poi = four.get("poi") or {}
    pop = four.get("population") or {}
    housing = four.get("housing") or {}
    industry = four.get("industry") or {}

    poi_rad = _ring(poi.get("rings"), ac.RING_RADIATION)
    pop_rad = _ring(pop.get("rings"), ac.RING_RADIATION)
    house_rad = _ring(housing.get("rings"), ac.RING_RADIATION)
    ind_rad = _ring(industry.get("rings"), ac.RING_RADIATION)

    metrics = [
        _metric("poi_total_radiation", "辐射圈POI总数", poi_rad.get("total"), "个",
                _eid("poi", pid, ac.RING_RADIATION, "poi_total"), src),
        _metric("pop_grid_count_radiation", "辐射圈人口网格数", pop_rad.get("grid_count"), "个",
                _eid("population", pid, ac.RING_RADIATION, "pop_grid_count"), src),
        _metric("housing_sample_count_radiation", "辐射圈房价样本数", house_rad.get("sample_count"),
                "套", _eid("housing", pid, ac.RING_RADIATION, "housing_sample_count"), src),
        _metric("enterprise_count_radiation", "辐射圈企业数", ind_rad.get("enterprise_count"),
                "家", _eid("industry", pid, ac.RING_RADIATION, "enterprise_count"), src),
    ]
    allowed = full.get("allowed_splits", [])
    summary = (
        "分析覆盖 POI 配套、人口画像、房价交易、产业点位四类本地结构化数据，"
        f"按核心/近邻{project.nearby_buffer_m}m/辐射{project.radiation_buffer_m}m 三圈层归集；"
        f"数据划分仅使用 {('/'.join(allowed)) or 'train/val'}（不含 test）。"
    )
    key_findings = [
        f"辐射圈数据规模：POI {_num(poi_rad.get('total'))} 个、"
        f"人口网格 {_num(pop_rad.get('grid_count'))} 个、"
        f"房价样本 {_num(house_rad.get('sample_count'))} 套、"
        f"企业 {_num(ind_rad.get('enterprise_count'))} 家。",
        "所有指标均带 evidence_id，可回溯到底层记录与计算过程。",
    ]
    limitations = []
    if not poi_rad.get("total"):
        limitations.append("辐射圈 POI 记录不足，区位配套结论置信度有限。")
    if not house_rad.get("sample_count"):
        limitations.append("辐射圈房价样本不足，价值结论置信度有限。")
    if not limitations:
        limitations.append("四类数据均有覆盖，数据范围充分。")
    evid = [m["evidence_id"] for m in metrics]
    return {
        "section_id": "S2",
        "title": "数据来源与分析范围",
        "summary": summary,
        "key_findings": key_findings,
        "metrics": metrics,
        "evidence_ids": _dedup(evid),
        "data_limitations": limitations,
    }


def _sec_poi(project: Project, full: dict, src: dict, facts: dict) -> dict:
    pid = project.id
    poi = (full.get("four_dimension", {}).get("poi")) or {}
    rad = _ring(poi.get("rings"), ac.RING_RADIATION)
    nearby = _ring(poi.get("rings"), ac.RING_NEARBY)
    l_score = poi.get("score")
    shortboards = poi.get("shortboards_top5") or []
    recommend = poi.get("recommend_top5") or []

    metrics = [
        _metric("L_score", "区位配套评分 L_score", l_score, "分",
                _eid("poi", pid, None, "L_score"), src),
        _metric("poi_total_radiation", "辐射圈POI总数", rad.get("total"), "个",
                _eid("poi", pid, ac.RING_RADIATION, "poi_total"), src),
        _metric("poi_commercial_radiation", "辐射圈商业POI", rad.get("commercial"), "个",
                _eid("poi", pid, ac.RING_RADIATION, "poi_commercial"), src),
        _metric("poi_public_radiation", "辐射圈公共服务POI", rad.get("public"), "个",
                _eid("poi", pid, ac.RING_RADIATION, "poi_public"), src),
        _metric("poi_mix_index_nearby", "近邻圈功能混合度", nearby.get("mix_index"), "",
                _eid("poi", pid, ac.RING_NEARBY, "poi_mix_index"), src),
    ]
    summary = (
        f"区位配套维度评分 L_score={_num(l_score)}。辐射圈 POI 共 {_num(rad.get('total'))} 个，"
        f"其中商业 {_num(rad.get('commercial'))} 个、公共服务 {_num(rad.get('public'))} 个。"
    )
    key_findings = [f"L_score={_num(l_score)}，辐射圈 POI 总数 {_num(rad.get('total'))} 个。"]
    if shortboards:
        key_findings.append(f"配套短板（类别）：{('、'.join(shortboards[:3]))}。")
    if recommend:
        key_findings.append(f"建议补充业态：{('、'.join(recommend[:3]))}。")
    limitations = []
    if not rad.get("total"):
        limitations.append("辐射圈无 POI 记录，配套分析置信度低。")
    if not limitations:
        limitations.append("POI 数据覆盖充分。")
    evid = [m["evidence_id"] for m in metrics]
    return {
        "section_id": "S3",
        "title": "区位与POI配套分析",
        "summary": summary,
        "key_findings": key_findings,
        "metrics": metrics,
        "evidence_ids": _dedup(evid),
        "data_limitations": limitations,
    }


def _sec_population(project: Project, full: dict, src: dict, facts: dict) -> dict:
    pid = project.id
    pop = (full.get("four_dimension", {}).get("population")) or {}
    rad = _ring(pop.get("rings"), ac.RING_RADIATION)
    p_score = pop.get("score")
    main_segment = pop.get("main_segment")
    facts["main_segment"] = main_segment

    metrics = [
        _metric("P_score", "人口潜力评分 P_score", p_score, "分",
                _eid("population", pid, None, "P_score"), src),
        _metric("pop_residential_radiation", "辐射圈居住人口", rad.get("residential"), "人",
                _eid("population", pid, ac.RING_RADIATION, "pop_residential"), src),
        _metric("pop_worker_radiation", "辐射圈工作人口", rad.get("worker"), "人",
                _eid("population", pid, ac.RING_RADIATION, "pop_worker"), src),
    ]
    if rad.get("job_housing_ratio") is not None:
        metrics.append(
            _metric("job_housing_ratio_radiation", "辐射圈职住比", rad.get("job_housing_ratio"),
                    "", _eid("population", pid, ac.RING_RADIATION, "job_housing_ratio"), src)
        )
    summary = (
        f"人口潜力维度评分 P_score={_num(p_score)}。辐射圈居住人口约 {_num(rad.get('residential'))} 人，"
        f"工作人口约 {_num(rad.get('worker'))} 人。主力客群：{main_segment or '数据不足'}。"
    )
    key_findings = [
        f"P_score={_num(p_score)}，居住人口 {_num(rad.get('residential'))} 人。",
        f"主力客群：{main_segment or '数据不足，需补充调研'}。",
    ]
    limitations = ["收入字段数据缺失/不适用（数据集未提供，未编造）。"]
    if not main_segment:
        limitations.append("人口画像字段不足，客群定位需结合补充调研。")
    return {
        "section_id": "S4",
        "title": "人口画像与客群分析",
        "summary": summary,
        "key_findings": key_findings,
        "metrics": metrics,
        "evidence_ids": _dedup([m["evidence_id"] for m in metrics]),
        "data_limitations": limitations,
    }


def _sec_housing(project: Project, full: dict, src: dict, facts: dict) -> dict:
    pid = project.id
    housing = (full.get("four_dimension", {}).get("housing")) or {}
    rad = _ring(housing.get("rings"), ac.RING_RADIATION)
    h_score = housing.get("score")
    interval = housing.get("baseline_interval") or {}
    model = housing.get("model_metrics") or {}
    facts["housing_model_type"] = model.get("model_type")
    facts["housing_val_mape"] = model.get("val_mape")

    metrics = [
        _metric("H_score", "房价价值评分 H_score", h_score, "分",
                _eid("housing", pid, None, "H_score"), src),
        _metric("housing_avg_unit_price_radiation", "辐射圈平均单价", rad.get("avg_unit_price"),
                "元/㎡", _eid("housing", pid, ac.RING_RADIATION, "avg_unit_price"), src),
        _metric("housing_median_unit_price_radiation", "辐射圈单价中位数",
                rad.get("median_unit_price"), "元/㎡",
                _eid("housing", pid, ac.RING_RADIATION, "median_unit_price"), src),
    ]
    for bound in ("low", "mid", "high"):
        if interval.get(bound) is not None:
            metrics.append(
                _metric(f"baseline_{bound}", f"房价基线区间-{bound}", interval.get(bound),
                        "元/㎡", _eid("housing", pid, None, f"baseline_{bound}"), src)
            )
    summary = (
        f"房价价值维度评分 H_score={_num(h_score)}。辐射圈平均单价约 {_num(rad.get('avg_unit_price'))} 元/㎡，"
        f"基线中枢约 {_num(interval.get('mid'))} 元/㎡。"
        f"房价基线模型类型：{model.get('model_type') or '数据不足'}，val_mape={_num(model.get('val_mape'))}。"
    )
    key_findings = [
        f"H_score={_num(h_score)}，辐射圈均价 {_num(rad.get('avg_unit_price'))} 元/㎡。",
        f"房价模型 {model.get('model_type') or '数据不足'}（val_mape={_num(model.get('val_mape'))}，仅 train/val 训练）。",
    ]
    limitations = []
    if not rad.get("sample_count"):
        limitations.append("辐射圈房价样本不足，价值判断置信度有限。")
    if model.get("degraded"):
        limitations.append("房价模型已降级为统计基线，预测区间参考性有限。")
    if not limitations:
        limitations.append("房价样本与模型指标可用。")
    return {
        "section_id": "S5",
        "title": "房价与价值潜力分析",
        "summary": summary,
        "key_findings": key_findings,
        "metrics": metrics,
        "evidence_ids": _dedup([m["evidence_id"] for m in metrics]),
        "data_limitations": limitations,
    }


def _sec_industry(project: Project, full: dict, src: dict, facts: dict) -> dict:
    pid = project.id
    industry = (full.get("four_dimension", {}).get("industry")) or {}
    rad = _ring(industry.get("rings"), ac.RING_RADIATION)
    i_score = industry.get("score")
    dominant = industry.get("dominant_industry")
    suggestions = industry.get("adaptation_suggestions") or []
    facts["dominant_industry"] = dominant

    metrics = [
        _metric("I_score", "产业经济评分 I_score", i_score, "分",
                _eid("industry", pid, None, "I_score"), src),
        _metric("enterprise_count_radiation", "辐射圈企业数", rad.get("enterprise_count"), "家",
                _eid("industry", pid, ac.RING_RADIATION, "enterprise_count"), src),
    ]
    if rad.get("density_per_km2") is not None:
        metrics.append(
            _metric("industry_density_radiation", "辐射圈产业密度", rad.get("density_per_km2"),
                    "家/km²", _eid("industry", pid, ac.RING_RADIATION, "density_per_km2"), src)
        )
    summary = (
        f"产业经济维度评分 I_score={_num(i_score)}。辐射圈企业 {_num(rad.get('enterprise_count'))} 家，"
        f"产业密度约 {_num(rad.get('density_per_km2'))} 家/km²。主导产业（一级类目）：{dominant or '数据不足'}。"
    )
    key_findings = [f"I_score={_num(i_score)}，辐射圈企业 {_num(rad.get('enterprise_count'))} 家。"]
    if suggestions:
        key_findings.append(f"功能适配方向：{suggestions[0]}")
    limitations = ["数据集产业为单一类目，细分行业字段缺失，不可细分（未编造行业分类）。"]
    if not rad.get("enterprise_count"):
        limitations.append("辐射圈无产业点位，产业方向需结合外部产业规划确认。")
    return {
        "section_id": "S6",
        "title": "产业基础与功能适配分析",
        "summary": summary,
        "key_findings": key_findings,
        "metrics": metrics,
        "evidence_ids": _dedup([m["evidence_id"] for m in metrics]),
        "data_limitations": limitations,
    }


def _sec_type_score(project: Project, full: dict, src: dict, facts: dict) -> dict:
    pid = project.id
    ptype = full.get("project_type")
    conf = full.get("project_type_confidence")
    f_score = full.get("F_score")
    level = full.get("score_level")
    scores = full.get("scores") or {}
    weights = full.get("weights") or {}
    facts["project_type"] = ptype
    facts["score_level"] = level
    facts["weights"] = weights

    type_eid = _eid("classification", pid, None, "project_type")
    conf_eid = _eid("classification", pid, None, "type_confidence")
    f_eid = _eid("scoring", pid, None, "F_score")

    metrics = [
        _metric("type_confidence", "类型识别置信度", conf, "",
                conf_eid, src),
        _metric("F_score", "综合评分 F_score", f_score, "分", f_eid, src),
    ]
    for k in ("P_score", "H_score", "L_score", "I_score"):
        if k in scores:
            dim = {"P_score": "population", "H_score": "housing",
                   "L_score": "poi", "I_score": "industry"}[k]
            metrics.append(
                _metric(k, f"{k} 维度分", scores.get(k), "分",
                        _eid(dim, pid, None, k), src)
            )
    weight_str = (
        f"P={weights.get('P')}, H={weights.get('H')}, L={weights.get('L')}, I={weights.get('I')}"
        if weights else "数据不足"
    )
    summary = (
        f"项目识别类型「{ptype or '数据不足'}」（置信度 {_num(conf)}）。"
        f"按类型权重（{weight_str}）加权四维分得综合评分 F_score={_num(f_score)}（{level or '数据不足'}）。"
    )
    key_findings = [
        f"类型：{ptype or '数据不足'}（置信度 {_num(conf)}）。",
        f"综合评分 F_score={_num(f_score)}（{level or '数据不足'}）。",
        f"四维分：P={_num(scores.get('P_score'))} / H={_num(scores.get('H_score'))} / "
        f"L={_num(scores.get('L_score'))} / I={_num(scores.get('I_score'))}。",
    ]
    limitations = []
    if conf is not None and conf < 0.25:
        limitations.append("类型识别置信度偏低，建议补齐项目输入字段后复判。")
    if not limitations:
        limitations.append("类型与评分依据充分，可复算。")
    return {
        "section_id": "S7",
        "title": "项目类型识别与综合评分",
        "summary": summary,
        "key_findings": key_findings,
        "metrics": metrics,
        "evidence_ids": _dedup([f_eid, type_eid, conf_eid]),
        "data_limitations": limitations,
    }


def _sec_strategy(project: Project, full: dict, src: dict, facts: dict) -> dict:
    pid = project.id
    strat = full.get("strategy_result") or {}
    count = full.get("strategy_count")
    facts["strategy_count"] = count
    positioning = strat.get("update_positioning")
    opportunities = strat.get("key_opportunities") or []
    risks = strat.get("key_risks") or []
    directions = strat.get("recommended_directions") or []
    actions = strat.get("priority_actions") or []

    pos_eid = _eid("strategy", pid, None, "update_positioning")
    cnt_eid = _eid("strategy", pid, None, "strategy_count")

    metrics = [
        _metric("strategy_count", "策略建议条目数", count, "条", cnt_eid, src),
    ]
    summary = (
        f"更新定位：{positioning or '数据不足'}。共形成 {_num(count)} 条结构化策略建议"
        "（机会/风险/方向/优先行动）。"
    )
    key_findings = []
    if opportunities:
        key_findings.append(f"关键机会：{opportunities[0]}")
    if risks:
        key_findings.append(f"关键风险：{risks[0]}")
    if directions:
        key_findings.append(f"推荐方向：{('、'.join(directions[:2]))}。")
    if actions:
        key_findings.append(f"优先行动：{actions[0]}")
    if not key_findings:
        key_findings.append("策略建议数据不足，需结合补充调研。")
    limitations = list(strat.get("data_limitations") or [])
    if not limitations:
        limitations.append("策略依据充分，无重大数据局限。")
    return {
        "section_id": "S8",
        "title": "更新策略与实施建议",
        "summary": summary,
        "key_findings": key_findings,
        "metrics": metrics,
        "evidence_ids": _dedup([pos_eid, cnt_eid]),
        "data_limitations": limitations,
    }


def _sec_risk(project: Project, full: dict, sections: list[dict], src: dict, facts: dict) -> dict:
    pid = project.id
    f_eid = _eid("scoring", pid, None, "F_score")
    strat_eid = _eid("strategy", pid, None, "strategy_count")

    # 汇总各章数据局限（去重），形成全局风险提示
    all_limits: list[str] = []
    for s in sections:
        for lim in s.get("data_limitations", []):
            if "无重大" not in lim and "充分" not in lim and lim not in all_limits:
                all_limits.append(lim)

    four = full.get("four_dimension", {})
    conf = four.get("confidence") or {}
    low_conf = [k for k, v in conf.items() if isinstance(v, (int, float)) and v < 0.3]

    metrics = [
        _metric("low_confidence_dimension_count", "低置信度维度数", float(len(low_conf)), "个",
                f_eid, src),
    ]
    summary = (
        "本章汇总报告数据局限与风险提示。报告所有数字均来自本地确定性计算并带 evidence_id，"
        f"当前存在 {len(low_conf)} 个低置信度维度。缺失数据均已显式标注，未做编造。"
    )
    key_findings = [
        f"低置信度维度：{('、'.join(low_conf)) if low_conf else '无'}。",
        "报告未使用 test 数据、未调用外部 API、未使用大模型生成事实数字。",
    ]
    limitations = all_limits or ["未发现重大数据局限。"]
    return {
        "section_id": "S9",
        "title": "数据局限与风险提示",
        "summary": summary,
        "key_findings": key_findings,
        "metrics": metrics,
        "evidence_ids": _dedup([f_eid, strat_eid]),
        "data_limitations": limitations,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_report_content(
    db: Session, project: Project, include_test: bool = False
) -> dict[str, Any]:
    """生成结构化报告内容 JSON（确定性模板 + 结构化数据填充）。

    include_test 仅作签名兼容，本阶段强制 false（报告默认仅 train/val）。
    """
    # 确保 run-full 已完成；幂等运行获取 canonical 完整结构（仅 train/val）。
    full = orch.run_full_analysis(db, project, include_test=False)

    source_metrics: dict[str, Any] = {}
    source_facts: dict[str, Any] = {}

    sections = [
        _sec_overview(project, full, source_metrics, source_facts),
        _sec_data_scope(project, full, source_metrics, source_facts),
        _sec_poi(project, full, source_metrics, source_facts),
        _sec_population(project, full, source_metrics, source_facts),
        _sec_housing(project, full, source_metrics, source_facts),
        _sec_industry(project, full, source_metrics, source_facts),
        _sec_type_score(project, full, source_metrics, source_facts),
        _sec_strategy(project, full, source_metrics, source_facts),
    ]
    sections.append(_sec_risk(project, full, sections, source_metrics, source_facts))

    all_evidence: list[str] = []
    for s in sections:
        all_evidence.extend(s["evidence_ids"])
    all_evidence = _dedup(all_evidence)

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_id = f"report:p{project.id}:{now}"

    notes = [
        "报告由确定性模板 + 结构化数据填充生成，未使用大模型撰写、未调用外部 API。",
        "所有数字来自 AnalysisResult / full analysis / Project，均带 evidence_id 可溯源。",
        "默认仅 train/val，未触碰 test；不含原始JSON、原始点位、企业名、小区名、地址明细。",
    ]
    if full.get("used_test"):
        notes.append("警告：本次 full analysis 标记 used_test=true（非默认）。")

    content = {
        "report_id": report_id,
        "project_id": project.id,
        "project_name": project.name,
        "project_type": full.get("project_type"),
        "generated_at": now,
        "sections": sections,
        "notes": notes,
        # ---- 供第7.5门禁回比（不进入 docx 正文，仅服务内部使用）----
        "source_metrics": source_metrics,
        "source_facts": source_facts,
        "allowed_splits": full.get("allowed_splits", []),
        "used_test": full.get("used_test", False),
        "evidence_ids": all_evidence,
        "evidence_ids_count": len(all_evidence),
        "sections_count": len(sections),
        "required_sections_count": REQUIRED_SECTIONS,
    }

    _persist(content)
    logger.info(
        "report content built project_id=%s report_id=%s sections=%s evidence=%s used_test=%s",
        project.id, report_id, len(sections), len(all_evidence), content["used_test"],
    )
    return content


# --------------------------------------------------------------------------- #
# 落盘（backend/data/outputs/reports/{project_id}/，已 gitignore）
# --------------------------------------------------------------------------- #
def _report_dir(project_id: int):
    d = settings.data_dir / "outputs" / "reports" / str(project_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(report_id: str) -> str:
    return report_id.replace(":", "_")


def _persist(content: dict[str, Any]) -> None:
    d = _report_dir(content["project_id"])
    fname = _safe_name(content["report_id"]) + ".json"
    (d / fname).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "latest.json").write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_latest(project_id: int) -> dict[str, Any] | None:
    """读取最近一次生成的报告内容（无则 None）。"""
    path = settings.data_dir / "outputs" / "reports" / str(project_id) / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
