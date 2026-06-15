"""策略方向建议（第6阶段，规则模板组装，绝不生成最终报告/不调大模型）。

基于项目类型 + F_score/score_level + 四维短板（POI 配套短板、人口客群、房价区间、
产业适配）用**规则模板**组装结构化前期策划建议。

红线：
- 仅使用已有分析结果与规则模板生成；不调用外部 API；不使用大模型；不编造数据。
- 输出为结构化 JSON（定位/机会/风险/方向/优先行动/数据局限），**不是最终 Word 报告**。
- 维度无数据时进入 data_limitations，不臆测。
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.models import Project
from app.services import analysis_common as ac
from app.services import project_type_service as pts

logger = logging.getLogger("cityrenew.strategy")

DIMENSION = "strategy"

# 各项目类型的定位模板与方向候选（经验规则，可在 train/val 校准）
TYPE_POSITIONING: dict[str, str] = {
    pts.TYPE_OLD: "以居住环境改善与存量空间提质为核心的老旧片区更新",
    pts.TYPE_INDUSTRIAL: "以工业遗存保护性再利用与产业功能转型为核心的更新",
    pts.TYPE_BLOCK: "以商业活力提升与街区风貌优化为核心的街区更新",
    pts.TYPE_PUBLIC_SPACE: "以公共空间品质提升与开放共享为核心的更新",
    pts.TYPE_COMMUNITY: "以社区配套补短板与民生服务升级为核心的更新",
    pts.TYPE_MIXED: "以多元功能统筹与片区综合价值提升为核心的更新",
}

TYPE_DIRECTIONS: dict[str, list[str]] = {
    pts.TYPE_OLD: ["居住品质改善与基础设施更新", "存量建筑功能置换与微更新", "公共空间与配套补足"],
    pts.TYPE_INDUSTRIAL: ["工业遗存活化与文化记忆保留", "产业载体升级与新功能导入", "厂区开放与城市缝合"],
    pts.TYPE_BLOCK: ["商业业态优化与消费场景营造", "街区风貌与慢行环境提升", "公共活动与夜间经济培育"],
    pts.TYPE_PUBLIC_SPACE: ["开放空间系统化与可达性提升", "景观生态与滨水慢行", "全龄友好公共活动场所"],
    pts.TYPE_COMMUNITY: ["公共服务设施补短板", "便民生活圈与一刻钟服务", "停车与适老化改造"],
    pts.TYPE_MIXED: ["功能混合与片区统筹", "产城融合与职住平衡", "公共空间与配套协同"],
}


def _ring(rings: list[dict] | None, ring_name: str) -> dict:
    for r in rings or []:
        if r.get("ring") == ring_name:
            return r
    return {}


def build(
    db: Session,
    project: Project,
    four_dim: dict[str, Any],
    project_type: str,
    score_result: dict[str, Any],
    persist: bool = True,
) -> dict[str, Any]:
    """组装结构化策略建议（规则模板 + 四维短板）。"""
    poi = four_dim.get("poi") or {}
    population = four_dim.get("population") or {}
    housing = four_dim.get("housing") or {}
    industry = four_dim.get("industry") or {}

    f_score = score_result.get("F_score")
    level = score_result.get("score_level")
    scores = score_result.get("scores") or {}

    update_positioning = TYPE_POSITIONING.get(project_type, TYPE_POSITIONING[pts.TYPE_MIXED])

    key_opportunities: list[str] = []
    key_risks: list[str] = []
    recommended_directions: list[str] = list(TYPE_DIRECTIONS.get(project_type, []))
    priority_actions: list[str] = []
    data_limitations: list[str] = []

    # ---- 机会：由相对高分维度驱动 ----
    contrib = sorted(score_result.get("contributions") or [],
                     key=lambda d: d.get("contribution", 0), reverse=True)
    if contrib:
        lead = contrib[0]
        if lead.get("score", 0) >= 60:
            key_opportunities.append(
                f"{lead['label']}维度基础较好（得分 {lead['score']}），可作为更新的价值支点。"
            )
    main_segment = population.get("main_segment")
    if main_segment:
        key_opportunities.append(f"人口客群特征明确：{main_segment}，利于精准配置功能与业态。")
    interval = housing.get("baseline_interval") or {}
    if interval.get("mid"):
        key_opportunities.append(
            f"房价基线中枢约 {int(interval['mid'])} 元/㎡，具备改造后价值提升空间。"
        )

    # ---- 风险：由短板与低分维度驱动 ----
    if contrib:
        drag = contrib[-1]
        if drag.get("score", 0) < 40:
            key_risks.append(
                f"{drag['label']}维度偏弱（得分 {drag['score']}），更新需重点补强。"
            )
    shortboards = poi.get("shortboards_top5") or []
    if shortboards:
        key_risks.append(f"配套短板集中在：{('、'.join(shortboards[:3]))}，影响居住与服务品质。")
    if (f_score is not None) and f_score < 50:
        key_risks.append(f"综合评分偏低（{f_score}，{level}），需谨慎评估更新强度与投入产出。")

    # ---- 优先行动：结合类型与短板 ----
    if shortboards:
        recommend = poi.get("recommend_top5") or []
        if recommend:
            priority_actions.append(f"优先补足配套：{('、'.join(recommend[:3]))}。")
    rad_ind = _ring(industry.get("rings"), ac.RING_RADIATION)
    if project_type == pts.TYPE_INDUSTRIAL:
        priority_actions.append("梳理工业遗存建筑可保留可利用部分，明确再利用路径与功能导入。")
    if project_type == pts.TYPE_COMMUNITY:
        priority_actions.append("以一刻钟便民生活圈为目标布局社区服务设施。")
    if project_type == pts.TYPE_OLD:
        priority_actions.append("分期推进基础设施更新与公共空间整治，兼顾居民诉求。")
    if not priority_actions:
        priority_actions.append("结合四维分析短板，分期制定功能补足与空间提升计划。")

    # ---- 数据局限（缺数据维度显式标注，不编造）----
    if not population.get("main_segment"):
        data_limitations.append("人口客群结构数据不足，客群定位需结合补充调研。")
    if industry.get("dominant_industry") is None or rad_ind.get("enterprise_count", 0) == 0:
        data_limitations.append("范围内产业点位不足或细分行业字段缺失，产业方向需结合外部产业规划确认。")
    if not (housing.get("baseline_interval") or {}).get("mid"):
        data_limitations.append("房价样本不足，价值判断置信度有限。")
    income = population.get("income_structure")
    if income and "缺失" in str(income):
        data_limitations.append("收入字段数据缺失/不适用（数据集未提供，未编造）。")

    notes = [
        "策略建议由项目类型 + F_score + 四维短板按规则模板组装，非大模型生成，非最终报告。",
    ]

    strategy_count = (
        len(key_opportunities) + len(key_risks) + len(recommended_directions)
        + len(priority_actions)
    )

    evidence_ids: list[str] = list(score_result.get("evidence_ids") or [])

    if persist:
        ac.clear_dimension_results(db, project.id, DIMENSION)
        s_evid = ac.record_metric(
            db, project_id=project.id, dimension=DIMENSION, ring=None,
            metric_key="update_positioning", text=update_positioning,
            summary="更新定位（规则模板，依据项目类型与四维结果）",
            confidence=None,
            metadata={"project_type": project_type, "score_level": level},
        )
        c_evid = ac.record_metric(
            db, project_id=project.id, dimension=DIMENSION, ring=None,
            metric_key="strategy_count", value=float(strategy_count), unit="条",
            summary="结构化策略建议条目数（机会/风险/方向/行动合计）",
            confidence=None,
        )
        evidence_ids = [s_evid, c_evid, *evidence_ids]
        db.commit()

    logger.info(
        "strategy project_id=%s type=%s count=%s f_score=%s used_test=%s",
        project.id, project_type, strategy_count, f_score, four_dim.get("used_test", False),
    )

    return {
        "project_id": project.id,
        "project_type": project_type,
        "update_positioning": update_positioning,
        "key_opportunities": key_opportunities,
        "key_risks": key_risks,
        "recommended_directions": recommended_directions,
        "priority_actions": priority_actions,
        "data_limitations": data_limitations,
        "strategy_count": strategy_count,
        "allowed_splits": four_dim.get("allowed_splits", []),
        "include_test": four_dim.get("include_test", False),
        "used_test": four_dim.get("used_test", False),
        "evidence_ids": evidence_ids,
        "notes": notes,
    }
