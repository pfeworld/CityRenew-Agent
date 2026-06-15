"""项目类型识别（第6阶段，规则 + 指标，绝不使用大模型）。

目标：基于项目输入字段（用地性质/建成年代/面积/更新诉求/期望方向）与第5阶段
四维 P/H/L/I 分析结果（含 POI 短板、人口结构、房价价值、产业密度），用
**关键词词典 + 指标阈值打分**判定 6 大项目类型，输出可解释的命中规则与置信度。

红线：
- 不读取 test；不调用外部 API；不使用大模型；不产生事实数字（分数来自确定性规则）。
- 输入信息不足时降低 confidence 并在 notes 标注"数据不足"，绝不编造类型。
- 仅输出枚举/分数/规则名/脱敏短语/evidence_id，不返回 raw_json/原始明细。

校准说明（第6.5 阶段）：
- 词典与阈值集中为模块顶部常量，可在 **train/val** 案例上校准（禁用 test）。
- 当前为经验值，预留扩展位，不为校准读取 test。
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.models import Project
from app.services import analysis_common as ac

logger = logging.getLogger("cityrenew.classifier")

DIMENSION = "classification"

# --------------------------------------------------------------------------- #
# 6 大项目类型枚举
# --------------------------------------------------------------------------- #
TYPE_OLD = "老旧片区/存量地块"
TYPE_INDUSTRIAL = "工业遗存"
TYPE_BLOCK = "街区提升"
TYPE_PUBLIC_SPACE = "公共空间优化"
TYPE_COMMUNITY = "社区配套升级"
TYPE_MIXED = "综合功能地块"

ALL_TYPES: tuple[str, ...] = (
    TYPE_OLD,
    TYPE_INDUSTRIAL,
    TYPE_BLOCK,
    TYPE_PUBLIC_SPACE,
    TYPE_COMMUNITY,
    TYPE_MIXED,
)

# --------------------------------------------------------------------------- #
# 关键词词典（经验值；可在 train/val 案例语料上校准，禁用 test）
# 作用字段：land_use（用地性质）/ update_demand（更新诉求）/ expected_direction（期望方向）
# --------------------------------------------------------------------------- #
LAND_USE_KEYWORDS: dict[str, list[str]] = {
    TYPE_OLD: ["居住", "住宅", "商住", "二类居住", "一类居住", "城镇住宅", "宅基"],
    TYPE_INDUSTRIAL: ["工业", "仓储", "厂房", "厂", "物流", "m1", "m2", "产业用地"],
    TYPE_BLOCK: ["商业", "商务", "零售", "批发", "商服", "金融", "办公"],
    TYPE_PUBLIC_SPACE: ["绿地", "广场", "公园", "公共绿地", "防护绿地", "水域", "滨水"],
    TYPE_COMMUNITY: ["公共管理", "公共服务", "教育", "医疗", "文化", "行政", "社区"],
    TYPE_MIXED: ["综合", "混合", "商住混合", "多功能", "复合"],
}

# 更新诉求 / 期望方向 共用同一套语义词典
DEMAND_KEYWORDS: dict[str, list[str]] = {
    TYPE_OLD: ["老旧", "老化", "危旧", "旧改", "棚改", "存量", "年久", "破旧", "老小区", "改造提升"],
    TYPE_INDUSTRIAL: ["工业", "厂房", "遗存", "退二进三", "腾退", "转型", "老厂", "工业遗产", "产业转型"],
    TYPE_BLOCK: ["街区", "商圈", "活力", "风貌", "商业提升", "业态", "街道", "消费场景", "夜经济"],
    TYPE_PUBLIC_SPACE: ["公共空间", "绿地", "景观", "开放空间", "广场", "滨水", "慢行", "口袋公园", "生态"],
    TYPE_COMMUNITY: ["配套", "便民", "服务设施", "社区", "养老", "托育", "菜场", "停车", "民生", "一刻钟"],
    TYPE_MIXED: ["综合", "混合", "多元", "复合", "统筹", "片区统筹", "多功能"],
}

# --------------------------------------------------------------------------- #
# 规则权重（经验值，可解释；可在 train/val 校准）
# --------------------------------------------------------------------------- #
W_LAND_USE = 3.0          # 用地性质命中（强信号）
W_DEMAND = 2.0            # 更新诉求命中
W_DIRECTION = 2.0         # 期望方向命中
W_BUILD_YEAR_OLD = 2.0    # 建成年代偏早 -> 老旧
W_HOUSING_OLD = 1.0       # 房价样本房龄偏老 -> 老旧
W_BIG_AREA = 1.5          # 用地面积大 -> 综合功能
W_IND_DENSITY = 2.0       # 产业密度高 -> 工业遗存
W_POI_COMMERCIAL = 1.5    # 商业 POI 占比高 -> 街区提升
W_POI_SHORTBOARD = 1.5    # 公共/便民短板明显 -> 社区配套
W_POP_RESIDENTIAL = 1.0   # 居住人口规模较高 -> 社区配套/老旧

# 指标阈值（经验值；预留 train/val 校准位，禁用 test）
OLD_BUILD_YEAR_MAX = 2000          # 建成年代早于该值视为偏老
OLD_HOUSING_MEDIAN_YEAR_MAX = 2005  # 圈层房价样本中位房龄偏老阈值
BIG_AREA_M2 = 50000.0              # 用地面积大阈值（㎡）
IND_DENSITY_HI = 18.0             # 辐射圈产业密度高阈值（家/km²）
POI_COMMERCIAL_SHARE_HI = 0.35    # 近邻圈商业 POI 占比高阈值
POP_RESIDENTIAL_HI = 20000        # 辐射圈居住人口规模较高阈值
MARGIN_TIE_RATIO = 0.15           # 与次高分差距比例小于该值视为"类型特征接近"

# 关键输入字段（用于数据完整度评估）
KEY_FIELDS = ("land_use", "build_year", "project_area_or_building", "update_demand", "expected_direction")


def _contains(text: str | None, keywords: list[str]) -> str | None:
    """返回命中的第一个关键词（脱敏：仅返回词典词，不回写原文整段）。"""
    if not text:
        return None
    low = text.lower()
    for kw in keywords:
        if kw.lower() in low:
            return kw
    return None


def _field_completeness(project: Project) -> tuple[float, list[str]]:
    """关键字段完整度（0~1）与缺失字段清单。"""
    present: list[str] = []
    missing: list[str] = []
    checks = {
        "land_use": bool(project.land_use),
        "build_year": bool(project.build_year),
        "project_area_or_building": bool(project.project_area or project.building_area),
        "update_demand": bool(project.update_demand),
        "expected_direction": bool(project.expected_direction),
    }
    for field, ok in checks.items():
        (present if ok else missing).append(field)
    return len(present) / len(KEY_FIELDS), missing


def _add(scores: dict[str, float], rules: dict[str, list[dict]], ptype: str,
         rule: str, weight: float, detail: str) -> None:
    scores[ptype] = scores.get(ptype, 0.0) + weight
    rules.setdefault(ptype, []).append(
        {"rule": rule, "weight": round(weight, 2), "detail": detail}
    )


def _apply_text_rules(project: Project, scores: dict, rules: dict) -> None:
    """用地性质 / 更新诉求 / 期望方向 三个文本字段的关键词命中规则。"""
    for ptype in ALL_TYPES:
        hit = _contains(project.land_use, LAND_USE_KEYWORDS.get(ptype, []))
        if hit:
            _add(scores, rules, ptype, "land_use_keyword", W_LAND_USE,
                 f"用地性质命中关键词「{hit}」")
        hit = _contains(project.update_demand, DEMAND_KEYWORDS.get(ptype, []))
        if hit:
            _add(scores, rules, ptype, "update_demand_keyword", W_DEMAND,
                 f"更新诉求命中关键词「{hit}」")
        hit = _contains(project.expected_direction, DEMAND_KEYWORDS.get(ptype, []))
        if hit:
            _add(scores, rules, ptype, "expected_direction_keyword", W_DIRECTION,
                 f"期望方向命中关键词「{hit}」")


def _apply_indicator_rules(project: Project, four_dim: dict, scores: dict, rules: dict) -> None:
    """基于项目数值字段 + 四维指标的阈值规则。"""
    # 建成年代偏老 -> 老旧片区
    if project.build_year and 0 < project.build_year < OLD_BUILD_YEAR_MAX:
        _add(scores, rules, TYPE_OLD, "build_year_old", W_BUILD_YEAR_OLD,
             f"建成年代 {project.build_year} 早于 {OLD_BUILD_YEAR_MAX}")

    # 用地面积大 -> 综合功能地块
    area = project.project_area or project.building_area
    if area and area >= BIG_AREA_M2:
        _add(scores, rules, TYPE_MIXED, "large_area", W_BIG_AREA,
             f"用地/建筑面积 {area:.0f}㎡ ≥ {BIG_AREA_M2:.0f}㎡")

    housing = four_dim.get("housing") or {}
    population = four_dim.get("population") or {}
    industry = four_dim.get("industry") or {}
    poi = four_dim.get("poi") or {}

    # 房价样本房龄偏老 -> 老旧片区
    rad_year = None
    for ring in housing.get("rings", []) or []:
        if ring.get("ring") == ac.RING_RADIATION:
            rad_year = (ring.get("year_summary") or {}).get("median_year")
    if rad_year and 0 < rad_year <= OLD_HOUSING_MEDIAN_YEAR_MAX:
        _add(scores, rules, TYPE_OLD, "housing_old_stock", W_HOUSING_OLD,
             f"辐射圈房价样本中位房龄 {int(rad_year)} ≤ {OLD_HOUSING_MEDIAN_YEAR_MAX}")

    # 产业密度高 -> 工业遗存
    rad_density = None
    for ring in industry.get("rings", []) or []:
        if ring.get("ring") == ac.RING_RADIATION:
            rad_density = ring.get("density_per_km2")
    if rad_density and rad_density >= IND_DENSITY_HI:
        _add(scores, rules, TYPE_INDUSTRIAL, "industry_density_high", W_IND_DENSITY,
             f"辐射圈产业密度 {rad_density:.1f} ≥ {IND_DENSITY_HI} 家/km²")

    # 商业 POI 占比高 -> 街区提升（近邻圈）
    for ring in poi.get("rings", []) or []:
        if ring.get("ring") == ac.RING_NEARBY and ring.get("total"):
            share = ring.get("commercial", 0) / ring["total"]
            if share >= POI_COMMERCIAL_SHARE_HI:
                _add(scores, rules, TYPE_BLOCK, "poi_commercial_share_high", W_POI_COMMERCIAL,
                     f"近邻圈商业 POI 占比 {share:.0%} ≥ {POI_COMMERCIAL_SHARE_HI:.0%}")

    # 公共/便民配套短板明显 -> 社区配套升级
    shortboards = poi.get("shortboards_top5") or []
    community_shortboards = [s for s in shortboards
                             if any(k in s for k in ("医疗", "科教", "生活", "公共", "体育"))]
    if community_shortboards:
        _add(scores, rules, TYPE_COMMUNITY, "poi_public_shortboard", W_POI_SHORTBOARD,
             f"存在公共/便民配套短板：{('、'.join(community_shortboards[:3]))}")

    # 居住人口规模较高 -> 社区配套升级 / 老旧片区
    rad_residential = None
    for ring in population.get("rings", []) or []:
        if ring.get("ring") == ac.RING_RADIATION:
            rad_residential = ring.get("residential")
    if rad_residential and rad_residential >= POP_RESIDENTIAL_HI:
        _add(scores, rules, TYPE_COMMUNITY, "population_residential_high", W_POP_RESIDENTIAL,
             f"辐射圈居住人口 {rad_residential} ≥ {POP_RESIDENTIAL_HI}")
        _add(scores, rules, TYPE_OLD, "population_residential_high", W_POP_RESIDENTIAL,
             f"辐射圈居住人口 {rad_residential} ≥ {POP_RESIDENTIAL_HI}")


def _score_evidence_ids(project_id: int, four_dim: dict) -> list[str]:
    """引用四维维度分的 evidence_id（作为类型判定的指标来源溯源）。"""
    out: list[str] = []
    for dim, key in (("poi", "L_score"), ("population", "P_score"),
                     ("housing", "H_score"), ("industry", "I_score")):
        if four_dim.get(dim):
            out.append(ac.make_evidence_id(dim, project_id, "all", key))
    return out


def identify(
    db: Session, project: Project, four_dim: dict[str, Any], persist: bool = True
) -> dict[str, Any]:
    """识别项目类型（规则 + 指标）。

    four_dim: analysis_orchestrator.run_four_dimension_analysis 的返回结构。
    persist=True 时写入 classification 维度的 AnalysisResult + EvidenceChain，并回写
    Project.project_type。
    """
    scores: dict[str, float] = {t: 0.0 for t in ALL_TYPES}
    rules: dict[str, list[dict]] = {}

    _apply_text_rules(project, scores, rules)
    _apply_indicator_rules(project, four_dim, scores, rules)

    completeness, missing = _field_completeness(project)
    dim_conf_vals = list((four_dim.get("confidence") or {}).values())
    dim_conf = round(sum(dim_conf_vals) / len(dim_conf_vals), 3) if dim_conf_vals else 0.0
    data_sufficiency = round(0.6 * completeness + 0.4 * dim_conf, 3)

    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    notes: list[str] = [
        "类型判定为规则+指标确定性推断，未使用大模型；分数来自经验权重（可在 train/val 校准）。",
    ]

    if total <= 0:
        project_type = TYPE_MIXED
        confidence = round(0.15 * data_sufficiency, 3)
        reason = "未命中任何用地/诉求/方向关键词与指标规则，信息不足，暂归为综合功能地块（兜底）。"
        notes.append("数据不足：无有效类型信号，建议补充用地性质/更新诉求/期望方向后重判。")
        matched = []
    else:
        project_type, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        base_conf = top_score / total
        margin_close = top_score > 0 and (top_score - second_score) / top_score < MARGIN_TIE_RATIO
        if margin_close:
            base_conf *= 0.85
            notes.append(f"多类型特征接近（次高：{ranked[1][0]}），置信度已下调。")
        # 数据稀缺时整体下调置信度
        confidence = round(max(0.0, min(1.0, base_conf * (0.5 + 0.5 * data_sufficiency))), 3)
        matched = rules.get(project_type, [])
        rule_summary = "；".join(r["detail"] for r in matched[:4]) or "（无）"
        reason = f"判定为「{project_type}」，主要依据：{rule_summary}。"

    if missing:
        notes.append(f"关键输入字段缺失：{', '.join(missing)}（已据此下调置信度，未编造）。")
    if dim_conf < 0.3:
        notes.append("四维分析置信度偏低（范围内数据较少），类型判定可靠性受限。")

    candidates = [
        {"project_type": t, "raw_score": round(s, 2), "matched_rule_count": len(rules.get(t, []))}
        for t, s in ranked
    ]

    evidence_ids = _score_evidence_ids(project.id, four_dim)

    if persist:
        ac.clear_dimension_results(db, project.id, DIMENSION)
        type_evid = ac.record_metric(
            db, project_id=project.id, dimension=DIMENSION, ring=None,
            metric_key="project_type", text=project_type,
            summary="项目类型识别结果（规则+指标，非大模型）",
            confidence=confidence,
            metadata={
                "matched_rules": [r["rule"] for r in matched],
                "candidates": {t: round(s, 2) for t, s in ranked},
                "allowed_splits": four_dim.get("allowed_splits"),
                "used_test": four_dim.get("used_test", False),
            },
        )
        conf_evid = ac.record_metric(
            db, project_id=project.id, dimension=DIMENSION, ring=None,
            metric_key="type_confidence", value=confidence, unit="ratio",
            summary="项目类型识别置信度（含数据完整度与四维置信度修正）",
            confidence=confidence,
            metadata={"data_sufficiency": data_sufficiency},
        )
        evidence_ids = [type_evid, conf_evid, *evidence_ids]
        project.project_type = project_type
        db.commit()

    logger.info(
        "classify project_id=%s type=%s conf=%.3f total_score=%.1f suff=%.3f used_test=%s",
        project.id, project_type, confidence, total, data_sufficiency,
        four_dim.get("used_test", False),
    )

    return {
        "project_id": project.id,
        "project_type": project_type,
        "confidence": confidence,
        "matched_rules": matched,
        "reason": reason,
        "candidates": candidates,
        "data_sufficiency": data_sufficiency,
        "missing_fields": missing,
        "allowed_splits": four_dim.get("allowed_splits", []),
        "include_test": four_dim.get("include_test", False),
        "used_test": four_dim.get("used_test", False),
        "evidence_ids": evidence_ids,
        "notes": notes,
    }
