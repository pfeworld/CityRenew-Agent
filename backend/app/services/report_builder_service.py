"""第12G：正式报告内容生成（ReportBuilder）。

对齐 SC/报告模版.docx 的「9 章固定结构 + 四张三圈层量化表」，生成正式城市更新前期
策划报告内容（结构化 JSON）。文字结论由确定性模板组织，所有数字来自本地分析结果
（analysis_orchestrator.run_full_analysis），案例参考来自 CaseLearningService。

正式报告口径（红线）：
- 严格 9 章，缺失章节（案例参考 / 附录）自动补齐；
- 四张量化表均按核心 / 近邻 / 辐射三圈层填写，有数据填数据，无数据写「暂无数据」；
- 全文中文业务表达：不出现 raw key、英文变量名、`____`、Markdown 星号 / #、训练评测字段；
- 仅 train/val，不触碰 test；不调用外部 API；大模型不参与事实数字生成。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Project
from app.services import analysis_orchestrator as orch
from app.services import case_learning_service as cases
from app.services import report_template_service as tpl

logger = logging.getLogger("cityrenew.report.builder")

MISSING = "暂无数据"
CURRENT_YEAR = datetime.now().year


# --------------------------------------------------------------------------- #
# 数值格式化（中文友好，绝不输出 raw key / ____ / 英文）
# --------------------------------------------------------------------------- #
def _ring(dim: dict, ring_name: str) -> dict:
    for r in dim.get("rings") or []:
        if r.get("ring") == ring_name:
            return r
    return {}


def _fmt_count(v: Any) -> str:
    if v is None:
        return MISSING
    try:
        return f"{int(round(float(v)))}"
    except (TypeError, ValueError):
        return MISSING


def _fmt_price(v: Any) -> str:
    if v is None:
        return MISSING
    try:
        return f"{int(round(float(v)))}"
    except (TypeError, ValueError):
        return MISSING


def _fmt_score(v: Any) -> str:
    if v is None:
        return MISSING
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return MISSING


def _fmt_ratio(v: Any) -> str:
    if v is None:
        return MISSING
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return MISSING


# --------------------------------------------------------------------------- #
# 四张量化表（核心 / 近邻 / 辐射）+ 缺失值分类（不以 0 冒充）
# --------------------------------------------------------------------------- #
# 缺失状态 → 单元格展示文案
TBD = "待补充"
NO_SAMPLE = "暂无有效样本"
NOT_APPLICABLE = "不适用"
_STATUS_TEXT = {
    "no_redline": MISSING,         # 无红线时核心范围已用中心点缓冲真实归集，不再占位
    "missing": MISSING,            # 应有但暂缺 → 据实标注暂无数据
    "no_sample": NO_SAMPLE,        # 圈层内无有效样本
    "not_applicable": NOT_APPLICABLE,
    "unavailable_field": MISSING,  # 数据集无该字段
}
# 各维度用于判断「圈层是否有有效样本」的计数字段
_COUNT_FIELD = {"location": "total", "population": "grid_count",
                "housing": "sample_count", "industry": "enterprise_count"}


def _building_age(ring: dict) -> Any:
    """建筑平均使用年限 = 当前年 - 有效建成年代中位数（基于该圈层可得 year 样本）。"""
    ys = ring.get("year_summary") or {}
    median_year = ys.get("median_year")
    if not median_year:
        return None
    return CURRENT_YEAR - int(median_year)


def _pct_text(struct: dict | None, order: list[tuple[str, str]]) -> Any:
    """把画像结构占比格式化为简短中文文本（真实分析占比，缺样本返回 None）。"""
    if not struct or not struct.get("available"):
        return None
    ratios = struct.get("ratios") or {}
    parts = [f"{label}{round(ratios.get(key, 0) * 100)}%" for key, label in order
             if ratios.get(key, 0) > 0]
    return " ".join(parts[:4]) if parts else None


_AGE_ORDER = [("<18", "<18岁"), ("18-24", "18-24岁"), ("25-34", "25-34岁"),
              ("35-44", "35-44岁"), ("45-54", "45-54岁"), ("55-64", "55-64岁"), ("65+", "65岁+")]
_CONSUME_ORDER = [("high", "高"), ("middle", "中"), ("low", "低")]


def _age_text(ring: dict) -> Any:
    return _pct_text(ring.get("age_structure"), _AGE_ORDER)


def _consume_text(ring: dict) -> Any:
    return _pct_text(ring.get("consumption_structure"), _CONSUME_ORDER)


def _econ_activity(ring: dict) -> Any:
    """区域经济活跃度指数：基于企业密度（家/km²）归一化到 0-100 的可解释指数。"""
    d = ring.get("density_per_km2")
    if not isinstance(d, (int, float)) or d <= 0:
        return None
    return round(min(100.0, d / 30.0 * 100.0), 1)  # 30 家/km² 为经验高位


def _classify(value, *, ring: str, has_redline: bool, ring_has_sample: bool,
              unavailable: bool, fmt) -> tuple[str, str]:
    """返回 (展示文案, 缺失状态)。绝不以 0 冒充缺失，也不再用「待补充」占位。

    核心范围即使暂无红线，也用中心点缓冲（默认150米）的真实归集结果填写；
    仅当某字段数据集层面确实缺失（unavailable）或圈层内无有效样本时，据实标注。
    """
    if unavailable:
        return MISSING, "unavailable_field"
    if isinstance(value, str):
        return (value, "value") if value.strip() else (MISSING, "missing")
    if value is None:
        return (NO_SAMPLE, "no_sample") if not ring_has_sample else (MISSING, "missing")
    try:
        num = float(value)
    except (TypeError, ValueError):
        return MISSING, "missing"
    if num == 0:
        return (NO_SAMPLE, "no_sample") if not ring_has_sample else ("0", "true_zero")
    return fmt(value), "value"


def _build_tables(full: dict, src: dict, emap: dict, *, has_redline: bool) -> dict[str, dict]:
    fd = full.get("four_dimension", {})
    dims = {"location": fd.get("poi") or {}, "population": fd.get("population") or {},
            "housing": fd.get("housing") or {}, "industry": fd.get("industry") or {}}
    rings = ["core", "nearby", "radiation"]

    def row(table_key, row_key, label, field, fmt, *, unavailable=False, value_fn=None):
        dim = dims[table_key]
        cnt_field = _COUNT_FIELD[table_key]
        cells = []
        for rg in rings:
            rd = _ring(dim, rg)
            cnt = rd.get(cnt_field)
            ring_has_sample = bool(isinstance(cnt, (int, float)) and cnt > 0)
            raw = value_fn(rd) if value_fn else (rd.get(field) if field else None)
            disp, status = _classify(raw, ring=rg, has_redline=has_redline,
                                     ring_has_sample=ring_has_sample, unavailable=unavailable, fmt=fmt)
            emap[f"{table_key}:{row_key}:{rg}"] = {
                "label": label, "ring": rg, "display": disp, "status": status,
                "value": round(float(raw), 4) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None,
            }
            # value 与 true_zero（真实计数为 0）均登记，保证量化表每个数值单元格可回比
            if status in ("value", "true_zero") and isinstance(raw, (int, float)) and not isinstance(raw, bool):
                src[f"{table_key}:{row_key}:{rg}"] = round(float(raw), 4)
            cells.append(disp)
        return {"label": label, "core": cells[0], "nearby": cells[1], "radiation": cells[2]}

    t1 = {"key": "location", "title": "配套/区位指标表", "columns": tpl.RING_COLUMNS, "rows": [
        row("location", "transport", "轨交/公交站点数量（个）", "transport", _fmt_count),
        row("location", "poi_total", "POI兴趣点总数（个）", "total", _fmt_count),
        row("location", "convenience", "便民服务类POI（个）", "convenience", _fmt_count),
        row("location", "commercial", "商业服务类POI（个）", "commercial", _fmt_count),
        row("location", "public", "公共服务类POI（个）", "public", _fmt_count),
    ]}

    t2 = {"key": "population", "title": "人口指标表", "columns": tpl.RING_COLUMNS, "rows": [
        row("population", "residential", "居住人口（人）", "residential", _fmt_count),
        row("population", "worker", "工作人口（人）", "worker", _fmt_count),
        row("population", "age_structure", "年龄结构（占比）", None, str, value_fn=_age_text),
        row("population", "consume_level", "消费能力分级（占比）", None, str, value_fn=_consume_text),
        row("population", "income_level", "收入水平分级（占比）", None, _fmt_count, unavailable=True),
    ]}

    # 房价/建成年代/出租：本地圈层无样本时，回落到「本区真实成交基线 / 全市租金参考」
    # （真实、可回溯、非编造；近邻/辐射用区级房价基线，建成年代用区级基线，出租用全市参考）。
    _hz = dims["housing"]
    _dpb = _hz.get("district_price_baseline") or {}
    _dyb = _hz.get("district_build_year_baseline") or {}
    _rent = _hz.get("rent_reference") or {}

    def _sh_price(rd):
        v = rd.get("avg_unit_price")
        if v is not None:
            return v
        if rd.get("ring") in ("nearby", "radiation") and _dpb.get("median"):
            return _dpb["median"]
        return None

    def _bage_fb(rd):
        v = _building_age(rd)
        if v is not None:
            return v
        return _dyb.get("median_building_age")

    def _rent_unit(rd):
        return _rent.get("median_rent_unit")

    t3 = {"key": "housing", "title": "房价与空间现状表", "columns": tpl.RING_COLUMNS, "rows": [
        row("housing", "secondhand_price", "二手房均价（元/㎡）", None, _fmt_price, value_fn=_sh_price),
        row("housing", "secondhand_listing", "二手房挂牌数（个）", "sample_count", _fmt_count),
        row("housing", "rent_price", "出租房均价（元/㎡·月）", None, _fmt_price,
            value_fn=_rent_unit, unavailable=not bool(_rent.get("median_rent_unit"))),
        row("housing", "rent_listing", "出租房挂牌数（个）", None, _fmt_count, unavailable=True),
        row("housing", "building_age", "建筑平均使用年限（年）", None, _fmt_count, value_fn=_bage_fb),
        row("housing", "price_growth", "房价历史涨幅（%）", None, _fmt_ratio, unavailable=True),
    ]}

    dominant = dims["industry"].get("dominant_industry")
    t4 = {"key": "industry", "title": "产业/经济指标表", "columns": tpl.RING_COLUMNS, "rows": [
        row("industry", "dominant", "主导产业类型", None, lambda v: str(v),
            value_fn=lambda rd: dominant),
        row("industry", "enterprise_count", "产业企业数量（家）", "enterprise_count", _fmt_count),
        row("industry", "economy_activity", "区域经济活跃度指数", None, _fmt_ratio, value_fn=_econ_activity),
        row("industry", "industry_pop_ratio", "产业人口占比", None, _fmt_ratio, unavailable=True),
    ]}

    return {"location": t1, "population": t2, "housing": t3, "industry": t4}


def _missing_summary(emap: dict) -> dict[str, list[str]]:
    """汇总各缺失状态对应的指标（供附录解释缺失原因）。"""
    out: dict[str, list[str]] = {}
    for info in emap.values():
        st = info.get("status")
        if st in ("value", "true_zero"):
            continue
        label = info.get("label", "")
        out.setdefault(st, [])
        if label and label not in out[st]:
            out[st].append(label)
    return out


# --------------------------------------------------------------------------- #
# 各章节文字（确定性模板，中文业务表达）
# --------------------------------------------------------------------------- #
def _ring_clause(project: Project) -> str:
    return (
        f"本报告统一采用三圈层口径：核心范围为项目红线内（城市更新核心改造区域），"
        f"近邻范围为周边{project.nearby_buffer_m or 500}米（更新直接辐射区域），"
        f"辐射范围为周边{project.radiation_buffer_m or 1500}米（更新全域影响区域）。"
    )


def _resolve_location(project) -> str:
    """解析项目位置（市/区/街道），优先结构化字段，缺失时从地址/项目名解析；绝不写「待补充」。"""
    addr = f"{project.address or ''} {project.name or ''}"
    city = (project.city or "").strip()
    if not city:
        m = re.search(r"([\u4e00-\u9fa5]{2,}?市)", addr)
        city = m.group(1) if m else ""
    district = (project.district or "").strip()
    if not district:
        m = re.search(r"([\u4e00-\u9fa5]{2,}?区)", addr)
        district = m.group(1) if m else ""
    street = (project.street or "").strip()
    if not street:
        m = re.search(r"区([\u4e00-\u9fa5]{2,}?(?:街道|镇|路|大道|弄))", addr)
        street = m.group(1) if m else ""
    return "".join(filter(None, [city, district, street])) or (project.address or "").strip() \
        or f"{project.name}（以项目定位点构建圈层分析）"


def _ch1(project, full, src, facts):
    ptype = full.get("project_type") or MISSING
    f_score = full.get("F_score")
    level = full.get("score_level") or MISSING
    if isinstance(f_score, (int, float)):
        src["comprehensive_score"] = round(float(f_score), 4)
    facts["project_type"] = full.get("project_type")
    facts["score_level"] = full.get("score_level")
    loc = _resolve_location(project)
    paras = [
        f"项目「{project.name}」位于{loc}。{_ring_clause(project)}",
        f"经综合研判，本项目识别更新类型为「{ptype}」，综合评分约 {_fmt_score(f_score)} 分（{level}）。"
        "该判断综合了区位配套、人口客群、房价空间与产业经济四个维度的量化结果，可作为城市更新方向判断的基础依据。",
        "城市更新适配性初步判断：项目所在区域具备开展前期策划的数据基础；在暂未取得项目红线矢量文件的情况下，"
        "核心范围已以项目中心点150米缓冲圆近似归集真实数据，并与500米、1500米圈层共同构成分析口径，"
        "后续可结合红线文件进一步校正核心范围指标。",
    ]
    return {"no": "1", "title": "项目基础概况", "paragraphs": paras,
            "bullets": [
                f"识别更新类型：{ptype}。",
                f"综合评分：{_fmt_score(f_score)} 分（{level}）。",
                f"圈层口径：核心（红线内）/ 近邻（{project.nearby_buffer_m or 500}米）/ 辐射（{project.radiation_buffer_m or 1500}米）。",
            ],
            "tables": []}


def _ch2(project, full, tables, src, facts):
    poi = full.get("four_dimension", {}).get("poi") or {}
    rad = _ring(poi, "radiation")
    score = poi.get("score")
    src["location_score"] = round(float(score), 4) if isinstance(score, (int, float)) else None
    shortboards = poi.get("shortboards_top5") or []
    recommend = poi.get("recommend_top5") or []
    paras = [
        f"区位与配套维度得分约 {_fmt_score(score)} 分。辐射范围内 POI 兴趣点共 {_fmt_count(rad.get('total'))} 个，"
        f"其中商业服务类 {_fmt_count(rad.get('commercial'))} 个、公共服务类 {_fmt_count(rad.get('public'))} 个、"
        f"便民服务类 {_fmt_count(rad.get('convenience'))} 个，交通站点 {_fmt_count(rad.get('transport'))} 个，"
        f"功能混合度约 {_fmt_ratio(rad.get('mix_index'))}。",
    ]
    if shortboards:
        paras.append(f"结合城市更新需求，当前配套短板集中在：{('、'.join(shortboards[:3]))}，"
                     f"建议在更新中优先补足{('、'.join(recommend[:3])) if recommend else '相关服务设施'}，"
                     "以提升片区居住与服务品质。")
    else:
        paras.append("整体配套结构较为均衡，更新中可在现有基础上强化商业活力与公共服务连续性。")
    return {"no": "2", "title": "区位与POI配套分析", "paragraphs": paras,
            "bullets": [], "tables": [tables["location"]]}


def _ch3(project, full, tables, src, facts):
    pop = full.get("four_dimension", {}).get("population") or {}
    rad = _ring(pop, "radiation")
    score = pop.get("score")
    src["population_score"] = round(float(score), 4) if isinstance(score, (int, float)) else None
    main_segment = pop.get("main_segment")
    facts["main_segment"] = main_segment
    grid = rad.get("grid_count")
    has_pop = isinstance(grid, (int, float)) and grid > 0

    if has_pop:
        paras = [
            f"人口与客群维度得分约 {_fmt_score(score)} 分。辐射范围内居住人口约 {_fmt_count(rad.get('residential'))} 人，"
            f"工作人口约 {_fmt_count(rad.get('worker'))} 人，职住比约 {_fmt_ratio(rad.get('job_housing_ratio'))}。",
            f"主力客群特征：{main_segment or '需结合补充调研明确'}。"
            "结合城市更新民生导向，建议围绕主力客群配置社区服务、便民商业与公共活动空间，"
            "提升一刻钟便民生活圈的覆盖与品质。",
            "说明：年龄结构、消费能力与收入水平的分级占比，受现有数据集字段限制暂无法量化，相关单元格标注为暂无数据，"
            "后续可结合补充调研完善。",
        ]
    else:
        # 外区无落圈人口网格：据实标注暂无，绝不以 0 或 POI 密度冒充人口（红线）
        paras = [
            f"人口与客群维度得分约 {_fmt_score(score)} 分。本项目所在范围暂无落圈人口画像网格样本，"
            "居住人口、工作人口与职住比均据实标注为暂无数据。",
            "当前人口画像网格数据仅覆盖部分区域，本片区人口与客群结构需后续接入人口画像或统计网格数据后补充；"
            "结合城市更新以人为本的导向，更新中应同步开展人口与客群调研，据实完善民生服务配置。",
            "说明：居住/工作人口、年龄结构、消费能力与收入水平受现有数据集覆盖范围限制暂无法量化，"
            "相关单元格据实标注为暂无数据，本报告不以 POI 密度等间接指标估算人口，避免伪造人口数据。",
        ]
    return {"no": "3", "title": "人口与客群画像分析", "paragraphs": paras,
            "bullets": [], "tables": [tables["population"]]}


def _ch4(project, full, tables, src, facts):
    housing = full.get("four_dimension", {}).get("housing") or {}
    rad = _ring(housing, "radiation")
    score = housing.get("score")
    src["housing_score"] = round(float(score), 4) if isinstance(score, (int, float)) else None
    interval = housing.get("baseline_interval") or {}
    dpb = housing.get("district_price_baseline") or {}
    dyb = housing.get("district_build_year_baseline") or {}
    rent = housing.get("rent_reference") or {}
    district = housing.get("district") or "本区"
    rad_price = rad.get("avg_unit_price")

    if rad_price is not None:
        p1 = (f"房价与空间现状维度得分约 {_fmt_score(score)} 分。辐射范围二手房均价约 {_fmt_price(rad_price)} 元/㎡，"
              f"单价中位数约 {_fmt_price(rad.get('median_unit_price'))} 元/㎡，样本约 {_fmt_count(rad.get('sample_count'))} 套，"
              f"户均面积约 {_fmt_score(rad.get('avg_area'))} ㎡。")
    else:
        p1 = (f"房价与空间现状维度得分约 {_fmt_score(score)} 分。辐射范围内暂无本地落圈成交样本，"
              f"价格水平采用本区（{district}）授权脱敏成交样本中位价约 {_fmt_price(dpb.get('median'))} 元/㎡作为基线"
              f"（样本约 {_fmt_count(dpb.get('count'))} 套，与房价模型同源、可回溯）。")

    age_txt = ""
    if dyb.get("median_building_age"):
        age_txt = (f"本区典型建成年代约 {int(dyb['median_build_year'])} 年、平均楼龄约 "
                   f"{int(dyb['median_building_age'])} 年，")
    p2 = (f"价值与空间诊断：区域价格基线中枢约 {_fmt_price(interval.get('mid'))} 元/㎡；{age_txt}"
          "既有建筑与低效空间具备城市更新存量盘活与品质提升的空间。")

    if rent.get("median_rent_unit"):
        p3 = (f"租赁市场参考：全市租金中位约 {_fmt_price(rent['median_rent_unit'])} 元/㎡·月"
              f"（来源科研授权租赁挂牌约 {_fmt_count(rent.get('sample_count'))} 条；该数据已去除小区信息，"
              "仅作全市参考、不分区）。出租房挂牌量与房价历史涨幅受现有数据集限制暂无法量化，相关单元格据实标注。")
    else:
        p3 = "说明：出租房均价与挂牌量、房价历史涨幅受现有数据集限制暂无法量化，相关单元格据实标注。"

    return {"no": "4", "title": "房价与空间现状诊断", "paragraphs": [p1, p2, p3],
            "bullets": [], "tables": [tables["housing"]]}


def _ch5(project, full, tables, src, facts):
    industry = full.get("four_dimension", {}).get("industry") or {}
    rad = _ring(industry, "radiation")
    score = industry.get("score")
    src["industry_score"] = round(float(score), 4) if isinstance(score, (int, float)) else None
    dominant = industry.get("dominant_industry")
    facts["dominant_industry"] = dominant
    suggestions = industry.get("adaptation_suggestions") or []
    paras = [
        f"产业与区域经济维度得分约 {_fmt_score(score)} 分。辐射范围内产业企业约 {_fmt_count(rad.get('enterprise_count'))} 家，"
        f"产业密度约 {_fmt_ratio(rad.get('density_per_km2'))} 家/平方公里，主导产业类型为「{dominant or '需结合区域产业规划确认'}」。",
        f"功能定位锚定：{(suggestions[0] if suggestions else '建议结合区域产业基础与更新目标，导入与片区能级匹配的复合功能。')}",
        "说明：区域经济活跃度指数与产业人口占比受现有数据集限制暂无法量化，相关单元格标注为暂无数据。",
    ]
    return {"no": "5", "title": "产业与区域经济分析", "paragraphs": paras,
            "bullets": [], "tables": [tables["industry"]]}


def _ch6(project, full, src, facts, dprof, refs):
    facts["case_ref_count"] = len(refs)
    facts["case_ref_names"] = [r["name"] for r in refs]
    ctype = dprof.get("canonical_type") or full.get("project_type") or "城市更新"

    # 6.1 可参考案例筛选（不暴露内部文件名/链接）
    refs_bullets = [f"{r['name']}：{r['applicable']}" for r in refs]
    sec61 = {"title": "6.1 可参考案例筛选",
             "paragraphs": [f"结合本项目「{ctype}」的更新方向，选取若干相近的城市更新案例作为参考，"
                            "重点借鉴其空间、功能、运营与实施层面的可复用策略。"],
             "bullets": refs_bullets or ["暂未匹配到高度相近的案例，建议结合补充资料后进一步比选。"]}

    # 6.2 案例经验转译（按项目类型差异化）
    transl = list(dprof.get("emphasis") or [])
    if not transl:
        transl = ["存量空间盘活", "公共服务补短板", "功能复合与活力提升"]
    sec62 = {"title": "6.2 案例经验转译",
             "paragraphs": [f"将上述案例经验转译为适配本项目「{ctype}」的更新策略："],
             "bullets": transl[:6]}

    # 6.3 政策适配与合规路径
    sec63 = {"title": "6.3 政策适配与合规路径",
             "paragraphs": ["结合现行城市更新政策导向，明确本项目的合规要点与实施路径："],
             "bullets": [
                 "存量空间盘活：坚持「留改拆」并举、以保护保留为主，优先盘活低效存量空间；",
                 "民生优先：把居民诉求与公共利益放在首位，更新过程不降低基本居住与服务保障；",
                 "公共服务补短板：依据控规与公共要素配置要求，补足教育、医疗、养老、文体等设施；",
                 "历史文化与风貌保护：对历史建筑、风貌肌理依法保护，避免大拆大建；",
                 "产业导入合规：导入功能与业态须符合用地性质与产业准入要求；",
                 "分阶段实施：按近期、中期、远期分期推进，确保资金、产权与运营可落地。",
             ]}
    return {"no": "6", "title": "案例参考与政策适配分析",
            "paragraphs": ["本章选取与本项目更新方向相近的城市更新案例作为参考，"
                           "并结合现行城市更新政策明确合规路径。"],
            "bullets": [], "tables": [], "sections": [sec61, sec62, sec63]}


def _ch7(project, full, src, facts):
    strat = full.get("strategy_result") or {}
    opportunities = strat.get("key_opportunities") or []
    risks = strat.get("key_risks") or []
    paras = [
        "综合区位配套、人口客群、房价空间与产业经济四维结果，对项目在城市更新导向下的核心需求与核心潜力研判如下。",
    ]
    need_bullets = [f"核心需求：{r}" for r in risks[:3]] or ["核心需求：补足配套短板、提升公共服务与空间品质。"]
    pot_bullets = [f"核心潜力：{o}" for o in opportunities[:3]] or ["核心潜力：存量空间盘活与价值提升空间明确。"]
    return {"no": "7", "title": "城市更新导向下的需求研判与潜力分析", "paragraphs": paras,
            "bullets": need_bullets + pot_bullets, "tables": []}


def _ch8(project, full, src, facts, dprof):
    strat = full.get("strategy_result") or {}
    positioning = strat.get("update_positioning")
    actions = strat.get("priority_actions") or []
    count = full.get("strategy_count")
    if isinstance(count, (int, float)):
        src["strategy_count"] = round(float(count), 4)
    facts["strategy_count"] = count
    facts["update_positioning"] = positioning
    ctype = dprof.get("canonical_type") or full.get("project_type") or "城市更新"
    core_logic = dprof.get("core_logic")
    goal = project.expected_direction
    emphasis = list(dprof.get("emphasis") or [])

    sec81 = {"title": "8.1 总体定位建议",
             "paragraphs": [
                 f"结合项目「{ctype}」类型、用户更新目标"
                 + (f"（{goal}）" if goal else "") + "、本地多维分析与案例经验，"
                 f"建议总体定位为：{positioning or ('以' + (core_logic or '存量盘活与民生服务升级') + '为核心的城市更新')}。"
             ], "bullets": []}

    sp = ["建筑更新：对老旧、低效建筑分级实施修缮、改造与功能置换；",
          "街区界面：优化沿街界面与第五立面，强化可识别性与连续性；",
          "公共空间：增补口袋公园、广场与活动场地，提升公共空间品质；",
          "慢行系统：织补步行与骑行网络，改善可达性与体验；",
          "红线内外衔接：统筹红线内更新与周边街区联动，避免孤岛式改造；",
          "低效空间盘活：挖掘低效用地与闲置空间的更新潜力。"]
    if dprof.get("canonical_type") == "老旧仓库":
        sp.insert(3, "外挂连廊/立体动线：以连廊、平台串联多栋既有建筑，提升上层可达性；")
    sec82 = {"title": "8.2 空间优化建议", "paragraphs": [], "bullets": sp}

    sec83 = {"title": "8.3 配套完善建议", "paragraphs": [], "bullets": [
        "公共服务：补足教育、医疗、养老等基本公共服务设施；",
        "便民生活：完善一刻钟便民生活圈，提升日常服务可达性；",
        "社区服务：增设社区活动、文体与综合服务空间；",
        "商业配套：优化便民商业与体验业态结构；",
        "停车与静态交通：增补停车供给、优化静态交通组织；",
        "适老化：推进无障碍与适老化改造，关注全龄友好；",
        "文化活动：预留文化展示与公共活动场所。"]}

    fn = ["业态导入：导入与片区能级匹配的复合业态；",
          "产业适配：结合区域产业基础推进产城融合；",
          "文化消费：营造文化体验与消费场景；",
          "商业活力：激活首层界面与沿街商业活力；",
          "社区复合：促进居住、办公、服务与休闲复合；",
          "运营导入：前置运营策划，推动招商与内容运营联动。"]
    sec84 = {"title": "8.4 功能提升建议", "paragraphs": [], "bullets": fn}

    impl = []
    if actions:
        impl.append(f"近期优先事项：{actions[0]}；")
    else:
        impl.append("近期优先事项：补足民生短板、改善公共空间与街区界面；")
    impl += [
        "中期提升内容：推进功能置换、业态升级与运营导入；",
        "后续复核资料：补齐项目红线、建筑安全、产权与控规等资料；",
        "合规重点：落实留改拆、风貌保护与公共要素配置要求；",
        "运营策略：明确运营主体与全时段内容运营机制；",
        "投资实施节奏：按分期开发与现金流平衡安排实施节奏。"]
    if emphasis:
        impl.append(f"差异化重点：本项目应突出{('、'.join(emphasis[:3]))}。")
    sec85 = {"title": "8.5 实施重点建议", "paragraphs": [], "bullets": impl}

    return {"no": "8", "title": "项目前策核心建议", "paragraphs": [],
            "bullets": [], "tables": [],
            "sections": [sec81, sec82, sec83, sec84, sec85]}


def _ch9(project, full, src, facts, emap, has_redline):
    fd = full.get("four_dimension", {})
    poi_rad = _ring(fd.get("poi") or {}, "radiation")
    house_rad = _ring(fd.get("housing") or {}, "radiation")
    ind_rad = _ring(fd.get("industry") or {}, "radiation")
    pop_rad = _ring(fd.get("population") or {}, "radiation")
    miss = _missing_summary(emap)

    sec91 = {"title": "9.1 数据来源说明",
             "paragraphs": ["本报告的数字与结论来自下列来源，均可回溯，未作任何编造："],
             "bullets": [
                 "用户输入：项目位置、现状问题与更新目标；",
                 "上传资料：用户在对话中提供并解析的项目文件；",
                 "黑客松比赛提供专用数据库（房价、人口、产业等多源数据）；",
                 "POI 兴趣点数据（比赛专用数据库 + 高德开放POI，GCJ02，覆盖上海全市）；",
                 "人口画像数据（比赛专用数据库）；",
                 "房价交易数据（比赛专用数据库；本地无落圈成交样本的区域，价格基线采用科研授权脱敏成交样本按行政区中位价，与正式房价模型同源）；",
                 "建成年代/楼龄基线（科研授权脱敏成交样本按行政区中位，覆盖上海全市16区）；",
                 "租金参考（科研授权租赁挂牌，全市口径；该数据已去除小区信息，不分区，仅作全市参考）；",
                 "产业经济数据（比赛专用数据库 + 高德开放企业POI，覆盖上海全市）；",
                 "城市更新案例样本；",
                 "本地模型分析结果（项目类型识别与房价分析等）。"]}

    if has_redline:
        ring_paras = [
            "核心范围：项目红线内（城市更新核心改造区域）；",
            "近邻范围：周边500米（更新直接辐射区域）；",
            "辐射范围：周边1500米（更新全域影响区域）。"]
    else:
        ring_paras = [
            "核心范围：当前暂未获得项目红线矢量文件，核心范围以项目定位点150米缓冲圆近似（城市更新核心改造区域），"
            "相关数据均由该缓冲范围内真实归集计算得出；待红线文件补齐后可进一步校正；",
            "近邻范围：周边500米（更新直接辐射区域）；",
            "辐射范围：周边1500米（更新全域影响区域）。"]
    sec92 = {"title": "9.2 圈层口径说明", "paragraphs": [], "bullets": ring_paras}

    gaps: list[str] = []
    if not has_redline:
        gaps.append("项目红线：暂未获得矢量红线文件，核心范围已以项目定位点150米缓冲圆近似，"
                    "其指标由该范围内真实数据归集得出，并非占位。")
    if miss.get("unavailable_field"):
        gaps.append("数据集未提供的字段（不可编造）：" + "、".join(miss["unavailable_field"])
                    + "，据实标注为暂无数据；其中收入水平、出租房挂牌量、房价历史涨幅、产业人口占比"
                    "为本次比赛数据集未覆盖字段（出租房均价已以全市租金参考补充，建成年代/楼龄已以行政区基线补充）。")
    if miss.get("no_sample"):
        gaps.append("圈层内暂无有效样本：" + "、".join(miss["no_sample"]) + "，据实标注为暂无有效样本。")
    # 人口网格覆盖限制（外区诚实说明，不以 POI 密度估算人口冒充真实人口）
    pop_grid = pop_rad.get("grid_count")
    if not pop_grid and (pop_rad.get("residential") in (None, 0)):
        gaps.append("人口网格：当前人口画像网格数据仅覆盖部分区域，本项目所在范围暂无落圈人口网格样本，"
                    "居住/工作人口据实标注为暂无数据，需后续接入人口画像或统计网格数据后补充；"
                    "本报告不以 POI 密度等间接指标估算人口，避免伪造人口数据。")
    gaps.append("以上均按真实数据与缺失类型据实标注，未以 0 或臆测数值填补，也未使用任何占位填充。")
    sec93 = {"title": "9.3 数据缺失说明", "paragraphs": [], "bullets": gaps}

    sec94 = {"title": "9.4 可信边界说明",
             "paragraphs": ["本报告为前策阶段辅助研判，后续仍需结合实地踏勘、控规、产权、建筑安全、"
                            "招商运营、实施主体等资料进一步复核，再形成正式实施方案。"],
             "bullets": []}

    facts["sample_radiation"] = {
        "poi": poi_rad.get("total"), "population_grid": pop_rad.get("grid_count"),
        "housing": house_rad.get("sample_count"), "industry": ind_rad.get("enterprise_count")}
    return {"no": "9", "title": "附录：数据口径与样本说明", "paragraphs": [],
            "bullets": [], "tables": [], "sections": [sec91, sec92, sec93, sec94]}


# --------------------------------------------------------------------------- #
# 模板「就地填空」用的分析填充值（每个 ____ 横线对应一条结论）
# --------------------------------------------------------------------------- #
GENERATOR_NAME = "CityRenew Agent 城市更新前期策划智能体"
# 项目类型 → 模板「项目类型」复选项
_TYPE_CHOICE = {
    "老旧仓库": "城市更新类", "工业遗存": "城市更新类", "老旧社区": "社区配套升级",
    "商业街区": "街区提升", "公共空间优化": "公共空间优化", "综合功能地块": "综合功能地块",
}


def _present(disp: str) -> bool:
    return bool(disp) and disp not in (MISSING, TBD, NO_SAMPLE, NOT_APPLICABLE)


def _trim(s: Any) -> str:
    return str(s or "").strip().rstrip("。；;，,、 ")


def _strip_suffix(text: str | None, *sufs: str) -> str:
    t = (text or "").strip()
    for s in sufs:
        if t.endswith(s):
            return t[: -len(s)]
    return t


def _template_fills(project: Project, full: dict, dprof: dict) -> dict[str, Any]:
    """生成模板每个横线/复选项对应的分析填充文本（全部可回溯，不编造）。"""
    fd = full.get("four_dimension", {})
    poi = fd.get("poi") or {}
    pr = _ring(poi, "radiation")
    pop = fd.get("population") or {}
    por = _ring(pop, "radiation")
    hou = fd.get("housing") or {}
    hr = _ring(hou, "radiation")
    ind = fd.get("industry") or {}
    ir = _ring(ind, "radiation")
    strat = full.get("strategy_result") or {}

    ctype = dprof.get("canonical_type") or full.get("project_type") or "城市更新"
    emphasis = list(dprof.get("emphasis") or [])
    emph_txt = "、".join(emphasis[:3]) if emphasis else "存量盘活、配套补短板、功能与活力提升"
    core_logic = dprof.get("core_logic") or "存量盘活与民生服务升级"
    goal = (project.expected_direction or "").strip()
    demand = (project.update_demand or "").strip()
    shortboards = poi.get("shortboards_top5") or []
    recommend = poi.get("recommend_top5") or []
    main_segment = pop.get("main_segment")
    dominant = ind.get("dominant_industry")
    suggestions = ind.get("adaptation_suggestions") or []
    interval = hou.get("baseline_interval") or {}
    risks = strat.get("key_risks") or []
    opps = strat.get("key_opportunities") or []
    positioning = strat.get("update_positioning")

    poi_total = _fmt_count(pr.get("total"))
    transport = _fmt_count(pr.get("transport"))
    mix = _fmt_ratio(pr.get("mix_index"))
    # 人口：仅在有落圈网格样本时取值，外区无样本据实暂无（不以 0 冒充）
    pop_grid = por.get("grid_count")
    has_pop = isinstance(pop_grid, (int, float)) and pop_grid > 0
    res = _fmt_count(por.get("residential")) if has_pop else MISSING
    worker = _fmt_count(por.get("worker")) if has_pop else MISSING
    # 房价：本地无落圈成交时回落本区授权脱敏成交中位价基线（与房价模型同源）
    dpb = hou.get("district_price_baseline") or {}
    rad_price = hr.get("avg_unit_price")
    price_is_baseline = rad_price is None and dpb.get("median") is not None
    price = _fmt_price(rad_price if rad_price is not None else dpb.get("median"))
    mid = _fmt_price(interval.get("mid"))
    sample = _fmt_count(hr.get("sample_count"))
    ent = _fmt_count(ir.get("enterprise_count"))

    # 优势配套类别（取便民/商业/公共中数量最高者）
    cat_map = {"便民服务": pr.get("convenience"), "商业服务": pr.get("commercial"),
               "公共服务": pr.get("public")}
    cat_valid = {k: v for k, v in cat_map.items() if isinstance(v, (int, float)) and v > 0}
    adv_cat = max(cat_valid, key=cat_valid.get) if cat_valid else None

    def grounded(num: str, tmpl: str, fb: str) -> str:
        return tmpl if _present(num) else fb

    conc: dict[str, str] = {}
    # 2.2 区位与POI
    conc["区位通达性与POI配套密度"] = grounded(
        poi_total,
        f"辐射范围内POI兴趣点共约{poi_total}个、交通站点约{transport}个，功能混合度约{mix}，"
        f"区位通达性与配套密度具备开展{ctype}更新的基础。",
        "辐射范围内POI与交通站点暂无有效样本，区位通达性以现有数据据实研判。")
    conc["配套优势与缺失环节"] = (
        (f"现状配套优势集中在{adv_cat}类设施" if adv_cat else "现状配套结构总体均衡")
        + ("，主要缺失环节为" + "、".join(shortboards[:3]) + "。" if shortboards
           else "，缺失环节需结合实地踏勘进一步确认。"))
    conc["配套优化方向"] = (
        "建议在更新中优先补足" + ("、".join(recommend[:3]) if recommend else "民生与公共服务类配套")
        + ("，并紧扣" + demand + "的现状诉求" if demand else "")
        + "，提升片区服务连续性与可达性。")
    # 3.2 人口
    conc["核心人群构成与特征"] = grounded(
        res,
        f"辐射范围常住人口约{res}人、工作人口约{worker}人；"
        + (f"主力客群为{main_segment}。" if main_segment else "客群结构需结合补充调研明确。"),
        "本项目所在范围暂无落圈人口画像网格样本，居住/工作人口据实标注为暂无数据，"
        "核心人群构成需后续接入人口画像或统计网格数据后研判（不以 POI 密度估算人口）。")
    conc["人群行为与核心需求"] = (
        "结合城市更新以人为本逻辑，"
        + (f"{main_segment}客群" if main_segment else "片区客群")
        + "对便民服务、公共空间与品质生活的需求突出，"
        + (f"并与本项目「{goal}」的目标相呼应。" if goal else "需在更新中重点回应。"))
    conc["客群定位适配建议"] = (
        f"建议围绕主力客群，配置与{ctype}相适配的社区服务、便民商业与公共活动空间，"
        f"突出{emph_txt}。")
    # 4.2 房价与空间
    conc["区域房价水平与走势"] = grounded(
        price,
        (f"辐射范围二手房均价约{price}元/㎡"
         if not price_is_baseline
         else f"辐射范围内暂无本地落圈成交样本，价格水平采用本区授权脱敏成交中位价约{price}元/㎡作为基线（与房价模型同源）")
        + (f"，价格基线中枢约{mid}元/㎡" if _present(mid) else "")
        + "；历史涨幅数据暂缺，走势以现状价格水平研判。",
        "辐射范围内房价交易暂无有效样本，区域价格水平以现有数据据实研判。")
    conc["居住空间的供需关系"] = grounded(
        sample,
        f"二手房有效样本约{sample}套，反映一定的存量交易活跃度；"
        "出租房供需数据暂缺，后续可补充以完善供需研判。",
        "辐射范围内房价样本暂无有效样本，供需关系以现有数据据实研判。")
    conc["空间改造与价值提升潜力"] = (
        "既有建筑与低效空间具备更新盘活潜力，"
        + (f"结合约{mid}元/㎡的价格基线，" if _present(mid) else "")
        + f"更新后在{core_logic}方向具备明确的价值释放空间。")
    # 5.2 产业
    conc["区域产业布局与经济特征"] = grounded(
        ent,
        f"辐射范围产业企业约{ent}家，"
        + (f"主导产业类型为「{dominant}」。" if dominant else "主导产业有待结合区域产业规划确认。"),
        "辐射范围内产业点位暂无有效样本，区域产业布局以现有数据据实研判。")
    conc["产业与项目的适配性"] = (
        "结合产城融合逻辑，"
        + (f"建议功能定位锚定{_trim(suggestions[0])}。" if suggestions
           else f"建议导入与片区能级匹配、契合{ctype}方向的复合功能。"))
    conc["产业带动项目发展的潜力"] = (
        (f"可依托「{dominant}」相关产业升级" if dominant else "可依托区域产业基础")
        + "带动项目功能重塑与片区活力提升，形成产业与空间更新的良性循环。")
    # 6.1 核心需求 / 6.2 核心潜力（模板正文「需求研判」章）
    conc["基础民生需求"] = (
        "补足" + ("、".join(shortboards[:3]) if shortboards else "公共服务与便民配套")
        + "等基础民生设施，完善一刻钟便民生活圈。")
    conc["品质提升需求"] = "提升公共空间、街区界面与环境品质，改善慢行体验与整体风貌。"
    conc["功能适配需求"] = f"结合客群与产业需求，推进功能重塑与业态升级，突出{emph_txt}。"
    conc["空间利用潜力"] = "老旧与低效空间存在更新盘活潜力，可通过修缮、改造与功能置换释放价值。"
    conc["功能升级潜力"] = (
        (f"依托{dominant}及" if dominant else "依托") + "客群基础，片区具备功能升级与业态优化潜力。")
    conc["价值释放潜力"] = grounded(
        mid,
        f"价格基线中枢约{mid}元/㎡，城市更新对资产价值与片区活力的提升潜力明确。",
        "结合房价与经济数据，城市更新具备价值释放潜力，待数据补齐后进一步量化。")

    # 1.2 适配性
    ch1_adapt = (
        f"经多源数据与本地模型研判，本项目识别为「{ctype}」更新方向，"
        f"核心更新逻辑为{core_logic}"
        + (f"；结合用户目标「{goal}」，" if goal else "，")
        + f"宜重点把握{emph_txt}。")

    # 7.1-7.5 前策核心建议（按 guidance 关键词匹配填充）
    advice = {
        "明确项目总体定位": (positioning or f"以{core_logic}为核心的{ctype}更新")
        + (f"，紧扣用户目标「{goal}」。" if goal else "。"),
        "针对老旧建筑、低效空间": (
            "对老旧、低效建筑分级实施修缮、改造与功能置换，盘活存量空间，"
            "优化沿街界面与公共空间，统筹红线内外更新衔接"
            + ("；本项目宜突出" + "、".join(emphasis[:2]) + "。" if emphasis else "。")),
        "结合城市更新民生需求": (
            "补足教育、医疗、养老与文体等公共服务，完善一刻钟便民生活圈，"
            "增补停车与静态交通供给，推进无障碍与适老化改造。"),
        "结合产业适配性": (
            "导入与片区能级匹配的复合业态"
            + (f"（如{_trim(suggestions[0])}）" if suggestions else "")
            + "，推进产城融合，激活首层界面与文化消费场景，前置运营策划。"),
        "明确城市更新实施优先级": (
            "按近期补短板、中期提功能、远期强运营分期推进；"
            + (f"近期重点推进{_trim(strat['priority_actions'][0])}；"
               if strat.get("priority_actions") else "")
            + "落实留改拆与风貌保护要求，明确运营主体与投资实施节奏。"),
    }

    location_text = _resolve_location(project)

    return {
        "subtitle": f"【{project.name}】前策大数据分析报告（含城市更新逻辑）",
        "generator": GENERATOR_NAME,
        "location_text": location_text,
        "type_choice": _TYPE_CHOICE.get(ctype, "城市更新类"),
        "ch1_adapt": ch1_adapt,
        "conclusions": conc,
        "advice": advice,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_report(db: Session, project: Project, *, case_style_key: str | None = None) -> dict[str, Any]:
    full = orch.run_full_analysis(db, project, include_test=False)

    src: dict[str, Any] = {}
    facts: dict[str, Any] = {}
    emap: dict[str, Any] = {}
    has_redline = bool(getattr(project, "boundary_geojson", None))
    tables = _build_tables(full, src, emap, has_redline=has_redline)

    user_text = " ".join(filter(None, [project.name, project.update_demand,
                                        project.expected_direction, project.land_use]))
    dprof = cases.map_to_type_profile(full.get("renewal_type") or full.get("project_type"), user_text)
    refs = cases.select_case_refs_v2(dprof.get("canonical_type"), user_text, limit=3)
    fills = _template_fills(project, full, dprof)

    chapters = [
        _ch1(project, full, src, facts),
        _ch2(project, full, tables, src, facts),
        _ch3(project, full, tables, src, facts),
        _ch4(project, full, tables, src, facts),
        _ch5(project, full, tables, src, facts),
        _ch6(project, full, src, facts, dprof, refs),
        _ch7(project, full, src, facts),
        _ch8(project, full, src, facts, dprof),
        _ch9(project, full, src, facts, emap, has_redline),
    ]

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_id = f"report:p{project.id}:{now}"

    content = {
        "report_id": report_id,
        "project_id": project.id,
        "project_name": project.name,
        "project_type": full.get("project_type"),
        "renewal_type": full.get("renewal_type") or full.get("project_type"),
        "canonical_type": dprof.get("canonical_type"),
        "has_redline": has_redline,
        "generated_at": now,
        "title": "城市更新前期策划报告",
        "ring_columns": tpl.RING_COLUMNS,
        "chapters": chapters,
        "template_fills": fills,
        "directory": [f"{c['no']}. {c['title']}" for c in chapters],
        "tables_index": [t["title"] for t in tables.values()],
        "case_style": case_style_key,
        # ---- 内部回比 / 合规（不进入 docx / pdf 正文）----
        "source_metrics": {k: v for k, v in src.items() if v is not None},
        "source_facts": facts,
        "evidence_map": emap,
        "missing_summary": _missing_summary(emap),
        "case_refs": [{"name": r["name"], "category": r.get("category", "")} for r in refs],
        "evidence_ids": full.get("evidence_ids", []),
        "allowed_splits": full.get("allowed_splits", ["train", "val"]),
        "used_test": full.get("used_test", False),
        "required_chapters": 9,
        "required_tables": 4,
        "chapters_count": len(chapters),
    }
    _persist(content)
    logger.info(
        "report v2 built project_id=%s report_id=%s chapters=%s tables=%s used_test=%s",
        project.id, report_id, len(chapters), len(tables), content["used_test"],
    )
    return content


# --------------------------------------------------------------------------- #
# 落盘 backend/data/outputs/reports_v2/{project_id}/（gitignore）
# --------------------------------------------------------------------------- #
def _report_dir(project_id: int):
    d = settings.data_dir / "outputs" / "reports_v2" / str(project_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_name(report_id: str) -> str:
    return report_id.replace(":", "_")


def _persist(content: dict[str, Any]) -> None:
    d = _report_dir(content["project_id"])
    (d / (safe_name(content["report_id"]) + ".json")).write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "latest.json").write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def load_latest(project_id: int) -> dict[str, Any] | None:
    path = settings.data_dir / "outputs" / "reports_v2" / str(project_id) / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_by_report_id(report_id: str) -> dict[str, Any] | None:
    # report_id 形如 report:p{pid}:{ts}
    try:
        pid = int(report_id.split(":p", 1)[1].split(":", 1)[0])
    except (IndexError, ValueError):
        return None
    path = _report_dir(pid) / (safe_name(report_id) + ".json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
