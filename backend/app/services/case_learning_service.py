"""第12G：案例学习服务（CaseLearningService）。

职责（内部能力，不直接暴露前台）：
- 解析 SC/华建集团相关案例.docx：抽取正式城市更新案例（分类 / 名称 / 摘要），
  用于学习案例写法、分析逻辑与可复用表达，并作为报告第6章「案例参考」的候选库
  与回归测试样本。
- 维护 SC/【2024最新稿】鲁商1992项目（…）V2-图片版 的案例画像。该材料为图片/PPT，
  无可解析文本，故以「策略级」结构化画像沉淀（历史建筑活化、商业更新、外挂连廊、
  空间复合、场所记忆等方向），不编造任何数字。
- 提供按项目类型 / 策略方向 / 关键词的案例检索与「按某案例风格」的表达参考。

红线：
- 仅本地读取 SC；不外发、不入库；日志只输出统计量，不输出案例原文整段。
- 案例仅用于「方向/写法/结构」参考；任何数字仍以本地分析结果为准，不得借案例编造。
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from app.config import settings

logger = logging.getLogger("cityrenew.case.learning")

HUAJIAN_FILE = "华建集团相关案例.docx"
LUSHANG_DIR = "【2024最新稿】鲁商1992项目（完整整合外挂连廊方案）V2-图片版"

_CATEGORY_RE = re.compile(r"^[一二三四五六七八九十]+\s*[、.．]")
_CASE_RE = re.compile(r"^\d+\s*[、.．]")


# --------------------------------------------------------------------------- #
# 华建集团案例解析
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


@lru_cache(maxsize=1)
def parse_huajian_cases() -> list[dict[str, Any]]:
    """解析华建案例 docx → [{category, name, summary, keywords}]。"""
    path = settings.sc_path / HUAJIAN_FILE
    cases: list[dict[str, Any]] = []
    if not path.exists():
        logger.warning("华建案例文件不存在：%s", HUAJIAN_FILE)
        return cases
    try:
        from docx import Document

        doc = Document(str(path))
        category = ""
        current: dict[str, Any] | None = None
        buf: list[str] = []

        def flush() -> None:
            nonlocal current, buf
            if current is not None:
                summary = _clean(" ".join(buf))
                current["summary"] = summary[:400]
                cases.append(current)
            current, buf = None, []

        for para in doc.paragraphs:
            t = _clean(para.text)
            if not t:
                continue
            if _CATEGORY_RE.match(t):
                flush()
                category = re.sub(_CATEGORY_RE, "", t).strip() or t
                continue
            if _CASE_RE.match(t):
                flush()
                name = re.sub(_CASE_RE, "", t).strip().strip("（）()")
                current = {"category": category, "name": name, "summary": "", "keywords": []}
                continue
            if current is not None:
                buf.append(t)
            else:
                # 分类下的引言段也并入当前分类的"概述伪案例"
                buf.append(t)
        flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("华建案例解析失败：%s", type(exc).__name__)
        return cases

    for c in cases:
        c["keywords"] = _keywords_for(c["category"], c["name"], c["summary"])
        c["source"] = "华建集团相关案例"
    cases = [c for c in cases if c.get("name")]
    logger.info("huajian cases parsed: %s", len(cases))
    return cases


def _keywords_for(category: str, name: str, summary: str) -> list[str]:
    text = f"{category} {name} {summary}"
    kw_map = {
        "老旧": "老旧片区改善",
        "存量": "存量盘活",
        "工业遗存": "工业遗存活化",
        "街区": "街区活力提升",
        "历史": "历史风貌保护",
        "公共空间": "公共空间提升",
        "社区": "社区配套升级",
        "综合": "综合功能统筹",
        "商业": "商业更新",
        "文化": "场所文化表达",
    }
    hits = [v for k, v in kw_map.items() if k in text]
    return list(dict.fromkeys(hits))


# --------------------------------------------------------------------------- #
# 鲁商1992 案例：自动解析图片文件夹 + 同名 PPTX（提取每页文字与可视要点）
# --------------------------------------------------------------------------- #
LUSHANG_PPTX = "【2024最新稿】鲁商1992项目（完整整合外挂连廊方案）V2.pptx"

# 可视要点关键词（用于从每页文字归纳 visual_notes，不编造看不到的内容）
_VISUAL_KEYWORDS = {
    "外挂连廊": ["连廊", "外廊", "外挂"],
    "慢行系统": ["慢行", "步行", "漫游", "通道", "垂直交通", "楼梯", "露台"],
    "首层界面激活": ["首层", "沿街", "门头", "橱窗", "外摆", "界面", "展示面"],
    "历史记忆转译": ["石库门", "海派", "历史", "修旧如旧", "肌理", "记忆", "文脉", "保留"],
    "业态组合": ["业态", "餐饮", "零售", "文化艺术", "娱乐", "首店", "网红", "品牌"],
    "工业遗存活化": ["厂房", "船厂", "工业", "烟囱", "钢楼梯", "管道", "结构体"],
    "公共空间复合": ["中庭", "花园", "屋顶", "平台", "公共空间", "口袋"],
    "分级保护改造": ["留改拆", "留、改、拆", "分级", "修缮", "拆除重建", "再生性改造"],
}


def _lushang_cache_path():
    d = settings.data_dir / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d / "lushang_pages.json"


def _extract_pptx_pages(pptx_path) -> dict[int, list[str]]:
    """从 PPTX 直接解析每页 <a:t> 文本（避开 python-pptx 对个别形状的兼容问题）。"""
    import zipfile

    pages: dict[int, list[str]] = {}
    try:
        z = zipfile.ZipFile(str(pptx_path))
        slide_names = sorted(
            [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
            key=lambda n: int(re.search(r"slide(\d+)", n).group(1)),
        )
        at = re.compile(r"<a:t>(.*?)</a:t>", re.S)
        for n in slide_names:
            idx = int(re.search(r"slide(\d+)", n).group(1))
            xml = z.read(n).decode("utf-8", "ignore")
            runs = [re.sub(r"<.*?>", "", t).strip() for t in at.findall(xml)]
            pages[idx] = [r for r in runs if r]
    except Exception as exc:  # noqa: BLE001
        logger.warning("鲁商 PPTX 解析失败：%s", type(exc).__name__)
    return pages


def _visual_notes(text: str) -> list[str]:
    return [tag for tag, kws in _VISUAL_KEYWORDS.items() if any(k in text for k in kws)]


@lru_cache(maxsize=1)
def parse_lushang_pages() -> list[dict[str, Any]]:
    """遍历鲁商1992图片文件夹 + 解析同名 PPTX，逐页建立结构化样本。

    每页：page_id / page_no / image_path / ocr_text(来自 PPTX 文本，非编造) / visual_notes。
    无 OCR 环境时，文本来自 PPTX 幻灯片，图片仅建立索引；解析结果缓存到 processed。
    """
    img_dir = settings.sc_path / LUSHANG_DIR
    pptx = settings.sc_path / LUSHANG_PPTX
    pages: list[dict[str, Any]] = []
    if not img_dir.exists():
        logger.warning("鲁商图片文件夹不存在：%s", LUSHANG_DIR)
        return pages

    images = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda p: p.name,
    )
    slide_text = _extract_pptx_pages(pptx) if pptx.exists() else {}

    for i, img in enumerate(images, start=1):
        runs = slide_text.get(i, [])
        ocr_text = _clean(" ".join(runs))
        pages.append({
            "page_id": f"lushang_p{i:02d}",
            "page_no": i,
            "image_path": str(img.relative_to(settings.sc_path.parent)) if str(img).startswith(str(settings.sc_path.parent)) else img.name,
            "image_name": img.name,
            "ocr_text": ocr_text[:1200],
            "ocr_source": "pptx_text" if runs else "image_index_only",
            "visual_notes": _visual_notes(ocr_text),
        })

    # 缓存（落在 gitignore 的 data 目录，不外发原图）
    try:
        import json as _json

        _lushang_cache_path().write_text(
            _json.dumps({"page_count": len(pages), "pages": pages}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass
    logger.info("鲁商 pages parsed: %s (pptx_text=%s)", len(pages), bool(slide_text))
    return pages


@lru_cache(maxsize=1)
def lushang_profile() -> dict[str, Any]:
    """由解析出的页内容归纳鲁商1992策略级画像（聚合 visual_notes，不编造数字）。"""
    pages = parse_lushang_pages()
    present = (settings.sc_path / LUSHANG_DIR).exists()
    note_freq: dict[str, int] = {}
    for pg in pages:
        for n in pg.get("visual_notes", []):
            note_freq[n] = note_freq.get(n, 0) + 1
    derived = [k for k, _ in sorted(note_freq.items(), key=lambda x: x[1], reverse=True)]

    return {
        "name": "鲁商1992项目（外挂连廊整合方案）",
        "source": "鲁商1992案例（图片版+PPTX）",
        "material_present": present,
        "material_type": "正式项目策划方案（图文/PPT，逐页解析后按策略级沉淀）",
        "page_count": len(pages),
        "background": "依托存量历史建筑与工业遗存的城市更新，以历史记忆为底色，"
                      "通过外挂连廊整合多栋既有建筑、重塑片区商业活力与公共体验。",
        "problems": [
            "既有建筑老化、动线割裂、上层商业可达性弱；",
            "沿街界面封闭、公共空间与场所体验不足；",
            "建筑价值与历史记忆未被充分激活。",
        ],
        "goals": ["商业活力再生", "历史记忆延续", "空间体验升级", "存量价值提升"],
        "spatial_strategy": [
            "通过外挂连廊、平台与楼梯串联多栋既有建筑，重构垂直与水平动线、提升上层可达性；",
            "织补街区慢行网络，形成连续可漫步的公共空间体系；",
            "复合利用既有结构，植入共享中庭、口袋花园与多义性公共空间。",
        ],
        "function_strategy": [
            "首层界面激活，沿街强化门头、橱窗、外摆与体验业态；",
            "中高层导入主题化、体验化、复合化业态（首店/网红/文化艺术）；",
            "以公共活动与文化场景带动全时段人气。",
        ],
        "business_import": ["体验零售", "特色餐饮", "文化展演", "社交休闲", "首店/主理人业态"],
        "public_space": ["连廊+中庭复合公共空间", "屋顶与平台活化", "连续步行漫游系统"],
        "heritage": "采取留改拆并举与分级保护改造，保留石库门/海派/工业遗存特征并加以转译，新旧对话延续文脉。",
        "operation": ["主题化运营与内容策划", "全时段活动运营", "存量空间分期更新与招商联动"],
        "expression": "以'问题诊断—更新目标—空间策略—功能业态—公共空间—历史表达—运营策略'的逻辑递进表达。",
        "reusable_writing": [
            "先点明存量痛点与价值支点，再给出空间—功能—公共—运营的系统性策略；",
            "强调'外挂连廊/空间复合/首层激活/场所记忆/分级保护'等可识别更新手法；",
            "结论落到可实施的更新路径与运营逻辑，而非泛泛而谈。",
        ],
        "benchmarks": _lushang_benchmarks(pages),
        "derived_visual_notes": derived,
        "keywords": [
            "历史建筑活化", "商业更新", "外挂连廊", "空间复合",
            "场所记忆", "首层激活", "存量盘活", "公共空间提升",
        ],
    }


def _lushang_benchmarks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从解析页中识别其对标的上海更新案例（老码头/上生新所/船厂1862/武夷MIX/今潮8弄等）。"""
    marks: list[dict[str, Any]] = []
    seen: set[str] = set()
    name_map = {
        "老码头": "老码头2号仓库（杜月笙油脂仓库活化）",
        "幸福里": "G-ART 幸福里（橡胶研究所改造）",
        "上生新所": "上生新所（生物制品研究所改造）",
        "船厂": "船厂1862（上海船厂改造）",
        "武夷": "上海·武夷MIX320（第一泵厂改造）",
        "旭辉天地": "恒基·旭辉天地",
        "今潮": "今潮8弄（石库门里弄活化）",
    }
    for pg in pages:
        txt = pg.get("ocr_text", "")
        for key, full in name_map.items():
            if key in txt and full not in seen:
                seen.add(full)
                marks.append({"name": full, "page_no": pg["page_no"],
                              "visual_notes": pg.get("visual_notes", [])})
    return marks


# --------------------------------------------------------------------------- #
# 检索 / 风格参考
# --------------------------------------------------------------------------- #
def select_case_refs(
    project_type: str | None,
    directions: list[str] | None,
    keywords: list[str] | None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """按项目类型/策略方向/关键词，选出最相关的案例（华建 + 鲁商）作为第6章参考。"""
    want = " ".join(filter(None, [project_type or ""] + (directions or []) + (keywords or [])))
    refs: list[dict[str, Any]] = []

    # 鲁商画像：当涉及商业/历史/存量/街区/公共空间时优先纳入
    lushang = lushang_profile()
    if any(k in want for k in ["商业", "历史", "存量", "街区", "活力", "公共空间", "文化"]) or not want.strip():
        refs.append({
            "name": lushang["name"],
            "category": "存量商业更新 / 历史建筑活化",
            "applicable": "可参考其外挂连廊、空间复合、首层激活与场所记忆表达，提升商业活力与公共体验。",
            "keywords": lushang["keywords"][:5],
            "source": lushang["source"],
        })

    scored: list[tuple[int, dict[str, Any]]] = []
    for c in parse_huajian_cases():
        score = sum(1 for kw in c.get("keywords", []) if any(part and part in kw for part in want.split()))
        score += sum(1 for kw in (keywords or []) if kw and kw in (c.get("summary", "") + c.get("name", "")))
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    for _, c in scored:
        refs.append({
            "name": c["name"],
            "category": c.get("category", ""),
            "applicable": f"该案例属「{c.get('category','')}」，其更新思路可为本项目提供{('、'.join(c.get('keywords', [])[:3])) or '更新路径'}方面的参考。",
            "keywords": c.get("keywords", [])[:5],
            "source": c.get("source", "华建集团相关案例"),
        })
        if len(refs) >= limit:
            break

    # 兜底：若无匹配华建案例，补一个分类代表
    if len(refs) < limit:
        for c in parse_huajian_cases():
            if all(c["name"] != r["name"] for r in refs):
                refs.append({
                    "name": c["name"],
                    "category": c.get("category", ""),
                    "applicable": f"可作为「{c.get('category','')}」方向的更新案例参考。",
                    "keywords": c.get("keywords", [])[:5],
                    "source": c.get("source", "华建集团相关案例"),
                })
            if len(refs) >= limit:
                break
    return refs[:limit]


def style_reference(style_key: str | None) -> dict[str, Any] | None:
    """返回「按某案例风格」的可复用表达参考。当前支持鲁商1992。"""
    if not style_key:
        return None
    if any(k in style_key for k in ["鲁商", "1992", "lushang"]):
        p = lushang_profile()
        return {
            "case": p["name"],
            "directions": p["goals"],
            "spatial_strategy": p["spatial_strategy"],
            "function_strategy": p["function_strategy"],
            "public_space": p["public_space"],
            "heritage": p["heritage"],
            "keywords": p["keywords"],
            "reusable_writing": p["reusable_writing"],
        }
    return None


def case_corpus_status() -> dict[str, Any]:
    """内部状态（供回归测试 / 内部接口），不暴露前台。"""
    hj = parse_huajian_cases()
    ls = lushang_profile()
    lp = parse_lushang_pages()
    return {
        "huajian_present": (settings.sc_path / HUAJIAN_FILE).exists(),
        "huajian_case_count": len(hj),
        "huajian_categories": sorted({c.get("category", "") for c in hj if c.get("category")}),
        "huajian_fixture_count": len(HUAJIAN_FIXTURES),
        "huajian_with_source_url": sum(1 for f in HUAJIAN_FIXTURES if f.get("source_url")),
        "lushang_present": ls["material_present"],
        "lushang_page_count": len(lp),
        "lushang_pages_with_text": sum(1 for p in lp if p.get("ocr_source") == "pptx_text"),
        "lushang_benchmarks": [b["name"] for b in ls.get("benchmarks", [])],
        "lushang_keywords": ls["keywords"],
        "type_profiles": sorted(TYPE_PROFILES.keys()),
        "fewshot_sample_count": len(FEWSHOT_SAMPLES),
    }


# =========================================================================== #
# 华建集团案例内部 fixtures（含 source_url / 类型标签 / 可提取策略）
# 仅作内部 few-shot / 回归 / 第6章参考；不在前台展示文件名与链接。
# =========================================================================== #
HUAJIAN_FIXTURES: list[dict[str, Any]] = [
    {"name": "万荣路467号改扩建项目", "category": "城市更新老旧片区/存量地块",
     "type_tags": ["存量地块", "综合功能地块"],
     "source_url": "http://www.archina.com/index.php?g=Works&m=index&a=show&id=159126",
     "strategies": ["存量地块高强度复合开发", "创新研发功能导入", "生态社区与立体田园",
                    "高容积率下的公共空间组织", "商办/研发/社区复合更新"]},
    {"name": "张园城市更新项目", "category": "城市更新老旧片区/存量地块",
     "type_tags": ["历史建筑", "存量地块", "商业街区"],
     "source_url": "http://stock.10jqka.com.cn/20260409/c675866277.shtml",
     "strategies": ["历史建筑保护性开发", "地下空间复合利用", "商业文化公共活动复合",
                    "风貌保护与功能升级并重", "高密度中心城区更新"]},
    {"name": "东斯文里项目", "category": "城市更新老旧片区/存量地块",
     "type_tags": ["历史建筑", "商业街区"],
     "source_url": "https://www.thepaper.cn/newsDetail_forward_27807384",
     "strategies": ["成片历史街区保护", "金融与文创功能导入", "苏河滨水空间联动",
                    "风貌保护与产业升级", "片区级城市更新"]},
    {"name": "蕃瓜弄小区旧住房改建工程", "category": "城市更新老旧片区/存量地块",
     "type_tags": ["老旧社区"],
     "source_url": "https://www.gzw.sh.gov.cn/shgzw_zxzx_gqdt/20240111/3d4342199e334679a2dbed132a0a8bf1.html",
     "strategies": ["老旧住房拆落地改建", "公共设施补短板", "居住品质提升",
                    "社区配套完善", "民生导向更新"]},
    {"name": "三林滨江项目", "category": "城市更新老旧片区/存量地块",
     "type_tags": ["综合功能地块", "公共空间"],
     "source_url": "https://www.gzw.sh.gov.cn/shgzw_zxzx_gqdt/20240111/3d4342199e334679a2dbed132a0a8bf1.html",
     "strategies": ["成片区域综合开发", "滨江公共空间联动", "全过程城市更新服务",
                    "产业居住公共空间复合", "片区统筹实施"]},
    {"name": "西岸梦中心穹顶艺术中心", "category": "城市更新工业遗存",
     "type_tags": ["工业遗存", "公共空间"],
     "source_url": "https://www.thepaper.cn/newsDetail_forward_25247572",
     "strategies": ["工业遗存活化", "大跨度空间再利用", "文化演艺功能导入",
                    "城市地标塑造", "滨江文化消费场景"]},
    {"name": "威海至海港湾项目", "category": "城市更新工业遗存",
     "type_tags": ["工业遗存", "公共空间"],
     "source_url": "https://www.xinminweekly.com.cn",
     "strategies": ["工业遗存保护", "滨海空间活化", "文商旅融合",
                    "公共开放空间", "城市更新品牌塑造"]},
    {"name": "景德镇东三宝城市更新片区", "category": "城市更新工业遗存",
     "type_tags": ["工业遗存", "商业街区"],
     "source_url": "https://t.10jqka.com.cn/lgt/article_detail/index.html?contentId=c1bat9zasd7ebdb086506",
     "strategies": ["铁路/物流厂房遗存更新", "文创艺术导入", "陶艺产业与社区融合",
                    "商业居住文化复合", "创作聚落营造"]},
    {"name": "罗店古镇核心区提升工程", "category": "街区提升",
     "type_tags": ["商业街区", "历史建筑"],
     "source_url": "https://www.thepaper.cn/newsDetail_forward_33048031",
     "strategies": ["古镇微更新", "针灸式织补", "原住民生活延续",
                    "历史文化街区保护", "小尺度渐进式更新"]},
    {"name": "济南泉城路老旧街区更新改造", "category": "街区提升",
     "type_tags": ["商业街区"],
     "source_url": "http://sdenews.com/html/2025/8/377613.shtml",
     "strategies": ["商业主街更新", "街道空间提质", "业态迭代",
                    "消费场景升级", "步行体验优化"]},
    {"name": "烟台市朝阳街历史街区修缮设计", "category": "街区提升",
     "type_tags": ["历史建筑", "商业街区"],
     "source_url": "http://www.cupc.org.cn/cupc/dxyl/csgx/article/20230615105041392134059.html",
     "strategies": ["开埠历史街区修缮", "商业文化旅游复合", "历史街巷肌理保护",
                    "居住与旅游共存", "街区活化运营"]},
    {"name": "上海蟠龙天地", "category": "街区提升",
     "type_tags": ["历史建筑", "商业街区"],
     "source_url": "",
     "strategies": ["历史古镇商业化更新", "风貌保护", "文旅商业融合",
                    "水乡街区活化", "保护建筑再利用"]},
    {"name": "徐汇西岸传媒港及周边综合开发", "category": "综合功能地块",
     "type_tags": ["综合功能地块"],
     "source_url": "https://www.westbund.com",
     "strategies": ["综合功能地块开发", "站城一体化", "文化传媒产业导入",
                    "商办居住复合", "滨水片区更新"]},
    {"name": "上海世博文化公园", "category": "公共空间优化",
     "type_tags": ["公共空间"],
     "source_url": "https://www.shanghai.gov.cn",
     "strategies": ["大尺度公共空间优化", "生态休闲功能", "城市公园更新",
                    "公共开放性", "文化休闲复合"]},
]


# =========================================================================== #
# 项目类型 → 差异化策略画像（对话与报告据此体现项目差异，避免千篇一律）
# =========================================================================== #
TYPE_PROFILES: dict[str, dict[str, Any]] = {
    "老旧仓库": {
        "aliases": ["仓库", "仓储", "库房", "货仓", "老码头", "存量建筑", "历史建筑", "厂库",
                    "老旧片区/存量地块更新型", "老旧片区", "存量地块"],
        "core_logic": "存量空间盘活 + 历史记忆转译 + 文化商业复合",
        "emphasis": ["存量建筑活化与结构复用", "历史记忆与场所精神转译",
                     "首层界面激活与沿街体验", "外挂连廊/慢行与人流组织",
                     "文化商业复合业态导入", "运营前置与分期更新"],
        "case_hint": ["老码头", "张园", "今潮", "鲁商"],
    },
    "老旧社区": {
        "aliases": ["社区", "小区", "居住", "老旧住房", "住宅", "公房", "里弄住区",
                    "社区配套升级型", "社区配套升级"],
        "core_logic": "公共服务补短板 + 一刻钟生活圈 + 适老化",
        "emphasis": ["公共服务设施补短板", "一刻钟便民生活圈完善",
                     "适老化与无障碍改造", "停车与慢行系统改善",
                     "社区公共活动空间营造", "民生导向分期实施"],
        "case_hint": ["蕃瓜弄"],
    },
    "商业街区": {
        "aliases": ["商业街", "商街", "街区", "商圈", "步行街", "商业片区",
                    "商业活力提升型", "街区提升型", "商业活力", "街区提升"],
        "core_logic": "首层界面 + 业态组合 + 消费场景 + 运营导入",
        "emphasis": ["首层商业界面更新", "业态组合与迭代",
                     "消费场景与夜间经济", "公共空间与人流动线组织",
                     "步行体验优化", "商业运营与招商导入"],
        "case_hint": ["泉城路", "武夷", "幸福里", "罗店"],
    },
    "工业遗存": {
        "aliases": ["工业", "厂房", "船厂", "工厂", "遗存", "产业园", "锅炉", "车间",
                    "工业遗存活化型"],
        "core_logic": "工业遗存活化 + 功能复合 + 文创消费 + 产城融合",
        "emphasis": ["工业遗存与大空间活化再利用", "文创艺术与功能复合导入",
                     "产城融合与产业再生", "公共空间开放与地标塑造",
                     "历史工业符号转译", "分级保护改造"],
        "case_hint": ["西岸梦中心", "船厂1862", "东三宝", "上生新所"],
    },
    "公共空间优化": {
        "aliases": ["公共空间", "公园", "绿地", "广场", "滨水", "滨江", "慢行",
                    "公共空间优化型"],
        "core_logic": "公共空间补足 + 慢行系统 + 便民服务 + 人本尺度",
        "emphasis": ["公共活动空间补足", "慢行系统与步行网络优化",
                     "便民服务设施完善", "社区公共活动与文化场景",
                     "绿地休闲与生态品质", "人本尺度更新"],
        "case_hint": ["世博文化公园", "三林滨江"],
    },
    "综合功能地块": {
        "aliases": ["综合", "综合开发", "站城", "传媒港", "复合地块",
                    "综合功能地块型"],
        "core_logic": "功能复合 + 站城一体 + 产居融合 + 公共空间统筹",
        "emphasis": ["多功能复合开发", "站城一体化衔接",
                     "产业与居住复合", "公共空间统筹与开放",
                     "片区能级提升", "分期统筹实施"],
        "case_hint": ["徐汇西岸", "万荣路467号"],
    },
}

# 用户更新目标关键词 → 强调点（即使类型相同，目标不同也应体现差异）
_GOAL_KEYWORDS = {
    "适老": "适老化与无障碍改造",
    "养老": "养老服务与适老化配套",
    "停车": "停车供给与静态交通改善",
    "活力": "片区商业活力与消费场景营造",
    "业态": "业态组合优化与运营导入",
    "文化": "文化场景与历史记忆转译",
    "历史": "历史风貌保护与记忆转译",
    "产业": "产业导入与产城融合",
    "文创": "文创艺术功能导入",
    "夜间": "夜间经济与全时段运营",
    "连廊": "外挂连廊与慢行立体组织",
    "公共空间": "公共空间补足与开放",
    "民生": "民生服务补短板",
    "品质": "环境品质与街区风貌提升",
}


def classify_type_from_text(text: str) -> str | None:
    """从用户输入文本识别更新项目类型（命中 aliases 最多者）。"""
    if not text:
        return None
    best, best_hits = None, 0
    for canon, prof in TYPE_PROFILES.items():
        hits = sum(1 for a in prof["aliases"] if a in text)
        if hits > best_hits:
            best, best_hits = canon, hits
    return best if best_hits > 0 else None


def map_to_type_profile(project_type: str | None, user_text: str | None = None) -> dict[str, Any]:
    """把（模型识别的）项目类型 + 用户文本，映射到差异化策略画像。

    优先级：① project_type 命中某规范类型（含别名）→ 直接采用；
            ② 否则按 project_type + 用户文本中别名命中数最多者判定（max-hits）。
    """
    canon = None
    pt = project_type or ""
    # ① 显式类型优先（规范名或其别名命中）
    for c, prof in TYPE_PROFILES.items():
        if c in pt or any(a in pt for a in prof["aliases"]):
            canon = c
            break
    # ② 退化为按命中数最多判定（避免别名在文本中先后顺序造成误判）
    if canon is None:
        canon = classify_type_from_text(f"{pt} {user_text or ''}")
    prof = TYPE_PROFILES.get(canon or "", {})
    goal_focus = [v for k, v in _GOAL_KEYWORDS.items() if user_text and k in user_text]
    return {
        "canonical_type": canon,
        "core_logic": prof.get("core_logic", ""),
        "emphasis": prof.get("emphasis", []),
        "case_hint": prof.get("case_hint", []),
        "goal_focus": list(dict.fromkeys(goal_focus)),
    }


def goal_focus_from_text(user_text: str | None) -> list[str]:
    if not user_text:
        return []
    return list(dict.fromkeys(v for k, v in _GOAL_KEYWORDS.items() if k in user_text))


# =========================================================================== #
# 内部输入提示词 + 期望输出 few-shot 样本（用于 DeepSeek 受约束成文/兜底/回归校准）
# 不在前台展示。
# =========================================================================== #
FEWSHOT_SAMPLES: list[dict[str, Any]] = [
    {"id": "fewshot_warehouse",
     "type": "老旧仓库",
     "prompt": "项目为上海中心城区一处老旧仓储建筑，建筑具备历史记忆，但现状空间利用效率低，"
               "沿街界面封闭，商业活力不足，希望更新为复合型商业文化街区，请生成城市更新前策研判。",
     "expected_points": ["存量建筑活化", "历史记忆转译", "首层界面激活",
                          "商业文化复合", "慢行/连廊/人流组织", "运营导入", "分阶段实施"],
     "ref_cases": ["鲁商1992项目（外挂连廊整合方案）", "张园城市更新项目"]},
    {"id": "fewshot_community",
     "type": "老旧社区",
     "prompt": "项目为中心城区老旧居住片区，公共服务设施不足，停车困难，适老化不足，"
               "希望通过城市更新提升社区服务能力与生活品质。",
     "expected_points": ["公共服务补短板", "一刻钟生活圈", "适老化",
                          "停车改善", "社区活动空间", "民生导向实施路径"],
     "ref_cases": ["蕃瓜弄小区旧住房改建工程"]},
    {"id": "fewshot_commercial",
     "type": "商业街区",
     "prompt": "项目为老旧商业街区，沿街界面陈旧，业态同质化严重，人流停留不足，"
               "希望提升商业活力和消费体验。",
     "expected_points": ["首层界面更新", "业态迭代", "消费场景",
                          "夜间经济", "步行体验优化", "商业运营导入"],
     "ref_cases": ["济南泉城路老旧街区更新改造", "上海·武夷MIX320（第一泵厂改造）"]},
    {"id": "fewshot_industrial",
     "type": "工业遗存",
     "prompt": "项目为原工业厂房片区，具备工业遗存特色，但产业功能衰退，"
               "希望导入文创、艺术、商业和社区复合功能。",
     "expected_points": ["工业遗存活化", "文创艺术导入", "大空间再利用",
                          "商业文化复合", "产城融合", "公共空间开放"],
     "ref_cases": ["西岸梦中心穹顶艺术中心", "船厂1862（上海船厂改造）"]},
    {"id": "fewshot_publicspace",
     "type": "公共空间优化",
     "prompt": "项目周边居住人口较多，但公共活动空间不足，慢行体验较弱，"
               "便民服务设施不完善，希望通过城市更新改善片区公共空间品质。",
     "expected_points": ["公共空间补足", "慢行系统优化", "便民服务设施",
                          "社区公共活动", "绿地与休闲", "人本尺度更新"],
     "ref_cases": ["上海世博文化公园", "三林滨江项目"]},
]


def fewshot_samples() -> list[dict[str, Any]]:
    return FEWSHOT_SAMPLES


def huajian_fixtures() -> list[dict[str, Any]]:
    return HUAJIAN_FIXTURES


def select_case_refs_v2(canonical_type: str | None, user_text: str | None, limit: int = 3) -> list[dict[str, Any]]:
    """按类型标签 + 用户文本，从华建 fixtures + 鲁商画像选案例（含可提取策略，不暴露链接到前台）。"""
    blob = f"{canonical_type or ''} {user_text or ''}"
    refs: list[dict[str, Any]] = []
    # 鲁商优先用于仓库/商业/历史/工业类
    if canonical_type in ("老旧仓库", "商业街区", "工业遗存") or any(
            k in blob for k in ["仓库", "历史", "商业", "工业", "连廊", "活力"]):
        lp = lushang_profile()
        refs.append({
            "name": lp["name"], "category": "存量历史建筑/商业更新",
            "applicable": "可借鉴外挂连廊串联、首层界面激活、空间复合与场所记忆转译，提升商业活力与公共体验。",
            "strategies": (lp["spatial_strategy"][:1] + lp["function_strategy"][:1]),
            "source": lp["source"],
        })
    scored: list[tuple[int, dict[str, Any]]] = []
    for f in HUAJIAN_FIXTURES:
        score = 0
        if canonical_type and canonical_type in f.get("type_tags", []):
            score += 3
        score += sum(1 for t in f.get("type_tags", []) if t in blob)
        score += sum(1 for s in f.get("strategies", []) if any(w and w in s for w in blob.split()))
        if score > 0:
            scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    for _, f in scored:
        refs.append({
            "name": f["name"], "category": f.get("category", ""),
            "applicable": f"可借鉴：{('、'.join(f.get('strategies', [])[:3]))}。",
            "strategies": f.get("strategies", [])[:4],
            "source": "华建集团相关案例",
        })
        if len(refs) >= limit:
            break
    # 至少补足到 2 个可参考案例（命中不足时用相近华建案例补位，避免「挂名不引用」）
    if len(refs) < min(2, limit):
        have = {r["name"] for r in refs}
        for f in HUAJIAN_FIXTURES:
            if f["name"] in have:
                continue
            refs.append({"name": f["name"], "category": f.get("category", ""),
                         "applicable": f"可借鉴：{('、'.join(f.get('strategies', [])[:3]))}。",
                         "strategies": f.get("strategies", [])[:4], "source": "华建集团相关案例"})
            if len(refs) >= max(2, min(2, limit)):
                break
    return refs[:limit]
