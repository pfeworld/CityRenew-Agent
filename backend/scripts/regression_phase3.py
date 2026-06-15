"""第三阶段返工内部回归：案例样本 / 对话去模板化相似度 / 9章完整 / 缺失值 / 禁用词。

仅本地、仅 train/val；不触碰 test；不外发；输出 PASS/FAIL 汇总。
运行： .venv/bin/python -m scripts.regression_phase3
"""
from __future__ import annotations

import os
import sys
import tempfile

# 回归隔离：会话元数据写入临时文件，绝不污染正式 conversations.json。
# 必须在导入 conversation_service 之前设置（模块在导入时读取该路径）。
os.environ["CITYRENEW_AGENT_CONV_PATH"] = os.path.join(
    tempfile.gettempdir(), "cityrenew_conversations_test.json")

from app.database import SessionLocal
from app.models import Project
from app.services import case_learning_service as cl
from app.services import conversation_service as cs
from app.services import report_builder_service as rb
from app.api.routes_report import _plain_text

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, PASS if ok else FAIL, detail))


# 1) 案例样本 -------------------------------------------------------------- #
st = cl.case_corpus_status()
check("华建案例 fixtures≥14", st["huajian_fixture_count"] >= 14, str(st["huajian_fixture_count"]))
check("华建案例含 source_url", st["huajian_with_source_url"] >= 13, str(st["huajian_with_source_url"]))
check("鲁商图片/PPT 解析≥18页", st["lushang_page_count"] >= 18, str(st["lushang_page_count"]))
check("鲁商页含PPTX文本", st["lushang_pages_with_text"] >= 18, str(st["lushang_pages_with_text"]))
check("鲁商对标案例已识别", len(st["lushang_benchmarks"]) >= 5, str(st["lushang_benchmarks"]))
check("few-shot 样例≥5", st["fewshot_sample_count"] >= 5, str(st["fewshot_sample_count"]))

# few-shot 类型映射覆盖
for fs in cl.fewshot_samples():
    prof = cl.map_to_type_profile(fs["type"], fs["prompt"])
    hit = sum(1 for pt in fs["expected_points"]
              if any(pt[:2] in e or e[:2] in pt for e in prof["emphasis"]))
    check(f"few-shot[{fs['id']}]类型/侧重命中", prof["canonical_type"] == fs["type"] and hit >= 2,
          f"type={prof['canonical_type']} hit={hit}")


# 2) 对话去模板化（不同项目相似度 < 阈值，且体现用户现状/目标）------------------ #
def fake_ar(name, rtype, poi, pop, price, ent, score, risk):
    return {"project_understanding": {"name": name}, "renewal_type": rtype,
            "comprehensive_score": score, "score_level": "中等",
            "location_poi_analysis": {"rings": [{"ring": "radiation", "total": poi}]},
            "population_analysis": {"rings": [{"ring": "radiation", "residential": pop}]},
            "housing_space_analysis": {"rings": [{"ring": "radiation", "avg_unit_price": price}]},
            "industry_analysis": {"rings": [{"ring": "radiation", "enterprise_count": ent}]},
            "demand_potential_analysis": {"key_risks": [risk]}}


# 不同项目使用不同地点的真实量级（模拟不同坐标下的数据差异）
cases_in = [
    ("老码头仓储片区", "老旧仓库", "沿街界面封闭、商业活力不足", "更新为复合型商业文化街区",
     3806, 51301, 91677, 207, 53.6, "上层商业可达性弱、动线割裂"),
    ("龙华老旧小区", "老旧社区", "停车困难、适老化不足", "提升社区服务与生活品质",
     1290, 78400, 62300, 31, 71.2, "公共服务与适老设施不足"),
    ("某老工业厂房", "工业遗存", "产业功能衰退、大空间闲置", "导入文创艺术与商业复合",
     880, 22600, 48900, 540, 64.8, "产业能级偏弱、大空间利用低"),
    ("某商业街区", "商业街区", "业态同质化、人流停留不足", "提升商业活力与夜间经济",
     2150, 33700, 75200, 96, 58.4, "业态同质化、夜间活力不足"),
]
texts = []
for name, rtype, demand, goal, poi, pop, price, ent, score, risk in cases_in:
    conv = cs.create_conversation()
    conv["profile"].update({"name": name, "update_demand": demand, "expected_direction": goal})
    conv["messages"] = [{"role": "user", "text": f"{name} {demand} {goal}"}]
    dprof = cl.map_to_type_profile(rtype, cs._user_text(conv))
    t = cs._deterministic_diagnosis(fake_ar(name, rtype, poi, pop, price, ent, score, risk), conv, dprof)
    texts.append((name, demand, goal, t))
    # 必须包含用户现状/目标关键片段
    check(f"回答含用户现状[{name}]", demand[:4] in t, demand[:8])
    check(f"回答含用户目标[{name}]", goal[:4] in t, goal[:8])

maxsim = 0.0
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        s = cs._similarity(texts[i][3], texts[j][3])
        maxsim = max(maxsim, s)
check("不同项目两两相似度<0.58", maxsim < cs._SIM_THRESHOLD, f"max_sim={round(maxsim,3)}")


# 3) 报告 9 章完整 / 缺失值 / 禁用词 ---------------------------------------- #
db = SessionLocal()
p = db.query(Project).first()
content = rb.build_report(db, p)
chs = content["chapters"]
check("报告 9 章", len(chs) == 9, str([c["no"] for c in chs]))
check("第6章有案例子节", any(c["no"] == "6" and len(c.get("sections", [])) >= 3 for c in chs))
check("第8章有8.1-8.5", any(c["no"] == "8" and len(c.get("sections", [])) >= 5 for c in chs))
check("第9章有9.1-9.4", any(c["no"] == "9" and len(c.get("sections", [])) >= 4 for c in chs))
check("四张量化表", sum(len(c.get("tables", [])) for c in chs) == 4)

# 缺失值：无红线时核心范围用中心点150米缓冲真实归集，不得出现「待补充」占位
emap = content["evidence_map"]
core_displays = [v["display"] for k, v in emap.items() if k.endswith(":core")]
check("无红线时核心范围无待补充占位", all(d != "待补充" for d in core_displays))
core_filled = sum(1 for d in core_displays if d not in ("待补充", "暂无数据", "暂无有效样本", "不适用"))
check("无红线时核心范围已填真实值", core_filled >= 3, f"已填{core_filled}项")
statuses = {v["status"] for v in emap.values()}
check("evidence_map不再出现no_redline占位", "no_redline" not in statuses, str(sorted(statuses)))

text = _plain_text(content)
FORBIDDEN = ["一、报告封面", "二、报告目录", "三、报告正文", "标准模板",
             "三圈层量化表", "四张量化表", "完整性检查", "一致性检查", "#", "待补充"]
bad = [w for w in FORBIDDEN if w in text]
check("正文无结构说明/验收口径/待补充/#", not bad, f"命中:{bad}")
check("封面保留【】格式", f"【{content['project_name']}】" in text)
check("封面保留黑客松数据来源", "数据来源：黑客松比赛提供专用数据库" in text)
db.close()


# 汇总 -------------------------------------------------------------------- #
print("\n================ 第三阶段回归结果 ================")
npass = sum(1 for _, r, _ in results if r == PASS)
for name, r, detail in results:
    print(f"[{r}] {name}" + (f"  · {detail}" if detail else ""))
print(f"------------------------------------------------\n通过 {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
