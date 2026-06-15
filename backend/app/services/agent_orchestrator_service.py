"""第12E：智能体编排层（Agent Orchestrator）。

流程：
  用户输入 → 任务理解（推断 task_type）→ 选择并运行自研只读工具
  → 汇总结构化结果 / 证据 / 可信边界 → DeepSeek 基于结构化结果组织自然语言
  → 若大模型不可用则 graceful fallback（仍输出基于自研模型的结构化回答）。

红线：
- DeepSeek 仅做表达；所有事实数字来自工具返回的结构化结果。
- 大模型不可用/失败时不阻断；不伪造城市更新结论；保留可信边界提示。
- 不输出 raw chain-of-thought，仅输出工具编排概述（thinking_summary）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.services import agent_tools_service as tools
from app.services import deepseek_service

logger = logging.getLogger("cityrenew.agent.orchestrator")

# 项目类型英文枚举 → 业务化中文（与前端 businessLabels 对齐）
TYPE_LABELS = {
    "commercial_vitality_upgrade": "商业活力提升型更新",
    "block_renewal": "商业活力提升型更新",
    "public_service_improvement": "公共服务补短板型更新",
    "community_renewal": "社区复合更新型",
    "community": "社区复合更新型",
    "industry_upgrade": "产业功能提升型更新",
    "industrial": "产业功能提升型更新",
    "old_district": "老旧片区改善型更新",
    "old": "老旧片区改善型更新",
    "public_space": "公共空间提升型更新",
    "mixed": "复合功能统筹型更新",
}

# 任务类型 → 工具组合
TASK_TOOLS: dict[str, list[str]] = {
    "project_diagnosis": ["get_full_summary", "get_poi_summary", "get_score_result", "get_project_type", "get_risk_summary"],
    "poi_analysis": ["get_poi_summary", "get_project_features", "get_feature_quality"],
    "renewal_type": ["get_project_type", "get_poi_summary", "get_score_result"],
    "strategy": ["get_full_summary", "get_project_type", "get_score_result", "get_risk_summary"],
    "report_outline": ["get_full_summary", "get_score_result", "get_project_type", "get_evidence_lineage"],
    "evidence_trace": ["get_evidence_lineage", "get_trust_summary", "get_risk_summary"],
}

TASK_LABELS = {
    "project_diagnosis": "项目初判",
    "poi_analysis": "周边配套分析",
    "renewal_type": "更新类型判断",
    "strategy": "更新策略建议",
    "report_outline": "报告大纲生成",
    "evidence_trace": "证据来源追溯",
}

# 任务 → 下一步动作（label + 前端路由）
TASK_NEXT_ACTIONS: dict[str, list[dict[str, str]]] = {
    "project_diagnosis": [
        {"label": "查看项目研判", "target": "/project-analysis"},
        {"label": "生成更新策略", "target": "/strategy"},
        {"label": "查看证据链", "target": "/evidence-lineage"},
    ],
    "poi_analysis": [
        {"label": "查看项目研判", "target": "/project-analysis"},
        {"label": "生成更新策略", "target": "/strategy"},
    ],
    "renewal_type": [
        {"label": "生成更新策略", "target": "/strategy"},
        {"label": "查看项目研判", "target": "/project-analysis"},
    ],
    "strategy": [
        {"label": "生成报告草稿", "target": "/report-studio"},
        {"label": "查看证据链", "target": "/evidence-lineage"},
    ],
    "report_outline": [
        {"label": "前往报告生成", "target": "/report-studio"},
        {"label": "查看证据链", "target": "/evidence-lineage"},
    ],
    "evidence_trace": [
        {"label": "进入可信度中心", "target": "/trust-center"},
        {"label": "查看项目研判", "target": "/project-analysis"},
    ],
}


def infer_task_type(message: str | None) -> str:
    """从用户消息推断任务类型（规则匹配，缺省 project_diagnosis）。"""
    text = (message or "").lower()
    if any(k in text for k in ["证据", "来源", "血缘", "依据从哪", "可追溯"]):
        return "evidence_trace"
    if any(k in text for k in ["报告", "大纲", "outline"]):
        return "report_outline"
    if any(k in text for k in ["策略", "建议", "方向", "怎么做", "如何更新"]):
        return "strategy"
    if any(k in text for k in ["配套", "周边", "poi", "圈层", "设施"]):
        return "poi_analysis"
    if any(k in text for k in ["类型", "适合做什么", "适合什么", "哪种更新", "什么更新"]):
        return "renewal_type"
    return "project_diagnosis"


def _find(obj: Any, keys: tuple[str, ...]) -> Any:
    """在嵌套 dict/list 中深度查找首个命中键的值。"""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] not in (None, "", [], {}):
                return obj[k]
        for v in obj.values():
            found = _find(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find(v, keys)
            if found is not None:
                return found
    return None


def _type_label(raw_type: Any) -> str | None:
    if not raw_type:
        return None
    key = str(raw_type).strip().lower()
    return TYPE_LABELS.get(key, str(raw_type))


def _collect_evidence(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从工具结果中收集证据引用（evidence_id / 来源 / 摘要），去重限量。"""
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tr in tool_results:
        data = tr.get("data")
        ev = _find(data, ("evidence_ids", "evidence_refs", "evidences"))
        if isinstance(ev, list):
            for item in ev:
                if isinstance(item, str):
                    eid = item
                    entry = {"evidence_id": eid}
                elif isinstance(item, dict):
                    eid = str(item.get("evidence_id") or item.get("id") or item.get("source") or "")
                    entry = {k: item.get(k) for k in ("evidence_id", "source_file", "summary", "source") if item.get(k)}
                else:
                    continue
                if eid and eid not in seen:
                    seen.add(eid)
                    refs.append(entry)
                if len(refs) >= 8:
                    return refs
    return refs


def _build_structured_result(task_type: str, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    """把工具结果压缩为前端可展示、可传给大模型的结构化摘要（业务化）。

    为保证数字一致性：每个字段优先从其权威工具提取，避免跨工具首次命中导致口径不一。
    """
    by_tool = {tr["tool"]: tr.get("data") for tr in tool_results if tr.get("status") == "ok"}

    # 更新类型：仅取类型识别工具（其次研判汇总），不跨工具乱取。
    type_src = by_tool.get("get_project_type")
    raw_type = _find(type_src, ("predicted_type", "final_type", "project_type")) if type_src else None
    if raw_type is None:
        raw_type = _find(by_tool.get("get_full_summary"), ("predicted_type", "final_type", "project_type"))
    confidence = _find(type_src, ("confidence", "confidence_score")) if type_src else None

    # 综合评分：仅取评分工具（其次研判汇总）。
    score_src = by_tool.get("get_score_result")
    score = _find(score_src, ("comprehensive_score", "final_score", "f_score")) if score_src else None
    if score is None:
        score = _find(by_tool.get("get_full_summary"), ("comprehensive_score", "final_score", "f_score"))

    # POI 总数：优先 1500m 辐射圈层；其次评分工具中的 poi_total。
    poi_total = _ring_poi_total(by_tool.get("get_poi_summary"))
    if poi_total is None and score_src is not None:
        poi_total = _find(score_src, ("poi_total", "total_poi"))

    structured: dict[str, Any] = {
        "task": TASK_LABELS.get(task_type, task_type),
        "renewal_type_label": _type_label(raw_type),
        "renewal_type_source": "规则辅助识别",
        "comprehensive_score": score,
        "poi_total_around": poi_total,
        "type_confidence": confidence,
    }
    return {k: v for k, v in structured.items() if v is not None}


def _ring_poi_total(poi_data: Any) -> Any:
    """从 POI 摘要中取 1500m 辐射圈层 POI 总数（口径与报告模板一致）。"""
    if not isinstance(poi_data, dict):
        return None
    ring = poi_data.get("ring_summary") or {}
    for key in ("ring_1500m", "ring_3000m", "ring_5000m", "ring_500m", "core"):
        node = ring.get(key)
        if isinstance(node, dict) and node.get("poi_total") is not None:
            return node["poi_total"]
    return _find(poi_data, ("poi_total", "total_poi"))


def _build_limitations(tool_results: list[dict[str, Any]]) -> list[str]:
    """汇总可信边界提示（业务化）：仅列出不可用 / 待补充的工具数据。

    详细的内部门禁 warning 不在此堆砌，统一在可信度中心查看。
    """
    limits: list[str] = []
    for tr in tool_results:
        if tr.get("status") in ("error", "empty"):
            limits.append(f"{tr['name']}：{tr.get('message') or '该数据暂不可用'}")
    out: list[str] = []
    for x in limits:
        if x not in out:
            out.append(x)
    return out[:6]


def _fallback_answer(task_type: str, structured: dict[str, Any], limitations: list[str]) -> str:
    """大模型不可用时，基于自研模型结构化结果生成业务化回答（不编造）。"""
    rtype = structured.get("renewal_type_label")
    score = structured.get("comprehensive_score")
    poi = structured.get("poi_total_around")

    parts: list[str] = []
    if rtype:
        parts.append(
            f"根据当前项目周边配套、综合评分与规则辅助识别结果，系统初步判断该项目更适合「{rtype}」。"
        )
    else:
        parts.append("系统已读取当前项目的自研模型结果，但更新类型识别结果暂不可用。")

    detail = []
    if poi:
        detail.append(f"周边圈层 POI 约 {poi} 个")
    if score is not None:
        try:
            detail.append(f"综合评分约 {float(score):.1f}")
        except (TypeError, ValueError):
            detail.append(f"综合评分 {score}")
    if detail:
        parts.append("，".join(detail) + "，可作为研判与策略的量化基础。")

    if task_type == "strategy":
        parts.append("建议围绕「优势提升 + 短板补齐 + 复合功能」形成前期策划方向，具体清单见更新策略页。")
    elif task_type == "report_outline":
        parts.append("可据此生成报告大纲，所有数字将从自研模型结果注入，结论可追溯至证据链。")
    elif task_type == "evidence_trace":
        parts.append("各项结论均可追溯至来源数据与证据链，详见证据追溯页。")

    parts.append("（当前大模型思考层暂未连接，以上为基于自研城市更新模型结果的结构化回答。）")
    # 仅提示数据缺口数量，不在回答正文堆砌内部门禁术语；详情见可信度中心。
    gaps = [x for x in limitations if "暂不可用" in x]
    if gaps:
        parts.append(f"另有 {len(gaps)} 项数据仍需补充，详见证据追溯与可信度中心。")
    return "".join(parts)


def _compose_llm_prompt(message: str, task_type: str, structured: dict[str, Any],
                        tool_results: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> str:
    """构造给 DeepSeek 的用户消息：用户问题 + 工具结构化结果（JSON）。"""
    tool_payload = [
        {"tool": tr["name"], "status": tr["status"], "data": tr.get("data")}
        for tr in tool_results
    ]
    payload = {
        "用户问题": message,
        "任务类型": TASK_LABELS.get(task_type, task_type),
        "自研模型结构化结果": structured,
        "工具返回明细": tool_payload,
        "证据引用": evidence,
    }
    body = json.dumps(payload, ensure_ascii=False, default=str)
    # 控制体量，避免超长。
    if len(body) > 12000:
        body = body[:12000] + " ...(已截断)"
    return (
        "以下是 CityRenew 自研城市更新模型与数据工具针对该项目返回的结构化结果。"
        "请仅基于这些结果，用业务化中文回答用户问题，给出结论、依据、建议与限制；"
        "不要引入结果之外的事实数字；数据不足处明确说明。\n\n"
        f"{body}"
    )


def run_agent(db: Session, *, message: str, project_id: int = 1,
              task_type: str | None = None, thinking: bool = False) -> dict[str, Any]:
    """执行一次智能体交互，返回 AgentChatResponse 兼容 dict。

    参数 thinking：True 时使用深度思考模型（pro 推理），False 使用常规对话模型（flash）。
    """
    resolved_task = task_type if task_type in TASK_TOOLS else infer_task_type(message)
    tool_names = TASK_TOOLS.get(resolved_task, TASK_TOOLS["project_diagnosis"])

    steps: list[str] = [
        f"理解到任务类型：{TASK_LABELS.get(resolved_task, resolved_task)}",
        f"计划调用自研工具：{', '.join(tools.TOOL_REGISTRY[t].__name__ if t in tools.TOOL_REGISTRY else t for t in tool_names)}",
    ]

    tool_results: list[dict[str, Any]] = []
    for tname in tool_names:
        fn = tools.TOOL_REGISTRY.get(tname)
        if fn is None:
            continue
        tool_results.append(fn(db, project_id))

    ok_count = sum(1 for t in tool_results if t.get("status") == "ok")
    steps.append(f"工具返回：{ok_count}/{len(tool_results)} 项可用")

    structured = _build_structured_result(resolved_task, tool_results)
    evidence = _collect_evidence(tool_results)
    limitations = _build_limitations(tool_results)
    next_actions = TASK_NEXT_ACTIONS.get(resolved_task, TASK_NEXT_ACTIONS["project_diagnosis"])

    # 工具调用结果（脱敏摘要供前端工具卡）
    used_tools = [
        {
            "tool": tr["tool"],
            "name": tr["name"],
            "status": tr["status"],
            "message": tr.get("message", ""),
            "summary": _tool_card_summary(tr),
        }
        for tr in tool_results
    ]

    deepseek_configured = deepseek_service.is_configured()
    deepseek_used = False
    model_used: str | None = None
    answer = ""

    mode_label = "深度思考（pro 推理）" if thinking else "常规对话（flash）"
    if deepseek_configured:
        prompt = _compose_llm_prompt(message, resolved_task, structured, tool_results, evidence)
        steps.append(f"大模型基于结构化结果组织表达 · {mode_label}")
        llm = deepseek_service.generate(prompt, thinking=thinking)
        if llm.get("ok"):
            answer = llm["text"]
            deepseek_used = True
            model_used = llm.get("model")
        else:
            steps.append(f"大模型暂不可用（{llm.get('error')}），切换为自研模型结构化回答")
            answer = _fallback_answer(resolved_task, structured, limitations)
    else:
        steps.append("大模型思考层未配置，输出自研模型结构化回答")
        answer = _fallback_answer(resolved_task, structured, limitations)

    confidence_note = (
        "结论基于 CityRenew 自研模型与真实数据，可追溯至证据链；"
        "更新类型为规则辅助识别，最终冻结评估未参与训练或调参。"
    )

    return {
        "answer": answer,
        "task_type": resolved_task,
        "project_id": project_id,
        "used_tools": used_tools,
        "structured_result": structured,
        "evidence_refs": evidence,
        "next_actions": next_actions,
        "limitations": limitations,
        "confidence_note": confidence_note,
        "thinking_summary": steps,
        "deepseek_configured": deepseek_configured,
        "deepseek_used": deepseek_used,
        "thinking_mode": thinking,
        "model_used": model_used,
        "degraded": not deepseek_used,
    }


def _tool_card_summary(tr: dict[str, Any]) -> dict[str, Any]:
    """为工具卡抽取极简摘要（避免回灌大对象到前端）。"""
    if tr.get("status") != "ok":
        return {}
    data = tr.get("data")
    summary: dict[str, Any] = {}
    score = _find(data, ("comprehensive_score", "final_score", "f_score"))
    if score is not None:
        summary["综合评分"] = score
    rtype = _find(data, ("predicted_type", "final_type", "project_type"))
    if rtype is not None:
        summary["识别类型"] = _type_label(rtype)
    poi = _find(data, ("poi_total", "total_poi"))
    if poi is not None:
        summary["POI 数"] = poi
    return summary
