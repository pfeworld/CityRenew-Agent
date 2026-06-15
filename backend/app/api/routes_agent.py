"""第12E：智能体（Agent）交互接口。

GET  /api/agent/health        智能体健康（大模型是否配置 / 工具可用度 / 可信信号）
GET  /api/agent/capabilities  能力清单（技能 / 工具 / 任务类型 / 模型分工）
POST /api/agent/chat          智能体对话（理解→调用只读工具→大模型组织表达）
POST /api/agent/run-task      快捷任务（与 chat 同一编排，按 task_type 触发）

红线：
- 仅智能体交互，绝不触发训练/评测/导出等写操作；工具层只读。
- 不返回 / 不打印 DeepSeek API key；大模型不可用时 graceful fallback。
- 不输出 raw chain-of-thought。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.agent import (
    TASK_TYPES,
    AgentCapabilitiesResponse,
    AgentChatRequest,
    AgentChatResponse,
    AgentHealthResponse,
    AgentRunTaskRequest,
    AgentToolMeta,
    ModelLayer,
)
from app.services import (
    agent_orchestrator_service,
    agent_tools_service,
    attachment_service,
    conversation_service,
    deepseek_service,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 单文件 20MB 上限

logger = logging.getLogger("cityrenew.agent.api")

router = APIRouter(prefix="/api/agent", tags=["agent"])


_SKILLS = [
    {"key": "project_diagnosis", "name": "项目初判", "description": "综合 POI / 评分 / 类型给出项目初步研判"},
    {"key": "poi_analysis", "name": "圈层配套分析", "description": "核心/近邻/辐射圈层 POI 配套结构与短板"},
    {"key": "renewal_type", "name": "更新类型判断", "description": "规则辅助识别项目更新类型与依据"},
    {"key": "score", "name": "综合评分", "description": "十维评分与综合评分（可复算）"},
    {"key": "strategy", "name": "策略建议", "description": "基于研判结果生成结构化更新策略方向"},
    {"key": "report", "name": "报告生成", "description": "生成报告大纲，数字注入自研模型结果"},
    {"key": "evidence", "name": "证据追溯", "description": "结论到来源数据与证据链的追溯"},
    {"key": "trust", "name": "可信度校验", "description": "数据一致性、证据链与最终冻结评估状态"},
]


@router.get("/health", response_model=AgentHealthResponse)
def agent_health(db: Session = Depends(get_db)) -> AgentHealthResponse:
    """智能体健康检查：大模型配置状态 + 只读工具可用度 + 关键可信信号。"""
    cfg = deepseek_service.get_config()
    ok_tools, total_tools = agent_tools_service.available_tool_count(db, project_id=1)

    self_model = False
    evidence_ok = False
    warning_count = 0
    try:
        score = agent_tools_service.get_score_result(db, 1)
        ptype = agent_tools_service.get_project_type(db, 1)
        self_model = score.get("status") == "ok" or ptype.get("status") == "ok"
        ev = agent_tools_service.get_evidence_lineage(db, 1)
        evidence_ok = ev.get("status") == "ok"
        risk = agent_tools_service.get_risk_summary(db, 1)
        if risk.get("status") == "ok":
            warns = agent_orchestrator_service._find(risk.get("data"), ("warnings", "warning_list"))
            warning_count = len(warns) if isinstance(warns, list) else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_health 采集可信信号失败：%s", type(exc).__name__)

    return AgentHealthResponse(
        status="ok",
        deepseek_configured=cfg["configured"],
        model_name=cfg["model"] if cfg["configured"] else None,
        thinking_enabled=cfg["thinking_enabled"],
        tools_available=ok_tools,
        tools_total=total_tools,
        self_model_available=self_model,
        evidence_available=evidence_ok,
        warning_count=warning_count,
        message="大模型思考层已配置" if cfg["configured"] else "大模型思考层未配置，自研模型结果仍可用",
    )


@router.get("/capabilities", response_model=AgentCapabilitiesResponse)
def agent_capabilities(db: Session = Depends(get_db)) -> AgentCapabilitiesResponse:
    """能力清单：技能 / 只读工具状态 / 任务类型 / 模型分工。"""
    cfg = deepseek_service.get_config()
    tool_metas: list[AgentToolMeta] = []
    for meta in agent_tools_service.TOOL_META:
        fn = agent_tools_service.TOOL_REGISTRY.get(meta["tool"])
        status = "ready"
        available = True
        if fn is not None:
            try:
                res = fn(db, 1)
                if res.get("status") == "ok":
                    status = "ready"
                elif res.get("status") == "empty":
                    status, available = "need_data", False
                else:
                    status, available = "error", False
            except Exception:  # noqa: BLE001
                status, available = "error", False
        tool_metas.append(AgentToolMeta(
            tool=meta["tool"], name=meta["name"], description=meta["description"],
            source=meta.get("source", ""), available=available, status=status,
        ))

    return AgentCapabilitiesResponse(
        deepseek_configured=cfg["configured"],
        model_name=cfg["model"] if cfg["configured"] else None,
        thinking_enabled=cfg["thinking_enabled"],
        skills=_SKILLS,
        tools=tool_metas,
        task_types=list(TASK_TYPES),
        model_layer=ModelLayer(),
    )


@router.post("/chat", response_model=AgentChatResponse)
def agent_chat(payload: AgentChatRequest, db: Session = Depends(get_db)) -> AgentChatResponse:
    """智能体对话：理解任务 → 调用自研只读工具 → 大模型基于结构化结果组织表达。"""
    result = agent_orchestrator_service.run_agent(
        db, message=payload.message, project_id=payload.project_id,
        task_type=payload.task_type, thinking=payload.thinking,
    )
    return AgentChatResponse(**result)


class ConversationCreateRequest(BaseModel):
    project_id: int | None = Field(default=None)


class ConversationChatRequest(BaseModel):
    message: str = Field(..., description="用户自然语言输入（支持短输入如『1』『继续』）")
    project_id: int | None = Field(default=None)


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=40)


class FlagRequest(BaseModel):
    value: bool = Field(default=True)


@router.post("/conversations")
def create_conversation(payload: ConversationCreateRequest) -> dict:
    """创建一个对话会话（多轮上下文 / 报告状态记忆）。"""
    conv = conversation_service.create_conversation(payload.project_id)
    return {
        "conversation_id": conv["conversation_id"],
        "project_id": conv["project_id"],
        "title": conv["title"],
        "stage": conv["state"]["stage"],
        "created_at": conv["created_at"],
    }


@router.get("/conversations")
def list_conversations(query: str | None = None, include_archived: bool = False) -> dict:
    """历史会话列表（支持搜索 / 是否含归档；置顶在前，按更新时间倒序）。"""
    return {"conversations": conversation_service.list_conversations(
        query=query, include_archived=include_archived)}


@router.post("/conversations/cleanup")
def cleanup_conversations() -> dict:
    """一次性清理历史中的空会话（无消息/无附件/无项目/无报告）。"""
    return conversation_service.cleanup_empty_conversations()


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    """读取会话（历史消息 + 当前阶段 + 报告状态）。"""
    conv = conversation_service.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期，请重新开始对话。")
    st = conv["state"]
    return {
        "conversation_id": conv["conversation_id"],
        "project_id": conv["project_id"],
        "title": conv["title"],
        "stage": st["stage"],
        "report_ready": st["report_ready"],
        "report_id": st["report_id"],
        "messages": conv["messages"],
    }


@router.patch("/conversations/{conversation_id}/rename")
def rename_conversation(conversation_id: str, payload: RenameRequest) -> dict:
    e = conversation_service.rename_conversation(conversation_id, payload.title)
    if e is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return {"id": conversation_id, "title": e["title"]}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    ok = conversation_service.delete_conversation(conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return {"id": conversation_id, "deleted": True}


@router.post("/conversations/{conversation_id}/pin")
def pin_conversation(conversation_id: str, payload: FlagRequest) -> dict:
    e = conversation_service.set_pinned(conversation_id, payload.value)
    if e is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return {"id": conversation_id, "pinned": e["pinned"]}


@router.post("/conversations/{conversation_id}/archive")
def archive_conversation(conversation_id: str, payload: FlagRequest) -> dict:
    e = conversation_service.set_archived(conversation_id, payload.value)
    if e is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return {"id": conversation_id, "archived": e["archived"]}


@router.get("/conversations/{conversation_id}/share")
def share_conversation(conversation_id: str) -> dict:
    payload = conversation_service.share_payload(conversation_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return payload


@router.post("/conversations/{conversation_id}/chat")
def conversation_chat(conversation_id: str, payload: ConversationChatRequest,
                      db: Session = Depends(get_db)) -> dict:
    """在会话内对话：多轮上下文 + 短输入理解 + 报告生成/导出状态记忆。"""
    return conversation_service.chat(
        db, message=payload.message, conversation_id=conversation_id,
        project_id=payload.project_id,
    )


@router.post("/conversations/{conversation_id}/attachments")
async def upload_attachment(conversation_id: str,
                            file: UploadFile = File(...)) -> dict:
    """上传项目资料附件：解析为文本 → 抽取项目要素 → 并入会话档案。"""
    conv = conversation_service.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期，请重新开始对话。")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件为空。")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件过大，请上传小于 20MB 的文件。")
    parsed = attachment_service.parse(file.filename or "未命名文件", data)
    result = conversation_service.ingest_attachment(conv, parsed)
    return {"conversation_id": conversation_id, **result}


@router.post("/run-task", response_model=AgentChatResponse)
def agent_run_task(payload: AgentRunTaskRequest, db: Session = Depends(get_db)) -> AgentChatResponse:
    """快捷任务：与 chat 共用编排，按 task_type 触发对应工具组合。"""
    message = payload.message or _default_task_message(payload.task_type)
    result = agent_orchestrator_service.run_agent(
        db, message=message, project_id=payload.project_id,
        task_type=payload.task_type, thinking=payload.thinking,
    )
    return AgentChatResponse(**result)


def _default_task_message(task_type: str) -> str:
    return {
        "project_diagnosis": "请给出这个项目的初步研判。",
        "poi_analysis": "请分析这个项目的周边配套情况。",
        "renewal_type": "这个项目适合做什么类型的城市更新？",
        "strategy": "请基于研判结果给出更新策略建议。",
        "report_outline": "请生成这个项目的城市更新前期策划报告大纲。",
        "evidence_trace": "这些结论的证据来源和数据血缘是什么？",
    }.get(task_type, "请给出这个项目的初步研判。")
