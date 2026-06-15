"""第12E：智能体（Agent）层数据模型。

仅承载「智能体交互」：理解用户问题 → 调用自研只读工具 → 由大模型（DeepSeek）
基于结构化结果组织自然语言回答。

红线：
- 不承载训练/评测/导出等写操作；工具层仅只读 GET。
- 不向前端返回任何 API key；不返回 raw chain-of-thought。
- 大模型仅做理解与表达，专业结论以自研模型与真实数据为准。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# 支持的快捷任务类型（与前端快捷任务对齐）
TASK_TYPES = [
    "project_diagnosis",   # 项目初判
    "poi_analysis",        # 周边配套分析
    "renewal_type",        # 更新类型判断
    "strategy",            # 更新策略
    "report_outline",      # 报告大纲
    "evidence_trace",      # 证据来源
]


class AgentChatRequest(BaseModel):
    """智能体对话请求。"""

    message: str = Field(..., description="用户自然语言输入")
    project_id: int = Field(default=1, description="目标项目（当前仅示例项目 1）")
    task_type: str | None = Field(
        default=None, description="可选任务类型；缺省时由后端按消息推断"
    )
    thinking: bool = Field(
        default=False, description="是否开启深度思考（pro 推理模型）；否则用常规对话（flash）"
    )


class AgentRunTaskRequest(BaseModel):
    """快捷任务请求。"""

    task_type: str = Field(..., description="快捷任务类型，见 TASK_TYPES")
    project_id: int = Field(default=1, description="目标项目")
    message: str | None = Field(default=None, description="可选补充说明")
    thinking: bool = Field(
        default=False, description="是否开启深度思考（pro 推理模型）；否则用常规对话（flash）"
    )


class ToolCallResult(BaseModel):
    """单个工具调用的脱敏结果（供前端「工具调用过程」展示）。"""

    tool: str = Field(..., description="工具内部标识")
    name: str = Field(..., description="工具中文名")
    status: str = Field(..., description="ok / empty / error")
    message: str = Field(default="", description="状态说明（失败时为原因，不含原文）")
    summary: dict[str, Any] = Field(
        default_factory=dict, description="脱敏后的关键字段摘要"
    )


class ModelLayer(BaseModel):
    """模型分工说明。"""

    llm: str = Field(default="deepseek", description="大模型思考层")
    professional_engine: str = Field(
        default="CityRenew self-trained models",
        description="专业判断层（自研城市更新模型）",
    )
    llm_role: str = "理解需求、拆解任务、组织表达"
    engine_role: str = "项目研判、评分、类型识别、策略方向、证据链"


class AgentChatResponse(BaseModel):
    """智能体回答（已格式化，前端可直接展示）。"""

    answer: str = Field(..., description="面向用户的自然语言回答")
    task_type: str = Field(..., description="本次解析的任务类型")
    project_id: int
    used_tools: list[ToolCallResult] = Field(default_factory=list)
    structured_result: dict[str, Any] = Field(
        default_factory=dict, description="自研模型/数据工具的结构化结果（脱敏）"
    )
    evidence_refs: list[dict[str, Any]] = Field(
        default_factory=list, description="证据来源（evidence_id / 来源文件名 / 摘要）"
    )
    next_actions: list[dict[str, str]] = Field(
        default_factory=list, description="建议的下一步动作（label + target 路由）"
    )
    limitations: list[str] = Field(
        default_factory=list, description="可信边界 / 数据不足提示"
    )
    confidence_note: str = Field(default="", description="可信度说明")
    thinking_summary: list[str] = Field(
        default_factory=list,
        description="思考过程摘要（非 raw chain-of-thought，仅工具编排概述）",
    )
    model_layer: ModelLayer = Field(default_factory=ModelLayer)
    deepseek_configured: bool = Field(
        default=False, description="大模型思考层是否已配置"
    )
    deepseek_used: bool = Field(
        default=False, description="本次回答是否实际经过大模型组织"
    )
    thinking_mode: bool = Field(
        default=False, description="本次是否使用深度思考（pro 推理）模型"
    )
    model_used: str | None = Field(
        default=None, description="本次实际使用的大模型名（降级时为空）"
    )
    degraded: bool = Field(
        default=False, description="是否处于降级模式（大模型不可用，仅自研模型结果）"
    )


class AgentToolMeta(BaseModel):
    """能力清单中的单个工具描述。"""

    tool: str
    name: str
    description: str
    source: str = Field(default="", description="数据/接口来源（脱敏）")
    available: bool = True
    status: str = Field(default="ready", description="ready / need_data / error")


class AgentCapabilitiesResponse(BaseModel):
    """智能体能力清单。"""

    agent_name: str = "CityRenew Agent"
    summary: str = (
        "面向城市更新前期策划的专业智能体：大模型负责理解与表达，"
        "自研城市更新模型负责项目研判、评分、类型识别与策略方向，"
        "结论可追溯至真实数据与证据链。"
    )
    deepseek_configured: bool = False
    model_name: str | None = None
    thinking_enabled: bool = False
    skills: list[dict[str, str]] = Field(default_factory=list)
    tools: list[AgentToolMeta] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=lambda: list(TASK_TYPES))
    model_layer: ModelLayer = Field(default_factory=ModelLayer)


class AgentHealthResponse(BaseModel):
    """智能体健康检查。"""

    status: str = "ok"
    deepseek_configured: bool = False
    model_name: str | None = None
    thinking_enabled: bool = False
    tools_available: int = 0
    tools_total: int = 0
    self_model_available: bool = False
    evidence_available: bool = False
    warning_count: int = 0
    message: str = ""
