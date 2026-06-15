"""第12E/12F：DeepSeek 大模型思考层封装。

唯一允许出网的智能体模块。职责严格限定为「理解 + 表达」：
- 理解用户问题、拆解任务、组织自然语言回答；
- 基于后端自研工具返回的结构化结果进行表达，不得脱离工具结果编造城市更新结论；
- 不生成事实数字、不读取语料原文、不输出 raw chain-of-thought。

双模型（第12F）：
- 常规对话：deepseek-v4-flash（快）；
- 深度思考：deepseek-v4-pro（推理模型，思考链写入 reasoning_content，需更大 max_tokens）。

红线：
- API key 仅从 .env 经 pydantic settings 读取，绝不写死、绝不入库、绝不返回前端、绝不打印。
- key 缺失或调用失败时返回结构化的「不可用」状态，由编排层 graceful fallback。
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from app.config import settings

logger = logging.getLogger("cityrenew.agent.deepseek")

# 系统提示词：明确大模型与自研模型的分工边界（写入每次请求）。
SYSTEM_PROMPT = (
    "你是 CityRenew Agent（城市更新前期策划智能体）的大模型思考层。\n"
    "你负责理解用户问题、拆解任务、组织面向用户的自然语言回答。\n"
    "你不能脱离工具结果编造城市更新结论。\n"
    "专业判断必须基于 CityRenew 自研模型、POI/房价/产业/人口数据、综合评分结果与证据链；"
    "这些结构化结果会在用户消息中以 JSON 形式提供给你，你只能基于它们组织表达。\n"
    "所有数字、比例、距离、均价、评分、类型结论必须直接来自工具结果，"
    "不得自行推算或虚构；工具未提供的事实一律说明「该数据暂不可用」。\n"
    "如果数据不足，必须明确说明不足与可信边界。\n"
    "不要输出内部训练指标、变量名或评测术语（如 weak_label、MAPE、final test、train/val）给普通用户，"
    "应使用业务化中文表达。\n"
    "不要输出你的内部推理过程（raw chain-of-thought），只输出给用户看的结论、依据、建议和限制。\n"
    "回答使用简体中文，语气专业、克制、可信。"
)

_DEFAULT_BASE_URL = "https://api.deepseek.com"


def get_config() -> dict[str, Any]:
    """读取 DeepSeek 配置（不含 key 明文，仅返回是否已配置与模型名）。

    供 health / capabilities 接口使用，绝不暴露 key 本身。
    """
    api_key = (settings.deepseek_api_key or "").strip()
    base_url = (settings.deepseek_base_url or _DEFAULT_BASE_URL).strip()
    chat_model = (settings.deepseek_model_chat or "").strip()
    think_model = (settings.deepseek_model_think or "").strip()
    legacy = (settings.deepseek_model or "").strip()

    # 回退：未显式配置 chat/think 时，用旧单模型或默认值兜底。
    if not chat_model:
        chat_model = legacy or "deepseek-v4-flash"
    if not think_model:
        think_model = legacy or "deepseek-v4-pro"

    return {
        "configured": bool(api_key),
        "base_url": base_url,
        "chat_model": chat_model,
        "think_model": think_model,
        "model": chat_model,  # 兼容旧字段（默认展示常规模型）
        "thinking_enabled": bool(settings.deepseek_thinking_enabled),
    }


def is_configured() -> bool:
    return bool((settings.deepseek_api_key or "").strip())


def generate(
    user_content: str,
    *,
    thinking: bool = False,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """调用 DeepSeek 生成回答。

    参数 thinking：
    - False（默认）→ 常规对话模型（flash），快、max_tokens 较小；
    - True → 深度思考模型（pro，推理），思考链会占用 max_tokens，需更大额度。

    返回统一结构：
      {ok: bool, text: str, error: str, model: str, thinking: bool, reasoning_used: bool}
    任何异常都被捕获并转为 ok=False，绝不向上抛出，绝不泄露 key。
    """
    cfg = get_config()
    api_key = (settings.deepseek_api_key or "").strip()
    model = cfg["think_model"] if thinking else cfg["chat_model"]
    if not api_key:
        return {"ok": False, "text": "", "error": "not_configured",
                "model": model, "thinking": thinking, "reasoning_used": False}

    # 推理模型的 reasoning_content 与最终 content 共享 max_tokens，故思考模式给更大额度。
    if max_tokens is None:
        max_tokens = 4000 if thinking else 1500

    url = f"{cfg['base_url'].rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = int(settings.deepseek_request_timeout_s or 60)
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            # 不记录响应正文（可能回显敏感请求头），仅记录状态码。
            logger.warning("DeepSeek 调用返回非 200：status=%s model=%s", resp.status_code, model)
            return {"ok": False, "text": "", "error": f"http_{resp.status_code}",
                    "model": model, "thinking": thinking, "reasoning_used": False}
        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message", {})
        text = (msg.get("content") or "").strip()
        reasoning_content = (msg.get("reasoning_content") or "").strip()
        finish_reason = (data.get("choices") or [{}])[0].get("finish_reason", "")

        if not text:
            if reasoning_content:
                logger.warning(
                    "DeepSeek 推理模型 content 为空（finish_reason=%s）：max_tokens 可能不足", finish_reason,
                )
                return {"ok": False, "text": "", "error": "reasoning_truncated",
                        "model": model, "thinking": thinking, "reasoning_used": True}
            return {"ok": False, "text": "", "error": "empty_response",
                    "model": model, "thinking": thinking, "reasoning_used": False}
        return {"ok": True, "text": text, "error": "",
                "model": model, "thinking": thinking, "reasoning_used": bool(reasoning_content)}
    except requests.Timeout:
        logger.warning("DeepSeek 调用超时（model=%s）", model)
        return {"ok": False, "text": "", "error": "timeout",
                "model": model, "thinking": thinking, "reasoning_used": False}
    except requests.RequestException as exc:
        logger.warning("DeepSeek 调用网络异常：%s", type(exc).__name__)
        return {"ok": False, "text": "", "error": "network_error",
                "model": model, "thinking": thinking, "reasoning_used": False}
    except Exception as exc:  # noqa: BLE001 - 兜底，绝不向上抛
        logger.warning("DeepSeek 调用未知异常：%s", type(exc).__name__)
        return {"ok": False, "text": "", "error": "unknown_error",
                "model": model, "thinking": thinking, "reasoning_used": False}
