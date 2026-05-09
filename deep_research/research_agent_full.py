#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full Multi-Agent Research System
修复内容：
1. 严格集成用户指定的数据库分析 Prompt 和路由 Prompt。
2. retrieve_battery_node 节点仅负责执行查询并调用 LLM 生成分析，不直接返回 JSON。
3. 保持动态 SQL 构建逻辑。
4. [新增] 全局历史记录管理：无论简单/复杂任务，均自动保存并加载对话历史。
"""

import os
import json
import asyncio
import re
import shutil
import time
import sqlite3
import ast
from typing import Tuple, Dict, Any, List, TypedDict, Annotated, Optional
from operator import add # 必须导入 add 用于 state 更新
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# === LangSmith 环境配置 ===
from dotenv import load_dotenv
env_path = str(_PROJECT_ROOT / ".env")
load_dotenv(env_path, override=True)

if os.environ.get("LANGSMITH_TRACING") == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
if os.environ.get("LANGSMITH_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGSMITH_PROJECT")

from langsmith import traceable

# 原始导入
from .gemini_chat import *
import deep_research.gemini_chat
# 从 prompts 导入提示词
from .prompts import *
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, RemoveMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# 尝试引入绘图工具
try:
    from .research_agent_draw import *
    from .research_agent_analyze import *
except ImportError:
    generate_chart_url = None
    print("[System] Warning: research_agent_draw module not found. Charting disabled.")
    print("[System] Warning: research_agent_analyze module not found. analyze disabled.")

MAX_TOKENS = 15000

# 项目内依赖
import deep_research.utils as dr_utils
from deep_research.utils import get_today_str

# [关键修复] 从统一的状态文件导入 State
from .state_scope import AgentState

from deep_research.research_agent_scope import (
    clarify_with_user,
    pre_brief_retrieval,
    write_draft_report,
    solve_simple_task,
)
from deep_research.multi_agent_supervisor import supervisor_agent
from deep_research.single_agent_supervisor import single_agent


# =============================================================================
# 数据库查询配置
# =============================================================================
DB_PATH = str(_PROJECT_ROOT / "dbdata" / "JN1-BA_results.db")
TABLE_NAME = "alarm_events"


# =============================================================================
# 兼容补丁
# =============================================================================
def _enable_ai_message_lite_support():
    try:
        import langchain_core.messages.utils as msg_utils
        from deep_research.research_agent import AIMessageLite
    except Exception:
        return
    def _adapt_one(m):
        try:
            from deep_research.research_agent import AIMessageLite as _AIL
        except Exception:
            _AIL = None
        if _AIL and isinstance(m, _AIL):
            content = getattr(m, "content", None)
            if content is None:
                content = str(m)
            return AIMessage(content=str(content))
        return m
    def _patch_func(func):
        def _wrapped(message_or_messages, *args, **kwargs):
            try:
                if isinstance(message_or_messages, (list, tuple)):
                    message_or_messages = [_adapt_one(x) for x in message_or_messages]
                else:
                    message_or_messages = _adapt_one(message_or_messages)
            except Exception:
                pass
            return func(message_or_messages, *args, **kwargs)
        return _wrapped
    for name in ["coerce_message_like_to_message", "coerce_maybe_message_to_message", "coerce_message_to_dict", "coerce_messages_to_dicts", "convert_to_messages", "convert_to_message", "coerce_message_like"]:
        try:
            import langchain_core.messages.utils as msg_utils
            obj = getattr(msg_utils, name, None)
            if callable(obj):
                setattr(msg_utils, name, _patch_func(obj))
        except Exception:
            pass
_enable_ai_message_lite_support()


# =============================================================================
# 状态定义与工具函数
# =============================================================================
class AgentInputState(TypedDict, total=False):
    messages: List[BaseMessage]
    raw_notes: List[str]
    notes: List[str]
    supervisor_messages: List[str]
    research_brief: str
    draft_report: str
    user_request: str
    pre_brief_cases: str
    # 允许输入包含历史
    history: List[str] 


def _messages_to_gemini_io(messages) -> Tuple[str, str]:
    system_instruction_parts: List[str] = []
    user_text_parts: List[str] = []
    if not isinstance(messages, (list, tuple)):
        messages = [messages]
    for m in messages:
        if isinstance(m, SystemMessage):
            system_instruction_parts.append(m.content)
            continue
        if isinstance(m, (HumanMessage, AIMessage)):
            user_text_parts.append(m.content)
            continue
        role = getattr(m, "role", None) or getattr(m, "type", None)
        content = getattr(m, "content", None)
        if content is None:
            if isinstance(m, dict):
                content = m.get("content") or json.dumps(m, ensure_ascii=False)
            else:
                content = str(m)
        if role and str(role).lower() == "system":
            system_instruction_parts.append(str(content))
        else:
            user_text_parts.append(str(content))
    return "\n".join(system_instruction_parts).strip(), "\n".join(user_text_parts).strip()

@traceable(name="Gemini Chat Sync", run_type="llm")
def _gemini_chat_sync(messages: List[BaseMessage]) -> AIMessage:
    sys_inst, user_text = _messages_to_gemini_io(messages)
    try:
        text, _usage = gemini_chat_once(user_text, sys_inst)
        return AIMessage(content=text)
    except Exception as e:
        raise RuntimeError(f"_gemini_chat_sync 失败：{e}")

async def _gemini_chat_async(messages) -> AIMessage:
    return await asyncio.to_thread(_gemini_chat_sync, messages)

writer_model_default = RunnableLambda(func=_gemini_chat_sync, afunc=_gemini_chat_async)
writer_model = writer_model_default
dr_utils.set_models(writer_model)
dr_utils.set_tool_audit(enabled=True)


# =============================================================================
# 历史记录管理器 [增强版]
# =============================================================================
class HistoryManager:
    def __init__(self, state: AgentState):
        self.state = state
        self._messages = state.get("messages", [])
        # 优先使用专门的 history 字段
        self._text_history = state.get("history", [])

    def get_recent_history_str(self, k: int = 5) -> str:
        """
        获取最近的 k 轮对话。优先使用 history 列表，如果没有则回退到 messages 解析。
        """
        # 1. 优先尝试读取 List[str] 格式的 history
        if self._text_history and len(self._text_history) > 0:
            # 取最近的 k 条记录
            recent = self._text_history[-k:]
            return "\n\n".join(recent)

        # 2. 回退逻辑：解析 messages 对象
        chat_msgs = [m for m in self._messages if not isinstance(m, SystemMessage)]
        if len(chat_msgs) > 0 and isinstance(chat_msgs[-1], HumanMessage):
             previous_msgs = chat_msgs[:-1]
        else:
             previous_msgs = chat_msgs
        
        if not previous_msgs:
            return "无历史对话。"
            
        recent_msgs = previous_msgs[-(k * 2):]
        history_text = []
        for m in recent_msgs:
            role = "User" if isinstance(m, HumanMessage) else "AI"
            content = str(m.content).strip().replace("\n", " ")[:800] 
            history_text.append(f"- {role}: {content}")
        return "\n".join(history_text)


# =============================================================================
# 最终报告生成节点 (复杂任务路径)
# =============================================================================
@traceable(name="final_report_generation", run_type="chain")
async def final_report_generation(state: AgentState):
    print("--- Executing Node: final_report_generation (Memory Enabled) ---", flush=True)
    notes = state.get("notes", []) or []
    findings = "\n".join(notes)
    
    # [修改] 传入整个 state 以支持从 history 字段读取
    history_mgr = HistoryManager(state)
    chat_history_str = history_mgr.get_recent_history_str(k=5) 
    
    current_request = state.get("user_request", "")
    extended_user_request = current_request
    
    if chat_history_str and chat_history_str != "无历史对话。":
        extended_user_request = (
            f"{current_request}\n\n"
            f"=== [Context from Previous Conversations] ===\n"
            f"{chat_history_str}"
        )

    final_report_prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
        user_request=extended_user_request 
    )
    
    try:
        # 在后台线程中运行生成器并收集结果
        def _run_rpo_generator():
            result = deep_research.gemini_chat.gemini_chat_once_rpo(
                user_text=final_report_prompt,
                system_instruction="Act as a senior research consultant.",
                max_tokens=MAX_TOKENS
            )
            
            # 兼容处理：如果返回的是生成器，遍历它；如果返回的是元组（已被server适配器处理），直接使用
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, tuple)):
                # 是生成器/迭代器
                final_text = ""
                final_usage = {}
                for text_chunk, usage in result:
                    final_text = text_chunk
                    final_usage = usage
                return final_text, final_usage
            elif isinstance(result, tuple) and len(result) == 2:
                # 已经被适配器处理过，直接返回元组
                return result
            else:
                # 其他情况，包装成元组
                return str(result), {}

        text, _ = await asyncio.to_thread(_run_rpo_generator)
        final_report_content = text
    except Exception as e:
        final_report_content = f"Error generating final report: {e}"

    try:
        log_path = str(_PROJECT_ROOT / "log" / "final_report_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(final_report_content + "\n")
    except Exception:
        pass

    # 注意：这里不再直接返回 history，而是交由 update_history_node 统一处理
    return {
        "final_report": final_report_content,
        "messages": [AIMessage(content=final_report_content)],
    }


# =============================================================================
# [新增] 历史记录更新节点
# =============================================================================
def update_history_node(state: AgentState):
    """
    汇聚节点：无论是 Simple 还是 Complex 路径，结束前都会经过这里。
    功能：将本次问答追加到 history 列表中。
    """
    print("--- Executing Node: update_history_node (Saving Context) ---", flush=True)
    
    user_req = state.get("user_request", "Unknown User Request")
    
    # 获取最终回答，solve_simple_task 和 final_report_generation 都应该写入 'final_report'
    ai_resp = state.get("final_report", "")
    
    # 兜底：如果 final_report 为空，尝试从 messages 拿
    if not ai_resp:
        msgs = state.get("messages", [])
        if msgs and isinstance(msgs[-1], AIMessage):
            ai_resp = msgs[-1].content
        else:
            ai_resp = "(No response generated)"

    # 格式化历史条目
    new_entry = f"User: {user_req}\nAI: {ai_resp}"
    
    # 返回字典，State 中的 history (Annotated[List, operator.add]) 会自动追加此列表
    return {"history": [new_entry]}


def prepare_initial_state(question: str, input_type: str = "messages") -> Dict[str, Any]:
    q = (question or "").strip()
    base: Dict[str, Any] = {
        "raw_notes": [],
        "notes": [],
        "supervisor_messages": [],
        "db_query_params": {},
        "history": [] # 初始化
    }
    if input_type == "user_request":
        base["user_request"] = q
    else:
        base["messages"] = [HumanMessage(content=q)]
        base["user_request"] = q
    return base


# =============================================================================
# Router 函数
# =============================================================================
@traceable(name="route_after_retrieval", run_type="chain")
async def route_after_retrieval(state: AgentState):
    if state.get("is_complex_task", False):
        return "complex"
    else:
        return "simple"
    # 节点判断过程，前端可展示

# =============================================================================
# Graph Builder
# =============================================================================
def deep_researcher_builder(llm: RunnableLambda = None, checkpointer=None):
    global writer_model
    if llm is not None:
        writer_model = llm
    dr_utils.set_models(writer_model)

    builder = StateGraph(AgentState, input_schema=AgentInputState)

    # 1. 添加所有节点
    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("pre_brief_retrieval", pre_brief_retrieval)
    builder.add_node("write_draft_report", write_draft_report)
    builder.add_node("supervisor_subgraph", supervisor_agent)
    builder.add_node("final_report_generation", final_report_generation)
    builder.add_node("single_agent", single_agent)
    
    # [新增] 添加历史记录更新节点
    builder.add_node("update_history", update_history_node)

    # 2. 定义边 (Edges)
    builder.add_edge(START, "clarify_with_user")

    # 3. 路由逻辑
    builder.add_conditional_edges(
        "clarify_with_user",
        route_after_retrieval,
        {                        
            "complex": "pre_brief_retrieval",
            "simple": "single_agent"
        }
    )
    builder.add_edge("pre_brief_retrieval", "write_draft_report")
    # Complex Path 继续
    builder.add_edge("write_draft_report", "supervisor_subgraph")
    builder.add_edge("supervisor_subgraph", "final_report_generation")
    
    # 4. [修改] 将原本的结束点重定向到 update_history
    # 无论是 Complex 路径的终点 (final_report_generation)
    # 还是 Simple 路径的终点 (single_agent)
    # 都必须流向 update_history，确保存储记忆
    builder.add_edge("final_report_generation", "update_history")
    builder.add_edge("single_agent", "update_history")
    
    # 5. 最后结束
    builder.add_edge("update_history", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    
    return builder

try:
    agent = deep_researcher_builder().compile()
except Exception:
    agent = None
