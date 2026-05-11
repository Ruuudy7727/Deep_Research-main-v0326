#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import requests
from datetime import datetime
from typing import Any, List, Tuple, Dict, Optional

from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import HumanMessage, get_buffer_string, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
# 注意：虽然导入了 Command，但在配合 research_agent_full.py 使用时，
# scope 内的节点应尽量避免直接使用 Command 进行路由，而是通过返回状态让 Router 决定。
from langgraph.types import Command, StreamWriter
from .prompts import *
import re

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from deep_research import utils as dr_utils
dr_utils.set_tool_audit(enabled=True)

# 引入本地知识库（计划专用）的实现
from .local_db import init_local_kb, search_local_kb

# =============================================================================
# [核心修复] 强制从 state_scope 导入 State 和 Models
# 禁止在此处重新定义 AgentState，否则会导致并行写入时的 InvalidUpdateError
# =============================================================================
from .state_scope import (
    AgentState, 
    ResearchQuestion, 
    DraftReport, 
    DetermineComplexity,
    ChartIntent
)

# [新增] 尝试引入绘图工具
try:
    from .research_agent_draw import generate_chart_url
except ImportError:
    generate_chart_url = None
    print("[System] Warning: research_agent_draw module not found. Charting disabled.")

from pydantic import BaseModel, Field, field_validator
MAX_TOKENS=15000
from .gemini_chat import *
import deep_research.gemini_chat  # 引入模块对象，以支持 server.py 的流式适配器


# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    return datetime.now().strftime("%a %b %-d, %Y")


def _messages_to_user_text(messages: List[BaseMessage]) -> str:
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(m.content)
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
        else:
            parts.append(str(getattr(m, "content", m)))
    return "\n".join(parts).strip()

# ===== 预检索辅助函数 =====
def _generate_retrieval_topics_en(messages_str: str, n: int = 3) -> List[str]:
    # 注意：根据要求，此函数现在强制生成中文主题
    system_instruction = (
        "你是一个以案例为中心的搜索关键词生成器。"
        "返回且仅返回一行或多行、每行一个的简洁中文关键词短语。"
        "每个短语必须专注于失效模式或根本原因，并包含 '案例' 一词。"
        "不要添加编号、项目符号、标点或任何解释。"
        "专注于现实世界的失效模式、根本原因和纠正措施。"
        "如果主题与电池失效分析相关，请遵循'工况 → 机理 → 现象'的映射链构建关键词。"
    )
    user_text = (
        "根据下面的对话，生成多个以案例为中心的中文搜索关键词短语。\n\n"
        f"对话内容:\n'''\n{messages_str}\n'''\n\n"
        "要求:\n"
        f"- 返回 {n} 行，每行一个短语。\n"
        "- 每行都必须包含 '案例'、'研究' 或 '排查' 等词。\n"
        "- 保持简洁具体，不要有任何额外文本。"
    )


    text, _ = gemini_chat_once(
        user_text=user_text,
        system_instruction=system_instruction,
        temperature=0.5,
        max_tokens=512
    )
    print(f"LLM 原始输出:\n---\n{text}\n---", flush=True)

    raw_lines = (text or "").splitlines()
    cleaned = []
    for ln in raw_lines:
        s = (ln or "").strip()
        s = re.sub(r"^[\-\*\d\.\)\s]+", "", s)
        if s:
            cleaned.append(s)

    topics = []
    seen = set()
    for s in cleaned:
        s_norm = s.strip()
        key = s_norm.lower()
        if key and key not in seen:
            seen.add(key)
            topics.append(s_norm)

    print(f"处理后的最终 Topics:\n{topics}", flush=True)
    return topics

# ===== 工具函数封装 =====

def init_local_retrievers():
    result = init_local_kb()
    print(result, flush=True)
    return result

def local_search(query: str, top_k: int) -> str:
    return search_local_kb(query=query, top_k=top_k)

def tavily_search(query: str, max_results: int) -> str:
    if os.getenv("EXTERNAL_SEARCH_ENABLED", "1") == "0" or not os.getenv("TAVILY_API_KEY"):
        print("[TOOL] Tavily 搜索未配置或已禁用，跳过。")
        return "External web search disabled or not configured."
    try:
        result = dr_utils.tavily_search.invoke({"query": query, "max_results": max_results, "topic": "general"})
    except AttributeError:
        result = dr_utils.tavily_search(query=query, max_results=max_results, topic="general")
    except Exception as e:
        print(f"[TOOL] Tavily 搜索异常：{e}")
        result = "External web search failed."
    return result

# ===== 公共检索逻辑 =====
def _execute_retrieval_logic(messages_str: str) -> str:
    """
    执行生成关键词、本地搜索的公共逻辑，返回汇总文本。
    """
    topics = _generate_retrieval_topics_en(messages_str)
    print(topics)
    print("Retrieving cases for generated topics:", flush=True)
    for i, t in enumerate(topics, 1):
        print(f"  {i}. {t}", flush=True)

    init_local_retrievers()

    blocks = []
    for idx, topic in enumerate(topics, 1):
        print(f"Web search for topic {idx} completed.", flush=True)
        local_cases_result = local_search(query=topic, top_k=3)
        print(f"Local search for topic {idx} completed.", flush=True)

        block = (
            f"=== Topic {idx} ===\n"
            f"Query: {topic}\n\n"
            f"--- Local Knowledge Base Examples ---\n{local_cases_result}\n"
        )
        blocks.append(block)

    combined_cases = "\n\n".join(blocks)
    return combined_cases

# ===== JSON结构化包装 =====

def _cleanup_json_text(text: str) -> str:
    s = str(text).strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "").replace("JSON\n", "")
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last >= first:
        s = s[first:last+1]
    return s.strip()

def _pydantic_fields(cls: Any) -> List[str]:
    try:
        fields = getattr(cls, "model_fields", {})
        if isinstance(fields, dict) and fields:
            return list(fields.keys())
    except Exception:
        pass
    try:
        fields_v1 = getattr(cls, "__fields__", {})
        if isinstance(fields_v1, dict) and fields_v1:
            return list(fields_v1.keys())
    except Exception:
        pass
    return []

def _pydantic_construct(cls: Any, data: Dict[str, Any]) -> Any:
    if hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    if hasattr(cls, "parse_obj"):
        return cls.parse_obj(data)
    return cls(**data)

class _StructuredWrapper:
    def __init__(self, temperature: float, max_tokens: int, output_cls: Any):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_cls = output_cls

    def invoke(self, messages: List[BaseMessage]) -> Any:
        user_text = _messages_to_user_text(messages)
        fields = _pydantic_fields(self.output_cls)
        fields_desc = ", ".join(fields) if fields else "请只输出所需字段的 JSON 对象"

        type_constraints = ""
        if self.output_cls is ResearchQuestion:
            type_constraints = " For the field 'research_brief', the value MUST be a single plain string (not an object or array)."
        elif self.output_cls is DraftReport:
            type_constraints = " For the field 'draft_report', the value MUST be a single plain string (not an object or array)."

        system_instruction = (
            "You are a strict JSON generator. "
            "Return ONLY a valid JSON object with keys: " + fields_desc + "."
            + type_constraints +
            " Do not include any extra text, commentary, or code fences. "
            "Ensure the JSON is syntactically valid."
        )
        text, _usage = gemini_chat_once(
            user_text=user_text,
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        cleaned = _cleanup_json_text(text)
        data: Dict[str, Any] = {}
        try:
            data = json.loads(cleaned)
        except Exception:
            if fields:
                if len(fields) == 1:
                    data = {fields[0]: text}
                else:
                    data = {}
                    for f in fields:
                        data[f] = text

        return _pydantic_construct(self.output_cls, data)

class _MideaGeminiChatModel:
    def __init__(self, temperature: float = 0.3, max_tokens: int = MAX_TOKENS):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        user_text = _messages_to_user_text(messages)
        text, _usage = gemini_chat_once(
            user_text=user_text,
            system_instruction="",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return AIMessage(content=text)

    def with_structured_output(self, output_cls: Any) -> _StructuredWrapper:
        return _StructuredWrapper(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            output_cls=output_cls
        )

def init_chat_model(model: str = GEMINI_MODEL, temperature: float = 0.3, max_tokens: int = MAX_TOKENS) -> _MideaGeminiChatModel:
    return _MideaGeminiChatModel(temperature=temperature, max_tokens=max_tokens)

# ===== CONFIGURATION =====
model = init_chat_model(model=GEMINI_MODEL, temperature=0, max_tokens=MAX_TOKENS)
creative_model = init_chat_model(model=GEMINI_MODEL, temperature=0.7, max_tokens=MAX_TOKENS)


def clarify_with_user(state: AgentState) -> dict:
    """
    第一步：仅进行意图分析和复杂性判断。
    只负责区分简单/复杂，并提取简单任务的 ToT 思维链，不涉及任何工具参数的具体提取。
    """
    print("[clarify_with_user] state keys:", list(state.keys()), flush=True)

    # 1. 定义解析模型 (已删除 DB 和 Chart 字段，仅保留核心判断和思维链)
    class DetermineComplexityLoose(BaseModel):
        need_deepresearch: bool = Field(
            description="True for complex tasks (Multi-Agent), False for simple tasks."
        )
        question: Optional[str] = Field(
            default="",
            description="For simple tasks: The ToT analysis string. For complex: empty."
        )

    # 2. 构建消息上下文 (保持不变)
    msgs = state.get("messages") or []
    if not msgs:
        ur = (state.get("user_request") or "").strip()  # 尝试从 user_request 获取
        if ur:
            msgs = [HumanMessage(content=ur)]
            state["messages"] = msgs
        else:
            raw_notes = state.get("raw_notes") or []  # 尝试从 notes 拼接
            notes = state.get("notes") or []
            merged = "\n".join([*map(str, raw_notes), *map(str, notes)]).strip()
            if merged:
                msgs = [HumanMessage(content=merged)]
                state["messages"] = msgs

    messages_str = get_buffer_string(messages=msgs) or ""  # 转为字符串格式

    # 3. 调用 LLM
    try:
        structured_output_model = model.with_structured_output(DetermineComplexityLoose)
        # 虽然提示词里要求了 db/chart，但通过 Pydantic 强行只接收 deepresearch 和 question
        response = structured_output_model.invoke([
            HumanMessage(content=classify_complexity_instructions.format(
                messages=messages_str,
                date=get_today_str()
            ))
        ])
    except Exception as e:
        print(f"--- [LLM Invoke Error] 自动降级为简单模式: {e}", flush=True)
        response = DetermineComplexityLoose(
            need_deepresearch=False,
            question=f"Error analyzing request: {e}. Treating as simple query."
        )

    # 4. 提取核心字段
    is_deep_research = response.need_deepresearch  # 是否复杂任务

    # 提取思维链：仅在简单任务且有内容时提取
    tot_outline = ""
    if not is_deep_research and response.question:
        tot_outline = response.question.strip()

    # 日志
    print(f"--- 判定结果: 复杂={is_deep_research} ---", flush=True)
    if tot_outline:
        print(f"ToT Plan: {tot_outline[:60].replace(chr(10), ' ')}...", flush=True)

    # 5. 更新状态 (只更新复杂性标记和思维链)
    # 复杂任务（深度研究分支）一律标记为 deep_research，前端使用深度检索卡片；
    # 简单任务的 task_type 留给 single_agent 内部 supervisor 路由后再写。
    update: Dict[str, Any] = {
        "user_request": messages_str,
        "is_complex_task": is_deep_research,
        "current_tot": tot_outline,
        "supervisor_messages": state.get("supervisor_messages", []) + [
            f"Classify: Complex={is_deep_research}"
        ],
    }
    if is_deep_research:
        update["task_type"] = "deep_research"
    return update


def pre_brief_retrieval(state: AgentState) -> dict:
    """
    第二步：通用的检索节点。
    [关键修改] 返回 Dict 而非 Command。
    research_agent_full.py 中的 conditional_edges 会根据 state 自动路由。
    """
    print("--- Executing Node: pre_brief_retrieval (Common Retrieval) ---", flush=True)
    
    # 1. 如果是明确的数据库查询任务，跳过通用检索以节省时间（可选策略）
    if state.get("is_use_db"):
         print("--- Skipping general retrieval for DB task ---", flush=True)
         return {"supervisor_messages": state.get("supervisor_messages", []) + ["Skipped retrieval for DB task."]}

    messages_str = get_buffer_string(state.get("messages", [])) or ""
    
    # 2. 执行公共检索逻辑
    combined_cases = _execute_retrieval_logic(messages_str)
    
    print("--- pre_brief_retrieval 完成，已汇总案例 ---", flush=True)

    existing_msgs = state.get("supervisor_messages", [])
    log_cases = combined_cases if len(combined_cases) <= 4000 else combined_cases[:4000] + "\n...[truncated]..."
    updated_msgs = (existing_msgs or []) + ["[pre_brief_retrieval] Search completed.", log_cases]
    
    # 3. 准备更新的数据 (不进行路由)
    return {
        "pre_brief_cases": combined_cases,
        "supervisor_messages": updated_msgs
    }

# =============================================================================
# [核心修改] solve_simple_task 增加 writer 参数注入
# LangGraph 在运行时会自动注入 StreamWriter 对象到 writer 参数
# =============================================================================
async def solve_simple_task(state: AgentState, writer: StreamWriter) -> dict:
    """
    第三步（分支A）：简单任务处理。
    基于检索到的案例(pre_brief_cases)或数据库结果(db_query_result)生成专家级诊断。
    """
    import asyncio
    import os
    import sys

    print("--- Executing Node: solve_simple_task (Simple Path) ---", flush=True)

    # 1. 提取上下文信息
    user_req = state.get("user_request", "")
    tot_plan = state.get("current_tot", "")
    rag_info = state.get("pre_brief_cases", "")
    db_info = state.get("db_query_result", "")  # [新增] 确保也能看到数据库查询结果
    
    # 2. 检查是否有图表
    chart_context = ""
    if state.get("chart_output"):
        chart_context = f"\n[辅助信息]: 系统已生成可视化图表，路径/描述为: {state.get('chart_output')}。请在报告中结合图表趋势进行说明。"

    # 3. 数据存在性检查 (用于日志提示)
    has_data = bool(db_info or "Source: database" in rag_info)
    if not rag_info and not db_info:
        print("Warning: No context data (RAG or DB) found.", flush=True)
    else:
        print(f"--- [Context Ready] RAG length: {len(rag_info)}, DB length: {len(str(db_info))} ---", flush=True)

    # 4. 定义专家级系统指令 (System Prompt)
    system_instruction = (
        "身份定义：你是一名资深储能系统与电池技术专家，精通电芯电化学机理、BMS算法逻辑及电站运维策略。"
        "任务目标：基于提供的上下文信息，为用户生成一份专业的排查诊断报告。"
        "\n"
        "核心处理逻辑："
        "1. 【数据敏感性】：首先扫描上下文。若存在具体运行数据（如电压、温度、电流、SOC、压差、报警日志等），"
        "   必须进行定量分析（指出极值、平均值、异常波动点、一致性偏差）。"
        "   若无具体数据，则根据问题描述进行理论分析或给出通用的排查SOP。"
        "2. 【诊断风格】：结论先行，拒绝模棱两可。使用专业术语（如：析锂、SEI膜生长、内阻增大、热失控前兆、采样漂移）。"
        "3. 【结构要求】："
        "   - 第一部分：诊断结论 (一句话概括核心问题)"
        "   - 第二部分：详细分析 (分为'数据洞察'或'机理分析'，视有无数据而定)"
        "   - 第三部分：处置建议 (具体的运维动作，如：下电检查、均衡维护、更换模组)"
        "\n"
        "格式约束："
        "1. 严禁使用 Markdown 加粗符号（即文中不要出现 ** 符号），可使用序号或标点分层。"
        "2. 去除冗余的客套话，直接输出干货。"
        "3. 输出纯文本，不要包含 JSON。"
    )

    # 5. 构建用户输入 (User Prompt)
    # 将 RAG 和 DB 数据合并展示，给 LLM 最全的视野
    user_prompt = (
        f"【用户请求】\n{user_req}\n\n"
        f"【思维链规划 (ToT)】\n{tot_plan}\n\n"
        f"【数据库查询结果】\n{db_info if db_info else '（本次未进行数据库查询或无结果）'}\n\n"
        f"【知识库检索/案例参考】\n{rag_info}\n\n"
        f"{chart_context}\n\n"
        "请依据上述信息，以专家视角生成回复："
    )

    LOCAL_MAX_TOKENS = 4096
    final_answer = ""

    # =========================================================================
    # [流式处理] Wrapper 函数：保持不变
    # =========================================================================
    def _sync_stream_consumption(u_text, sys_inst, max_tok, stream_writer):
        print("\n⚡ [RPO Stream Start] ...", flush=True)
        try:
            generator = deep_research.gemini_chat.gemini_chat_once_rpo(
                user_text=u_text,
                system_instruction=sys_inst,
                temperature=0.3, # 降低温度，增加分析的严谨性
                max_tokens=max_tok
            )
            
            final_txt = ""
            final_usage = {}
            last_len = 0
            
            print(">>> ", end="", flush=True)
            
            if isinstance(generator, tuple):
                return generator[0], generator[1]

            for text_chunk, usage in generator:
                diff = text_chunk[last_len:]
                if diff:
                    sys.stdout.write(diff)
                    sys.stdout.flush()
                    if stream_writer:
                        stream_writer({"token": diff})
                last_len = len(text_chunk)
                final_txt = text_chunk
                final_usage = usage
            
            print("\n\n✅ [Stream End]\n", flush=True)
            return final_txt, final_usage
            
        except Exception as inner_e:
            print(f"\n❌ [Stream Error in Thread]: {inner_e}", flush=True)
            return final_txt if 'final_txt' in locals() else str(inner_e), {}
    # =========================================================================
    
    try:
        text, _ = await asyncio.to_thread(
            _sync_stream_consumption,
            user_prompt,
            system_instruction,
            LOCAL_MAX_TOKENS,
            writer
        )
        final_answer = text
    except Exception as e:
        error_msg = f"Error generating report: {e}"
        print(f"[Simple Task] Generation Failed: {e}")
        final_answer = error_msg

    # 日志记录
    try:
        log_path = str(_PROJECT_ROOT / "log" / "simple_task_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"=== User: {user_req[:50]}... ===\n")
            f.write(final_answer + "\n\n")
    except Exception as e:
        print(f"[Simple Task] Log write failed: {e}")

    print("--- [Simple Path] Answer Generated ---", flush=True)

    return {
        "final_report": final_answer,
        "supervisor_messages": state.get("supervisor_messages", []) + ["Expert analysis completed."]
    }


def write_draft_report(state: AgentState) -> dict:
    #复杂任务的研究初稿生成智能体
    print("--- Executing Node: write_draft_report ---", flush=True)
    retrieved_cases = state.get("pre_brief_cases")
    structured_output_model = creative_model.with_structured_output(DraftReport)
    messages_str = get_buffer_string(state.get("messages", []))
    draft_report_prompt = draft_report_generation_prompt.format(
        user_request = messages_str,
        retrieved_cases=retrieved_cases,
        date=get_today_str()
    )

    final_prompt = (
        draft_report_prompt
        + "\n\nOutput strictly as a JSON object with a single key 'draft_report' whose value is a plain string."
    )

    response = structured_output_model.invoke([HumanMessage(content=final_prompt)])

    draft_report_content = (
        response.get("draft_report", "")
        if isinstance(response, dict)
        else getattr(response, "draft_report", "")
    )

    return {
        # "research_brief": research_brief,
        "draft_report": draft_report_content,   # 复杂问题的解决大纲，前端可展示
        "supervisor_messages": ["Here is the draft report: " + draft_report_content]
    }
