#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import requests
import sqlite3 
from datetime import datetime
from typing import Any, List, Tuple, Dict, Optional

from typing_extensions import TypedDict, Annotated
import operator

from langchain_core.messages import HumanMessage, get_buffer_string, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from .prompts import *
import re
from .gemini_chat import *

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from deep_research import utils as dr_utils
dr_utils.set_tool_audit(enabled=True)

# 引入本地知识库（计划专用）的实现
from .local_db import init_local_kb, search_local_kb

from pydantic import BaseModel, field_validator, Field


# =============================================================================
# 数据库查询工具配置
# =============================================================================
DB_PATH = str(_PROJECT_ROOT / "dbdata" / "JN1-BA_results.db")
TABLE_NAME = "alarm_events"

# =============================================================================
# [核心修复] 定义覆盖策略，防止并行写入报错
# =============================================================================
def overwrite(left: Any, right: Any) -> Any:
    """
    当两个节点同时写入同一个字段时，使用后者（右值）覆盖前者。
    这能解决 InvalidUpdateError: Can receive only one value per step.
    对于非列表类型的状态更新，通常建议使用此策略。
    """
    return right

# =============================================================================
# AgentState 定义 (已更新以支持 Supervisor 路由模式)
# =============================================================================

class AgentState(TypedDict, total=False):
    """
    图在运行过程中的状态容器。
    """
    # === 累加型字段 (使用 operator.add) ===
    messages: Annotated[List[BaseMessage], operator.add]
    raw_notes: Annotated[List[str], operator.add]
    notes: Annotated[List[str], operator.add]
    supervisor_messages: Annotated[List[str], operator.add] # 存储 Supervisor 和 Tools 的交互日志
    
    # [新增] 对话历史字段
    # 建议存储格式为 List[str] 或 List[BaseMessage]，每次对话追加新的 Human/AI 记录
    # 使用 operator.add 确保历史记录不断累积而不丢失
    history: Annotated[List[str], operator.add]

    # === 覆盖型字段 (使用 overwrite) ===
    # 基础输入
    user_request: Annotated[str, overwrite]
    
    # 思维链/前置规划 (对应 DetermineComplexity 中的 question 字段)
    question: Annotated[str, overwrite] 
    current_tot: Annotated[str, overwrite] # 兼容旧逻辑

    # 上下文积累 (Execute Tools Node 会读取并更新此字段)
    pre_brief_cases: Annotated[str, overwrite]
    
    # 最终报告
    final_report: Annotated[str, overwrite]
    draft_report: Annotated[str, overwrite]
    research_brief: Annotated[str, overwrite]

    # === [新增] Supervisor 路由专用字段 ===
    # 存储 single_supervisor_node 的决策结果 ("DATABASE" | "CHART" | "DIRECT")
    supervisor_route: Annotated[str, overwrite]
    # Supervisor 提取出的参数，供下游工具节点直接复用
    supervisor_params: Annotated[Dict[str, Any], overwrite]
    # 前端展示分类：direct / kb_retrieval / station_device_td / alerting / troubleshooting / deep_research
    # 由 supervisor 路由 LLM 显式输出，或在缺失时由 _infer_task_type 推断；
    # 深度研究分支会被强制覆盖为 deep_research。
    task_type: Annotated[str, overwrite]

    # === 工具输出字段 ===
    # 图表 URL (由 generate_chart_node 更新)
    chart_output: Annotated[str, overwrite]
    # 数据库参数 (由 retrieve_battery_node 更新)
    db_query_params: Annotated[Dict[str, Any], overwrite]
    # 数据库原始结果 (可选)
    db_query_result: Annotated[str, overwrite]
    # DATABASE 子链路：路由与执行信息
    db_route: Annotated[str, overwrite]
    db_query_plans: Annotated[List[Dict[str, Any]], overwrite]
    db_executed_sqls: Annotated[List[str], overwrite]
    db_evidence_bundle: Annotated[Dict[str, Any], overwrite]
    db_raw_results: Annotated[List[Dict[str, Any]], overwrite]

    # Web / CLI：本轮 run_id 与 DB 链路日志路径（server_plus / thinkdepth_test 注入）
    run_id: Annotated[str, overwrite]
    db_chain_log_path: Annotated[str, overwrite]
    db_plan_sanitizer_log_path: Annotated[str, overwrite]
    db_sanitizer_log_enabled: Annotated[bool, overwrite]

    # 标记位
    is_complex_task: Annotated[bool, overwrite]
    is_chart_needed: Annotated[bool, overwrite]
    is_use_db: Annotated[bool, overwrite]

# =============================================================================
# Pydantic 输出模型
# =============================================================================

class ResearchQuestion(BaseModel):
    research_brief: str

    @field_validator('research_brief', mode='before')
    def normalize_research_brief(cls, v):
        if isinstance(v, dict):
            title = v.get('title') or ''
            summary = v.get('summary') or v.get('content') or ''
            sections = v.get('sections') or v.get('deliverables') or []
            sources = v.get('sources') or []
            parts = []
            if title:
                parts.append(str(title))
            if summary:
                parts.append(str(summary))
            if sections:
                parts.append('Sections:\n' + '\n'.join(map(str, sections)))
            if sources:
                parts.append('Sources:\n' + '\n'.join(map(str, sources)))
            s = '\n\n'.join([p for p in parts if p])
            return s if s else json.dumps(v, ensure_ascii=False)
        if isinstance(v, list):
            return '\n'.join(map(str, v))
        if v is None:
            return ''
        return str(v)

class DraftReport(BaseModel):
    draft_report: str

    @field_validator('draft_report', mode='before')
    def normalize_draft_report(cls, v):
        if isinstance(v, dict) or isinstance(v, list):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        if v is None:
            return ''
        return str(v)


class DetermineComplexity(BaseModel):
    need_deepresearch: bool = Field(
        description="Set to True (1) if the task is complex/requires deep research. Set to False (0) if simple."
    )
    question: str = Field(
        description="If need_deepresearch is False, provide the preliminary ToT here. Otherwise empty."
    )
    verification: str = Field(
        description="Confirmation message if research is starting."
    )

    @field_validator('need_deepresearch', mode='before')
    def to_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ('true', 'yes', 'y', '1'):
                return True
            if s in ('false', 'no', 'n', '0'):
                return False
        if isinstance(v, (int, float)):
            return bool(v)
        return False

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
    def __init__(self, temperature: float = 0.3, max_tokens: int = 4096):
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

def init_chat_model(model: str = GEMINI_MODEL, temperature: float = 0.3, max_tokens: int = 4096) -> _MideaGeminiChatModel:
    return _MideaGeminiChatModel(temperature=temperature, max_tokens=max_tokens)

# ===== CONFIGURATION =====
model = init_chat_model(model=GEMINI_MODEL, temperature=0.2, max_tokens=4096)
creative_model = init_chat_model(model=GEMINI_MODEL, temperature=0.7, max_tokens=15000)

# =============================================================================
# 绘图参数提取模型
# =============================================================================

class ChartIntent(BaseModel):
    needs_chart: bool = Field(description="Set to True if extracted data is sufficient and user implies visualization.")
    chart_type: str = Field(description="One of: 'line_chart', 'bar_chart', 'pie_chart', 'area_chart'.")
    data_json: str = Field(description="A valid JSON string representing the list of dictionaries for the chart data. Example: '[{\"name\":\"A\",\"val\":10}]'")
    x_field: str = Field(description="Field name for X axis in data items")
    y_field: str = Field(description="Field name for Y axis in data items")
    title: str = Field(description="Chart title")
