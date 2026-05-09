#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import requests
import asyncio
import datetime
import re
import time  # 引入 time 模块用于计时
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

# 加载 .env 环境变量（不覆盖已有环境）
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"), override=False)

# 环境变量设置
os.environ["MIDEA_API_KEY"] = os.getenv("MIDEA_API_KEY", "")
os.environ["MIDEA_AIGC_USER"] = os.getenv("MIDEA_AIGC_USER", "user")
os.environ["GEMINI_AIMP_BIZ_ID"] = "gemini-2.5-flash"
os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"
os.environ["GEMINI_URL_SYNC"] = "https://aimpapi.midea.com/t-aigc/mip-chat-app/gemini/official/standard/sync/v1/chat/completions"

MIDEA_API_KEY = os.getenv("MIDEA_API_KEY", "")
MIDEA_AIGC_USER = os.getenv("MIDEA_AIGC_USER", "user")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GEMINI_URL_SYNC = os.environ["GEMINI_URL_SYNC"]
GEMINI_AIMP_BIZ_ID = os.environ["GEMINI_AIMP_BIZ_ID"]
GEMINI_MODEL = os.environ["GEMINI_MODEL"]

# --- 安全 JSON dump ---
def safe_dump(obj: Any) -> str:
    """安全地将对象序列化为 JSON 字符串，处理无法序列化的类型。"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)

def gemini_chat_once(
    user_text: str,
    system_instruction: str,
    temperature: float = 0.3,
    max_tokens: int = 4096
) -> Tuple[str, Dict[str, Any]]:
    """
    同步调用美的 Gemini API。
    """
    if not MIDEA_API_KEY:
        print("[Gemini 配置错误] 未设置 MIDEA_API_KEY，无法调用美的 Gemini。")
        return "错误：未设置 MIDEA_API_KEY，无法调用美的 Gemini。", {}
    headers = {
        "Authorization": f"Bearer {MIDEA_API_KEY}",
        "Aimp-Biz-Id": GEMINI_AIMP_BIZ_ID,
        "AIGC-USER": MIDEA_AIGC_USER,
        "Content-Type": "application/json",
    }
    body = {
        "model": GEMINI_MODEL,
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    try:
        resp = requests.post(
            GEMINI_URL_SYNC,
            headers=headers,
            data=json.dumps(body),
            timeout=180,
            proxies={"http": None, "https": None}
        )
        if not (200 <= resp.status_code < 300):
            req_id = resp.headers.get("X-Request-Id") or resp.headers.get("x-request-id") or ""
            snippet = ""
            try:
                snippet = resp.text[:2000]
            except Exception:
                snippet = "<响应体读取失败>"
            print(f"[Gemini HTTP错误] status={resp.status_code}, request-id={req_id}, url={GEMINI_URL_SYNC}")
            return f"错误：Gemini HTTP {resp.status_code}, request-id={req_id}, body={resp.text}", {}
        data = resp.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not text:
            print("[Gemini 响应解析警告] 响应中未找到文本字段。")
        return text, data.get("usageMetadata", {})
    except requests.RequestException as e:
        print(f"[Requests 异常] 类型={type(e).__name__}, 详情={e}")
        return f"错误：API调用时发生错误: {type(e).__name__}: {e}", {}
    except Exception as e:
        print(f"[非预期异常] 类型={type(e).__name__}, 详情={e}")
        return f"错误：非预期异常 {type(e).__name__}: {e}", {}

def _messages_to_gemini_io(messages: List[BaseMessage]) -> Tuple[str, str]:
    system_instruction_parts = []
    user_text_parts = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_instruction_parts.append(m.content)
        elif isinstance(m, HumanMessage):
            user_text_parts.append(m.content)
        elif isinstance(m, AIMessage):
            user_text_parts.append(f"Assistant: {m.content}")
        else:
            user_text_parts.append(str(getattr(m, "content", m)))
    return "\n".join(system_instruction_parts).strip(), "\n".join(user_text_parts).strip()

def _gemini_chat_sync(messages: List[BaseMessage]) -> AIMessage:
    sys_inst, user_text = _messages_to_gemini_io(messages)
    text, _usage = gemini_chat_once(user_text, sys_inst)
    try:
        if isinstance(text, str) and (text.startswith("错误：") or text.startswith("错误:")):
            print("[Gemini 返回错误文本] ", text)
    except Exception:
        pass
    return AIMessage(content=text)

async def _gemini_chat_async(messages: List[BaseMessage]) -> AIMessage:
    return await asyncio.to_thread(_gemini_chat_sync, messages)

gemini_chat_runnable = RunnableLambda(func=_gemini_chat_sync, afunc=_gemini_chat_async)

# 防御
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")

# 注入模型
import deep_research.utils as dr_utils
dr_utils.set_models(gemini_chat_runnable)

# 导入简单任务链路
try:
    from deep_research.single_agent_supervisor import single_agent
except ImportError:
    single_agent = None
    print("警告：无法导入 single_agent，简单任务链路不可用。")

def build_inputs_for_graph(question: str) -> Dict[str, Any]:
    return {
        "messages": [
            SystemMessage(content=""),
            HumanMessage(content=question)
        ],
        "raw_notes": [],
        "notes": [],
        "supervisor_messages": [],
    }


def _get_log_profile() -> str:
    """
    日志档位：
    - lite: 仅保留核心日志（main_run/db_chain/final_report）
    - full: 额外保留 node_outputs、db_debug、sanitizer 明细
    """
    profile = str(os.getenv("LOG_PROFILE", "lite") or "lite").strip().lower()
    return profile if profile in {"lite", "full"} else "lite"


def _apply_runtime_logging_defaults(profile: str) -> None:
    # 仅在用户未显式设置时注入默认值，避免覆盖手工调试配置
    if profile == "lite":
        os.environ.setdefault("DB_CONSOLE_VERBOSE", "0")
        os.environ.setdefault("SQL_PARSER_VERBOSE", "0")
    else:
        os.environ.setdefault("DB_CONSOLE_VERBOSE", "1")
        os.environ.setdefault("SQL_PARSER_VERBOSE", "1")

def safe_get_msg_content(msg: Any) -> str:
    try:
        if isinstance(msg, (AIMessage, HumanMessage, SystemMessage)):
            return msg.content
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"])
        return str(msg)
    except Exception:
        return str(msg)


def find_recursive(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        if key in data and data[key]:
            return data[key]
        for v in data.values():
            res = find_recursive(v, key)
            if res:
                return res
    return None


def _truncate_text(text: Any, max_len: int = 160) -> str:
    s = str(text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _append_text_file(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        print(f"[DB DEBUG WRITE ERROR] {e}")


def print_db_intermediate(output_data: Any, prefix: str = "DB", debug_log_path: Optional[Path] = None) -> None:
    if not isinstance(output_data, dict):
        return
    db_route = find_recursive(output_data, "db_route")
    db_plans = find_recursive(output_data, "db_query_plans")
    db_sqls = find_recursive(output_data, "db_executed_sqls")
    db_warnings = find_recursive(output_data, "db_plan_sanitizer_warnings")
    db_params = find_recursive(output_data, "db_query_params")
    raw_rows = find_recursive(output_data, "raw_db_results")

    has_db_signal = bool(db_route or db_plans or db_sqls or db_warnings or raw_rows)
    if not has_db_signal:
        return

    lines: List[str] = []
    lines.append(f"[{prefix}] ===== DATABASE 中间信息 =====")
    if isinstance(db_route, str):
        lines.append(f"[{prefix}] route: {db_route}")
    if isinstance(db_params, dict) and db_params:
        lines.append(f"[{prefix}] params: {safe_dump(db_params)}")
    if isinstance(db_plans, list) and db_plans:
        plan_tables = [
            str(p.get("table", "")) for p in db_plans
            if isinstance(p, dict) and p.get("table")
        ]
        if plan_tables:
            lines.append(f"[{prefix}] plans: {len(db_plans)} | tables={sorted(set(plan_tables))}")
        lines.append(f"[{prefix}] plans_detail: {safe_dump(db_plans)}")
    if isinstance(db_sqls, list) and db_sqls:
        for i, sql in enumerate(db_sqls[:3], 1):
            lines.append(f"[{prefix}] sql[{i}]: {_truncate_text(sql, 220)}")
        if len(db_sqls) > 3:
            lines.append(f"[{prefix}] sql: ...(+{len(db_sqls)-3} more)")
    if isinstance(db_warnings, list) and db_warnings:
        lines.append(f"[{prefix}] sanitizer_warnings: {len(db_warnings)}")
        for w in db_warnings[:2]:
            if isinstance(w, dict):
                idx = w.get("plan_index", "?")
                warn_items = w.get("warnings", [])
                if isinstance(warn_items, list) and warn_items:
                    lines.append(f"[{prefix}]   - plan#{idx}: {_truncate_text('; '.join(map(str, warn_items)), 220)}")
    if isinstance(raw_rows, list):
        lines.append(f"[{prefix}] raw_db_results: {len(raw_rows)}")
    lines.append(f"[{prefix}] =============================")

    text_block = "\n" + "\n".join(lines) + "\n"
    print(text_block)
    if debug_log_path:
        _append_text_file(debug_log_path, f"[{datetime.datetime.now().isoformat(timespec='seconds')}]")
        _append_text_file(debug_log_path, "\n".join(lines))

async def run_full_agent(question: str = None):
    """
    运行完整 Agent 并记录时间，最后统一打印
    
    Args:
        question: 用户问题，如果为 None 则使用默认问题
    """
    if not TAVILY_API_KEY:
        print("警告：未设置 TAVILY_API_KEY，使用外部搜索工具时可能出现连接错误。")

    # --- 日志记录设置 ---
    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / f"thinkdepth_run_{current_time_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_profile = _get_log_profile()
    _apply_runtime_logging_defaults(log_profile)
    
    main_log_path = run_dir / f"main_run.log"
    print(f"日志将保存至: {run_dir}")
    print(f"日志档位: {log_profile}")
    
    with open(main_log_path, "w", encoding="utf-8") as f:
        f.write(f"--- Log started at {current_time_str} ---\n\n")
        f.write(f"Log profile: {log_profile}\n")

    nodes_log_dir = run_dir / "nodes"
    node_outputs_dir = run_dir / "node_outputs"
    db_debug_log_path = run_dir / "db_debug.log"
    db_chain_log_path = run_dir / "db_chain.jsonl"
    db_plan_sanitizer_log_path = run_dir / "db_plan_sanitizer.jsonl"
    if log_profile == "full":
        nodes_log_dir.mkdir(parents=True, exist_ok=True)
        node_outputs_dir.mkdir(parents=True, exist_ok=True)

    def write_node_event_log(node_name: Optional[str], kind: Optional[str], data: Optional[Dict[str, Any]]):
        if log_profile != "full":
            return
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node_name or "unnamed"))
        node_log_path = nodes_log_dir / f"node_{safe_name}.log"
        event_record = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "name": node_name,
            "data": data if isinstance(data, dict) else (data or {}),
        }
        try:
            with open(node_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record, ensure_ascii=False, indent=2, default=str) + "\n\n")
        except Exception as e:
            print(f">>> [节点日志写入失败] name={node_name}, error={e}")

    def save_node_output(node_name: Optional[str], output_data: Any):
        if log_profile != "full":
            return
        if output_data is None:
            return
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node_name or "unnamed_output"))
        output_path = node_outputs_dir / f"{safe_name}_output.txt"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(safe_dump(output_data))
        except Exception as e:
            print(f">>> [保存节点输出失败] name={node_name}, error={e}")

    try:
        from deep_research.research_agent_full import deep_researcher_builder
        from langgraph.checkpoint.memory import InMemorySaver
        import inspect
    except Exception as e:
        print("无法导入 deep_researcher_builder 或 langgraph，跳过。错误：", e)
        return

    checkpointer = InMemorySaver()

    try:
        if callable(deep_researcher_builder):
            try:
                sig = inspect.signature(deep_researcher_builder)
                if "llm" in sig.parameters:
                    graph = deep_researcher_builder(llm=gemini_chat_runnable)
                else:
                    graph = deep_researcher_builder()
            except Exception:
                graph = deep_researcher_builder()
        else:
            graph = deep_researcher_builder
    except Exception as e:
        print("处理 deep_researcher_builder 失败：", e)
        return

    try:
        full_agent = graph.compile(checkpointer=checkpointer)
    except TypeError:
        full_agent = graph.compile()
    except Exception as e:
        print("编译 StateGraph 失败：", e)
        return

    # --- 保持用户要求的配置不做任何修改 ---
    question = (
            # "在分析电压数据时，有些电芯并不是全程充电电压偏高(偏低)，放电电压偏低(偏高)，例如：在磷酸铁锂-石墨电池包里，电池串联，电芯2充电<80%SOC电压最低，>80%SOC正常；放电电压全程最高，电芯2的故障是什么？"
            # "在磷酸铁锂-石墨电池包中，电池串联，某电芯充电过程全程最低，电压放电过程由最高变化到最低，放电>20%SOC最高，<20%SOC电压最低，放电后的静置过程中电压也是最低，但充电后静置过程中电压排名会升高。此外，该电池的阻抗在整个电池包中最小，其阻抗值波动性最大。该电芯可能发生了什么故障？"
            # "在磷酸铁锂-石墨电池包中，电池串联，某电芯放电全程偏低，充电<80%SOC偏高，>80%SOC正常。且该电芯的内阻偏大。该电池是什么故障？ "
            # "在磷酸铁锂-石墨电池包中，电池串联，某电芯充电<80%SOC正常，>80%SOC偏高，放电全程正常。且该电池阻抗中等。该电芯是什么故障？"
            # "电压异常故障有哪几种潜在原因？"
            # "介质强度实验按照哪些步骤进行？"
            # "耐湿热实验有哪些步骤?"
            # "电站运维实际碰到的情况都有哪些"
            # "你好"
            # "请帮我画一个柱状图，展示2023年四个季度的营收数据。第一季度营收150万元，第二季度营收230万元，第三季度营收180万元，第四季度营收300万元。"
        )

#     question = """请帮我画一个折线图，展示电站7月6日的光伏数据。已知其数据如下所示：
# local_time	time	pv
# 2024/7/6 8:00	2024-07-06 00:00:00+00:00	13953.33
# 2024/7/6 8:15	2024-07-06 00:15:00+00:00	12770
# 2024/7/6 8:30	2024-07-06 00:30:00+00:00	7270
# 2024/7/6 8:45	2024-07-06 00:45:00+00:00	10130
# 2024/7/6 9:00	2024-07-06 01:00:00+00:00	17303.33
# 2024/7/6 9:15	2024-07-06 01:15:00+00:00	18480
# 2024/7/6 9:30	2024-07-06 01:30:00+00:00	15716.67
# 2024/7/6 9:45	2024-07-06 01:45:00+00:00	18676.67
# 2024/7/6 10:00	2024-07-06 02:00:00+00:00	21330
# 2024/7/6 10:15	2024-07-06 02:15:00+00:00	21786.67
# 2024/7/6 10:30	2024-07-06 02:30:00+00:00	22040
# 2024/7/6 10:45	2024-07-06 02:45:00+00:00	22000
# 2024/7/6 11:00	2024-07-06 03:00:00+00:00	22526.67
# 2024/7/6 11:15	2024-07-06 03:15:00+00:00	24283.33
# 2024/7/6 11:30	2024-07-06 03:30:00+00:00	24936.67
# 2024/7/6 11:45	2024-07-06 03:45:00+00:00	25260
# 2024/7/6 12:00	2024-07-06 04:00:00+00:00	24310
# 2024/7/6 12:15	2024-07-06 04:15:00+00:00	17923.33
# 2024/7/6 12:30	2024-07-06 04:30:00+00:00	24606.67
# 2024/7/6 12:45	2024-07-06 04:45:00+00:00	24600
# 2024/7/6 13:00	2024-07-06 05:00:00+00:00	20603.33
# 2024/7/6 13:15	2024-07-06 05:15:00+00:00	17796.67
# 2024/7/6 13:30	2024-07-06 05:30:00+00:00	22700
# 2024/7/6 13:45	2024-07-06 05:45:00+00:00	23336.67
# 2024/7/6 14:00	2024-07-06 06:00:00+00:00	22983.33
# 2024/7/6 14:15	2024-07-06 06:15:00+00:00	20996.67
# 2024/7/6 14:30	2024-07-06 06:30:00+00:00	19970
# 2024/7/6 14:45	2024-07-06 06:45:00+00:00	19420
# 2024/7/6 15:00	2024-07-06 07:00:00+00:00	19143.33
# 2024/7/6 15:15	2024-07-06 07:15:00+00:00	18773.33
# 2024/7/6 15:30	2024-07-06 07:30:00+00:00	18583.33
# 2024/7/6 15:45	2024-07-06 07:45:00+00:00	17260
# 2024/7/6 16:00	2024-07-06 08:00:00+00:00	15613.33
# 2024/7/6 16:15	2024-07-06 08:15:00+00:00	13316.67
# 2024/7/6 16:30	2024-07-06 08:30:00+00:00	12150
# 2024/7/6 16:45	2024-07-06 08:45:00+00:00	10206.67
# 2024/7/6 17:00	2024-07-06 09:00:00+00:00	9916.67
# 2024/7/6 17:15	2024-07-06 09:15:00+00:00	9026.67
# 2024/7/6 17:30	2024-07-06 09:30:00+00:00	7646.67
# 2024/7/6 17:45	2024-07-06 09:45:00+00:00	5983.33
# 2024/7/6 18:00	2024-07-06 10:00:00+00:00	4876.67
# 2024/7/6 18:15	2024-07-06 10:15:00+00:00	3663.33
# 2024/7/6 18:30	2024-07-06 10:30:00+00:00	2516.67
# 2024/7/6 18:45	2024-07-06 10:45:00+00:00	1550
# 2024/7/6 19:00	2024-07-06 11:00:00+00:00	806.67
# 2024/7/6 19:15	2024-07-06 11:15:00+00:00	246.67
# 2024/7/6 19:30	2024-07-06 11:30:00+00:00	43.33
# 2024/7/6 19:45	2024-07-06 11:45:00+00:00	20
# 2024/7/6 20:00	2024-07-06 12:00:00+00:00	20
# 2024/7/7 5:15	2024-07-06 21:15:00+00:00	20
# 2024/7/7 5:30	2024-07-06 21:30:00+00:00	160
# 2024/7/7 5:45	2024-07-06 21:45:00+00:00	566.67
# 2024/7/7 6:00	2024-07-06 22:00:00+00:00	1233.33
# 2024/7/7 6:15	2024-07-06 22:15:00+00:00	2210
# 2024/7/7 6:30	2024-07-06 22:30:00+00:00	2633.33
# 2024/7/7 6:45	2024-07-06 22:45:00+00:00	2873.33
# 2024/7/7 7:00	2024-07-06 23:00:00+00:00	4960
# 2024/7/7 7:15	2024-07-06 23:15:00+00:00	8386.67
# 2024/7/7 7:30	2024-07-06 23:30:00+00:00	7916.67
# 2024/7/7 7:45	2024-07-06 23:45:00+00:00	8423.33"""

    # 使用传入的问题，如果没有则使用默认问题
    if not question:
        question = "帮我分析station-00256 下 169号bmu 的 pack-7 电池的运行状态"

    inputs = build_inputs_for_graph(question)
    inputs["db_chain_log_path"] = str(db_chain_log_path)
    inputs["db_plan_sanitizer_log_path"] = str(db_plan_sanitizer_log_path)
    inputs["db_sanitizer_log_enabled"] = bool(log_profile == "full")
    inputs["run_id"] = run_dir.name
    print(f"DB链路日志: {db_chain_log_path}")
    print(f"run_id: {run_dir.name}")
    if log_profile == "full":
        print(f"DB清洗日志: {db_plan_sanitizer_log_path}")
    config = {"configurable": {"thread_id": "1"}}
    final_state: Optional[Dict[str, Any]] = None
    
    # === 时间统计变量 ===
    node_start_times = {} # 临时存开始时间
    step_execution_records = [] # 存最终结果 (step_name, duration)

    try:
        async for event in full_agent.astream_events(inputs, config=config, version="v1"):
            kind = event.get("event")
            name = event.get("name")
            run_id = event.get("run_id")

            # --- 收集时间逻辑 ---
            if kind == "on_chain_start":
                node_start_times[run_id] = time.perf_counter()
                if str(name) == "LangGraph":
                     print(">>> LangGraph on_chain_start data.input =", safe_dump(event.get("data", {}).get("input")))
            
            elif kind == "on_chain_end":
                if run_id in node_start_times:
                    start_t = node_start_times.pop(run_id)
                    duration = time.perf_counter() - start_t
                    
                    # 过滤掉杂讯，只记录主要节点的耗时，供最后打印
                    main_nodes = [
                        "clarify_with_user", 
                        "pre_brief_retrieval", 
                        "write_research_brief", 
                        "solve_simple_task", 
                        "write_draft_report", 
                        "supervisor_subgraph", 
                        "final_report_generation",
                        "supervisor"
                    ]
                    if name in main_nodes:
                        step_execution_records.append({"name": name, "duration": duration})
                        # 记录到文件日志
                        write_node_event_log(name, "node_execution_time", {"duration_s": duration})

                data = event.get("data", {})
                if name in ("LangGraph", "final_report_generation", "supervisor_subgraph"):
                    keys = list(data.keys()) if isinstance(data, dict) else []
                    print(f">>> [{name}] end - state keys: {keys}")
                
                if isinstance(data, dict):
                    if name == "LangGraph":
                        final_state = data.get("output", data)
                    elif name in ("agent", "final_report_generation"):
                        if data.get("output") or data.get("final_output"):
                            final_state = data

                if name in ("retrieve_battery_node", "execute_tools_node", "solve_simple_task"):
                    out_for_debug = data.get("output") if isinstance(data, dict) else None
                    if isinstance(out_for_debug, dict) and log_profile == "full":
                        print_db_intermediate(out_for_debug, prefix=f"DB/{name}", debug_log_path=db_debug_log_path)
                
                if str(name) != "LangGraph":
                    output_data = event.get("data", {}).get("output")
                    save_node_output(name, output_data)

            # --- 记录所有事件到日志 ---
            write_node_event_log(name, kind, event.get("data"))

            if kind not in ["on_chain_start", "on_chain_end"]:
                 print(f"[EVENT] kind={kind}, name={name}")
            
            if kind in ("on_llm_error", "on_chain_error", "on_tool_error"):
                print(f">>> [{kind.upper()}] name={name}, error={safe_dump(event.get('data', {}).get('error'))}")
            
    except Exception as e:
        import traceback
        print("运行完整 Agent 失败：", repr(e))
        traceback.print_exc()

    finally:
        print("\n" + "=" * 50)
        print("📊  各步骤耗时统计 (Execution Breakdown)")
        print("-" * 50)
        
        # 统一打印耗时
        if step_execution_records:
            total_step_time = 0.0
            for record in step_execution_records:
                n = record["name"]
                d = record["duration"]
                total_step_time += d
                print(f"🔹 步骤: {n:<25} 耗时: {d:8.4f} s")
            print("-" * 50)
            print(f"∑  主要步骤总计耗时: {total_step_time:8.4f} s")
        else:
            print("未捕获到主要步骤的耗时数据。")
        print("=" * 50 + "\n")

        print("==== 最终输出 ====")
        try:
            if isinstance(final_state, dict):
                # print(final_state) # 调试用，可注释
                
                report_txt = None

                # 1. 优先检查 "solve_simple_task" 节点 (简单路径)
                # 你的日志显示数据在这个key下
                if "solve_simple_task" in final_state and isinstance(final_state["solve_simple_task"], dict):
                    node_data = final_state["solve_simple_task"]
                    # 尝试获取 final_report 或 draft_report
                    report_txt = node_data.get("final_report") or node_data.get("draft_report")

                # 2. 检查 "final_report_generation" 节点 (复杂路径)
                elif "final_report_generation" in final_state and isinstance(final_state["final_report_generation"], dict):
                    report_txt = final_state["final_report_generation"].get("final_report")

                # 3. 检查顶层状态 (防止状态被扁平化的情况)
                elif "final_report" in final_state and final_state["final_report"]:
                    report_txt = final_state["final_report"]
                elif "draft_report" in final_state and final_state["draft_report"]:
                    report_txt = final_state["draft_report"]

                # 4. 兜底：从 messages 获取
                elif "messages" in final_state and final_state["messages"]:
                    report_txt = safe_get_msg_content(final_state["messages"][-1])
                
                # 5. 实在没有，才转存整个 JSON
                else:
                    report_txt = safe_dump(final_state)
                
                # 打印纯文本内容
                if report_txt:
                    print(report_txt)

                # 保存到文件
                if report_txt and isinstance(report_txt, str):
                    out_path = run_dir / "final_report.txt"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(report_txt)
                    print(f"\n>>> final report saved to: {out_path}")
            else:
                print(final_state)
        except Exception as e:
            print("打印或保存最终输出时出错：", e)


async def run_simple_agent(question: str):
    """
    运行简单任务 Agent（single_agent_supervisor 链路）。
    适用于快速响应场景（数据库查询、本地知识库检索等）。
    """
    try:
        from deep_research.single_agent_supervisor import single_agent
    except Exception as e:
        print("无法导入 single_agent，跳过。错误：", e)
        return

    # 日志目录准备
    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / f"thinkdepth_run_{current_time_str}_simple"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_profile = _get_log_profile()
    _apply_runtime_logging_defaults(log_profile)
    
    main_log_path = run_dir / "main_run.log"
    print(f"日志将保存至: {run_dir}")
    print(f"日志档位: {log_profile}")
    
    with open(main_log_path, "w", encoding="utf-8") as f:
        f.write(f"--- Log started at {current_time_str} ---\n")
        f.write(f"Mode: SIMPLE (Fast Response)\n")
        f.write(f"Log profile: {log_profile}\n")
        f.write(f"Question: {question}\n\n")

    node_outputs_dir = run_dir / "node_outputs"
    db_debug_log_path = run_dir / "db_debug.log"
    db_chain_log_path = run_dir / "db_chain.jsonl"
    db_plan_sanitizer_log_path = run_dir / "db_plan_sanitizer.jsonl"
    if log_profile == "full":
        node_outputs_dir.mkdir(parents=True, exist_ok=True)

    def save_node_output(node_name: Optional[str], output_data: Any):
        if log_profile != "full":
            return
        if output_data is None:
            return
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node_name or "unnamed_output"))
        output_path = node_outputs_dir / f"{safe_name}_output.txt"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(safe_dump(output_data))
        except Exception as e:
            print(f">>> [保存节点输出失败] name={node_name}, error={e}")

    # 构建输入状态
    inputs = {
        "messages": [HumanMessage(content=question)],
        "user_request": question,
        "raw_notes": [],
        "notes": [],
        "supervisor_messages": [],
        "db_chain_log_path": str(db_chain_log_path),
        "db_plan_sanitizer_log_path": str(db_plan_sanitizer_log_path),
        "db_sanitizer_log_enabled": bool(log_profile == "full"),
        "run_id": run_dir.name,
    }
    print(f"DB链路日志: {db_chain_log_path}")
    print(f"run_id: {run_dir.name}")
    if log_profile == "full":
        print(f"DB清洗日志: {db_plan_sanitizer_log_path}")

    start_time = time.perf_counter()
    final_state = None

    try:
        print("\n⚡ 运行简单任务链路...")
        final_state = await single_agent.ainvoke(inputs)
        duration = time.perf_counter() - start_time

        if isinstance(final_state, dict) and log_profile == "full":
            print_db_intermediate(final_state, prefix="DB/simple_final", debug_log_path=db_debug_log_path)

        # 保存节点输出
        if final_state:
            save_node_output("final_state", final_state)

        # 提取最终报告
        report_txt = None
        if isinstance(final_state, dict):
            if "final_report" in final_state and final_state["final_report"]:
                report_txt = final_state["final_report"]
            elif "solve_simple_task" in final_state and isinstance(final_state["solve_simple_task"], dict):
                report_txt = final_state["solve_simple_task"].get("final_report", "")
            elif "messages" in final_state and final_state["messages"]:
                report_txt = safe_get_msg_content(final_state["messages"][-1])
            else:
                report_txt = safe_dump(final_state)
        else:
            report_txt = safe_dump(final_state)

        # 保存报告
        if report_txt and isinstance(report_txt, str):
            out_path = run_dir / "final_report.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report_txt)
            print(f"\n>>> final report saved to: {out_path}")
            print("\n" + "=" * 40)
            print("📄 最终报告:")
            print("=" * 40)
            print(report_txt)

        print(f"\n⏱️  简单任务耗时: {duration:.2f} 秒")
        return duration

    except Exception as e:
        import traceback
        print(f"\n❌ 简单任务运行出错: {e}")
        traceback.print_exc()
        with open(main_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[ERROR] {repr(e)}\n")
        raise


async def interactive_mode():
    """
    交互模式：询问用户选择运行模式，然后执行。
    """
    print("\n" + "=" * 50)
    print("🧠 Deep Research Agent - 交互模式")
    print("=" * 50)
    
    # 获取用户问题
    question = input("\n请输入您的问题: ").strip()
    if not question:
        print("问题不能为空，退出。")
        return
    
    # 询问模式
    print("\n请选择运行模式:")
    print("  1. 快速模式 (Simple) - 直接查询，快速响应 (~1-5秒)")
    print("  2. 深度研究模式 (Deep Research) - 完整链路，深度分析 (~10-60秒)")
    
    choice = input("\n请输入选项 (1/2，默认 1): ").strip()
    
    start_time = time.perf_counter()
    
    if choice == "2":
        print("\n🚀 启动深度研究模式...")
        await run_full_agent()
    else:
        print("\n⚡ 启动快速模式...")
        await run_simple_agent(question)
    
    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print("\n" + "=" * 50)
    print("⏱️  全流程总耗时统计")
    print("-" * 20)
    if minutes > 0:
        print(f"总耗时: {minutes} 分 {seconds:.2f} 秒")
    else:
        print(f"总耗时: {seconds:.2f} 秒")
    print("=" * 50)


if __name__ == "__main__":
    if not MIDEA_API_KEY:
        print("错误：请先设置环境变量 MIDEA_API_KEY 才能调用美的 Gemini。")
    else:
        try:
            asyncio.run(interactive_mode())
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户手动中止了程序。")
        except Exception as e:
            print(f"\n\n❌ 程序运行出错: {e}")
