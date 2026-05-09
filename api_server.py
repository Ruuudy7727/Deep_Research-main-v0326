#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对外暴露 Deep_Research 项目的输入、输出、中间过程的 API 服务。

文件位置（请确保确实在项目根目录）：
    <PROJECT_ROOT>/api_server.py

启动方式（务必在项目根目录下执行）：
    cd <PROJECT_ROOT>
    uvicorn api_server:app --host 0.0.0.0 --port 8000

依赖：
    pip install fastapi uvicorn python-dotenv
"""

import os
import sys
import asyncio
import datetime
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# ===================== 关键：路径设置，保证和你平时运行方式一致 =====================

# 当前文件所在目录 = 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 确保项目根在 sys.path[0]，与在根目录直接 python / uvicorn 一致
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 日志根目录
LOG_ROOT = PROJECT_ROOT / "log"

# 载入 .env
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"), override=False)
PUBLIC_API_TOKEN = os.getenv("PUBLIC_API_TOKEN", "changeme")  # 建议在 .env 里改掉
API_VERBOSE_EVENT_LOG = os.getenv("API_VERBOSE_EVENT_LOG", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

app = FastAPI(title="Deep Research API Server", version="0.1.0")

# ===================== 导入项目内部模块（不碰 research_agent_draw） =====================

try:
    # 从 thinkdepth_test 导入工具函数（你原来就有的）
    from thinkdepth_test import (
        build_inputs_for_graph,
        gemini_chat_runnable,
        safe_dump,
        safe_get_msg_content,
    )

    # 关键：只从 research_agent_full 导入 deep_researcher_builder
    # 不在这里 import research_agent_draw，也不做任何 draw 相关逻辑
    from deep_research.research_agent_full import deep_researcher_builder
    # 导入简单任务链路
    from deep_research.single_agent_supervisor import single_agent

    from langgraph.checkpoint.memory import InMemorySaver
    import inspect

except Exception as e:
    # 如果这里失败，说明你在命令行直接 import deep_research.research_agent_full 也会失败
    # 先打印出来方便你看错误
    print("导入 deep_research 或 thinkdepth_test 失败：", repr(e))
    deep_researcher_builder = None
    single_agent = None


# ===================== 通用工具 =====================

def verify_token(token: str):
    """简单鉴权：如果 token 不匹配，则拒绝访问。"""
    if PUBLIC_API_TOKEN and token != PUBLIC_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


def safe_listdir(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return [p for p in path.iterdir() if p.is_dir()]


def read_text_file(path: Path) -> str:
    try:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def read_json_lines(path: Path) -> List[Dict[str, Any]]:
    """读取节点日志（每个事件一段 JSON），返回 list。"""
    events = []
    if not path.exists():
        return events
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return events
    parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
    for part in parts:
        try:
            events.append(json.loads(part))
        except Exception:
            continue
    return events


# ===================== 核心：运行 Agent（简单模式） =====================

async def run_simple_agent_once(question: str, run_dir: Path) -> Dict[str, Any]:
    """
    运行简单任务 Agent（single_agent_supervisor 链路）。
    适用于快速响应场景（数据库查询、本地知识库检索等）。
    """
    if single_agent is None:
        raise RuntimeError("single_agent 未成功导入，无法运行简单任务链路。")

    import time
    import datetime as dt

    # 日志目录准备
    run_dir.mkdir(parents=True, exist_ok=True)
    current_time_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    main_log_path = run_dir / "main_run.log"
    with open(main_log_path, "w", encoding="utf-8") as f:
        f.write(f"--- Log started at {current_time_str} ---\n")
        f.write(f"Question: {question}\n")
        f.write(f"Mode: SIMPLE (Fast Response)\n\n")

    node_outputs_dir = run_dir / "node_outputs"
    node_outputs_dir.mkdir(parents=True, exist_ok=True)

    def save_node_output(node_name: Optional[str], output_data: Any):
        if output_data is None:
            return
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node_name or "unnamed_output"))
        output_path = node_outputs_dir / f"{safe_name}_output.txt"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(safe_dump(output_data))
        except Exception as e:
            print(f"[保存节点输出失败] name={node_name}, error={e}")

    # 构建输入状态
    from langchain_core.messages import HumanMessage
    inputs = {
        "messages": [HumanMessage(content=question)],
        "user_request": question,
        "raw_notes": [],
        "notes": [],
        "supervisor_messages": [],
    }

    start_time = time.perf_counter()
    final_state = None

    try:
        # 运行简单任务链路
        final_state = await single_agent.ainvoke(inputs)
        duration = time.perf_counter() - start_time

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
        report_path = run_dir / "final_report.txt"
        if report_txt and isinstance(report_txt, str):
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_txt)

        return {
            "log_dir": str(run_dir),
            "final_report_path": str(report_path) if report_txt else None,
            "duration_s": duration,
            "mode": "simple",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        with open(main_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[ERROR] {repr(e)}\n")
        raise


# ===================== 核心：运行 Agent（深度模式） =====================

async def run_full_agent_once(question: str, run_dir: Path) -> Dict[str, Any]:
    """
    运行完整 Agent（Deep Research 模式）。
    使用 deep_research.research_agent_full.deep_researcher_builder 构建图，
    包含 Supervisor 判断、多轮搜索、复杂报告生成等完整链路。
    """

    if deep_researcher_builder is None:
        # 这里说明 import 时就失败了，通常和 research_agent_full 的依赖有关
        raise RuntimeError(
            "deep_researcher_builder 未成功导入。"
            "请在 Python 交互环境中执行：\n"
            "  cd <PROJECT_ROOT>\n"
            "  python -c \"from deep_research.research_agent_full import deep_researcher_builder\"\n"
            "确认能否正常导入。如果这里都失败，就与 api_server 无关。"
        )

    import time
    import datetime as dt
    from typing import Optional as _Optional

    # 日志目录准备
    run_dir.mkdir(parents=True, exist_ok=True)
    current_time_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    main_log_path = run_dir / "main_run.log"
    with open(main_log_path, "w", encoding="utf-8") as f:
        f.write(f"--- Log started at {current_time_str} ---\n")
        f.write(f"Question: {question}\n\n")

    nodes_log_dir = run_dir / "nodes"
    node_outputs_dir = run_dir / "node_outputs"
    nodes_log_dir.mkdir(parents=True, exist_ok=True)
    node_outputs_dir.mkdir(parents=True, exist_ok=True)

    def write_node_event_log(node_name: _Optional[str], kind: _Optional[str], data: _Optional[Dict[str, Any]]):
        # 默认仅保留关键事件，避免高频全量写盘拖慢主链路。
        if not API_VERBOSE_EVENT_LOG:
            key_kinds = {"node_execution_time", "on_chain_end", "on_chain_error", "on_tool_error", "on_llm_error"}
            if kind not in key_kinds:
                return

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node_name or "unnamed"))
        node_log_path = nodes_log_dir / f"node_{safe_name}.log"
        event_record = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "name": node_name,
            "data": data if isinstance(data, dict) else (data or {}),
        }
        try:
            with open(node_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record, ensure_ascii=False, indent=2, default=str) + "\n\n")
        except Exception as e:
            print(f"[节点日志写入失败] name={node_name}, error={e}")

    def save_node_output(node_name: _Optional[str], output_data: Any):
        if output_data is None:
            return
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node_name or "unnamed_output"))
        output_path = node_outputs_dir / f"{safe_name}_output.txt"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(safe_dump(output_data))
        except Exception as e:
            print(f"[保存节点输出失败] name={node_name}, error={e}")

    # 构建 StateGraph
    checkpointer = InMemorySaver()

    # deep_researcher_builder 可能是函数，也可能是已构建的图
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

    # compile
    try:
        full_agent = graph.compile(checkpointer=checkpointer)
    except TypeError:
        full_agent = graph.compile()

    # 构建输入
    inputs = build_inputs_for_graph(question)
    config = {"configurable": {"thread_id": "api_server"}}

    final_state: _Optional[Dict[str, Any]] = None
    step_execution_records = []
    node_start_times: Dict[str, float] = {}

    try:
        async for event in full_agent.astream_events(inputs, config=config, version="v1"):
            kind = event.get("event")
            name = event.get("name")
            run_id = event.get("run_id")

            if kind == "on_chain_start":
                node_start_times[run_id] = time.perf_counter()

            elif kind == "on_chain_end":
                if run_id in node_start_times:
                    start_t = node_start_times.pop(run_id)
                    duration = time.perf_counter() - start_t

                    main_nodes = [
                        "clarify_with_user",
                        "pre_brief_retrieval",
                        "write_research_brief",
                        "solve_simple_task",
                        "write_draft_report",
                        "supervisor_subgraph",
                        "final_report_generation",
                        "supervisor",
                    ]
                    if name in main_nodes:
                        step_execution_records.append({"name": name, "duration": duration})
                        write_node_event_log(name, "node_execution_time", {"duration_s": duration})

                data = event.get("data", {})
                if isinstance(data, dict):
                    if name == "LangGraph":
                        final_state = data.get("output", data)
                    elif name in ("agent", "final_report_generation"):
                        if data.get("output") or data.get("final_output"):
                            final_state = data

                if str(name) != "LangGraph":
                    output_data = event.get("data", {}).get("output")
                    save_node_output(name, output_data)

            write_node_event_log(name, kind, event.get("data"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        with open(main_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[ERROR] {repr(e)}\n")

    # 提取最终报告
    report_txt = None
    if isinstance(final_state, dict):
        if "solve_simple_task" in final_state and isinstance(final_state["solve_simple_task"], dict):
            node_data = final_state["solve_simple_task"]
            report_txt = node_data.get("final_report") or node_data.get("draft_report")
        elif "final_report_generation" in final_state and isinstance(final_state["final_report_generation"], dict):
            report_txt = final_state["final_report_generation"].get("final_report")
        elif "final_report" in final_state and final_state["final_report"]:
            report_txt = final_state["final_report"]
        elif "draft_report" in final_state and final_state["draft_report"]:
            report_txt = final_state["draft_report"]
        elif "messages" in final_state and final_state["messages"]:
            report_txt = safe_get_msg_content(final_state["messages"][-1])
        else:
            report_txt = safe_dump(final_state)
    else:
        report_txt = safe_dump(final_state)

    report_path = run_dir / "final_report.txt"
    if report_txt and isinstance(report_txt, str):
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_txt)

    return {
        "log_dir": str(run_dir),
        "final_report_path": str(report_path),
        "step_execution_records": step_execution_records,
    }


# ===================== Pydantic 模型 =====================

class RunRequest(BaseModel):
    question: str
    token: str
    deep_mode: bool = False  # 新增：是否启用深度研究模式


class RunInfo(BaseModel):
    run_dir_name: str
    created_at: Optional[str] = None
    question: Optional[str] = None


# ===================== API 定义 =====================

@app.post("/api/v1/run")
async def api_run(req: RunRequest):
    """
    启动一次新的 deep_research 任务。
    
    根据 req.deep_mode 决定运行模式：
    - deep_mode=False (默认): 简单任务链路，快速响应
    - deep_mode=True: 深度研究链路，完整多轮研究
    """
    verify_token(req.token)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"thinkdepth_run_{timestamp}_api"
    run_dir = LOG_ROOT / run_dir_name

    async def _runner():
        if req.deep_mode:
            # 深度研究模式：走完整链路
            await run_full_agent_once(req.question, run_dir)
        else:
            # 简单任务模式：快速响应
            await run_simple_agent_once(req.question, run_dir)

    asyncio.create_task(_runner())

    run_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = run_dir / "main_run.log"
    with open(main_log_path, "a", encoding="utf-8") as f:
        f.write(f"Question: {req.question}\n")
        f.write(f"Mode: {'DEEP' if req.deep_mode else 'SIMPLE'}\n")

    return {
        "run_dir_name": run_dir_name,
        "status": "running",
        "log_dir": str(run_dir),
        "mode": "deep" if req.deep_mode else "simple",
    }


@app.get("/api/v1/runs", response_model=List[RunInfo])
async def list_runs(token: str = Query(..., description="API token")):
    """
    列出所有已有的运行记录（读取 log 目录下的 thinkdepth_run_*）。
    """
    verify_token(token)

    runs = []
    for d in safe_listdir(LOG_ROOT):
        name = d.name
        if not name.startswith("thinkdepth_run_"):
            continue
        main_log = d / "main_run.log"
        question = None
        created_at = None
        if main_log.exists():
            content = main_log.read_text(encoding="utf-8", errors="ignore")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            for line in lines:
                if line.startswith("--- Log started at"):
                    created_at = line.replace("--- Log started at", "").strip("- ").strip()
                if line.startswith("Question:"):
                    question = line.replace("Question:", "").strip()
        runs.append(RunInfo(run_dir_name=name, created_at=created_at, question=question))
    runs.sort(key=lambda x: x.run_dir_name, reverse=True)
    return runs


@app.get("/api/v1/runs/{run_dir_name}/final_report")
async def get_final_report(run_dir_name: str, token: str = Query(...)):
    """
    获取某次运行的最终报告文本。
    """
    verify_token(token)

    run_dir = LOG_ROOT / run_dir_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run_dir not found")

    report_path = run_dir / "final_report.txt"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="final_report.txt not found")

    content = read_text_file(report_path)
    return {
        "run_dir_name": run_dir_name,
        "final_report": content,
    }


@app.get("/api/v1/runs/{run_dir_name}/events")
async def get_all_events(run_dir_name: str, token: str = Query(...)):
    """
    获取某次运行的所有节点事件日志（合并 nodes 下的所有 node_*.log）。
    """
    verify_token(token)

    run_dir = LOG_ROOT / run_dir_name
    nodes_dir = run_dir / "nodes"
    if not nodes_dir.exists():
        raise HTTPException(status_code=404, detail="nodes log dir not found")

    all_events = []
    for log_file in nodes_dir.glob("node_*.log"):
        events = read_json_lines(log_file)
        for e in events:
            e["_source_file"] = log_file.name
        all_events.extend(events)

    def _get_ts(e):
        return e.get("ts", "")
    all_events.sort(key=_get_ts)

    return {
        "run_dir_name": run_dir_name,
        "events": all_events,
    }


@app.get("/api/v1/runs/{run_dir_name}/nodes/{node_name}")
async def get_node_output(run_dir_name: str, node_name: str, token: str = Query(...)):
    """
    获取某次运行中，某个节点的输出（node_outputs/{node_name}_output.txt）。
    """
    verify_token(token)

    run_dir = LOG_ROOT / run_dir_name
    node_outputs_dir = run_dir / "node_outputs"
    if not node_outputs_dir.exists():
        raise HTTPException(status_code=404, detail="node_outputs dir not found")

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", node_name)
    file_path = node_outputs_dir / f"{safe_name}_output.txt"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="node output not found")

    content = read_text_file(file_path)
    return {
        "run_dir_name": run_dir_name,
        "node_name": node_name,
        "output": content,
    }


@app.get("/")
async def root():
    return {
        "message": "Deep Research API Server",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
