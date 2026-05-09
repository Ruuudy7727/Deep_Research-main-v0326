import asyncio
import json
import re
import time
from typing import Any, Dict, List, Tuple
from typing_extensions import Literal, TypedDict
import os
import aiofiles
from datetime import datetime

# 外部依赖
from deep_research.prompts import lead_researcher_with_multiple_steps_diffusion_double_check_prompt, supervisor_tool_schema_prompt
from deep_research.research_agent import researcher_agent
from deep_research.state_multi_agent_supervisor import (
    SupervisorState,
    ConductResearch,
    ResearchComplete,
)
from deep_research.utils import get_today_str, think_tool, refine_draft_report
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
    filter_messages,
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from deep_research.gemini_chat import gemini_chat_once

# Jupyter 环境兼容
try:
    import nest_asyncio
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass
except ImportError:
    pass

from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 定义日志文件路径
TOOL_LOG_FILE_PATH = str(_PROJECT_ROOT / "log" / "use_tool.txt")

async def log_tool_decision(tool_calls: List[Dict], assistant_response: str):
    if not tool_calls:
        return
    try:
        async with aiofiles.open(TOOL_LOG_FILE_PATH, mode='a', encoding='utf-8') as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "assistant_response": assistant_response,
                "decided_tool_calls": tool_calls
            }
            await f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"[ASYNC LOGGING ERROR] Failed to write to {TOOL_LOG_FILE_PATH}: {e}")

def get_notes_from_tool_calls(messages: List[BaseMessage]) -> List[str]:
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

def _render_messages_to_user_text(messages: List[BaseMessage]) -> str:
    lines = []
    for m in messages:
        role = "system" if isinstance(m, SystemMessage) else (
            "tool" if isinstance(m, ToolMessage) else (
                "assistant" if isinstance(m, AIMessage) else "user"
            )
        )
        if isinstance(m, ToolMessage):
            name = getattr(m, "name", "")
            tid = getattr(m, "tool_call_id", "")
            lines.append(f"[{role} name={name} id={tid}] {m.content}")
        else:
            lines.append(f"[{role}] {m.content}")
    return "\n".join(lines)

def _build_system_instruction(base_system: str, max_concurrent: int, max_iterations: int) -> str:
    tool_schema_instruction = supervisor_tool_schema_prompt.format(
        max_concurrent=max_concurrent,
        max_iterations=max_iterations
    ).strip()
    return (base_system or "") + "\n\n" + tool_schema_instruction

def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"{.*}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}

def _parse_tool_calls_from_output_text(text: str) -> Tuple[List[Dict[str, Any]], str]:
    obj = _extract_json(text)
    if not obj:
        return [], text
    tool_calls = obj.get("tool_calls", [])
    assistant_response = obj.get("assistant_response", "")
    normalized_calls = []
    for i, call in enumerate(tool_calls):
        cid = call.get("id") or f"tc{i+1}"
        name = call.get("name") or ""
        args = call.get("args") or {}
        normalized_calls.append({"id": cid, "name": name, "args": args})
    return normalized_calls, assistant_response

class GeminiToolCallingModel:
    def __init__(self, temperature: float = 0.3, max_tokens: int = 4096,
                 max_concurrent: int = 3, max_iterations: int = 15) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.max_iterations = max_iterations

    async def ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        base_system = ""
        if messages and isinstance(messages[0], SystemMessage):
            base_system = messages[0].content
            user_msgs = messages[1:]
        else:
            user_msgs = messages

        system_instruction = _build_system_instruction(
            base_system=base_system,
            max_concurrent=self.max_concurrent,
            max_iterations=self.max_iterations,
        )
        user_text = _render_messages_to_user_text(user_msgs)

        output_text, _usage = await asyncio.to_thread(
            gemini_chat_once,
            user_text,
            system_instruction,
            self.temperature,
            self.max_tokens,
        )
        tool_calls, assistant_response = _parse_tool_calls_from_output_text(output_text)
        return AIMessage(content=assistant_response or output_text, tool_calls=tool_calls)

# ===== 配置 =====
supervisor_tools = [ConductResearch, ResearchComplete, think_tool, refine_draft_report]
max_researcher_iterations = 1
max_concurrent_researchers = 2

supervisor_model_with_tools = GeminiToolCallingModel(
    temperature=0.3,
    max_tokens=4096,
    max_concurrent=max_concurrent_researchers,
    max_iterations=max_researcher_iterations,
)

# ===== SUPERVISOR 节点 =====
async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """协调研究活动：决定下一步工具调用计划"""
    supervisor_messages = state.get("supervisor_messages", [])
    current_iter = state.get("research_iterations", 0)
    next_iter = current_iter + 1

    print("\n" + "=" * 100)
    print(f">>> [SUPERVISOR CALLED] PLAN PHASE — Iteration #{next_iter}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # [修改] 强制逻辑：如果已达最大轮次，直接构造总结指令，跳过 LLM
    # -------------------------------------------------------------------------
    if current_iter >= max_researcher_iterations:
        print(f"🛑 [MAX ITERATIONS REACHED] ({current_iter}/{max_researcher_iterations})")
        print("⚡ Forcing 'ResearchComplete' execution without LLM inference.")
        
        # 构造伪造的 AIMessage，指令是执行 ResearchComplete
        forced_response = AIMessage(
            content="Max iterations reached. Forcing final report refinement.",
            tool_calls=[{
                "id": f"force_final_{int(time.time())}",
                "name": "ResearchComplete",
                "args": {} # 参数为空，让工具节点自己去取 state
            }]
        )
        
        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [forced_response],
                # 轮次加 1，确保进入工具节点后能触发结束条件
                "research_iterations": next_iter, 
            },
        )
    # -------------------------------------------------------------------------

    # 正常 LLM 流程
    system_message = lead_researcher_with_multiple_steps_diffusion_double_check_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations,
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    t_start_plan = time.perf_counter()
    response = await supervisor_model_with_tools.ainvoke(messages)
    t_cost_plan = time.perf_counter() - t_start_plan
    print(f"⏱️ [Timing] Supervisor Plan Logic took {t_cost_plan:.4f}s")

    print(">>> [SUPERVISOR PLAN OUTPUT] assistant_response:")
    print(response.content)
    print(">>> [SUPERVISOR PLAN OUTPUT] tool_calls:")
    tc_list = getattr(response, "tool_calls", []) or []
    if not tc_list:
        print("    (none)")
    else:
        for tc in tc_list:
            print(f"    - id={tc.get('id','')}, name={tc.get('name','')}, args={tc.get('args',{})}")

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],   #这里的response的content部分是SUPERVISOR的思考结果，前端可展示
            "research_iterations": next_iter,
        },
    )

# ===== SUPERVISOR TOOLS 节点 =====用于工具注册
async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """执行工具调用：思考、自检、并行研究、汇总与结束判断"""
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    # 因为 supervisor 节点在返回时已经把 research_iterations + 1 了，
    # 所以这里的 research_iterations 实际上是当前正在执行的轮次号。
    
    most_recent_message = supervisor_messages[-1] if supervisor_messages else AIMessage(content="", tool_calls=[])
    refine_report_calls: List[Dict[str, Any]] = []

    print("\n" + "#" * 100)
    print(f">>> [SUPERVISOR_TOOLS CALLED] EXECUTION PHASE — Iteration #{research_iterations}")
    print("#" * 100)

    tool_messages: List[ToolMessage] = []
    all_raw_notes: List[str] = []
    draft_report = ""
    should_end = False

    # -------------------------------------------------------------------------
    # [修改] 结束判断逻辑微调
    # -------------------------------------------------------------------------
    # 注意：现在如果是 supervisor 强制触发的总结，research_iterations 已经超过了 max。
    # 我们需要在执行完工具 *之后* 再判断是否结束，而不是之前。
    # 所以这里先设一个 flag，表示“执行完这波操作后是否应该结束”。
    
    is_last_run = research_iterations > max_researcher_iterations 
    # 这里的 > 是因为 supervisor 在强制那次把 iter 加了1，所以现在应该是 max + 1
    
    tool_calls = getattr(most_recent_message, "tool_calls", []) or []
    no_tool_calls = len(tool_calls) == 0
    research_complete = any(tc.get("name") == "ResearchComplete" for tc in tool_calls)

    print(">>> [RECEIVED TOOL_CALLS]:")
    if not tool_calls:
        print("    (none)")
    else:
        for tc in tool_calls:
            print(f"    - id={tc.get('id','')}, name={tc.get('name','')}, args={tc.get('args',{})}")

    # 只有在真的没有任何工具要调用，或者是模型主动说结束时，才立即结束
    # 如果是 is_last_run，我们依然要执行这次的 tool_calls (也就是 refine_draft_report)
    if no_tool_calls or research_complete:
        should_end = True
        print(">>> [DECISION] should_end = True (no_tool_calls or ResearchComplete)")
    else:
        try:
            think_tool_calls = [tc for tc in tool_calls if tc.get("name") == "think_tool"]
            conduct_research_calls = [tc for tc in tool_calls if tc.get("name") == "ConductResearch"]
            refine_report_calls = [tc for tc in tool_calls if tc.get("name") == "refine_draft_report"]

            # 1. Think Tool
            for tc in think_tool_calls:
                args = tc.get("args") or {}
                if "reflection" not in args and "thought" in args:
                    args["reflection"] = args.pop("thought")
                args.setdefault("reflection", "")
                
                t_start_think = time.perf_counter()
                observation = think_tool.invoke(args)
                print(f"⏱️ [Timing] think_tool took {time.perf_counter() - t_start_think:.4f}s")

                tool_messages.append(
                    ToolMessage(
                        content=str(observation),
                        name=tc.get("name", "think_tool"),
                        tool_call_id=tc.get("id", ""),
                    )
                )
                print(">>> [EXEC THINK_TOOL] observation:")
                print(str(observation))

            # 2. Conduct Research
            if conduct_research_calls:
                coros = []
                for tc in conduct_research_calls:
                    args = tc.get("args") or {}
                    topic = args.get("research_topic", "")
                    print(f">>> [DISPATCH ConductResearch] topic='{topic}'")
                    coros.append(
                        researcher_agent.ainvoke({
                            "researcher_messages": [HumanMessage(content=topic)],
                            "research_topic": topic,
                        })
                    )
                
                t_start_research = time.perf_counter()
                tool_results = await asyncio.gather(*coros)
                print(f"⏱️ [Timing] ConductResearch (Parallel Group) took {time.perf_counter() - t_start_research:.4f}s")

                research_tool_messages = [
                    ToolMessage(
                        content=str(result.get("compressed_research", "Error synthesizing research report")),
                        name=tc.get("name", "ConductResearch"),
                        tool_call_id=tc.get("id", ""),
                    )
                    for result, tc in zip(tool_results, conduct_research_calls)
                ]
                tool_messages.extend(research_tool_messages)
                all_raw_notes = ["\n".join(result.get("raw_notes", [])) for result in tool_results]
                
                for idx, (result, tc) in enumerate(zip(tool_results, conduct_research_calls), start=1):
                    print(f">>> [RESULT ConductResearch #{idx}]")
                    print(">>> compressed_research:", str(result.get("compressed_research", ""))[:200], "...")

            # 3. Refine Draft Report
            for tc in refine_report_calls:
                notes = get_notes_from_tool_calls(supervisor_messages + tool_messages)
                findings = "\n".join(notes)
                
                t_start_refine = time.perf_counter()
                draft_report = refine_draft_report.invoke({
                    "research_brief": state.get("research_brief", ""),
                    "findings": findings,
                    "draft_report": state.get("draft_report", ""),
                })
                print(f"⏱️ [Timing] refine_draft_report took {time.perf_counter() - t_start_refine:.4f}s")

                tool_messages.append(
                    ToolMessage(
                        content=str(draft_report),
                        name=tc.get("name", "refine_draft_report"),
                        tool_call_id=tc.get("id", ""),
                    )
                )
                print(">>> [EXEC refine_draft_report] draft_report length:", len(str(draft_report)))

        except Exception as e:
            print(f"Error during tool execution: {e}")
            should_end = True

    # -------------------------------------------------------------------------
    # [修改] 延迟判断结束：执行完工具后，再看是否是最后一轮强制执行
    # -------------------------------------------------------------------------
    if is_last_run:
        should_end = True
        print(">>> [DECISION] should_end = True (Forced last run completed)")

    print(">>> [EXECUTED TOOL MESSAGES COUNT]:", len(tool_messages))
    print(f">>> [DECISION] should_end = {should_end}")

    if should_end:
        notes_combined = get_notes_from_tool_calls(supervisor_messages + tool_messages)
        return Command(
            goto=END,
            update={
                "notes": notes_combined,
                "research_brief": state.get("research_brief", ""),
            },
        )
    else:
        # 如果不是强制结束，且有 refine 操作，通常意味着阶段性成果，更新 report 状态
        updates = {
            "supervisor_messages": tool_messages,
            "raw_notes": all_raw_notes,
        }
        if draft_report:
            updates["draft_report"] = draft_report
            
        return Command(
            goto="supervisor",
            update=updates,
        )

# ===== 构建图 =====
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_edge("supervisor", "supervisor_tools")

supervisor_agent = supervisor_builder.compile()

# ===== 演示运行 =====
if __name__ == "__main__":
    initial_state: SupervisorState = {
        "supervisor_messages": [],
        "research_iterations": 0,
        "research_brief": "请研究“AI agent 在企业知识管理中的应用前景”。",
        "draft_report": "初稿：这是一个示例草稿，待完善。",
    }

    final_state = asyncio.run(supervisor_agent.ainvoke(initial_state))
    print("\n" + "*" * 100)
    print("=== Graph 运行结束，最终状态 ===")
    # print(final_state) # 内容太多，这里仅打印状态对象结构
    print("KEYS:", final_state.keys())
    print("*" * 100 + "\n")
