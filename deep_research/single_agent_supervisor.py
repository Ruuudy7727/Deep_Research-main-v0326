import asyncio
import json
import os
import sys
import time
from typing import Literal, Dict, Any, List

# === LangGraph & LangChain 依赖 ===
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# === 项目内部依赖 ===
from .state_scope import AgentState
from .gemini_chat import gemini_chat_once, gemini_chat_once_rpo
from .prompts import *


# =============================================================================
# 对话历史工具函数（供 Supervisor / Solver 共享）
# =============================================================================
def _get_recent_history_str(state: Dict[str, Any], k: int = 5, max_chars: int = 1200) -> str:
    """
    从 state 中提取最近 k 轮对话，返回纯文本。
    优先使用 state["history"]（List[str]），否则回退到 state["messages"]。
    """
    entries: List[str] = []
    hist = state.get("history") or []
    if isinstance(hist, list) and hist:
        recent = [str(h) for h in hist[-k:] if h]
        entries.extend(recent)

    if not entries:
        msgs = state.get("messages") or []
        chat_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]
        if chat_msgs and isinstance(chat_msgs[-1], HumanMessage):
            chat_msgs = chat_msgs[:-1]
        for m in chat_msgs[-(k * 2):]:
            role = "User" if isinstance(m, HumanMessage) else "AI"
            content = str(getattr(m, "content", "") or "").strip().replace("\n", " ")
            if len(content) > 600:
                content = content[:600] + "..."
            entries.append(f"- {role}: {content}")

    if not entries:
        return ""

    text = "\n\n".join(entries)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text

# =============================================================================
# [核心修改 1] 导入工具函数 (含新增的 local_search)
# =============================================================================
try:
    # 导入数据库查询逻辑
    from .research_agent_analyze import retrieve_battery_node
    # 导入画图逻辑
    from .research_agent_draw import generate_chart_node
    # [新增] 导入知识库检索工具 (假设位于 utils 或同级目录下)
    from .utils import local_search 
except ImportError:
    # 防止因路径问题导致代码无法运行的 Mock (实际运行请确保文件存在)
    print("Warning: External modules not found, using mocks.", flush=True)
    async def retrieve_battery_node(state): return {"raw_db_results": [{"mock": "data"}]}
    def generate_chart_node(state): return {"chart_output": "http://mock.url/img.png", "pre_brief_cases": "Chart generated."}
    # Mock local_search
    class MockSearch:
        def invoke(self, x): return "Mock Retrieval Result: 这是一个测试检索结果。"
    local_search = MockSearch()

# =============================================================================
# 1. Tool Definitions (Tool Binding Schema)
# =============================================================================

#工具声明和定义
@tool
class BatteryDataTool(BaseModel):
    """用于查询电池站点的具体设备数据、电压、报警状态等信息的工具。"""
    query: str = Field(description="用户的查询描述")

@tool
class ChartGenerationTool(BaseModel):
    """用于将现有数据可视化、生成折线图、柱状图等的工具。"""
    data_context: str = Field(description="需要可视化的数据上下文")

# =============================================================================
# 3. Node: Supervisor (Router) 注册工具
# =============================================================================


def _extract_first_json_object(text: str) -> str:
    """
    从 LLM 文本中截取第一个完整 JSON 对象（按括号深度 + 尊重字符串内引号转义），
    避免 re 贪婪匹配把解释文字或多段内容拼进导致 json.loads 失败。
    """
    if not text:
        return ""
    s = text.strip()
    s = s.replace("```json", "").replace("```", "").strip()
    start = s.find("{")
    if start < 0:
        return ""
    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return ""


def _log_supervisor_parse_failure(hint: str, resp_text: str) -> None:
    snippet = (resp_text or "")[:2500]
    print(f"[Supervisor Error] {hint}", flush=True)
    print(f"[Supervisor] raw response (len={len(resp_text or '')}):\n{snippet}", flush=True)
    try:
        log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "supervisor_router_fail.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {hint} ===\n{resp_text or ''}\n\n"
            )
    except Exception:
        pass


# =============================================================================
# task_type 推断（6 类前端布局：direct / kb_retrieval / station_device_td / alerting / troubleshooting / deep_research）
# =============================================================================
_TASK_TYPE_VALID = {"direct", "kb_retrieval", "station_device_td", "alerting", "troubleshooting", "deep_research"}

# 告警关键词：覆盖中英文常见说法
_ALARM_KEYWORDS = (
    "告警", "报警", "警报", "预警", "异常事件",
    "alarm", "alarm_event", "alarm_events",
    "severity", "average_severity", "summary_cn",
    "严重度", "严重等级", "风险等级",
)

# 绘图关键词：与 _should_chain_chart_after_database 保持一致
_CHART_KEYWORDS = (
    "画图", "绘图", "作图", "图表", "可视化", "折线图", "柱状图",
    "条形图", "饼图", "雷达图", "散点图", "直方图", "趋势图",
    "曲线", "时序", "时间序列",
)

_FAULT_DIAG_KEYWORDS = (
    "内阻异常", "dcr", "dcr_abnormal", "abnormal_days",
    "微短路", "内短路", "isc", "microshort", "isc_score",
    "容量不一致", "capacity_inconsistent", "容量异常",
    "故障钻探", "根因", "异常电芯",
)


def _infer_task_type(next_action: str, params: Dict[str, Any], user_req: str) -> str:
    """
    后端兜底：当 LLM 未输出 task_type 或输出非法值时，按规则推断。
    优先级：alerting > troubleshooting > station_device_td > kb_retrieval > direct。
    """
    action = (next_action or "DIRECT").upper()
    req = str(user_req or "")
    has_alarm_kw = any(kw.lower() in req.lower() for kw in _ALARM_KEYWORDS)
    has_chart_kw = any(kw in req for kw in _CHART_KEYWORDS)
    has_fault_kw = any(kw.lower() in req.lower() for kw in _FAULT_DIAG_KEYWORDS)

    if not isinstance(params, dict):
        params = {}

    if action == "DATABASE":
        if has_alarm_kw or str(params.get("summary_keyword") or "").strip():
            return "alerting"
        if has_fault_kw:
            return "troubleshooting"
        # 图表需求并不改变主功能类别：仍属于设备实时查询场景
        if has_chart_kw or any(params.get(k) for k in ("chart_after_db", "chart_type", "x_field", "y_field")):
            return "station_device_td"
        return "station_device_td"
    if action == "CHART":
        return "station_device_td"
    if action == "RETRIEVE":
        return "kb_retrieval"
    return "direct"


def _normalize_task_type(raw: Any, next_action: str, params: Dict[str, Any], user_req: str) -> str:
    """
    把 LLM 输出的 task_type 规范化到合法枚举；非法 / 缺失 时退化到 _infer_task_type。
    """
    if isinstance(raw, str):
        candidate = raw.strip().lower().replace("-", "_")
        if candidate in _TASK_TYPE_VALID:
            return candidate
    return _infer_task_type(next_action, params, user_req)


def _should_chain_chart_after_database(user_req: str, params: Dict[str, Any], raw_data: Any) -> bool:
    if not isinstance(raw_data, list) or not any(isinstance(r, dict) and not r.get("error") for r in raw_data):
        return False

    if not isinstance(params, dict):
        params = {}

    explicit_flag = params.get("chart_after_db")
    if isinstance(explicit_flag, bool):
        return explicit_flag
    if isinstance(explicit_flag, str) and explicit_flag.strip().lower() in {"1", "true", "yes", "y"}:
        return True

    if any(params.get(k) for k in ("chart_type", "x_field", "y_field")):
        return True

    chart_keywords = (
        "画图", "绘图", "作图", "图表", "可视化", "折线图", "柱状图",
        "条形图", "饼图", "雷达图", "散点图", "直方图", "趋势图",
    )
    req = str(user_req or "")
    return any(keyword in req for keyword in chart_keywords)


async def single_supervisor_node(state: AgentState):
    """
    路由节点：决定是去执行工具(DB/Chart/Retrieve) 还是直接去回答。
    """
    print("--- Executing Node: Supervisor (Intent Routing) ---", flush=True)
    
    user_req = state.get("user_request", "")
    tot_plan = state.get("current_tot", "暂无前置规划，请根据当前问题进行自主决策")

    history_str = _get_recent_history_str(state, k=5, max_chars=1500)
    prev_cases = state.get("pre_brief_cases", "") or ""
    prev_cases_snippet = prev_cases[-1500:] if prev_cases else ""

    extra_context_parts: List[str] = []
    if history_str:
        extra_context_parts.append(
            '【最近对话历史（越后越新，用于理解 上文/以上/之前 等指代）】：\n' + history_str
        )
    if prev_cases_snippet:
        extra_context_parts.append(
            '【上轮已检索到的数据/图表上下文（若用户要求 绘图/再次分析/继续 等，可直接复用）】：\n'
            + prev_cases_snippet
        )

    contextual_user_req = user_req
    if extra_context_parts:
        contextual_user_req = (
            user_req
            + "\n\n"
            + "\n\n".join(extra_context_parts)
            + "\n\n【路由提示】：如果当前请求依赖上文（出现 以上/上面/刚才/之前/再画一下 等代词），"
              "且上方已有可用数据上下文，优先路由到 CHART；否则按正常规则路由。"
        )

    # 格式化 Prompt
    final_prompt = supervisor_system_prompt.format(user_req=contextual_user_req, tot_plan=tot_plan)

    resp_text = ""
    try:
        # 调用 LLM 做决策
        resp_text, _ = await asyncio.to_thread(gemini_chat_once, final_prompt, "You are a JSON router.", 0.0)
        
        clean_json = _extract_first_json_object(resp_text or "")
        if not clean_json:
            _log_supervisor_parse_failure("no JSON object found in LLM output", resp_text or "")
            return {
                "supervisor_route": "DIRECT",
                "supervisor_params": {},
                "task_type": _infer_task_type("DIRECT", {}, user_req),
            }

        try:
            decision_data = json.loads(clean_json)
        except json.JSONDecodeError as e:
            _log_supervisor_parse_failure(f"json.JSONDecodeError: {e}", resp_text or "")
            return {
                "supervisor_route": "DIRECT",
                "supervisor_params": {},
                "task_type": _infer_task_type("DIRECT", {}, user_req),
            }

        next_action = decision_data.get("next_action", "DIRECT")
        if isinstance(next_action, str):
            next_action = next_action.upper()
        else:
            next_action = "DIRECT"
        params = decision_data.get("params", {})
        if not isinstance(params, dict):
            params = {}

        task_type = _normalize_task_type(decision_data.get("task_type"), next_action, params, user_req)

        print(f">>> [Supervisor Decision]: {next_action}")
        print(f">>> [Supervisor Params]: {params}")
        print(f">>> [Supervisor task_type]: {task_type}")

        return {
            "supervisor_route": next_action,
            "supervisor_params": params,
            "task_type": task_type,
        }

    except Exception as e:
        _log_supervisor_parse_failure(f"{type(e).__name__}: {e}", resp_text)
        print(f"[Supervisor Error]: {e}, defaulting to DIRECT.", flush=True)
        return {
            "supervisor_route": "DIRECT",
            "supervisor_params": {},
            "task_type": _infer_task_type("DIRECT", {}, user_req),
        }


# =============================================================================
# 4. Node: Execute Tools (The Tool Caller)
# =============================================================================

async def execute_tools_node(state: AgentState):
    """
    [关键节点] 工具执行层。
    根据 supervisor_route 调用 Database, Chart 或 Retrieve 工具。
    """
    route = state.get("supervisor_route", "DIRECT")
    params = state.get("supervisor_params", {})
    current_context = state.get("pre_brief_cases", "")
    user_req = state.get("user_request", "")
    current_db_rows = state.get("db_raw_results", [])
    
    print(f"--- Executing Node: Execute Tools ({route}) ---", flush=True)
    
    updates = {}

    def _summarize_db_results(raw_data: Any, max_rows: int = 8) -> str:
        if not isinstance(raw_data, list):
            return f"数据库返回非列表结构: {str(raw_data)[:500]}"
        if not raw_data:
            return "数据库无匹配记录。"

        total = len(raw_data)
        error_rows = [r for r in raw_data if isinstance(r, dict) and r.get("error")]
        valid_rows = [r for r in raw_data if isinstance(r, dict) and not r.get("error")]

        lines = [
            f"总记录数: {total}",
            f"有效记录: {len(valid_rows)}",
            f"错误记录: {len(error_rows)}",
        ]

        if valid_rows:
            stations = sorted({str(r.get("station_code", "")) for r in valid_rows if r.get("station_code")})
            bmus = sorted({str(r.get("bmu_code", "")) for r in valid_rows if r.get("bmu_code")})
            cells = sorted({str(r.get("cell_id", "")) for r in valid_rows if r.get("cell_id")})
            tables = sorted({str(r.get("table_name", "")) for r in valid_rows if r.get("table_name")})
            if tables:
                lines.append(f"来源表: {', '.join(tables[:8])}")
            if stations:
                lines.append(f"station_code: {', '.join(stations[:5])}")
            if bmus:
                lines.append(f"bmu_code: {', '.join(bmus[:8])}")
            if cells:
                lines.append(f"cell_id样本: {', '.join(cells[:12])}")

            max_summary_chars = 4000
            lines.append("\n样本记录:")
            for row in valid_rows[:max_rows]:
                if row.get("summary_cn"):
                    # 保留长文本，避免误导下游模型“数据为空/信息不足”
                    summary = str(row.get("summary_cn", "")).strip()
                    if len(summary) > max_summary_chars:
                        summary = summary[:max_summary_chars] + "...(truncated)"
                    lines.append(
                        "- table={table}, station={station}, bmu={bmu}, cell={cell}, summary={summary}".format(
                            table=row.get("table_name", ""),
                            station=row.get("station_code", ""),
                            bmu=row.get("bmu_code", ""),
                            cell=row.get("cell_id", ""),
                            summary=summary or "(无摘要)",
                        )
                    )
                else:
                    compact = {}
                    for key in (
                        "table_name", "ts", "bmu_code", "cluster_code", "cell", "cell_id",
                        "microshort_score", "microshortscore", "diagnosis_result",
                        "delta_v", "delta_t", "soc", "soh", "voltage", "current", "power",
                        "cell_avg_vol", "cell_avg_temp", "vmax", "vmin", "tmax", "tmin",
                    ):
                        if row.get(key) is not None:
                            compact[key] = row.get(key)
                    if not compact:
                        compact = {k: row.get(k) for k in list(row.keys())[:6]}
                    lines.append(f"- {json.dumps(compact, ensure_ascii=False, default=str)}")

        if error_rows:
            lines.append("\n错误样本:")
            for row in error_rows[:3]:
                lines.append(f"- {row.get('error')}")

        return "\n".join(lines)
    
    # === 分支 1: 调用外部数据库工具 (DATABASE) ===
    if route == "DATABASE":
        print(">>> Calling External Tool: retrieve_battery_node ...")
        try:
            # retrieve_battery_node 内部可能需要用到 params，确保 state 中已包含
            # 如果 retrieve_battery_node 依赖特定的 keys，这里可以手动合并 params 到 state
            # 此处假设 retrieve_battery_node 会读取 state 或 params
            
            tool_result = await retrieve_battery_node(state)
            
            raw_data = tool_result.get("raw_db_results", [])
            data_str = _summarize_db_results(raw_data)
            route_name = tool_result.get("db_route", "unknown")
            # DATABASE 子链路最终功能类型以 db_route 为准（而不是顶层粗粒度 task_type）
            if route_name == "station_device_td":
                updates["task_type"] = "station_device_td"
            elif route_name == "alerting":
                updates["task_type"] = "alerting"
            elif route_name == "troubleshooting":
                updates["task_type"] = "troubleshooting"
            elif route_name == "clarification_needed":
                # 仅查库链路会触发 clarification；维持数据库主场景语义
                fallback_tt = str(state.get("task_type", "") or "").strip()
                if fallback_tt in {"station_device_td", "alerting", "troubleshooting"}:
                    updates["task_type"] = fallback_tt
                else:
                    updates["task_type"] = "station_device_td"
            evidence_bundle = tool_result.get("db_evidence_bundle", {})
            evidence_str = ""
            if isinstance(evidence_bundle, dict) and evidence_bundle:
                evidence_str = f"\n\n【Database Evidence Bundle】:\n{json.dumps(evidence_bundle, ensure_ascii=False, default=str)}"

            updates["pre_brief_cases"] = current_context + f"\n\n【Database Query Result | route={route_name}】:\n{data_str}{evidence_str}"
            updates["supervisor_messages"] = ["Database tool executed successfully."]
            updates["db_raw_results"] = raw_data if isinstance(raw_data, list) else []
            
            if "db_query_params" in tool_result:
                updates["db_query_params"] = tool_result["db_query_params"]
            if "db_route" in tool_result:
                updates["db_route"] = tool_result["db_route"]
            if "db_query_plans" in tool_result:
                updates["db_query_plans"] = tool_result["db_query_plans"]
            if "db_executed_sqls" in tool_result:
                updates["db_executed_sqls"] = tool_result["db_executed_sqls"]
            if "db_evidence_bundle" in tool_result:
                updates["db_evidence_bundle"] = tool_result["db_evidence_bundle"]

            if _should_chain_chart_after_database(user_req, params, raw_data):
                print(">>> Chaining External Tool: generate_chart_node (after DATABASE) ...")
                chart_state = dict(state)
                chart_state.update(updates)
                chart_state["supervisor_route"] = "CHART"
                chart_state["supervisor_params"] = params

                try:
                    if asyncio.iscoroutinefunction(generate_chart_node):
                        chart_result = await generate_chart_node(chart_state)
                    else:
                        chart_result = await asyncio.to_thread(generate_chart_node, chart_state)

                    if chart_result:
                        updates.update(chart_result)
                        existing_msgs = updates.get("supervisor_messages", [])
                        if not isinstance(existing_msgs, list):
                            existing_msgs = [str(existing_msgs)]
                        if "Database tool executed successfully." not in existing_msgs:
                            existing_msgs.append("Database tool executed successfully.")
                        if "Chart tool executed after database query." not in existing_msgs:
                            existing_msgs.append("Chart tool executed after database query.")
                        updates["supervisor_messages"] = existing_msgs
                except Exception as chart_e:
                    print(f"[Tool Error] Post-DB chart generation failed: {chart_e}")
                    existing_msgs = updates.get("supervisor_messages", [])
                    if not isinstance(existing_msgs, list):
                        existing_msgs = [str(existing_msgs)]
                    existing_msgs.append(f"Chart tool failed after database query: {chart_e}")
                    updates["supervisor_messages"] = existing_msgs

        except Exception as e:
            print(f"[Tool Error] Database call failed: {e}")
            updates["pre_brief_cases"] = current_context + f"\n[System]: Database query failed: {e}"
            updates["db_raw_results"] = current_db_rows if isinstance(current_db_rows, list) else []

    # === 分支 2: 调用外部绘图工具 (CHART) ===
    elif route == "CHART":
        print(">>> Calling External Tool: generate_chart_node ...")
        updates["task_type"] = "station_device_td"
        try:
            if asyncio.iscoroutinefunction(generate_chart_node):
                tool_result = await generate_chart_node(state)
            else:
                tool_result = await asyncio.to_thread(generate_chart_node, state)
            
            if tool_result:
                updates.update(tool_result)
                if "db_raw_results" not in updates:
                    updates["db_raw_results"] = current_db_rows if isinstance(current_db_rows, list) else []
                
        except Exception as e:
            print(f"[Tool Error] Chart generation failed: {e}")
            updates["supervisor_messages"] = [f"Chart tool failed: {e}"]
            updates["db_raw_results"] = current_db_rows if isinstance(current_db_rows, list) else []

    # === [新增] 分支 3: 知识库检索 (RETRIEVE) ===
    elif route == "RETRIEVE":
        print(">>> Calling Tool: Local Knowledge Base Search ...")
        try:
            # 1. 获取查询词，优先使用 Supervisor 提取的 search_query
            query_term = params.get("search_query")
            if not query_term:
                query_term = user_req # 降级策略
            
            print(f"    -> Query: {query_term}")

            # 2. 调用 local_search 工具
            # 注意：local_search.invoke 通常接受 {"query": ...} 字典
            # 如果 local_search 是异步的需 await，这里假设它是标准的 LangChain Chain (同步) 或 异步
            
            search_input = {"query": query_term}
            
            if hasattr(local_search, "ainvoke"):
                search_res = await local_search.ainvoke(search_input)
            else:
                search_res = await asyncio.to_thread(local_search.invoke, search_input)

            # 3. 处理检索结果
            # 结果可能是 string 或 Document list
            final_obs = ""
            if isinstance(search_res, list):
                final_obs = "\n".join([getattr(doc, "page_content", str(doc)) for doc in search_res])
            elif isinstance(search_res, dict) and "result" in search_res:
                 final_obs = search_res["result"]
            else:
                final_obs = str(search_res)

            if not final_obs: 
                final_obs = "未能在知识库中检索到相关信息。"

            # 4. 更新 Context
            log_msg = f"Retrieval tool executed. Query: {query_term}"
            updates["pre_brief_cases"] = current_context + f"\n\n【Local Knowledge Base Result】:\n{final_obs}"
            updates["supervisor_messages"] = [log_msg]
            updates["db_raw_results"] = current_db_rows if isinstance(current_db_rows, list) else []

        except Exception as e:
            print(f"[Tool Error] Retrieval failed: {e}")
            updates["pre_brief_cases"] = current_context + f"\n[System]: Knowledge base search failed: {e}"
            updates["db_raw_results"] = current_db_rows if isinstance(current_db_rows, list) else []

    # === 分支 4: 直接通过 (DIRECT) ===
    else:
        # DIRECT 模式，无需操作，数据流直接通过到 Answer Node
        updates["db_raw_results"] = current_db_rows if isinstance(current_db_rows, list) else []
        
    return updates


# =============================================================================
# 5. Node: Solve Simple Task (Final Answer with RPO)
# =============================================================================

async def solve_simple_task(state: AgentState, writer=None) -> dict:
    """
    第三步：生成最终回答。
    接收 execute_tools_node 准备好的数据 (pre_brief_cases / chart_output)，
    使用 RPO 流式生成最终回答。
    """
    print("--- Executing Node: solve_simple_task (Simple Path) ---", flush=True)

    messages_str = state.get("user_request", "")
    current_tot = (state.get("current_tot") or state.get("question") or "")
    retrieved_info = state.get("pre_brief_cases", "")
    
    # 检查是否有图表 URL
    chart_info = ""
    if state.get("chart_output"):
        co = str(state.get("chart_output") or "").strip()
        pub = f"/figure/{os.path.basename(co.split('?', 1)[0])}" if co and not co.startswith("http") else (co if co.startswith("http") else "")
        chart_info = (
            "\n\n[Visual Content]: A trend chart has been generated and is shown in the UI. "
            "Reference the chart in your answer; do not print local filesystem paths."
            + (f" Public path hint: {pub}" if pub else "")
        )

    if not retrieved_info:
        print("Warning: No retrieved info found in state (Pure Chat Mode).", flush=True)

    # 读取最近对话历史
    history_str = _get_recent_history_str(state, k=5, max_chars=1500)
    history_block = ""
    if history_str:
        history_block = (
            '\n\n最近对话历史（用于理解"以上/上面/之前"等指代）:\n' + history_str + "\n"
        )

    print("--- [Simple Path] Synthesizing Answer... ---", flush=True)

    route = str(state.get("supervisor_route") or "DIRECT").upper()
    task_type = str(state.get("task_type") or "").strip().lower()
    use_direct_demo_few_shots = route == "DIRECT" or task_type == "direct"

    # 定义系统指令
    system_instruction = (
        "你是一名资深技术顾问。请根据用户的查询、初步分析(ToT)、检索到的证据(Database/KnowledgeBase)"
        "以及最近对话历史，生成最终回答。必须遵守以下格式约束："
        "1. 【直接结论】：第一行直接给出核心结论。"
        "2. 【逻辑分析】：结合检索到的数据或知识进行分析。"
        "3. 【禁止格式】：严禁使用 Markdown 加粗符号（即不要出现 ** 符号）。"
        "4. 输出纯文本，不要包含任何 JSON 格式。"
        '5. 若用户提及"以上/上面/之前"等代词，请结合对话历史正确理解其指代。'
    )
    if use_direct_demo_few_shots:
        fs = simple_direct_demo_few_shots
        if isinstance(fs, str) and fs.strip():
            system_instruction = system_instruction + "\n\n" + fs.strip()

    closing = (
        "基于以上信息，生成精炼回答。"
        if use_direct_demo_few_shots
        else "基于以上信息，生成一份精炼的诊断报告。"
    )
    # 定义用户提示
    user_prompt = (
        f"用户查询:\n{messages_str}\n\n"
        f"初步分析 (ToT):\n{current_tot}\n\n"
        f"检索到的证据/案例/数据:\n{retrieved_info}\n{chart_info}"
        f"{history_block}\n"
        f"{closing}"
    )

    LOCAL_MAX_TOKENS = 4096
    final_answer = ""

    # =========================================================================
    # [流式处理] Wrapper 函数
    # =========================================================================
    def _sync_stream_consumption(u_text, sys_inst, max_tok, stream_writer):
        print("\n⚡ [RPO Stream Start] ...", flush=True)
        try:
            generator = gemini_chat_once_rpo(
                user_text=u_text,
                system_instruction=sys_inst,
                temperature=0.3,
                max_tokens=max_tok
            )
            
            final_txt = ""
            final_usage = {}
            last_len = 0
            
            print(">>> ", end="", flush=True)
            
            if isinstance(generator, tuple): return generator[0], generator[1]

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
        error_msg = f"Error generating simple task report with RPO: {e}"
        print(f"[Simple Task] Generation Failed: {e}")
        final_answer = error_msg

    # 写入日志
    try:
        log_path = os.path.join(os.getcwd(), "log", "simple_task_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(final_answer + "\n\n")
    except Exception as e:
        print(f"[Simple Task] Log write failed: {e}")

    return {
        "final_report": final_answer,
        "supervisor_messages": state.get("supervisor_messages", []) + ["Simple task resolved via RPO."]
    }


# =============================================================================
# 5.5 历史记录更新节点（Fast 模式也要记住上下文）
# =============================================================================
def update_history_node(state: AgentState) -> Dict[str, Any]:
    """
    简单链路的历史收尾节点：把本轮 (User, AI) 追加到 state["history"]。
    state["history"] 的 reducer 是 operator.add，所以返回 list 会自动追加。
    """
    print("--- Executing Node: update_history (single_agent) ---", flush=True)

    user_req = state.get("user_request", "") or ""
    ai_resp = state.get("final_report", "") or ""

    if not ai_resp:
        msgs = state.get("messages", []) or []
        if msgs and isinstance(msgs[-1], AIMessage):
            ai_resp = str(msgs[-1].content or "")

    if not user_req and not ai_resp:
        return {}

    trimmed_user = str(user_req).strip()
    trimmed_ai = str(ai_resp).strip()
    if len(trimmed_ai) > 1500:
        trimmed_ai = trimmed_ai[:1500] + "...(truncated)"

    new_entry = f"User: {trimmed_user}\nAI: {trimmed_ai}"
    return {"history": [new_entry]}


# =============================================================================
# 6. Graph Logic (构建图)
# =============================================================================

def build_single_supervisor_graph(checkpointer=None, enable_history: bool = False):
    """
    构建单 Agent 监督者图。

    参数:
        checkpointer: 传入 LangGraph 兼容的 checkpointer（如 MemorySaver）可启用跨轮状态持久化
        enable_history: 是否在链路末尾挂接 update_history 节点，把每轮 (User, AI) 写入 state["history"]
    """
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", single_supervisor_node)
    builder.add_node("execute_tools_node", execute_tools_node)
    builder.add_node("solve_simple_task", solve_simple_task)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "execute_tools_node")
    builder.add_edge("execute_tools_node", "solve_simple_task")

    if enable_history:
        builder.add_node("update_history", update_history_node)
        builder.add_edge("solve_simple_task", "update_history")
        builder.add_edge("update_history", END)
    else:
        builder.add_edge("solve_simple_task", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()

# 默认导出：无 checkpointer 版本（供 research_agent_full 作为 sub-graph 使用）
single_agent = build_single_supervisor_graph()

# =============================================================================
# 7. 测试入口
# =============================================================================
if __name__ == "__main__":
    async def main():
        print("=== Test 1: Database Query ===")
        mock_state_db = {
            "user_request": "查一下 station-001 下 cluster-005 的电压",
            "question": "### 任务规划\n1. 意图：查库\n2. ID：cluster-005"
        }
        await single_agent.ainvoke(mock_state_db)
        
        print("\n=== Test 2: Knowledge Retrieval ===")
        mock_state_kb = {
            "user_request": "电池发生热失控的主要原因有哪些？",
            "question": "### 任务规划\n1. 意图：概念检索\n2. 关键词：热失控"
        }
        await single_agent.ainvoke(mock_state_kb)

    asyncio.run(main())
