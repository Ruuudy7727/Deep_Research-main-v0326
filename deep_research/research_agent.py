import os
import re
import sys
import json
import logging
import time
from typing import Any, Dict, List
from typing_extensions import Literal

# =================================================================================
# ===== 路径配置 (防止 ModuleNotFoundError) =======================================
# =================================================================================
try:
    from pathlib import Path
    # 获取当前文件的父目录的父目录 (即项目根目录 PROJECT_ROOT)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception as e:
    print(f"CRITICAL: Failed to configure sys.path. Imports may fail. Error: {e}", file=sys.stderr)

# =================================================================================
# ===== 导入依赖 ==================================================================
# =================================================================================
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

# --- 自定义模块 (使用绝对导入，确保稳定) ---
from deep_research.state_research import ResearcherState, ResearcherOutputState
# [修改] 注释掉 tavily_search 导入
from deep_research.utils import get_today_str, local_search # , tavily_search 
from deep_research.gemini_chat import *

# 导入 Prompts (包含你新添加的 search_decision_prompt)
from deep_research.prompts import (
    search_decision_prompt, 
    compress_research_system_prompt, 
    compress_research_human_message
)

# =================================================================================
# ===== 全局配置 ==================================================================
# =================================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ENABLE_LOCAL_KB = True   # 是否开启本地知识库补充
MAX_ROUNDS = 2           # 最大搜索轮次 (防止死循环)
ENABLE_COMPRESSION = False # 是否压缩搜索结果 (节省 Token)

# =================================================================================
# ===== 辅助工具函数 ==============================================================
# =================================================================================

def format_messages_for_llm(messages: List[Any]) -> str:
    """将 LangChain 的消息历史格式化为纯文本，供 LLM 分析决策"""
    lines: List[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            lines.append(f"[系统指令]\n{m.content}")
        elif isinstance(m, HumanMessage):
            lines.append(f"[用户任务]\n{m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(f"[搜索/工具结果]\n{m.content}")
        elif isinstance(m, AIMessage):
            # 过滤掉空的 tool_calls 消息，只保留有文本内容的
            if m.content:
                lines.append(f"[AI 思考/回复]\n{m.content}")
    return "\n\n".join(lines)

def extract_json_decision(text: str) -> Dict[str, Any]:
    """鲁棒的 JSON 解析器，从 LLM 回复中提取决策"""
    try:
        text = text.strip()
        # 1. 移除 Markdown 代码块标记 (```json ... ```)
        if "```" in text:
            text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
        
        # 2. 尝试提取最外层的 {}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
        
        # 3. 如果找不到 JSON 结构，默认结束
        logger.warning(f"未在回复中找到 JSON，原始内容: {text[:100]}...")
        return {"decision": "finish", "thought": "解析失败，默认结束"}
        
    except Exception as e:
        logger.error(f"JSON解析异常: {e}")
        return {"decision": "finish", "thought": "解析异常，强制结束"}

def compress_search_result(query: str, raw_content: str, research_topic: str = "") -> str:
    """(可选) 使用 LLM 压缩冗长的网页内容"""
    if not raw_content or len(raw_content) < 500:
        return raw_content
    
    # 简单的压缩 Prompt
    sys_prompt = "你是一个信息提取专家。请从以下文本中提取与查询词最相关的事实、数据和细节。去除无关广告和导航信息。"
    user_prompt = f"查询词: {query}\n研究主题: {research_topic}\n\n原文:\n{raw_content[:8000]}" # 限制输入长度
    
    try:
        compressed, _ = gemini_chat_once(user_text=user_prompt, system_instruction=sys_prompt)
        if not compressed: 
            return raw_content[:1000] + "..."
        return f"[摘要内容]\n{compressed}"
    except Exception:
        return raw_content[:1000] + "..."

# =================================================================================
# ===== 核心节点 1: 决策节点 (Decision Node) =======================================
# =================================================================================
def decision_node(state: ResearcherState):
    """
    分析历史，决定是 'search' 还是 'finish'。
    这一步只负责【决策】，不实际执行搜索。
    """
    messages = state.get("researcher_messages", [])
    research_topic = state.get("research_topic", "")
    
    # 1. 检查是否达到最大轮次
    tool_msgs_count = sum(1 for m in messages if isinstance(m, ToolMessage))
    if tool_msgs_count >= MAX_ROUNDS:
        logger.warning(f"🛑 已达到最大搜索轮次 ({MAX_ROUNDS})，强制停止。")
        return {"researcher_messages": [AIMessage(content="已达到最大搜索限制，停止搜索。")]}

    # 2. 准备 Prompt
    context_text = format_messages_for_llm(messages)
    # 使用 prompts.py 中新增的 search_decision_prompt
    system_instruction = search_decision_prompt.format(date=get_today_str())
    
    logger.info(f"🤔 正在进行决策 (第 {tool_msgs_count + 1} 轮)...")

    # 3. 调用 LLM
    start_t = time.perf_counter()
    llm_response, _ = gemini_chat_once(user_text=context_text, system_instruction=system_instruction)
    duration = time.perf_counter() - start_t
    print(f"⏱️ 决策耗时: {duration:.2f}s | LLM输出: {llm_response[:100]}...")

    # 4. 解析决策 JSON
    decision_data = extract_json_decision(llm_response)
    action = decision_data.get("decision", "finish").lower()
    query = decision_data.get("search_query", "")
    thought = decision_data.get("thought", "无思考过程")

    # 5. 根据决策构建输出
    if action == "search" and query:
        logger.info(f"🟢 决定搜索: {query}")
        
        # 关键：我们手动构造一个“假”的 tool_calls 结构，骗过后续的路由和工具节点
        # 这样就不依赖 LLM 输出标准的 OpenAI function call 格式了，稳定性更高
        ai_msg = AIMessage(
            content=f"决策: {thought}\n准备搜索: {query}",
            tool_calls=[{
                "id": f"call_{int(time.time())}", # 伪造一个唯一 ID
                "name": "tavily_search",           # 必须对应 tool_node 里的处理逻辑
                "args": {"query": query}
            }]
        )
        return {"researcher_messages": [ai_msg]}
        
    else:
        # 结束搜索
        logger.info(f"🔴 决定结束 (Finish). 原因: {thought}")
        return {"researcher_messages": [AIMessage(content=f"决策: {thought}\n搜索结束。")]}

# =================================================================================
# ===== 核心节点 2: 工具执行节点 (Tool Node) =======================================
# =================================================================================
def tool_node(state: ResearcherState):
    """
    执行决策节点传递下来的搜索任务。
    """
    messages = state.get("researcher_messages", [])
    research_topic = state.get("research_topic", "")
    
    # 获取最后一条消息 (应该是 decision_node 生成的带 tool_calls 的 AIMessage)
    last_msg = messages[-1]
    
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        logger.error("❌ ToolNode 被调用但没有找到 tool_calls")
        return {"researcher_messages": []}

    tool_outputs = []
    
    for tool_call in last_msg.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        call_id = tool_call["id"]
        
        query = args.get("query", "")
        
        if name == "tavily_search":
            logger.info(f"🔎 执行 Tavily 搜索: {query}")
            
            # --- 1. 网络搜索 ---
            # [修改] 注释掉 Tavily 搜索逻辑，防止未配置 API Key 报错
            # try:
            #     web_res = tavily_search.invoke(query)
            # except Exception as e:
            #     logger.error(f"Tavily 搜索失败: {e}")
            #     web_res = f"搜索出错: {e}"
            web_res = "Tavily search is disabled." # 占位符，保证变量已定义
            
            combined_res = f"[Internet Search Result]\n{web_res}"

            # --- 2. (可选) 本地知识库补充 ---
            if ENABLE_LOCAL_KB:
                try:
                    local_res = local_search.invoke({"query": query})
                    if local_res and "未找到" not in local_res:
                        combined_res += f"\n\n[Local Database Result]\n{local_res}"
                        logger.info(f"📚 本地库命中: {len(local_res)} 字符")
                except Exception as e:
                    logger.warning(f"本地搜索出错 (忽略): {e}")

            # --- 3. 结果压缩/摘录 (防止 Context 爆炸) ---
            if ENABLE_COMPRESSION:
                final_obs = compress_search_result(query, combined_res, research_topic)
            else:
                final_obs = combined_res
            
            # --- 4. 生成 ToolMessage ---
            tool_outputs.append(
                ToolMessage(content=final_obs, name=name, tool_call_id=call_id)
            )

    return {"researcher_messages": tool_outputs}

# =================================================================================
# ===== 核心节点 3: 报告生成节点 (Compress Research) ==============================
# =================================================================================
def compress_research(state: ResearcherState) -> dict:
    """
    搜索结束，汇总所有信息生成中间报告。
    """
    messages = state.get("researcher_messages", [])
    context_text = format_messages_for_llm(messages)
    research_topic = state.get("research_topic", "")
    
    # 注入 prompts.py 中的提示词
    system_message = compress_research_system_prompt.format(date=get_today_str())
    # 补充研究主题上下文
    user_text = f"Research Topic: {research_topic}\n\n" + context_text + "\n\n" + compress_research_human_message

    logger.info("📝 正在撰写最终汇总报告...")
    report, _ = gemini_chat_once(user_text=user_text, system_instruction=system_message)
    
    # 提取所有 ToolMessage 的原始内容作为笔记备份
    raw_notes = []
    for m in messages:
        if isinstance(m, ToolMessage):
            raw_notes.append(m.content)

    return {"compressed_research": report, "raw_notes": raw_notes}  # report是经过检索后，对检索到的信息进行压缩整合的结果，前端可展示

# =================================================================================
# ===== 路由逻辑 (Conditional Edges) ==============================================
# =================================================================================
def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """
    检查决策节点的输出：
    - 如果有 tool_calls -> 转去执行工具
    - 否则 -> 结束，转去写报告
    """
    messages = state.get("researcher_messages", [])
    if not messages:
        return "compress_research"
        
    last_msg = messages[-1]
    
    # decision_node 如果决定搜索，会生成带 tool_calls 的 AIMessage
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_node"
        
    return "compress_research"

# =================================================================================
# ===== 构建图 (Graph Construction) ===============================================
# =================================================================================
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# 1. 添加节点
agent_builder.add_node("decision_node", decision_node)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# 2. 定义边
# Start -> 决策
agent_builder.add_edge(START, "decision_node")

# 决策 -> (工具 OR 结束)
agent_builder.add_conditional_edges(
    "decision_node", 
    should_continue, 
    {
        "tool_node": "tool_node", 
        "compress_research": "compress_research"
    }
)

# 工具 -> 回到决策 (继续循环)
agent_builder.add_edge("tool_node", "decision_node")

# 报告 -> 结束
agent_builder.add_edge("compress_research", END)

# 3. 编译图
researcher_agent = agent_builder.compile()

# =================================================================================
# ===== 测试入口 (Main Test) ======================================================
# =================================================================================
if __name__ == "__main__":
    print("\n--- Researcher Agent (Binary Decision Logic) Test ---\n")
    
    test_topic = "AI Agent 在法律领域的最新应用案例"
    print(f"Testing Topic: {test_topic}")
    
    initial_state = {
        "researcher_messages": [HumanMessage(content=test_topic)],
        "research_topic": test_topic
    }
    
    try:
        for step in researcher_agent.stream(initial_state, {"recursion_limit": 20}):
            node_name = list(step.keys())[0]
            print(f"✅ 节点 '{node_name}' 执行完成")
            
            # 如果是总结节点，打印结果
            if node_name == "compress_research":
                res = step[node_name]
                print("\n\n====== 最终报告 ======\n")
                print(res.get("compressed_research"))
                print("\n======================\n")
                
    except Exception as e:
        print(f"❌ 运行出错: {e}")
