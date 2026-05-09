import os
import requests
import json
import logging
from langchain_core.tools import tool

# 获取 Docker Compose 中配置的环境变量
MCP_URL = os.getenv("MCP_CHART_SERVER_URL", "http://localhost:1122")
# MCP SSE 协议通常通过 POST /messages 发送 JSON-RPC
MCP_POST_ENDPOINT = f"{MCP_URL}/messages"

logging.basicConfig(level=logging.INFO)

def _send_mcp_request(method: str, params: dict = None):
    """发送 JSON-RPC 2.0 请求给 MCP Server"""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": 1
    }
    try:
        # 注意: 这里简化了 MCP 协议。
        # 标准 MCP 涉及 SSE 握手，但许多 MCP Server 实现也支持直接的 HTTP POST 调用工具
        # 如果 mcp-server-chart 严格只支持 SSE 流，这里可能需要调整为 SSEClient
        # 但根据 AntV 文档，我们主要尝试直接调用其 underlying API 或通过 HTTP 交互
        
        # 针对 AntV MCP 这种特殊情况，通常它们通过 HTTP 暴露工具调用
        # 如果这是基于标准 MCP SDK 构建的，我们尝试通过 session 初始化
        # 为简化演示，这里假设我们能通过 POST 发送工具调用请求
        
        # *修正策略*: 
        # 由于完全实现 MCP 客户端协议(握手->列出工具->调用)较复杂
        # 我们这里做一个 假设的 HTTP 桥接，或者建议直接使用 AntV 提供的 HTTP 接口
        # 既然是 Docker 部署，我们可以尝试构建一个简单的请求:
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{MCP_URL}/sse", headers=headers, json=payload, timeout=30)
        
        # 实际情况中，MCP over SSE 是一个长连接。
        # 这里的 hack 方法：如果 AntV 支持直接 HTTP 请求更好。
        # 鉴于其文档提到 "Run with SSE"，我们模拟一个工具调用消息。
        pass
    except Exception as e:
        return f"Connection Error: {e}"

@tool
def generate_chart_url(chart_type: str, data: list, x_field: str, y_field: str, title: str = "") -> str:
    """
    Generates a chart and returns an image URL.
    Useful when you need to visualize data.
    
    Args:
        chart_type: One of 'line_chart', 'bar_chart', 'pie_chart', 'area_chart'.
        data: A list of dictionaries, e.g., [{'date': '2023-01', 'value': 100}, ...]
        x_field: The key in the data dict for the X axis.
        y_field: The key in the data dict for the Y axis.
        title: The title of the chart.
    """
    
    # 映射 Agent 的 chart_type 到 MCP 工具名
    tool_mapping = {
        "line_chart": "generate_line_chart",
        "bar_chart": "generate_bar_chart",
        "pie_chart": "generate_pie_chart",
        "area_chart": "generate_area_chart"
    }
    
    mcp_tool_name = tool_mapping.get(chart_type, "generate_line_chart")
    
    # 构造 AntV 需要的参数格式
    chart_params = {
        "title": title,
        "xField": x_field,
        "yField": y_field,
        "data": data
    }
    
    # --- 核心调用逻辑 ---
    # 由于直接实现 Python MCP Client 很重，我们使用 requests 发送 JSON-RPC 'tools/call'
    # 注意：这取决于 mcp-server-chart 对 HTTP POST 的支持程度。
    # 如果失败，通常是因为需要完整的 SSE 会话 ID。
    
    logging.info(f"Generating {chart_type} via {MCP_URL}...")

    # 构造符合 MCP 规范的 JSON-RPC 请求
    json_rpc_req = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": mcp_tool_name,
            "arguments": chart_params
        },
        "id": "req-1"
    }

    try:
        # 尝试通过 SSE 端点发送 (有些实现允许 POST 消息到 SSE session)
        # 或者尝试直接 POST (取决于服务器实现)
        # 针对 AntV，我们尝试通过 HTTP POST 方式交互，如果它暴露了 streamable transport /mcp
        
        # 修正：docker-compose 中我们开启了 sse。
        # 如果要简单，我们可以把 command 改成 streamable 并用 POST /mcp
        target_url = f"{MCP_URL}/mcp" # 假设我们稍后会修改 docker command 为 streamable
        
        resp = requests.post(target_url, json=json_rpc_req, timeout=60)
        resp_data = resp.json()
        
        if "result" in resp_data:
            # 解析结果
            content = resp_data["result"].get("content", [])
            for item in content:
                if item.get("type") == "text":
                    # AntV 通常返回 JSON 字符串或 URL
                    text = item.get("text", "")
                    if "http" in text:
                        return f"Chart Generated: {text}"
                    return text
        return f"Chart generation failed: {resp.text}"
        
    except Exception as e:
        return f"Error calling chart server: {str(e)}\nMake sure Docker container is running."

