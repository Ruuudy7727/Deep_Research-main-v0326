import os
import json
import requests
import re
import shutil
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool

try:
    from .gemini_chat import gemini_chat_once
except ImportError:
    from deep_research.gemini_chat import gemini_chat_once
MCP_URL = os.getenv("MCP_CHART_SERVER_URL", "http://localhost:1122")
MCP_POST_ENDPOINT = f"{MCP_URL}/mcp"

logging.basicConfig(level=logging.INFO)

DEFAULT_THEME = "academy"
DEFAULT_PALETTE = ["#1677ff", "#13c2c2", "#52c41a", "#faad14", "#722ed1", "#eb2f96"]
SUPPORTED_CHART_TYPES = {
    "line_chart",
    "area_chart",
    "bar_chart",
    "column_chart",
    "pie_chart",
    "radar_chart",
    "word_cloud_chart",
    "scatter_chart",
    "histogram_chart",
}
FIELD_LABELS = {
    "ts": "Time",
    "time": "Time",
    "timestamp": "Time",
    "report_time": "Time",
    "write_time": "Time",
    "start_time": "Start Time",
    "end_time": "End Time",
    "voltage": "Voltage",
    "soc": "SOC",
    "soh": "SOH",
    "current": "Current",
    "power": "Power",
    "temp": "Temperature",
    "name": "Name",
    "category": "Category",
}

# =============================================================================
# [新增] 绘图意图与数据结构 (Pydantic)
# =============================================================================
class ChartIntent(BaseModel):
    needs_chart: bool = Field(description="Set to True if extracted data is sufficient.")
    chart_type: Optional[str] = Field(
        default="line_chart", 
        description="Type of chart: line_chart, bar_chart, pie_chart, area_chart, column_chart, radar_chart, word_cloud_chart"
    )
    data_json: Optional[str] = Field(default=None, description="JSON string of data list")
    x_field: Optional[str] = Field(default=None, description="X axis field name")
    y_field: Optional[str] = Field(default=None, description="Y axis field name")
    title: Optional[str] = Field(default="Analysis Chart", description="Chart title")


def _normalize_field_name(field: Optional[str]) -> str:
    if not field:
        return ""
    field = str(field).strip().lower()
    alias_map = {
        "时间": "ts",
        "timestamp": "ts",
        "time": "ts",
        "日期": "ts",
        "电压": "voltage",
        "总电压": "voltage",
        "pack电压": "voltage",
        "soc": "soc",
        "soh": "soh",
        "电流": "current",
        "功率": "power",
        "温度": "temp",
    }
    return alias_map.get(field, field)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    try:
        clean = str(value).replace(",", "").replace("¥", "").replace("$", "").replace("%", "").strip()
        return float(clean)
    except Exception:
        return None


def _normalize_chart_type(chart_type: Optional[str]) -> str:
    raw = str(chart_type or "line_chart").strip().lower()
    alias_map = {
        "line": "line_chart",
        "折线图": "line_chart",
        "area": "area_chart",
        "面积图": "area_chart",
        "bar": "bar_chart",
        "条形图": "bar_chart",
        "column": "column_chart",
        "柱状图": "column_chart",
        "pie": "pie_chart",
        "饼图": "pie_chart",
        "radar": "radar_chart",
        "雷达图": "radar_chart",
        "word_cloud": "word_cloud_chart",
        "wordcloud": "word_cloud_chart",
        "词云": "word_cloud_chart",
        "scatter": "scatter_chart",
        "散点图": "scatter_chart",
        "histogram": "histogram_chart",
        "直方图": "histogram_chart",
    }
    normalized = alias_map.get(raw, raw)
    if normalized not in SUPPORTED_CHART_TYPES:
        return "line_chart"
    return normalized


def _humanize_field_name(field: Optional[str]) -> str:
    if not field:
        return ""
    clean = str(field).strip()
    key = clean.lower()
    if key in FIELD_LABELS:
        return FIELD_LABELS[key]
    return clean.replace("_", " ").title()


def _parse_sortable_time(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _sort_records_for_chart(data: List[Dict[str, Any]], x_field: str) -> List[Dict[str, Any]]:
    if not data:
        return data
    parsed = [_parse_sortable_time(item.get(x_field)) for item in data]
    if all(p is not None for p in parsed):
        return sorted(data, key=lambda item: _parse_sortable_time(item.get(x_field)) or datetime.min)
    return data


def _default_chart_size(tool_name: str) -> Dict[str, int]:
    if "pie" in tool_name or "radar" in tool_name:
        return {"width": 760, "height": 560}
    if "word" in tool_name:
        return {"width": 900, "height": 560}
    if "bar" in tool_name:
        return {"width": 1000, "height": 620}
    return {"width": 1000, "height": 520}


def _default_chart_style(tool_name: str) -> Dict[str, Any]:
    style: Dict[str, Any] = {
        "backgroundColor": "#ffffff",
        "palette": DEFAULT_PALETTE,
        "texture": "default",
    }
    if any(name in tool_name for name in ("line", "area", "radar")):
        style["lineWidth"] = 3
    return style


def _build_common_chart_args(intent_data: Dict[str, Any], tool_name: str, x_title: str, y_title: str) -> Dict[str, Any]:
    size = _default_chart_size(tool_name)
    args: Dict[str, Any] = {
        "title": intent_data.get("title", "Chart"),
        "theme": intent_data.get("theme") or DEFAULT_THEME,
        "style": intent_data.get("style") or _default_chart_style(tool_name),
        "width": intent_data.get("width") or size["width"],
        "height": intent_data.get("height") or size["height"],
    }
    if not any(name in tool_name for name in ("pie", "radar", "word")):
        args["axisXTitle"] = intent_data.get("axisXTitle") or x_title
        args["axisYTitle"] = intent_data.get("axisYTitle") or y_title
    return args


def _transform_data_for_tool(
    raw_data: List[Dict[str, Any]],
    orig_x: str,
    orig_y: str,
    tool_name: str,
) -> Dict[str, Any]:
    sorted_data = _sort_records_for_chart(raw_data, orig_x)
    x_title = _humanize_field_name(orig_x)
    y_title = _humanize_field_name(orig_y)

    if "histogram" in tool_name:
        values = [_to_float(item.get(orig_y)) for item in sorted_data]
        histogram_data = [value for value in values if value is not None]
        if not histogram_data:
            raise ValueError("No numeric values available for histogram")
        return {"data": histogram_data, "x_title": y_title, "y_title": "Frequency"}

    transformed: List[Dict[str, Any]] = []
    for idx, item in enumerate(sorted_data):
        raw_x_val = item.get(orig_x, "")
        raw_y_val = item.get(orig_y, 0)
        clean_y_float = _to_float(raw_y_val)
        if clean_y_float is None:
            continue

        x_str = str(raw_x_val)
        if "scatter" in tool_name:
            x_num = _to_float(raw_x_val)
            if x_num is None:
                x_num = float(idx + 1)
            transformed.append({"x": x_num, "y": clean_y_float})
        elif "word" in tool_name:
            transformed.append({"text": x_str, "value": max(clean_y_float, 1.0)})
        elif "pie" in tool_name or "bar" in tool_name or "column" in tool_name:
            transformed.append({"category": x_str, "value": clean_y_float})
        elif "radar" in tool_name:
            transformed.append({"name": x_str, "value": clean_y_float})
        else:
            transformed.append({"time": x_str, "value": clean_y_float})

    if not transformed:
        raise ValueError("No valid numeric points available for chart")

    return {"data": transformed, "x_title": x_title, "y_title": y_title}


def _pick_x_field(sample_row: Dict[str, Any], preferred: str = "") -> str:
    if preferred and preferred in sample_row:
        return preferred
    for field in ("ts", "time", "report_time", "write_time", "start_time", "end_time"):
        if field in sample_row:
            return field
    return preferred or (next(iter(sample_row.keys())) if sample_row else "ts")


def _pick_y_field(sample_row: Dict[str, Any], user_req: str, preferred: str = "") -> str:
    if preferred and preferred in sample_row:
        return preferred

    keyword_candidates = [
        ("电压", "voltage"),
        ("soc", "soc"),
        ("soh", "soh"),
        ("电流", "current"),
        ("功率", "power"),
        ("温度", "temp"),
    ]
    req_lower = (user_req or "").lower()
    for keyword, field in keyword_candidates:
        if keyword in user_req or keyword in req_lower:
            if field in sample_row:
                return field

    for field in ("voltage", "soc", "soh", "current", "power", "temp", "vmax", "vmin", "tmax", "tmin"):
        if field in sample_row:
            return field
    return preferred


def _build_chart_intent_from_db_rows(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_rows = state.get("db_raw_results") or []
    if not isinstance(raw_rows, list):
        return None

    valid_rows = [r for r in raw_rows if isinstance(r, dict) and not r.get("error")]
    if not valid_rows:
        return None

    params = state.get("supervisor_params") or {}
    if not isinstance(params, dict):
        params = {}

    sample_row = valid_rows[0]
    pref_x = _normalize_field_name(params.get("x_field"))
    pref_y = _normalize_field_name(params.get("y_field"))
    x_field = _pick_x_field(sample_row, preferred=pref_x)
    y_field = _pick_y_field(sample_row, user_req=str(state.get("user_request", "")), preferred=pref_y)
    if not x_field or not y_field:
        return None

    data = []
    for row in valid_rows:
        x_val = row.get(x_field)
        y_val = _to_float(row.get(y_field))
        if x_val in (None, "") or y_val is None:
            continue
        data.append({x_field: x_val, y_field: y_val})

    if not data:
        return None

    return {
        "needs_chart": True,
        "chart_type": _normalize_chart_type(params.get("chart_type", "line_chart")),
        "title": params.get("title") or f"{y_field} trend",
        "data": data,
        "x_field": x_field,
        "y_field": y_field,
    }


def _call_mcp_chart(intent_data: Dict[str, Any]) -> Dict[str, str]:
    chart_log = ""
    chart_url_result = ""
    raw_data = intent_data.get("data") or intent_data.get("data_json")
    if not isinstance(raw_data, list):
        raise ValueError("Data is not a list")

    orig_x = intent_data.get("x_field")
    orig_y = intent_data.get("y_field")
    if (not orig_x or not orig_y) and len(raw_data) > 0:
        keys = list(raw_data[0].keys())
        if len(keys) >= 2:
            orig_x = keys[0]
            orig_y = keys[1]
    if not orig_x or not orig_y:
        raise ValueError("Missing x/y field for chart")

    chart_type_raw = _normalize_chart_type(intent_data.get("chart_type", "line_chart"))
    tool_name = chart_type_raw if chart_type_raw.startswith("generate_") else f"generate_{chart_type_raw}"
    mapped = _transform_data_for_tool(raw_data, orig_x, orig_y, tool_name)
    x_title = mapped["x_title"]
    y_title = mapped["y_title"]

    print(f"[Chart Node] Mapping User Data for {tool_name}: {orig_x} -> {x_title}, {orig_y} -> {y_title}", flush=True)

    mcp_args = _build_common_chart_args(intent_data, tool_name, x_title, y_title)
    mcp_args["data"] = mapped["data"]

    if "pie" in tool_name:
        mcp_args["innerRadius"] = intent_data.get("innerRadius", 0.58)
    elif "bar" in tool_name:
        mcp_args["group"] = False
        mcp_args["stack"] = False
    elif "column" in tool_name:
        mcp_args["group"] = False
        mcp_args["stack"] = False
    elif "area" in tool_name:
        mcp_args["stack"] = False
    elif "histogram" in tool_name:
        mcp_args["binNumber"] = intent_data.get("binNumber", min(12, max(6, len(mapped["data"]) // 8)))

    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": mcp_args
        },
        "id": 1
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }

    print(f"[Chart Node] Sending data to MCP ({tool_name})...", flush=True)
    response = requests.post(MCP_POST_ENDPOINT, json=payload, headers=headers, timeout=60)

    url_result = ""
    if response.status_code == 200:
        try:
            resp_json = response.json()
            if "result" in resp_json and "content" in resp_json["result"]:
                for c in resp_json["result"]["content"]:
                    if c["type"] == "text":
                        url_result = c["text"]
                        break
            elif "error" in resp_json:
                print(f"[Chart Node] MCP Error Body: {resp_json['error']}", flush=True)
        except Exception:
            print(f"[Chart Node] Non-JSON Response: {response.text[:100]}", flush=True)
    else:
        print(f"[Chart Node] HTTP Error {response.status_code}: {response.text}", flush=True)

    if url_result:
        print(f"[Chart Node] Success URL: {url_result}", flush=True)
        figure_dir = os.path.join(os.getcwd(), "figure")
        os.makedirs(figure_dir, exist_ok=True)

        safe_title = re.sub(r'[\\/*?:"<>|]', "", intent_data.get("title", "chart")).replace(" ", "_")
        file_name = f"{safe_title}_{int(time.time())}.png"
        dest_path = os.path.join(figure_dir, file_name)

        saved = False
        if url_result.startswith("http"):
            try:
                r = requests.get(url_result, timeout=30)
                if r.status_code == 200:
                    with open(dest_path, "wb") as f:
                        f.write(r.content)
                    saved = True
            except Exception as e:
                print(f"Download failed: {e}")
        elif os.path.exists(url_result):
            shutil.copy(url_result, dest_path)
            saved = True

        final_path = dest_path if saved else url_result
        chart_url_result = final_path
        chart_log = f"\n\n![Chart]({final_path})\n*Chart Generated: {intent_data.get('title')}*\n"
    else:
        chart_log = "\n[Chart Error] Valid chart URL not received."

    return {"chart_log": chart_log, "chart_url_result": chart_url_result}


# =============================================================================
# [核心] 图表生成节点逻辑
# =============================================================================
def generate_chart_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- Executing Node: generate_chart_node (Chart from structured DB rows first) ---", flush=True)

    user_req = state.get("user_request", "")
    retrieved = state.get("pre_brief_cases", "")
    context_preview = retrieved[:5000] if retrieved else "No retrieved data."
    raw_db_results = state.get("db_raw_results", [])

    chart_log = ""
    chart_url_result = ""
    intent_data = _build_chart_intent_from_db_rows(state)

    # 没有结构化明细时，才退回到从文本上下文中抽取绘图数据
    if intent_data is None:
        json_example = """
        {
            "needs_chart": true,
            "chart_type": "line_chart",
            "title": "Data Trend",
            "data": [{"label": "2020", "val": 100}, {"label": "2021", "val": 150}],
            "x_field": "label",
            "y_field": "val"
        }
        """

        system_inst = (
            "You are a Data Visualization Assistant. Extract data to generate a chart. "
            "Return STRICT JSON only.\n"
            f"Example:\n{json_example}\n"
            "RULES:\n"
            "1. 'data' must be a RAW JSON Array.\n"
            "2. Identify 'x_field' (category/time) and 'y_field' (numerical).\n"
        )

        user_prompt = (
            f"User Request: {user_req}\n\n"
            f"Context: {context_preview}\n\n"
            "Extract data. Return valid JSON."
        )

    try:
        if intent_data is None:
            raw_text, _ = gemini_chat_once(user_prompt, system_inst, temperature=0.1)
            clean_json = raw_text.strip()
            clean_json = re.sub(r"^```[a-zA-Z]*\n", "", clean_json)
            clean_json = re.sub(r"\n```$", "", clean_json)

            match = re.search(r"(\{.*\})", clean_json, re.DOTALL)
            if match:
                clean_json = match.group(1)

            data_dict = json.loads(clean_json)
            if "data" in data_dict and isinstance(data_dict["data"], str):
                try:
                    data_dict["data"] = json.loads(data_dict["data"])
                except Exception:
                    pass
            intent_data = data_dict

        if intent_data and intent_data.get("needs_chart"):
            render_result = _call_mcp_chart(intent_data)
            chart_log = render_result["chart_log"]
            chart_url_result = render_result["chart_url_result"]
        else:
            chart_log = "\n[Chart Error] No chartable data was found in current context."

    except Exception as e:
        print(f"[Chart Node] Execution Error: {e}", flush=True)
        chart_log = f"\n[Chart Error]: {str(e)}"

    updated_cases = state.get("pre_brief_cases", "") + chart_log

    return {
        "chart_output": chart_url_result,
        "pre_brief_cases": updated_cases,
        "db_raw_results": raw_db_results if isinstance(raw_db_results, list) else [],
        "supervisor_messages": ["Chart generation attempt finished."]
    }



# =============================================================================
# Tool Definition (Required for LangChain)
# =============================================================================
@tool
def generate_chart_url(chart_type: str, data: list, x_field: str, y_field: str, title: str = "") -> str:
    """
    Generates a chart and returns an image URL.
    
    Args:
        chart_type: One of 'line_chart', 'bar_chart', 'pie_chart', 'area_chart'.
        data: A list of dictionaries.
        x_field: The key for X axis.
        y_field: The key for Y axis.
        title: The title of the chart.
    """
    return "Please use the graph node execution path."
