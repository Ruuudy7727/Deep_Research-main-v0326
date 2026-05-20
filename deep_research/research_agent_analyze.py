import asyncio
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    from .sql_parser import getMySqlData
except ImportError:
    from deep_research.sql_parser import getMySqlData

try:
    from .sql_parser_td import getTdSqlData  # type: ignore
except Exception:
    getTdSqlData = None

from .gemini_chat import gemini_chat_once
from .prompts import (
    db_intent_router_prompt,
    db_query_planner_prompt,
    db_evidence_summarizer_prompt,
)
from .state_scope import AgentState
from .clarify_route import (
    assess_locked_route_readiness,
    build_locked_route_json,
    normalize_clarify_route,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_LIMIT = 500
DEFAULT_LIMIT = 80
DEFAULT_TIME_WINDOW_HOURS = 24
MAX_TIME_WINDOW_DAYS = 7
DEFAULT_DB_SANITIZER_LOG_PATH = _PROJECT_ROOT / "log" / "db_plan_sanitizer.jsonl"
DEFAULT_DB_CHAIN_LOG_PATH = _PROJECT_ROOT / "log" / "db_chain.jsonl"

TABLE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "alarm_event": {
        "aliases": ["alarm_event", "alarm_events"],
        "fields": {
            "id", "station_code", "bmu_id", "bmu_code", "cell_id", "source",
            "start_time", "end_time", "feature_type", "n_alert_days", "days_span",
            "alert_frequency", "grade", "n_alerts", "average_severity",
            "alert_statistic", "summary_cn", "status", "event_category",
        },
        # 与线上一致：用 start_time + end_time 做窗口；勿使用不存在的 report_time
        "time_fields": ["start_time", "end_time"],
        "time_range_mode": "start_end",
        "default_select": [
            "id", "station_code", "bmu_id", "bmu_code", "cell_id",
            "grade", "average_severity", "summary_cn", "start_time", "end_time",
        ],
    },
    "volt_temp_abnormal_result": {
        "aliases": ["volt_temp_abnormal_result"],
        "fields": {
            "id", "time", "bmu_code", "operation_condition", "type",
            "v_max", "v_min", "v_mean", "delta_v", "t_max", "t_min", "t_mean", "delta_t",
            "abnormal_voltage_cells", "abnormal_temp_cells", "write_time", "volt_value", "temp_value",
        },
        "time_fields": ["time", "write_time"],
        "default_select": [
            "time", "bmu_code", "operation_condition", "type", "delta_v", "delta_t",
            "v_max", "v_min", "t_max", "t_min", "abnormal_voltage_cells", "abnormal_temp_cells",
        ],
    },
    "capacity_inconsistent_cells": {
        "aliases": ["capacity_inconsistent_cells"],
        "fields": {
            "id", "bmu_code", "cell_index", "first_occurrence", "last_occurrence",
            "abnormal_reason", "confidence_score", "base_confidence", "direction", "has_self_discharge",
            "max_voltage_drop_rate_mvh", "self_discharge_count", "occurrence_count", "threshold_count",
            "risk_threshold_count", "main_abnormal_count", "risk_warning_count", "max_voltage_diff",
            "max_voltage_deviation_mv", "abnormal_types", "phases", "threshold_types", "reasons",
            "details_count", "write_time",
        },
        "time_fields": ["first_occurrence", "last_occurrence", "write_time"],
        # 点查 bmu+cell 时，默认 24h 窗会过滤掉 first_occurrence 在窗口外的历史行；无显式 time_filter 时不加时间条件
        "skip_time_window_when_identities": ["bmu_code", "cell_index"],
        "default_select": [
            "bmu_code", "cell_index", "direction", "has_self_discharge", "max_voltage_drop_rate_mvh",
            "risk_warning_count", "confidence_score", "abnormal_reason", "first_occurrence", "last_occurrence",
        ],
    },
    "dcr_abnormal_cells": {
        "aliases": ["dcr_abnormal_cells"],
        "fields": {
            "id", "bmu_code", "cell_index", "data_days", "abnormal_days", "period_r0_median_ohm",
            "period_iqr_ohm", "first_abnormal_time", "last_abnormal_time", "write_time",
        },
        "time_fields": ["first_abnormal_time", "last_abnormal_time", "write_time"],
        "skip_time_window_when_identities": ["bmu_code", "cell_index"],
        "default_select": [
            "bmu_code", "cell_index", "abnormal_days", "period_r0_median_ohm",
            "first_abnormal_time", "last_abnormal_time", "write_time",
        ],
    },
    "isc_score_result": {
        "aliases": ["isc_score_result"],
        "fields": {
            "id", "bmu_code", "window_id", "window_start", "window_end", "cell",
            "rank_bias_delta_r", "median_rank_charge", "median_rank_discharge", "hysteresis_area",
            "charging_ratio", "discharging_ratio", "r0_median_median",
            "s_rank_charge", "s_rank_discharge", "s_hysteresis_center", "s_charging_ratio",
            "s_discharging_ratio", "s_r0_low",
            "microshort_score", "microshort_score_pct", "volt_score", "capacity_score",
            "polarize_score", "resistance_score", "write_time", "diagnosis_result",
        },
        "time_fields": ["window_start", "window_end", "write_time"],
        "time_range_mode": "window_start_end",
        "default_select": [
            "bmu_code", "window_id", "cell", "microshort_score", "microshort_score_pct",
            "diagnosis_result", "window_start", "window_end", "write_time",
        ],
    },
    "box_data": {
        "aliases": ["box_data"],
        "fields": {
            "ts", "write_time", "report_time",
            "discharged_energy", "charged_energy", "day_discharged_energy", "day_charged_energy",
            "dischargable_energy", "chargable_energy", "power", "limit_chg_power", "limit_dchg_power",
            "soc", "online_state", "work_state",
            "vmax", "vmax_bms_idx", "vmax_bc_idx", "vmax_bmu_idx", "vmax_cell_idx",
            "vmin", "vmin_bms_idx", "vmin_bc_idx", "vmin_bmu_idx", "vmin_cell_idx",
            "cell_avg_vol", "vol_dif_max",
            "tmax", "tmax_bms_idx", "tmax_bc_idx", "tmax_bmu_idx", "tmax_cell_idx",
            "tmin", "tmin_bms_idx", "tmin_bc_idx", "tmin_bmu_idx", "tmin_cell_idx",
            "cell_avg_temp", "temp_dif_max",
            "imax", "imax_bms_idx", "imax_bc_idx", "imax_bmu_idx", "imax_cell_idx",
            "imin", "imin_bms_idx", "imin_bc_idx", "imin_bmu_idx", "imin_cell_idx",
            "cell_avg_cur",
            "soc_max", "soc_max_bms_idx", "soc_max_bc_idx",
            "soc_min", "soc_min_bms_idx", "soc_min_bc_idx",
            "box_code", "station_code", "box_type",
        },
        "time_fields": ["ts", "write_time", "report_time"],
        "default_select": [
            "ts", "box_code", "station_code", "box_type",
            "soc", "power", "charged_energy", "discharged_energy",
            "vmax", "vmin", "tmax", "tmin", "cell_avg_vol", "cell_avg_temp",
        ],
    },
    "cluster_data": {
        "aliases": ["cluster_data"],
        "fields": {
            "ts", "report_time", "write_time", "soc", "soh", "voltage", "current",
            "left_energy", "used_energy", "input_ah", "output_ah", "warn_state", "prt_state",
            "input_watt", "output_watt", "day_input_watt", "day_output_watt",
            "vmin", "vmin_bmu_idx", "vmin_cell_idx", "vmax", "vmax_bmu_idx", "vmax_cell_idx",
            "tmin", "tmin_bmu_idx", "tmin_cell_idx", "tmax", "tmax_bmu_idx", "tmax_cell_idx",
            "vmin_sec", "vmin_sec_bmu_idx", "vmin_sec_cell_idx",
            "vmax_sec", "vmax_sec_bmu_idx", "vmax_sec_cell_idx",
            "tmin_sec", "tmin_sec_bmu_idx", "tmin_sec_cell_idx",
            "tmax_sec", "tmax_sec_bmu_idx", "tmax_sec_cell_idx",
            "bmu_online_state", "run_state", "bc_prt_act_state", "brk_fault_state",
            "sa_online_state", "bc_fault_state", "balance_state", "balance_en",
            "brk_state", "bc_sys_state", "bmu_volt_sum", "power",
            "bcms_online_state", "bcu_online_state", "main_breaker_state",
            "balance_mode", "balance_manual_mode_en", "fan_state", "fan_fault",
            "limit_power_state", "limit_chg_power", "limit_dchg_power",
            "bmu_avg_vol", "cell_avg_vol", "cell_avg_temp", "vol_dif_max", "temp_dif_max",
            "soe", "sop", "sof", "pbus_resistor", "nbus_resistor",
            "correct", "temp", "once_input_watt", "once_output_watt",
            "tmax_brz", "tmax_brz_idx", "tmin_brz", "tmin_brz_idx",
            "max_bmu_volt", "max_bmu_volt_idx", "min_bmu_volt", "min_bmu_volt_idx",
            "max_bmu_volt_dif", "avg_bmu_volt", "imax", "imax_bmu_idx", "imax_cell_idx",
            "imin", "imin_bmu_idx", "imin_cell_idx", "bc_self_chk_state",
            "cluster_code", "bms_code", "box_code", "station_code",
        },
        "time_fields": ["ts", "write_time", "report_time"],
        "default_select": [
            "ts", "cluster_code", "bms_code", "box_code", "station_code",
            "soc", "soh", "voltage", "current", "power", "input_watt", "output_watt",
        ],
    },
    "bmu_data": {
        "aliases": ["bmu_data"],
        "fields": {
            "ts", "order_no", "report_time", "write_time",
            "bmu_code", "cluster_code", "bms_code", "box_code", "station_code",
            "soc", "soh", "voltage", "current", "bcurrent",
            "online_state", "temp", "cur_data", "temp_data", "volt_data",
            "balance_no", "fan_ctrl_pwm", "fan_ctrl_state", "balance_state",
            "cell_avg_vol", "cell_avg_temp", "vmax", "vmin", "vmax_idx", "vmin_idx",
            "tmax", "tmin", "tmax_idx", "tmin_idx", "bmu_balances",
            "imax", "imax_idx", "imin", "imin_idx",
        },
        "time_fields": ["ts", "write_time", "report_time"],
        "default_select": [
            "ts", "bmu_code", "cluster_code", "bms_code", "box_code", "station_code",
            "soc", "soh", "voltage", "cell_avg_vol", "cell_avg_temp", "tmax", "tmin", "vmax", "vmin",
        ],
    },
}

ROUTE_TABLE_ALLOWLIST: Dict[str, set] = {
    "alerting": {"alarm_event", "volt_temp_abnormal_result"},
    "troubleshooting": {
        "capacity_inconsistent_cells",
        "dcr_abnormal_cells",
        "isc_score_result",
        "volt_temp_abnormal_result",
    },
    "station_device_td": {"box_data", "cluster_data", "bmu_data"},
}

ROUTE_TABLE_DEFAULT: Dict[str, str] = {
    "alerting": "alarm_event",
    "troubleshooting": "capacity_inconsistent_cells",
    "station_device_td": "cluster_data",
}

# LLM/文档历史别名 -> 线上一致列名（须在对应表的 fields 中存在）
DB_FIELD_ALIASES: Dict[str, Dict[str, str]] = {
    "capacity_inconsistent_cells": {
        "maxvoltage_drop_rate_mvh": "max_voltage_drop_rate_mvh",
    },
    "isc_score_result": {
        "microshortscore": "microshort_score",
        "microshortscore_pct": "microshort_score_pct",
        "cell_id": "cell",
        "cell_index": "cell",
    },
    "volt_temp_abnormal_result": {
        "ts": "time",
        "report_time": "time",
    },
    "bmu_data": {
        "pcurrent": "bcurrent",
    },
}


def _apply_db_field_alias(table_name: str, field: str) -> str:
    m = DB_FIELD_ALIASES.get(table_name, {})
    return m.get(field, field)


# 前端指标卡片依赖字段（用于 SQL select_fields 自动补齐）
# 目标：即使 LLM 规划的 select_fields 合法但不完整，也能确保卡片所需字段被查询。
CARD_REQUIRED_SELECT_FIELDS: Dict[str, List[str]] = {
    "bmu_data": [
        "soc", "soh", "voltage", "current", "temp", "cell_avg_temp",
    ],
    "alarm_event": [
        "average_severity",
    ],
    "dcr_abnormal_cells": [
        "data_days", "abnormal_days", "period_r0_median_ohm",
        "period_iqr_ohm", "first_abnormal_time", "last_abnormal_time",
    ],
    "isc_score_result": [
        "microshort_score", "microshort_score_pct", "diagnosis_result",
    ],
    "capacity_inconsistent_cells": [
        "direction", "has_self_discharge", "max_voltage_drop_rate_mvh",
        "risk_warning_count", "confidence_score",
    ],
    "volt_temp_abnormal_result": [
        "delta_v", "delta_t", "v_max", "v_min", "t_max", "t_min",
    ],
}


def _extract_json_object_str(text: str) -> str:
    """从 LLM 输出中截取第一个平衡花括号 JSON 对象，避免贪婪正则把多段内容拼进导致解析失败。"""
    if not text:
        return ""
    s = text.strip().replace("```json", "").replace("```", "").strip()
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


def _extract_first_json(text: str) -> Dict[str, Any]:
    if not text or not str(text).strip():
        raise ValueError("empty LLM text")
    s = str(text)
    obj_str = _extract_json_object_str(s)
    if obj_str:
        return json.loads(obj_str)
    cleaned = s.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(cleaned)


def _normalize_dedup_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _row_dedup_key(row: Dict[str, Any]) -> Any:
    if not isinstance(row, dict) or row.get("error"):
        return ("__err__", str(row.get("error", "unknown")))

    table_name = _normalize_dedup_value(row.get("table_name", ""))
    has_event_window = any(
        _normalize_dedup_value(row.get(field, ""))
        for field in ("summary_cn", "start_time", "end_time")
    )
    if has_event_window:
        sc = _normalize_dedup_value(row.get("summary_cn", ""))[:4000]
        return (
            "event_window",
            table_name,
            _normalize_dedup_value(row.get("bmu_code", "")),
            _normalize_dedup_value(row.get("cell_id", "")),
            _normalize_dedup_value(row.get("start_time", "")),
            _normalize_dedup_value(row.get("end_time", "")),
            sc,
        )

    # 时序/指标表退回到“整行精确去重”，避免不同 ts 的采样点被误判为重复。
    row_items = tuple(
        (key, _normalize_dedup_value(row.get(key)))
        for key in sorted(row.keys())
        if key != "table_name"
    )
    return ("row_exact", table_name, row_items)


def _deduplicate_result_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    if not rows:
        return [], 0
    seen = set()
    out: List[Dict[str, Any]] = []
    dup = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        k = _row_dedup_key(r)
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        out.append(r)
    return out, dup


def _rebuild_per_table(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    d: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("table_name", "unknown"))
        d.setdefault(t, []).append(r)
    return d


def _rule_based_evidence_bundle(
    route: str,
    user_req: str,
    sample_rows: List[Dict[str, Any]],
    raw_n: int,
    dedup_n: int,
) -> Dict[str, Any]:
    """LLM 证据融合失败时的结构化兜底（偏 alarm_event 的 summary_cn 文本）。"""
    snippets: List[str] = []
    for r in sample_rows[:50]:
        if not isinstance(r, dict):
            continue
        s = r.get("summary_cn")
        if s:
            t = str(s).replace("\n", " ").strip()
            if t and t not in snippets:
                snippets.append(t[:500])
    term_hits: Counter = Counter()
    for sn in snippets:
        for m in re.findall(r"[\(（]([^)）]+)[\)）].*?[-高低调报报]+", sn):
            m = m.strip()
            if len(m) > 1 and m not in {"容量", "电压", "温度"}:
                term_hits[m] += 1
        for m2 in re.findall(r"(放电工况|充电工况|静置工况)[：:][^;；\n]+", sn):
            term_hits[m2[:80]] += 1
    top_terms = [t for t, _ in term_hits.most_common(8)]
    return {
        "route_used": route,
        "alarm_summary": {
            "risk_level": "未知",
            "high_freq_terms": top_terms,
            "repeat_patterns": [f"查询原始行数 {raw_n}，去重后 {dedup_n} 条；摘要模板重复时请结合 cell 粒度下钻。"],
            "key_points": snippets[:5],
        },
        "abnormal_topn": [],
        "td_metrics_summary": [],
        "station_td_query_purpose": "unknown",
        # 留空以免覆盖前端基于 alarm_summary 的用户向结论（JSON 解析失败见 db_llm_traces）
        "diagnosis_conclusion": "",
        "evidence_sufficiency": "一般",
        "next_action_suggestion": "若需更准的关键词统计，可缩小时间窗或加 cell 条件后重试。",
        "mcp_chart_hint": "",
        "confidence": 0.35,
    }


def _escape_sql_literal(value: str) -> str:
    return str(value).replace("'", "''")


class LlmJsonParseError(ValueError):
    """LLM 输出经 `_extract_first_json` 失败时抛出，并带上原始文本便于日志与重试后诊断。"""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw or ""


_SQL_FILTER_OPERATORS = frozenset({">", ">=", "<", "<=", "=", "!=", "<>"})


def _filter_value_to_sql_condition(field: str, value: Any) -> Optional[str]:
    """
    将 plan.filters 中的一项转为 WHERE 子句中的单个条件。
    支持标量等值，以及规划器产出的 { \"operator\": \">\", \"value\": 5 } 形比较条件。
    """
    if isinstance(value, dict) and "operator" in value and "value" in value:
        op = str(value.get("operator", "")).strip()
        if op not in _SQL_FILTER_OPERATORS:
            return None
        inner = value.get("value")
        if inner is None:
            return None
        if isinstance(inner, bool):
            return None
        if isinstance(inner, (int, float)):
            return f"{field} {op} {inner}"
        s = _normalize_text(inner)
        if s is None:
            return None
        return f"{field} {op} '{_escape_sql_literal(s)}'"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{field} = {value}"

    val = _normalize_text(value)
    if val is None:
        return None
    return f"{field} = '{_escape_sql_literal(val)}'"


def _safe_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(low, min(parsed, high))


def _normalize_text(value: Any) -> Any:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt or txt.lower() in {"null", "none", "nan"}:
        return None
    return txt


def _resolve_table_name(table_name: str) -> str:
    t = (table_name or "").strip()
    for canonical, schema in TABLE_SCHEMAS.items():
        if t in schema["aliases"]:
            return canonical
    return t


def _pick_time_field(schema: Dict[str, Any], preferred: Any) -> Any:
    if preferred and preferred in schema["fields"]:
        return preferred
    fields = schema.get("time_fields") or []
    return fields[0] if fields else None


def _time_filter_unspecified(tf: Dict[str, Any]) -> bool:
    """规划未给出起止时间（下游会默认 24h）；与显式窗口区分。"""
    return _normalize_text(tf.get("start_time")) is None and _normalize_text(tf.get("end_time")) is None


def _has_skip_time_identities(schema: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    keys = schema.get("skip_time_window_when_identities") or []
    if not keys:
        return False
    for k in keys:
        if k not in filters:
            return False
        if _normalize_text(filters.get(k)) is None:
            return False
    return True


def _normalize_time_range(start_time: Any, end_time: Any) -> Dict[str, str]:
    def _parse_now_token(raw: Any) -> Any:
        txt = _normalize_text(raw)
        if txt is None:
            return None
        if txt.lower() in ("now", "utcnow", "utc_now", "current"):
            return datetime.utcnow()
        return None

    def _parse_dt(raw: Any) -> Any:
        n = _parse_now_token(raw)
        if n is not None:
            return n
        txt = _normalize_text(raw)
        if txt is None:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
            try:
                return datetime.strptime(txt, fmt)
            except ValueError:
                continue
        return None

    def _parse_relative_start(raw: Any, anchor_end: datetime) -> Any:
        """解析规划器侧相对时间，如 P-1W、P-7D、P-24H（锚点为 end）。"""
        txt = _normalize_text(raw)
        if not txt:
            return None
        t = txt.strip().upper()
        m = re.match(r"^P-(\d+)W$", t)
        if m:
            return anchor_end - timedelta(weeks=int(m.group(1)))
        m = re.match(r"^P-(\d+)D$", t)
        if m:
            return anchor_end - timedelta(days=int(m.group(1)))
        m = re.match(r"^P-(\d+)H$", t)
        if m:
            return anchor_end - timedelta(hours=int(m.group(1)))
        return None

    parsed_end = _parse_dt(end_time)
    has_explicit_end = parsed_end is not None

    end_dt = parsed_end or datetime.utcnow()
    # 先按绝对时间解析起点；再尝试相对 P-*（依赖 end_dt 锚点）
    parsed_start = _parse_dt(start_time)
    if parsed_start is None and _normalize_text(start_time):
        parsed_start = _parse_relative_start(start_time, end_dt)
    has_explicit_start = parsed_start is not None

    start_dt = parsed_start or (end_dt - timedelta(hours=DEFAULT_TIME_WINDOW_HOURS))

    if end_dt < start_dt:
        start_dt, end_dt = end_dt - timedelta(hours=DEFAULT_TIME_WINDOW_HOURS), end_dt

    # 仅在缺省时间（用户未明确给全起止）时限制窗口，避免用户明确查询被意外裁剪
    if (not has_explicit_start or not has_explicit_end) and (end_dt - start_dt).days > MAX_TIME_WINDOW_DAYS:
        start_dt = end_dt - timedelta(days=MAX_TIME_WINDOW_DAYS)
    return {
        "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _get_db_sanitizer_log_path(state: Optional[AgentState] = None) -> Path:
    if isinstance(state, dict):
        s = str(state.get("db_plan_sanitizer_log_path", "") or "").strip()
        if s:
            return Path(s)
    raw = os.getenv("DB_PLAN_SANITIZER_LOG_PATH", "").strip()
    if raw:
        return Path(raw)
    return DEFAULT_DB_SANITIZER_LOG_PATH


def _get_db_chain_log_path(state: Optional[AgentState] = None) -> Path:
    if isinstance(state, dict):
        s = str(state.get("db_chain_log_path", "") or "").strip()
        if s:
            return Path(s)
    raw = os.getenv("DB_CHAIN_LOG_PATH", "").strip()
    if raw:
        return Path(raw)
    return DEFAULT_DB_CHAIN_LOG_PATH


def _write_jsonl(path: Path, event: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _db_sanitizer_log_enabled(state: Optional[AgentState] = None) -> bool:
    if isinstance(state, dict) and "db_sanitizer_log_enabled" in state:
        return bool(state.get("db_sanitizer_log_enabled"))
    return str(os.getenv("DB_SANITIZER_LOG_ENABLED", "0") or "0").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }


def _write_db_sanitizer_log(state: Optional[AgentState], event: Dict[str, Any]) -> None:
    if not _db_sanitizer_log_enabled(state):
        return
    try:
        path = _get_db_sanitizer_log_path(state)
        _write_jsonl(path, event)
    except Exception as e:
        print(f"[DB_PLAN_SANITIZER_LOG_ERROR] {e}", flush=True)


def _write_db_chain_log(state: Optional[AgentState], event: Dict[str, Any]) -> None:
    try:
        path = _get_db_chain_log_path(state)
        _write_jsonl(path, event)
    except Exception as e:
        print(f"[DB_CHAIN_LOG_ERROR] {e}", flush=True)


def _sanitize_plan(plan: Dict[str, Any], route: str) -> Dict[str, Any]:
    warnings: List[str] = []
    raw_table = str(plan.get("table", "")).strip()
    table_name = _resolve_table_name(raw_table)
    schema = TABLE_SCHEMAS.get(table_name)
    if not schema:
        raise ValueError(f"Unsupported table: {table_name}")

    allow_tables = ROUTE_TABLE_ALLOWLIST.get(route, set(TABLE_SCHEMAS.keys()))
    if table_name not in allow_tables:
        fallback_table = ROUTE_TABLE_DEFAULT.get(route, table_name)
        warnings.append(f"table_not_allowed_for_route:{table_name}->{fallback_table}")
        table_name = fallback_table
        schema = TABLE_SCHEMAS[table_name]

    raw_select = plan.get("select_fields")
    if not isinstance(raw_select, list):
        raw_select = schema["default_select"]
        warnings.append("select_fields_invalid_type:use_default")
    raw_select_norm: List[str] = []
    for f in raw_select:
        if not isinstance(f, str):
            continue
        c = _apply_db_field_alias(table_name, f)
        if c != f:
            warnings.append(f"select_field_alias:{f}->{c}")
        raw_select_norm.append(c)
    valid_select = [f for f in raw_select_norm if f in schema["fields"]]
    valid_set = set(valid_select)
    dropped_select: List[Any] = []
    for f in raw_select:
        if not isinstance(f, str):
            dropped_select.append(f)
        elif _apply_db_field_alias(table_name, f) not in valid_set:
            dropped_select.append(f)
    if dropped_select:
        warnings.append(f"select_fields_dropped:{dropped_select}")
    if not valid_select:
        valid_select = schema["default_select"]
        warnings.append("select_fields_empty_after_filter:use_default")

    # 卡片字段兜底补齐：防止 select_fields 合法但不包含前端指标卡片依赖字段。
    required_select = CARD_REQUIRED_SELECT_FIELDS.get(table_name, [])
    appended_for_cards: List[str] = []
    for field in required_select:
        cf = _apply_db_field_alias(table_name, field)
        if cf in schema["fields"] and cf not in valid_select:
            valid_select.append(cf)
            appended_for_cards.append(cf)
    if appended_for_cards:
        warnings.append(f"select_fields_auto_appended_for_cards:{appended_for_cards}")

    raw_filters = plan.get("filters")
    if not isinstance(raw_filters, dict):
        raw_filters = {}
        warnings.append("filters_invalid_type:use_empty")
    valid_filters: Dict[str, Any] = {}
    dropped_filters: List[str] = []
    for field, value in raw_filters.items():
        fk = str(field) if field is not None else ""
        cf = _apply_db_field_alias(table_name, fk) if fk else fk
        if fk and cf != fk:
            warnings.append(f"filter_field_alias:{fk}->{cf}")
        if not cf or cf not in schema["fields"]:
            if fk:
                dropped_filters.append(fk)
            continue
        valid_filters[cf] = value
    if dropped_filters:
        warnings.append(f"filters_dropped:{dropped_filters}")

    raw_tf = plan.get("time_filter")
    if not isinstance(raw_tf, dict):
        raw_tf = {}
        warnings.append("time_filter_invalid_type:use_empty")
    preferred_time_field = raw_tf.get("time_field")
    if table_name == "alarm_event" and preferred_time_field in ("report_time", "time", "ts", "write_time"):
        preferred_time_field = "start_time"
        warnings.append("time_field_alias:legacy->start_time")
    if table_name == "isc_score_result" and preferred_time_field in (
        "report_time", "ts", "time", "start_time", "end_time", "",
    ):
        # 规划器若误用通用时间名，SQL 仍由 window_start/window_end 子句处理
        preferred_time_field = None
        warnings.append("time_field_alias:legacy->null_for_isc_window")
    picked_time_field = _pick_time_field(schema, preferred_time_field)
    if preferred_time_field and preferred_time_field != picked_time_field:
        warnings.append(f"time_field_adjusted:{preferred_time_field}->{picked_time_field}")
    valid_time_filter = {
        "time_field": picked_time_field,
        "start_time": raw_tf.get("start_time"),
        "end_time": raw_tf.get("end_time"),
    }

    raw_order = plan.get("order_by")
    if not isinstance(raw_order, list):
        raw_order = []
        warnings.append("order_by_invalid_type:use_empty")
    valid_order: List[Dict[str, Any]] = []
    dropped_order: List[Any] = []
    for item in raw_order:
        if not isinstance(item, dict):
            dropped_order.append(item)
            continue
        field = item.get("field")
        if table_name == "alarm_event" and field == "report_time":
            field = "start_time"
            warnings.append("order_by_field_alias:report_time->start_time")
        if isinstance(field, str):
            nf = _apply_db_field_alias(table_name, field)
            if nf != field:
                warnings.append(f"order_by_field_alias:{field}->{nf}")
            field = nf
        if field not in schema["fields"]:
            dropped_order.append(item)
            continue
        valid_order.append({"field": field, "desc": bool(item.get("desc"))})
    if dropped_order:
        warnings.append(f"order_by_dropped:{dropped_order}")

    limit = _safe_int(plan.get("limit"), DEFAULT_LIMIT, 1, MAX_LIMIT)
    if plan.get("limit") != limit:
        warnings.append(f"limit_adjusted:{plan.get('limit')}->{limit}")

    return {
        "table": table_name,
        "select_fields": valid_select,
        "filters": valid_filters,
        "time_filter": valid_time_filter,
        "order_by": valid_order,
        "limit": limit,
        "_sanitize_warnings": warnings,
    }


def _build_sql_from_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    table_name = _resolve_table_name(str(plan.get("table", "")))
    schema = TABLE_SCHEMAS.get(table_name)
    if not schema:
        raise ValueError(f"Unsupported table: {table_name}")

    select_fields = plan.get("select_fields") or schema["default_select"]
    select_fields = [f for f in select_fields if f in schema["fields"]]
    if not select_fields:
        select_fields = schema["default_select"]
    sql = f"SELECT {', '.join(select_fields)} FROM {table_name}"

    conditions: List[str] = []
    filters = plan.get("filters") or {}
    for field, value in filters.items():
        if field not in schema["fields"]:
            continue
        part = _filter_value_to_sql_condition(field, value)
        if part:
            conditions.append(part)

    tf = plan.get("time_filter") or {}
    rng = _normalize_time_range(tf.get("start_time"), tf.get("end_time"))
    tr_mode = schema.get("time_range_mode")
    time_field = _pick_time_field(schema, tf.get("time_field"))
    skip_simple_time = (
        tr_mode is None
        and _time_filter_unspecified(tf)
        and _has_skip_time_identities(schema, filters)
    )
    if tr_mode == "start_end" and "start_time" in schema["fields"] and "end_time" in schema["fields"]:
        # alarm_event：start_time / end_time 与查询窗口一致
        conditions.append(
            f"start_time >= '{_escape_sql_literal(rng['start'])}' AND end_time <= '{_escape_sql_literal(rng['end'])}'"
        )
    elif tr_mode == "window_start_end" and "window_start" in schema["fields"] and "window_end" in schema["fields"]:
        # isc_score_result：时间窗口落在线段 [window_start, window_end] 上且整体在查询窗内
        conditions.append(
            f"window_start >= '{_escape_sql_literal(rng['start'])}' AND window_end <= '{_escape_sql_literal(rng['end'])}'"
        )
    elif time_field and not skip_simple_time:
        conditions.append(
            f"{time_field} BETWEEN '{_escape_sql_literal(rng['start'])}' AND '{_escape_sql_literal(rng['end'])}'"
        )

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    order_by = plan.get("order_by") or []
    order_clauses = []
    for item in order_by:
        field = item.get("field")
        if field in schema["fields"]:
            desc = bool(item.get("desc"))
            order_clauses.append(f"{field} {'DESC' if desc else 'ASC'}")
    if order_clauses:
        sql += " ORDER BY " + ", ".join(order_clauses)

    limit = _safe_int(plan.get("limit"), DEFAULT_LIMIT, 1, MAX_LIMIT)
    sql += f" LIMIT {limit}"

    return {
        "table": table_name,
        "sql": sql,
        "limit": limit,
        "select_fields": select_fields,
    }


def _df_to_records(table_name: str, df: Any, limit: int) -> List[Dict[str, Any]]:
    if df is None or len(df) == 0:
        return []
    rows: List[Dict[str, Any]] = []
    for _, row in df.head(limit).iterrows():
        item = {k: row.get(k) for k in row.index}
        item["table_name"] = table_name
        rows.append(item)
    return rows


def _clip_preview_value(value: Any, max_chars: int) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        sample_n = 3
        sample = [_clip_preview_value(v, max_chars) for v in value[:sample_n]]
        return {
            "_type": "list",
            "len": len(value),
            "sample": sample,
        }
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 6:
                out["..."] = f"+{len(value) - 6} fields"
                break
            out[str(k)] = _clip_preview_value(v, max_chars)
        return out
    s = str(value)
    if len(s) <= max_chars:
        return value
    return s[:max_chars] + "...(truncated)"


def _clip_for_log(value: Any, max_chars: int = 12000) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) <= max_chars else (value[:max_chars] + "...(truncated)")
    return _clip_preview_value(value, max_chars=max_chars)


_STATION_TD_METRIC_PRIORITY = (
    "soc",
    "soh",
    "voltage",
    "current",
    "power",
    "temp",
    "cell_avg_vol",
    "cell_avg_temp",
    "vmax",
    "vmin",
    "tmax",
    "tmin",
)


def _safe_float_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _format_metric_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 6)


def _infer_station_td_query_purpose(user_req: str) -> str:
    req = str(user_req or "").lower()
    if any(k in req for k in ("画图", "绘图", "图表", "曲线", "趋势图", "可视化", "chart", "plot", "graph")):
        return "charting"
    if any(k in req for k in ("异常", "告警", "故障", "排查", "下钻", "定位", "根因")):
        return "fault_drill_down"
    return "direct_view"


def _pick_station_td_device_label(rows: List[Dict[str, Any]]) -> str:
    for key in ("bmu_code", "cluster_code", "box_code", "bms_code", "station_code"):
        vals = []
        for row in rows:
            v = _normalize_text(row.get(key))
            if v and v not in vals:
                vals.append(v)
        if vals:
            joined = ",".join(vals[:3])
            if len(vals) > 3:
                joined += f"...(+{len(vals) - 3})"
            return f"{key}={joined}"
    return "device=unknown"


def _pick_station_td_time_span(rows: List[Dict[str, Any]]) -> str:
    times: List[str] = []
    for row in rows:
        for key in ("ts", "time", "report_time", "write_time"):
            v = _normalize_text(row.get(key))
            if v:
                times.append(v)
                break
    if not times:
        return "unknown"
    return f"{min(times)} ~ {max(times)}"


def _build_station_td_evidence_bundle(
    user_req: str,
    per_table: Dict[str, List[Dict[str, Any]]],
    all_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    td_metrics_summary: List[Dict[str, Any]] = []
    diagnosis_parts: List[str] = []
    has_any_valid_metric = False
    has_valid_soh = False
    has_valid_voltage = False
    has_all_zero_soc = False

    for table_name, rows in per_table.items():
        valid_rows = [r for r in rows if isinstance(r, dict) and not r.get("error")]
        if not valid_rows:
            continue

        sample_row = valid_rows[0]
        metric_fields = [m for m in _STATION_TD_METRIC_PRIORITY if m in sample_row]
        if not metric_fields:
            continue

        device_label = _pick_station_td_device_label(valid_rows)
        time_span = _pick_station_td_time_span(valid_rows)
        local_parts = [f"{table_name}({device_label}) 共 {len(valid_rows)} 条记录，时间范围 {time_span}。"]

        for metric in metric_fields:
            values: List[float] = []
            missing_count = 0
            zero_count = 0
            for row in valid_rows:
                num = _safe_float_or_none(row.get(metric))
                if num is None:
                    missing_count += 1
                    continue
                values.append(num)
                if abs(num) < 1e-12:
                    zero_count += 1

            if not values:
                continue

            has_any_valid_metric = True
            metric_min = min(values)
            metric_max = max(values)
            metric_mean = sum(values) / len(values)
            metric_summary = {
                "metric": f"{table_name}.{metric}",
                "min": _format_metric_number(metric_min),
                "max": _format_metric_number(metric_max),
                "mean": _format_metric_number(metric_mean),
                "volatility": _format_metric_number(metric_max - metric_min),
                "abnormal_points": zero_count if zero_count == len(values) else missing_count,
                "time_span": time_span,
            }
            td_metrics_summary.append(metric_summary)

            if metric == "soh":
                has_valid_soh = True
            if metric == "voltage":
                has_valid_voltage = True
            if metric == "soc" and zero_count == len(values):
                has_all_zero_soc = True

            if zero_count == len(values):
                local_parts.append(f"{metric} 全部为 0。")
            elif metric_max == metric_min:
                local_parts.append(f"{metric} 恒定为 {_format_metric_number(metric_min)}。")
            else:
                local_parts.append(
                    f"{metric} 在 {_format_metric_number(metric_min)} ~ {_format_metric_number(metric_max)} 之间，均值 {_format_metric_number(metric_mean)}。"
                )
            if missing_count > 0:
                local_parts.append(f"{metric} 有 {missing_count} 条缺失。")

        diagnosis_parts.append(" ".join(local_parts))

    if not has_any_valid_metric:
        return {
            "route_used": "station_device_td",
            "station_td_query_purpose": _infer_station_td_query_purpose(user_req),
            "alarm_summary": {
                "risk_level": "未知",
                "high_freq_terms": [],
                "repeat_patterns": [],
                "key_points": [],
            },
            "abnormal_topn": [],
            "td_metrics_summary": [],
            "diagnosis_conclusion": "查询到了时序数据，但未能从结果中提取出可用的数值指标。",
            "evidence_sufficiency": "不足",
            "next_action_suggestion": "请检查查询字段是否包含数值列，或补充需要关注的指标。",
            "mcp_chart_hint": "",
            "confidence": 0.55,
        }

    conclusion = " ".join(diagnosis_parts)
    if has_all_zero_soc and (has_valid_soh or has_valid_voltage):
        conclusion += " 可以确认并非整行数据缺失：SOH/voltage 至少有一项存在有效值；当前更像是 SOC 估算、字段映射或特定采样链路异常，而不是通信中断。"

    first_valid = next((r for r in all_rows if isinstance(r, dict) and not r.get("error")), {})
    chart_x = "ts" if isinstance(first_valid, dict) and "ts" in first_valid else "time"
    chart_metrics = [m["metric"] for m in td_metrics_summary[:4]]

    return {
        "route_used": "station_device_td",
        "station_td_query_purpose": _infer_station_td_query_purpose(user_req),
        "alarm_summary": {
            "risk_level": "未知",
            "high_freq_terms": [],
            "repeat_patterns": [],
            "key_points": [],
        },
        "abnormal_topn": [],
        "td_metrics_summary": td_metrics_summary[:12],
        "diagnosis_conclusion": conclusion,
        "evidence_sufficiency": "充分",
        "next_action_suggestion": (
            "若需继续排查，可优先对恒零或恒值指标核对设备端采样、字段映射及协议解析；"
            "若需展示趋势，可直接按时间列绘制对应指标曲线。"
        ),
        "mcp_chart_hint": f"横轴建议使用 {chart_x}；可绘制的指标包括 {', '.join(chart_metrics) if chart_metrics else 'station_device_td 数值列'}。",
        "confidence": 0.88,
    }


def _finalize_evidence_bundle(
    route: str,
    user_req: str,
    per_table: Dict[str, List[Dict[str, Any]]],
    all_rows: List[Dict[str, Any]],
    llm_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    if route != "station_device_td":
        return llm_bundle
    return _build_station_td_evidence_bundle(user_req, per_table, all_rows)


def _db_console_verbose() -> int:
    raw = str(os.getenv("DB_CONSOLE_VERBOSE", "1") or "1").strip()
    try:
        return int(raw)
    except Exception:
        return 1


def _print_db_rows_preview(
    table_name: str,
    rows: List[Dict[str, Any]],
    max_rows: int = 5,
    max_chars_per_field: int = 800,
) -> None:
    lvl = _db_console_verbose()
    if lvl <= 0:
        return
    if not rows:
        print(f">>> [SQL结果预览] table={table_name} rows=0", flush=True)
        return
    print(f">>> [SQL结果预览] table={table_name} rows={len(rows)} preview={min(len(rows), max_rows)}", flush=True)
    if lvl < 2:
        return
    for i, row in enumerate(rows[:max_rows], 1):
        preview_row = {k: _clip_preview_value(v, max_chars_per_field) for k, v in row.items()}
        try:
            row_text = json.dumps(preview_row, ensure_ascii=False, default=str)
        except Exception:
            row_text = str(preview_row)
        print(f">>> [SQL行 {i}] {row_text}", flush=True)


def _print_db_llm_truncate(label: str, raw_text: str, max_len: int = 1500) -> None:
    lvl = _db_console_verbose()
    if lvl <= 0:
        return
    if lvl < 2:
        print(f">>> [LLM {label}] len={len(raw_text or '')}", flush=True)
        return
    t = (raw_text or "").replace("\n", " ").strip()
    if len(t) > max_len:
        t = t[:max_len] + "..."
    print(f">>> [LLM {label}] {t}", flush=True)


async def _invoke_json_llm_with_raw(
    user_text: str,
    system_instruction: str = "You are a strict JSON planner.",
    json_retry: bool = True,
) -> Tuple[Dict[str, Any], str]:
    last_raw = ""
    last_err: Optional[Exception] = None
    for attempt in (0, 1):
        u = user_text
        if attempt == 1:
            u = (
                user_text
                + "\n\n[硬性要求] 只输出**一个**合法 JSON 对象：禁止 markdown、禁止解释、"
                "不要代码围栏；所有键用双引号；字符串内换行用 \\n 转义。"
            )
        text, _ = await asyncio.to_thread(
            gemini_chat_once,
            user_text=u,
            system_instruction=system_instruction,
            temperature=0.0,
            max_tokens=2048,
        )
        last_raw = text if isinstance(text, str) else str(text)
        try:
            return _extract_first_json(last_raw), last_raw
        except Exception as e:
            last_err = e
            if attempt == 0 and json_retry:
                continue
            raise LlmJsonParseError(str(e), last_raw) from e
    raise LlmJsonParseError(str(last_err) if last_err else "json parse failed", last_raw)


async def _invoke_evidence_json_with_retry(
    user_text: str,
    system_instruction: str = "You are a strict JSON evidence summarizer.",
) -> Tuple[Dict[str, Any], str, str]:
    """
    证据融合 LLM：首解析失败时自动重试一次（更强硬约束），仍失败则抛错由上层做规则兜底。
    返回: (parsed, raw, status) status 为 ok | retry_ok
    """
    last_raw = ""
    last_err: Optional[Exception] = None
    for attempt in (0, 1):
        u = user_text
        if attempt == 1:
            u = (
                user_text
                + "\n\n[硬性要求] 只输出**一个**合法 JSON 对象：禁止 markdown、禁止解释文字、"
                "不要代码围栏；所有键用双引号；字符串内换行用 \\n 转义。"
            )
        text, _ = await asyncio.to_thread(
            gemini_chat_once,
            user_text=u,
            system_instruction=system_instruction,
            temperature=0.0,
            max_tokens=2560,
        )
        last_raw = text if isinstance(text, str) else str(text)
        try:
            return _extract_first_json(last_raw), last_raw, "retry_ok" if attempt else "ok"
        except Exception as e:
            last_err = e
    raise ValueError(f"evidence json parse failed after retry: {last_err}")


async def _invoke_json_llm(user_text: str, system_instruction: str = "You are a strict JSON planner.") -> Dict[str, Any]:
    data, _ = await _invoke_json_llm_with_raw(user_text, system_instruction)
    return data


def _merge_scope(base: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key in ["station_code", "bms_code", "box_code", "cluster_code", "bmu_code", "cell_id", "bmu_id"]:
        if _normalize_text(merged.get(key)) is None:
            merged[key] = fallback.get(key)
    return merged


def _align_supervisor_scope_with_user_text(supervisor_params: Dict[str, Any], user_req: str) -> Dict[str, Any]:
    """
    修正常见误抽取：用户明确写 bms_code=... 时，避免被错误填到 bmu_code。
    """
    params = dict(supervisor_params or {})
    req = str(user_req or "")
    m = re.search(r"\bbms(?:_code)?\s*[:=：]\s*([A-Za-z0-9_-]+)", req, flags=re.IGNORECASE)
    if not m:
        return params

    bms_val = m.group(1).strip()
    if bms_val:
        params["bms_code"] = bms_val
        if _normalize_text(params.get("bmu_code")) == bms_val:
            params["bmu_code"] = None
    return params


def _build_planner_schema_payload(route: str) -> Dict[str, Any]:
    allow_tables = sorted(ROUTE_TABLE_ALLOWLIST.get(route, set(TABLE_SCHEMAS.keys())))
    tables: Dict[str, Any] = {}
    for t in allow_tables:
        schema = TABLE_SCHEMAS.get(t, {})
        tables[t] = {
            "fields": sorted(list(schema.get("fields", []))),
            "time_fields": list(schema.get("time_fields", [])),
            "default_select": list(schema.get("default_select", [])),
        }
    return {
        "route": route,
        "allow_tables": allow_tables,
        "tables": tables,
    }


def _server_now_injection_text() -> str:
    """供 DB 路由/规划提示词注入，避免模型用训练截止日臆测「今天、此刻」。"""
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        tz_label = "Asia/Shanghai（业务默认：中国时区）"
    except Exception:
        now = datetime.utcnow()
        tz_label = "UTC（时区数据不可用时的回退）"
    weekdays = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
    wd = weekdays[int(now.weekday())]
    iso = now.strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"- 服务端当前：{iso}（{wd}）\n"
        f"- 时区：{tz_label}\n"
        f"- 使用规则：用户若说「今天、此刻、现在、本周、上周、最近 N 天、截止到当前」等，均以**本条服务端当前时间**为锚计算 "
        f"`time_range` 及 `now` 的含义；**禁止**把模型训练数据中的日期当作「今天」。"
    )


async def retrieve_battery_node(state: AgentState):
    print("--- Executing Node: retrieve_battery_node ---", flush=True)
    user_req = state.get("user_request", "")
    supervisor_params = state.get("supervisor_params") if isinstance(state.get("supervisor_params"), dict) else {}
    supervisor_params = _align_supervisor_scope_with_user_text(supervisor_params, str(user_req))
    run_id = str(state.get("run_id", "") or "").strip()
    db_llm_traces: Dict[str, Any] = {}
    raw_route_text = ""
    db_chain_path = _get_db_chain_log_path(state)

    def emit_chain_event(event: str, **payload: Any) -> None:
        row: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "event": event,
            "user_req": str(user_req),
            **payload,
        }
        if run_id:
            row["run_id"] = run_id
        _write_db_chain_log(state, row)

    emit_chain_event(
        "db_chain_start",
        db_chain_log_path=str(db_chain_path),
        supervisor_params=supervisor_params,
    )

    server_now_ctx = _server_now_injection_text()
    locked_clarify = normalize_clarify_route(state.get("clarify_route"))
    route_json: Dict[str, Any]

    if locked_clarify:
        route_json = build_locked_route_json(locked_clarify, supervisor_params)
        route_json["device_scope"] = _merge_scope(route_json.get("device_scope", {}), supervisor_params)
        db_llm_traces["1_intent_router"] = {
            "step": "db_intent_router",
            "skipped": True,
            "clarify_route_locked": locked_clarify,
            "parsed": route_json,
        }
        emit_chain_event(
            "llm_intent_router_skipped",
            clarify_route=locked_clarify,
            parsed=route_json,
        )
        ready, narrow_question = assess_locked_route_readiness(
            locked_clarify, route_json.get("device_scope") or {}, str(user_req)
        )
        if not ready:
            emit_chain_event(
                "clarification_needed",
                route="clarification_needed",
                clarify_route_locked=locked_clarify,
                clarify_question=narrow_question,
                clarify_candidates=[],
            )
            return {
                "db_route": "clarification_needed",
                "clarify_route": "",
                "db_query_params": {
                    "clarify_question": narrow_question,
                    "clarify_candidates": [],
                },
                "raw_db_results": [{"error": f"需要补充信息: {narrow_question}"}],
                "db_evidence_bundle": {
                    "route_used": route_json.get("route", "unknown"),
                    "station_td_query_purpose": "unknown",
                    "diagnosis_conclusion": "当前信息不足，未执行查询。",
                    "evidence_sufficiency": "不足",
                    "next_action_suggestion": narrow_question,
                    "mcp_chart_hint": "",
                    "confidence": 1.0,
                },
                "db_llm_traces": db_llm_traces,
            }
    else:
        try:
            route_prompt = (
                db_intent_router_prompt.replace("{server_now}", server_now_ctx).replace("{user_req}", str(user_req))
            )
            route_json, raw_route_text = await _invoke_json_llm_with_raw(route_prompt)
            db_llm_traces["1_intent_router"] = {
                "step": "db_intent_router",
                "raw_response": raw_route_text,
                "parsed": route_json,
            }
            emit_chain_event(
                "llm_intent_router",
                parsed=route_json,
                raw_response=_clip_for_log(raw_route_text),
            )
            _print_db_llm_truncate("1_intent_router", raw_route_text)
        except LlmJsonParseError as e:
            raw_route_text = e.raw
            db_llm_traces["1_intent_router"] = {
                "step": "db_intent_router",
                "error": str(e),
                "partial_raw": e.raw,
            }
            emit_chain_event(
                "llm_intent_router_error",
                error=str(e),
                raw_response=_clip_for_log(e.raw),
                partial_raw=_clip_for_log(e.raw),
            )
            return {
                "raw_db_results": [{"error": f"路由 JSON 解析失败: {e}"}],
                "db_route": "alerting",
                "db_llm_traces": db_llm_traces,
            }
        except Exception as e:
            db_llm_traces["1_intent_router"] = {
                "step": "db_intent_router",
                "error": str(e),
                "partial_raw": raw_route_text,
            }
            emit_chain_event(
                "llm_intent_router_error",
                error=str(e),
                partial_raw=_clip_for_log(raw_route_text),
            )
            return {
                "raw_db_results": [{"error": f"路由解析失败: {e}"}],
                "db_route": "alerting",
                "db_llm_traces": db_llm_traces,
            }

        route = str(route_json.get("route", "clarification_needed"))
        confidence = float(route_json.get("confidence", 0.0) or 0.0)
        need_clarification = bool(route_json.get("need_clarification")) or confidence < 0.65 or route == "clarification_needed"
        if need_clarification:
            question = route_json.get("clarify_question") or "请补充设备定位（box/cluster/pack/cell）以及时间范围。"
            clarify_candidates = [
                {
                    "id": "station_device_td",
                    "icon": "📊",
                    "title": "设备实时查询",
                    "desc": "查 box_data / cluster_data / bmu_data 时序趋势",
                    "hint": "请补充设备编码（bmu_code/cluster_code/box_code）和时间范围",
                },
                {
                    "id": "alerting",
                    "icon": "⚠️",
                    "title": "异常预警",
                    "desc": "查 alarm_event / volt_temp_abnormal_result 告警摘要",
                    "hint": "请补充设备/站点范围和关注的时间段",
                },
                {
                    "id": "troubleshooting_dcr",
                    "icon": "🔍",
                    "title": "故障钻探 · 内阻异常",
                    "desc": "查内阻异常电芯汇总表",
                    "hint": "请补充 pack/bmu 编码和时间范围",
                },
                {
                    "id": "troubleshooting_isc",
                    "icon": "🔍",
                    "title": "故障钻探 · ISC评分",
                    "desc": "查 ISC 评分表（微短路/内短路评分）",
                    "hint": "请补充 bmu_code 和滑窗时间范围",
                },
                {
                    "id": "troubleshooting_cap",
                    "icon": "🔍",
                    "title": "故障钻探 · 容量不一致",
                    "desc": "查容量不一致电芯表",
                    "hint": "请补充 bmu_code 和电芯号",
                },
            ]
            emit_chain_event(
                "clarification_needed",
                route=route,
                confidence=confidence,
                clarify_question=question,
                clarify_candidates=clarify_candidates,
            )
            return {
                "db_route": "clarification_needed",
                "db_query_params": {
                    "clarify_question": question,
                    "clarify_candidates": clarify_candidates,
                },
                "raw_db_results": [{"error": f"需要补充信息: {question}"}],
                "db_evidence_bundle": {
                    "route_used": "unknown",
                    "station_td_query_purpose": "unknown",
                    "diagnosis_conclusion": "当前信息不足，未执行查询。",
                    "evidence_sufficiency": "不足",
                    "next_action_suggestion": question,
                    "mcp_chart_hint": "",
                    "confidence": round(confidence, 3),
                },
                "db_llm_traces": db_llm_traces,
            }

    route = str(route_json.get("route", "clarification_needed"))

    route_json["device_scope"] = _merge_scope(route_json.get("device_scope", {}), supervisor_params)

    raw_planner_text = ""
    try:
        planner_schema_json = json.dumps(
            _build_planner_schema_payload(route),
            ensure_ascii=False,
        )
        planner_prompt = (
            db_query_planner_prompt.replace("{server_now}", server_now_ctx)
            .replace("{user_req}", str(user_req))
            .replace("{router_json}", json.dumps(route_json, ensure_ascii=False))
            .replace("{planner_schema_json}", planner_schema_json)
        )
        planner_json, raw_planner_text = await _invoke_json_llm_with_raw(planner_prompt)
        db_llm_traces["2_query_planner"] = {
            "step": "db_query_planner",
            "raw_response": raw_planner_text,
            "parsed": planner_json,
        }
        emit_chain_event(
            "llm_query_planner",
            route=route,
            parsed=planner_json,
            raw_response=_clip_for_log(raw_planner_text),
        )
        _print_db_llm_truncate("2_query_planner", raw_planner_text)
    except LlmJsonParseError as e:
        raw_planner_text = e.raw
        db_llm_traces["2_query_planner"] = {
            "step": "db_query_planner",
            "error": str(e),
            "partial_raw": e.raw,
        }
        emit_chain_event(
            "llm_query_planner_error",
            route=route,
            error=str(e),
            raw_response=_clip_for_log(e.raw),
            partial_raw=_clip_for_log(e.raw),
        )
        return {
            "raw_db_results": [{"error": f"查询规划 JSON 解析失败: {e}"}],
            "db_route": route,
            "db_llm_traces": db_llm_traces,
        }
    except Exception as e:
        db_llm_traces["2_query_planner"] = {
            "step": "db_query_planner",
            "error": str(e),
            "partial_raw": raw_planner_text,
        }
        emit_chain_event(
            "llm_query_planner_error",
            route=route,
            error=str(e),
            partial_raw=_clip_for_log(raw_planner_text),
        )
        return {
            "raw_db_results": [{"error": f"查询规划失败: {e}"}],
            "db_route": route,
            "db_llm_traces": db_llm_traces,
        }

    plans = planner_json.get("plans") or []
    if not isinstance(plans, list) or not plans:
        return {
            "raw_db_results": [{"error": "查询规划为空，无法执行。"}],
            "db_route": route,
            "db_llm_traces": db_llm_traces,
        }

    executed_sqls: List[str] = []
    all_rows: List[Dict[str, Any]] = []
    per_table: Dict[str, List[Dict[str, Any]]] = {}
    plan_sanitizer_warnings: List[Dict[str, Any]] = []  # 执行阶段仍按表暂存，最后在去重后覆盖

    for idx, plan in enumerate(plans):
        if not isinstance(plan, dict):
            continue
        try:
            sanitized_plan = _sanitize_plan(plan, route)
            sanitize_warnings = sanitized_plan.pop("_sanitize_warnings", [])
            if sanitize_warnings:
                plan_sanitizer_warnings.append({
                    "plan_index": idx,
                    "raw_table": plan.get("table"),
                    "sanitized_table": sanitized_plan.get("table"),
                    "warnings": sanitize_warnings,
                })
            _write_db_sanitizer_log(state, {
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
                "event": "plan_sanitized",
                "route": route,
                "plan_index": idx,
                "raw_plan": plan,
                "sanitized_plan": sanitized_plan,
                "warnings": sanitize_warnings,
            })
            emit_chain_event(
                "plan_sanitized",
                route=route,
                plan_index=idx,
                raw_plan=plan,
                sanitized_plan=sanitized_plan,
                warnings=sanitize_warnings,
            )

            built = _build_sql_from_plan(sanitized_plan)
            sql = built["sql"]
            table_name = built["table"]
            use_td = bool(route == "station_device_td" and getTdSqlData is not None)
            print(f">>> [实际执行 SQL | parser={'td' if use_td else 'mysql'}] {sql}", flush=True)
            emit_chain_event(
                "sql_built",
                route=route,
                plan_index=idx,
                table_name=table_name,
                parser="td" if use_td else "mysql",
                sql=sql,
                limit=built.get("limit"),
                select_fields=built.get("select_fields"),
            )

            if use_td:
                df = getTdSqlData(sql)
            else:
                df = getMySqlData(sql)

            rows = _df_to_records(table_name, df, built["limit"])
            _print_db_rows_preview(table_name, rows)
            emit_chain_event(
                "sql_result",
                route=route,
                plan_index=idx,
                table_name=table_name,
                rows_count=len(rows),
                rows_preview=[_clip_for_log(r, max_chars=2000) for r in rows[:5]],
            )
            all_rows.extend(rows)
            per_table[table_name] = per_table.get(table_name, []) + rows
            executed_sqls.append(sql)
        except Exception as e:
            _write_db_sanitizer_log(state, {
                "ts": datetime.utcnow().isoformat(timespec="seconds"),
                "event": "plan_execution_error",
                "route": route,
                "plan_index": idx,
                "raw_plan": plan,
                "error": str(e),
            })
            emit_chain_event(
                "plan_execution_error",
                route=route,
                plan_index=idx,
                raw_plan=plan,
                error=str(e),
            )
            all_rows.append({"table_name": plan.get("table", "unknown"), "error": str(e)})

    raw_row_count = len(all_rows)
    deduped_rows, dup_removed = _deduplicate_result_rows(
        [r for r in all_rows if isinstance(r, dict)]
    )
    per_table = _rebuild_per_table(deduped_rows)
    dedup_stats = {
        "raw_row_count": raw_row_count,
        "deduped_row_count": len(deduped_rows),
        "duplicates_removed": dup_removed,
    }
    emit_chain_event("rows_deduped", route=route, **dedup_stats)

    query_result_json = json.dumps(
        {
            "per_table_rows": per_table,
            "sample_rows": deduped_rows[:120],
            "dedup_stats": dedup_stats,
        },
        ensure_ascii=False,
        default=str,
    )

    raw_evidence_text = ""
    try:
        evidence_prompt = (
            db_evidence_summarizer_prompt
            .replace("{user_req}", str(user_req))
            .replace("{route}", str(route))
            .replace("{query_result_json}", query_result_json)
        )
        evidence_bundle, raw_evidence_text, evidence_parse_status = await _invoke_evidence_json_with_retry(
            evidence_prompt,
            system_instruction="You are a strict JSON evidence summarizer.",
        )
        db_llm_traces["3_evidence_summarizer"] = {
            "step": "db_evidence_summarizer",
            "raw_response": raw_evidence_text,
            "parsed": evidence_bundle,
            "parse_status": evidence_parse_status,
        }
        emit_chain_event(
            "llm_evidence_summarizer",
            route=route,
            parsed=evidence_bundle,
            parse_status=evidence_parse_status,
            raw_response=_clip_for_log(raw_evidence_text),
        )
        _print_db_llm_truncate("3_evidence_summarizer", raw_evidence_text)
    except Exception as e:
        evidence_bundle = _rule_based_evidence_bundle(
            route, str(user_req), deduped_rows, raw_row_count, len(deduped_rows)
        )
        err_detail = f"{e}; rule_fallback=1; raw_len={len(raw_evidence_text or '')}"
        db_llm_traces["3_evidence_summarizer"] = {
            "step": "db_evidence_summarizer",
            "error": str(e),
            "partial_raw": raw_evidence_text,
            "parse_status": "rule_fallback",
            "rule_fallback_bundle": True,
        }
        emit_chain_event(
            "llm_evidence_summarizer_error",
            route=route,
            error=err_detail,
            partial_raw=_clip_for_log(raw_evidence_text) if raw_evidence_text else "",
            rule_fallback="applied",
        )

    evidence_bundle = _finalize_evidence_bundle(
        route=route,
        user_req=str(user_req),
        per_table=per_table,
        all_rows=deduped_rows,
        llm_bundle=evidence_bundle if isinstance(evidence_bundle, dict) else {},
    )

    emit_chain_event(
        "db_chain_end",
        route=route,
        executed_sql_count=len(executed_sqls),
        total_rows_raw=raw_row_count,
        total_rows_deduped=len(deduped_rows),
        tables=list(per_table.keys()),
    )
    result = {
        "db_route": route,
        "db_query_params": route_json.get("device_scope", {}),
        "db_query_plans": plans,
        "db_executed_sqls": executed_sqls,
        "db_plan_sanitizer_warnings": plan_sanitizer_warnings,
        "db_result_stats": dedup_stats,
        "db_evidence_bundle": evidence_bundle,
        "db_llm_traces": db_llm_traces,
        "raw_db_results": (
            deduped_rows
            if deduped_rows
            else (all_rows if all_rows else [{"error": "未查询到结果。"}])
        ),
    }
    if locked_clarify:
        result["clarify_route"] = ""
    return result
