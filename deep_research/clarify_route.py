#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据库澄清卡片 → 锁定 db_route / target_tables（仍由 Supervisor/Planner 抽参）。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

CLARIFY_ROUTE_IDS = frozenset({
    "station_device_td",
    "alerting",
    "troubleshooting_dcr",
    "troubleshooting_isc",
    "troubleshooting_cap",
})

CLARIFY_ROUTE_META: Dict[str, Dict[str, Any]] = {
    "station_device_td": {
        "db_route": "station_device_td",
        "task_type": "station_device_td",
        "target_tables": ["cluster_data"],
        "title": "设备实时查询",
        "param_hint": "请补充设备编码（bmu_code/cluster_code/box_code）和时间范围",
    },
    "alerting": {
        "db_route": "alerting",
        "task_type": "alerting",
        "target_tables": ["alarm_event"],
        "title": "异常预警",
        "param_hint": "请补充设备/站点范围和关注的时间段",
    },
    "troubleshooting_dcr": {
        "db_route": "troubleshooting",
        "task_type": "troubleshooting",
        "target_tables": ["dcr_abnormal_cells"],
        "title": "故障钻探 · 内阻异常",
        "param_hint": "请补充 pack/bmu 编码和时间范围",
    },
    "troubleshooting_isc": {
        "db_route": "troubleshooting",
        "task_type": "troubleshooting",
        "target_tables": ["isc_score_result"],
        "title": "故障钻探 · ISC评分",
        "param_hint": "请补充 bmu_code 和滑窗时间范围",
    },
    "troubleshooting_cap": {
        "db_route": "troubleshooting",
        "task_type": "troubleshooting",
        "target_tables": ["capacity_inconsistent_cells"],
        "title": "故障钻探 · 容量不一致",
        "param_hint": "请补充 bmu_code 和电芯号",
    },
}

_DEVICE_KEYS = ("station_code", "bms_code", "box_code", "cluster_code", "bmu_code", "cell_id", "pack_code", "bmu_id")


def normalize_clarify_route(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cid = str(raw).strip().lower().replace("-", "_")
    return cid if cid in CLARIFY_ROUTE_IDS else None


def build_clarify_route_hint(clarify_route: Optional[str]) -> str:
    cid = normalize_clarify_route(clarify_route)
    if not cid:
        return ""
    meta = CLARIFY_ROUTE_META[cid]
    tables = ", ".join(meta.get("target_tables") or [])
    return (
        f"\n【澄清卡片已选场景（硬锁定，勿改路由）】\n"
        f"用户已通过澄清卡片选定：「{meta['title']}」\n"
        f"- 必须 next_action=DATABASE，task_type={meta['task_type']}\n"
        f"- 下游 db_route={meta['db_route']}，目标表：{tables}\n"
        f"- 请从用户原话中**完整抽取**设备 ID、时间范围及查库 params，不要留空能填的字段。\n"
    )


def build_locked_route_json(
    clarify_route: str,
    device_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cid = normalize_clarify_route(clarify_route)
    if not cid:
        raise ValueError(f"invalid clarify_route: {clarify_route}")
    meta = CLARIFY_ROUTE_META[cid]
    scope = dict(device_scope or {})
    return {
        "route": meta["db_route"],
        "reason": f"用户澄清卡片锁定场景: {cid}",
        "confidence": 1.0,
        "need_clarification": False,
        "clarify_question": "",
        "target_tables": list(meta["target_tables"]),
        "mode": "targeted",
        "device_scope": scope,
        "time_range": {"start_time": None, "end_time": None},
        "metrics_hint": [],
        "clarify_route_locked": cid,
    }


def locked_task_type(clarify_route: Optional[str]) -> Optional[str]:
    cid = normalize_clarify_route(clarify_route)
    if not cid:
        return None
    return str(CLARIFY_ROUTE_META[cid]["task_type"])


def _scope_has_device(scope: Dict[str, Any]) -> bool:
    for key in _DEVICE_KEYS:
        val = scope.get(key)
        if val is None:
            continue
        if str(val).strip():
            return True
    return False


_DATE_LIKE = re.compile(
    r"\b(20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}|\d{4}-\d{2}-\d{2}|"
    r"今天|昨天|本周|上周|最近\s*\d+\s*天|P-\d+[DWH])",
    re.IGNORECASE,
)


def _text_has_device_hint(text: str) -> bool:
    req = str(text or "")
    patterns = (
        r"\bbmu(?:_code)?\s*[:=：]?\s*[\w-]+",
        r"\bpack[-_]?\d+",
        r"\bcluster[-_]?\d+",
        r"\bbox[-_]?\d+",
        r"\bstation[-_]?\d+",
        r"\b00\d{10,}\b",
        r"站\s*\d+",
    )
    return any(re.search(p, req, re.IGNORECASE) for p in patterns)


def _text_has_time_hint(text: str) -> bool:
    return bool(_DATE_LIKE.search(str(text or "")))


def assess_locked_route_readiness(
    clarify_route: str,
    device_scope: Dict[str, Any],
    user_req: str,
) -> Tuple[bool, str]:
    """锁定场景后仅检查设备+时间是否足够，不再弹出 5 选 1。"""
    cid = normalize_clarify_route(clarify_route)
    if not cid:
        return False, "无效澄清场景"
    meta = CLARIFY_ROUTE_META[cid]
    if cid == "troubleshooting_cap":
        has_device = _scope_has_device(device_scope) or _text_has_device_hint(user_req)
        if has_device:
            return True, ""
        return False, (
            f"您已选择「{meta['title']}」，请补充设备范围。"
            f"（{meta['param_hint']}）"
        )

    has_device = _scope_has_device(device_scope) or _text_has_device_hint(user_req)
    has_time = _text_has_time_hint(user_req)
    if has_device and has_time:
        return True, ""
    missing = []
    if not has_device:
        missing.append("设备范围")
    if not has_time:
        missing.append("时间范围")
    question = (
        f"您已选择「{meta['title']}」，请补充{'、'.join(missing)}。"
        f"（{meta['param_hint']}）"
    )
    return False, question
