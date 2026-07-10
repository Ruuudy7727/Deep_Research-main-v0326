# -*- coding: utf-8 -*-
"""Per-answer LLM token usage accumulator.

Normalize Gemini ``usageMetadata`` and OpenAI/DashScope ``usage`` into one
summary that can be attached to SHARED_STATE / SSE / run logs / eval results.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_context: Dict[str, Any] = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "llm_calls": 0,
    "by_model": {},
    "by_stage": {},
    "calls": [],
}


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_usage(usage: Any) -> Dict[str, int]:
    """Normalize provider-specific usage dicts to prompt/completion/total."""
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt = _as_int(
        usage.get("prompt_tokens")
        or usage.get("promptTokenCount")
        or usage.get("input_tokens")
        or usage.get("inputTokens")
    )
    completion = _as_int(
        usage.get("completion_tokens")
        or usage.get("candidatesTokenCount")
        or usage.get("output_tokens")
        or usage.get("outputTokens")
        or usage.get("candidates_tokens")
    )
    # Gemini sometimes reports thoughts separately; fold into completion when present.
    thoughts = _as_int(usage.get("thoughtsTokenCount") or usage.get("reasoning_tokens"))
    if thoughts and not completion:
        completion = thoughts
    elif thoughts:
        completion = completion + thoughts

    total = _as_int(usage.get("total_tokens") or usage.get("totalTokenCount"))
    if total <= 0:
        total = prompt + completion

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def reset_usage() -> None:
    with _lock:
        _context["total_prompt_tokens"] = 0
        _context["total_completion_tokens"] = 0
        _context["total_tokens"] = 0
        _context["llm_calls"] = 0
        _context["by_model"] = {}
        _context["by_stage"] = {}
        _context["calls"] = []


def record_usage(
    usage: Any,
    *,
    model: str = "",
    stage: str = "",
) -> Dict[str, int]:
    """Accumulate one LLM call's usage. Returns the normalized counters."""
    normalized = normalize_usage(usage)
    model_key = (model or "unknown").strip() or "unknown"
    stage_key = (stage or "unspecified").strip() or "unspecified"

    with _lock:
        _context["total_prompt_tokens"] += normalized["prompt_tokens"]
        _context["total_completion_tokens"] += normalized["completion_tokens"]
        _context["total_tokens"] += normalized["total_tokens"]
        _context["llm_calls"] += 1

        by_model = _context["by_model"].setdefault(
            model_key,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
        )
        by_model["prompt_tokens"] += normalized["prompt_tokens"]
        by_model["completion_tokens"] += normalized["completion_tokens"]
        by_model["total_tokens"] += normalized["total_tokens"]
        by_model["calls"] += 1

        by_stage = _context["by_stage"].setdefault(
            stage_key,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
        )
        by_stage["prompt_tokens"] += normalized["prompt_tokens"]
        by_stage["completion_tokens"] += normalized["completion_tokens"]
        by_stage["total_tokens"] += normalized["total_tokens"]
        by_stage["calls"] += 1

        _context["calls"].append(
            {
                "model": model_key,
                "stage": stage_key,
                **normalized,
            }
        )

    return normalized


def get_usage_summary() -> Dict[str, Any]:
    with _lock:
        return {
            "total_prompt_tokens": int(_context["total_prompt_tokens"]),
            "total_completion_tokens": int(_context["total_completion_tokens"]),
            "total_tokens": int(_context["total_tokens"]),
            "llm_calls": int(_context["llm_calls"]),
            "by_model": {
                k: dict(v) for k, v in (_context.get("by_model") or {}).items()
            },
            "by_stage": {
                k: dict(v) for k, v in (_context.get("by_stage") or {}).items()
            },
        }


def usage_from_openai_response(resp: Any) -> Dict[str, Any]:
    """Extract usage dict from an OpenAI SDK response or stream chunk."""
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
