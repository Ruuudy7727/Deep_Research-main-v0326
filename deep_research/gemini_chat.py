# -*- coding: utf-8 -*-
"""LLM chat wrappers with DashScope (default) and Midea Gemini providers.

Public API stays stable for agents:
  - gemini_chat_once(...)
  - gemini_chat_once_rpo(...)  # streaming iterator
  - qwen_chat_once(...)       # legacy Midea Qwen helper
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from dotenv import load_dotenv

try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    _HAVE_RETRY = True
except Exception:
    HTTPAdapter = None  # type: ignore
    Retry = None  # type: ignore
    _HAVE_RETRY = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = str(_PROJECT_ROOT / ".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)

try:
    from deep_research.llm_usage import record_usage, usage_from_openai_response
except Exception:
    from llm_usage import record_usage, usage_from_openai_response  # type: ignore


# =============================================================================
# Shared HTTP session (Midea paths)
# =============================================================================
def _build_pooled_session() -> requests.Session:
    sess = requests.Session()
    if _HAVE_RETRY and HTTPAdapter is not None and Retry is not None:
        retry = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST", "GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            pool_connections=16,
            pool_maxsize=32,
            max_retries=retry,
        )
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
    return sess


_HTTP_SESSION: requests.Session = _build_pooled_session()


def get_http_session() -> requests.Session:
    return _HTTP_SESSION


GEMINI_TIMEOUT_FAST = float(os.getenv("GEMINI_TIMEOUT_FAST", "45"))
GEMINI_TIMEOUT_LONG = float(os.getenv("GEMINI_TIMEOUT_LONG", "120"))

# --- Provider selection ---
LLM_PROVIDER = (os.getenv("LLM_PROVIDER", "dashscope") or "dashscope").strip().lower()
if LLM_PROVIDER not in {"dashscope", "midea"}:
    LLM_PROVIDER = "dashscope"

# --- DashScope (OpenAI compatible) ---
DASHSCOPE_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY", "").strip()
    or os.getenv("QWEN_API_KEY", "").strip()
)
DASHSCOPE_BASE_URL = (
    os.getenv("DASHSCOPE_BASE_URL", "").strip()
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
).rstrip("/")
LLM_MODEL_FAST = os.getenv("LLM_MODEL_FAST", "").strip() or "qwen3.7-plus"
LLM_MODEL_DEEP = os.getenv("LLM_MODEL_DEEP", "").strip() or "qwen3.7-max"
LLM_ENABLE_THINKING_FAST = os.getenv("LLM_ENABLE_THINKING_FAST", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
LLM_ENABLE_THINKING_DEEP = os.getenv("LLM_ENABLE_THINKING_DEEP", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

# --- Midea Gemini ---
MIDEA_API_KEY = os.getenv("MIDEA_API_KEY", "")
MIDEA_AIGC_USER = os.getenv("MIDEA_AIGC_USER", "user")
GEMINI_URL_SYNC = "https://aimpapi.midea.com/t-aigc/mip-chat-app/gemini/official/standard/sync/v1/chat/completions"
GEMINI_AIMP_BIZ_ID = os.getenv("GEMINI_AIMP_BIZ_ID", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MIDEA_API_KEY_RPO = os.getenv("MIDEA_API_KEY_RPO", "")
GEMINI_AIMP_BIZ_ID_RPO = os.getenv("GEMINI_AIMP_BIZ_ID_RPO", "")
GEMINI_MODEL_RPO = os.getenv("GEMINI_MODEL_RPO", "")
GEMINI_URL_SYNC_RPO = "https://aimpapi.midea.com/t-aigc/mip-chat-app/gemini/official/standard/stream/v2/chat/completions"

# --- Legacy Midea Qwen ---
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_URL = os.getenv("QWEN_URL", "https://aimpapi.midea.com/t-aigc/aimp-qwen3-32b/v1/chat/completions")
QWEN_MODEL = os.getenv("QWEN_MODEL", "/model/qwen3-32b")

_dashscope_client = None


def get_llm_provider() -> str:
    return LLM_PROVIDER


def get_fast_model_name() -> str:
    if LLM_PROVIDER == "midea":
        return GEMINI_MODEL or "gemini-2.5-flash"
    return LLM_MODEL_FAST


def get_deep_model_name() -> str:
    if LLM_PROVIDER == "midea":
        return GEMINI_MODEL_RPO or GEMINI_MODEL or "gemini-rpo"
    return LLM_MODEL_DEEP


def _get_dashscope_client():
    global _dashscope_client
    if _dashscope_client is not None:
        return _dashscope_client
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY (or QWEN_API_KEY) is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package required for DashScope provider") from exc
    _dashscope_client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        timeout=GEMINI_TIMEOUT_LONG,
    )
    return _dashscope_client


def _build_messages(user_text: str, system_instruction: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": user_text or ""})
    return messages


# =============================================================================
# DashScope paths
# =============================================================================
def _dashscope_chat_once(
    user_text: str,
    system_instruction: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    *,
    model: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    stage: str = "sync",
) -> Tuple[str, Dict[str, Any]]:
    model_name = (model or LLM_MODEL_FAST).strip() or LLM_MODEL_FAST
    thinking = LLM_ENABLE_THINKING_FAST if enable_thinking is None else bool(enable_thinking)
    try:
        client = _get_dashscope_client()
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": _build_messages(user_text, system_instruction),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"enable_thinking": thinking},
        }
        resp = client.chat.completions.create(**kwargs)
        text = ""
        if resp.choices:
            text = (resp.choices[0].message.content or "").strip()
        usage = usage_from_openai_response(resp)
        record_usage(usage, model=model_name, stage=stage)
        if not text:
            print(
                f"[DashScope Sync] empty text (model={model_name!r}, thinking={thinking})",
                flush=True,
            )
        return text, usage
    except Exception as e:
        print(f"DashScope Sync Error: {e}", flush=True)
        traceback.print_exc()
        return f"Error: {str(e)}", {}


def _dashscope_chat_stream(
    user_text: str,
    system_instruction: str,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    *,
    model: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    stage: str = "stream",
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    model_name = (model or LLM_MODEL_DEEP).strip() or LLM_MODEL_DEEP
    thinking = LLM_ENABLE_THINKING_DEEP if enable_thinking is None else bool(enable_thinking)
    try:
        client = _get_dashscope_client()
    except Exception as e:
        yield f"[Config Error] {e}", {}
        return

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": _build_messages(user_text, system_instruction),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": {"enable_thinking": thinking},
    }

    final_text = ""
    usage_metadata: Dict[str, Any] = {}
    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            chunk_usage = usage_from_openai_response(chunk)
            if chunk_usage:
                usage_metadata = chunk_usage

            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            # Prefer answer content; ignore reasoning_content for the yielded answer text.
            content = getattr(delta, "content", None) or ""
            if content:
                final_text += content
                yield final_text, usage_metadata
    except Exception as e:
        err = f"DashScope stream failed: {e}"
        print(err, flush=True)
        traceback.print_exc()
        yield err, usage_metadata
        return

    if usage_metadata:
        record_usage(usage_metadata, model=model_name, stage=stage)
    else:
        print(
            f"[DashScope Stream] WARNING: no usage in final chunk "
            f"(model={model_name!r}). Ensure stream_options.include_usage=True.",
            flush=True,
        )
    yield final_text, usage_metadata


def dashscope_chat_once_multimodal(
    user_text: str,
    system_instruction: str,
    *,
    image_data_urls: List[str],
    temperature: float = 0.3,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    stage: str = "multimodal",
) -> Tuple[str, Dict[str, Any]]:
    """Sync multimodal call for DashScope (OpenAI image_url format)."""
    model_name = (model or LLM_MODEL_FAST).strip() or LLM_MODEL_FAST
    content: List[Dict[str, Any]] = []
    if user_text:
        content.append({"type": "text", "text": user_text})
    for url in image_data_urls:
        if not url:
            continue
        content.append({"type": "image_url", "image_url": {"url": url}})
    messages: List[Dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": content})

    try:
        client = _get_dashscope_client()
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
        )
        text = ""
        if resp.choices:
            text = (resp.choices[0].message.content or "").strip()
        usage = usage_from_openai_response(resp)
        record_usage(usage, model=model_name, stage=stage)
        return text, usage
    except Exception as e:
        print(f"DashScope Multimodal Error: {e}", flush=True)
        traceback.print_exc()
        return f"Error: {str(e)}", {}


# =============================================================================
# Midea Gemini paths (preserved)
# =============================================================================
def _midea_gemini_chat_once(
    user_text: str,
    system_instruction: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    *,
    stage: str = "sync",
) -> Tuple[str, Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {MIDEA_API_KEY}",
        "Aimp-Biz-Id": GEMINI_AIMP_BIZ_ID,
        "AIGC-USER": MIDEA_AIGC_USER,
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {
        "model": GEMINI_MODEL,
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    try:
        resp = _HTTP_SESSION.post(
            GEMINI_URL_SYNC,
            headers=headers,
            json=body,
            timeout=GEMINI_TIMEOUT_FAST,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            text = (
                candidates[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        usage = data.get("usageMetadata", {}) or {}
        record_usage(usage, model=GEMINI_MODEL or "gemini", stage=stage)
        if not (text or "").strip():
            finish = candidates[0].get("finishReason") if candidates else None
            block = data.get("promptFeedback") or data.get("blockReason")
            print(
                f"[Gemini Sync] empty text (status={resp.status_code}, "
                f"finishReason={finish!r}, block={block!r}, "
                f"model={GEMINI_MODEL!r}, key_set={bool(MIDEA_API_KEY)})",
                flush=True,
            )
            if not MIDEA_API_KEY or not GEMINI_AIMP_BIZ_ID:
                return "Error: MIDEA_API_KEY or GEMINI_AIMP_BIZ_ID not configured", {}
        return text, usage
    except Exception as e:
        print(f"Gemini Sync Error: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", {}


def _midea_gemini_chat_once_rpo(
    user_text: str,
    system_instruction: str,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    *,
    stage: str = "stream",
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if not MIDEA_API_KEY_RPO or not GEMINI_AIMP_BIZ_ID_RPO or not GEMINI_MODEL_RPO:
        error_msg = (
            "[Config Error] RPO 环境变量 "
            "(MIDEA_API_KEY_RPO, GEMINI_AIMP_BIZ_ID_RPO, GEMINI_MODEL_RPO) 未正确设置"
        )
        print(error_msg)
        yield error_msg, {}
        return

    headers = {
        "Authorization": f"Bearer {MIDEA_API_KEY_RPO}",
        "Aimp-Biz-Id": GEMINI_AIMP_BIZ_ID_RPO,
        "AIGC-USER": MIDEA_AIGC_USER,
        "Content-Type": "application/json",
    }
    body = {
        "model": GEMINI_MODEL_RPO,
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    try:
        resp = _HTTP_SESSION.post(
            GEMINI_URL_SYNC_RPO,
            headers=headers,
            json=body,
            timeout=GEMINI_TIMEOUT_LONG,
            stream=True,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_msg = f"RPO 请求失败: {str(e)}"
        if hasattr(e, "response") and e.response is not None:
            error_msg += f" (Status: {e.response.status_code}, Body: {e.response.text})"
        yield error_msg, {}
        return

    final_text = ""
    usage_metadata: Dict[str, Any] = {}

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            decoded_line = line.decode("utf-8").strip()
            if decoded_line.startswith("data: "):
                content_str = decoded_line[6:]
                if content_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(content_str)
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text_fragment = parts[0].get("text", "")
                            if text_fragment:
                                final_text += text_fragment
                                yield final_text, usage_metadata
                    if "usageMetadata" in chunk:
                        usage_metadata = chunk["usageMetadata"]
                        yield final_text, usage_metadata
                except json.JSONDecodeError:
                    print(f"[RPO Warning] JSON 解析错误, 数据片段: {content_str[:50]}...")
                    continue
        except Exception as e:
            print(f"[RPO Stream Error] Line processing failed: {e}")
            continue

    if usage_metadata:
        record_usage(usage_metadata, model=GEMINI_MODEL_RPO or "gemini-rpo", stage=stage)
    yield final_text, usage_metadata


# =============================================================================
# Public API (provider-routed)
# =============================================================================
def gemini_chat_once(
    user_text: str,
    system_instruction: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> Tuple[str, Dict[str, Any]]:
    stage = str(kwargs.get("stage") or "sync")
    if LLM_PROVIDER == "midea":
        return _midea_gemini_chat_once(
            user_text,
            system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            stage=stage,
        )
    return _dashscope_chat_once(
        user_text,
        system_instruction,
        temperature=temperature,
        max_tokens=max_tokens,
        model=kwargs.get("model") or LLM_MODEL_FAST,
        enable_thinking=kwargs.get("enable_thinking"),
        stage=stage,
    )


def gemini_chat_once_rpo(
    user_text: str,
    system_instruction: str,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    **kwargs: Any,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    stage = str(kwargs.get("stage") or "stream")
    if LLM_PROVIDER == "midea":
        yield from _midea_gemini_chat_once_rpo(
            user_text,
            system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            stage=stage,
        )
        return
    yield from _dashscope_chat_stream(
        user_text,
        system_instruction,
        temperature=temperature,
        max_tokens=max_tokens,
        model=kwargs.get("model") or LLM_MODEL_DEEP,
        enable_thinking=kwargs.get("enable_thinking"),
        stage=stage,
    )


def qwen_chat_once(
    user_text: str,
    system_instruction: str = "",
    temperature: float = 0,
    max_tokens: int = 4096,
    enable_thinking: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Legacy Midea Qwen3 non-streaming helper (unchanged endpoint)."""
    if not QWEN_API_KEY:
        print("[Warning] QWEN_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": user_text})

    body = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    try:
        resp = _HTTP_SESSION.post(
            QWEN_URL,
            headers=headers,
            json=body,
            timeout=GEMINI_TIMEOUT_FAST,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "", {}
        message = choices[0].get("message", {})
        content = message.get("content", "")
        usage = data.get("usage", {}) or {}
        record_usage(usage, model=QWEN_MODEL or "qwen", stage="qwen_legacy")
        return content, usage
    except Exception as e:
        print(f"Qwen API call failed: {e}")
        return "", {}


def test_stream():
    print("--- 开始测试 RPO / Deep 流式输出 ---")
    print(f"provider={LLM_PROVIDER} deep_model={get_deep_model_name()}")
    user_text = "请写一首关于春天的五言绝句，并逐句解释。"
    system_instruction = "你是中国古诗词专家。"
    stream_generator = gemini_chat_once_rpo(
        user_text=user_text,
        system_instruction=system_instruction,
    )
    start_time = time.time()
    last_text_len = 0
    current_text = ""
    usage: Dict[str, Any] = {}
    try:
        for current_text, usage in stream_generator:
            new_chars = current_text[last_text_len:]
            sys.stdout.write(new_chars)
            sys.stdout.flush()
            last_text_len = len(current_text)
            if current_text.startswith("[Config Error]") or current_text.startswith("RPO 请求失败"):
                print(f"\n\n❌ 测试失败: {current_text}")
                return
        print(f"\n\n--- 测试完成 ---")
        print(f"总耗时: {time.time() - start_time:.2f}秒")
        print(f"最终字数: {len(current_text)}")
        print(f"Token消耗: {usage}")
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")


if __name__ == "__main__":
    test_stream()
