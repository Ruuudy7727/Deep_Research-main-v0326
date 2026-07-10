#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立联调脚本：测试百炼公共 API（OpenAI 兼容）

验证：
  1) qwen3.7-plus 非流式
  2) qwen3.7-max 流式 + thinking
  3) 流式末帧 usage（必须 stream_options.include_usage=True）

不依赖本项目任何模块，仅需：pip install openai

用法：
  1. 在下方填入 API_KEY，或设置环境变量 DASHSCOPE_API_KEY
  2. python test.py
"""

from __future__ import annotations

import os

# ========== 在此填写你的配置 ==========
API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip() or "sk-xxx"
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).strip()
MODEL_FAST = os.getenv("LLM_MODEL_FAST", "qwen3.7-plus").strip() or "qwen3.7-plus"
MODEL_DEEP = os.getenv("LLM_MODEL_DEEP", "qwen3.7-max").strip() or "qwen3.7-max"
ENABLE_THINKING = True  # Deep 流式开启 thinking
# ======================================

QUESTION_SYNC = "你好，请用一句话介绍你自己。"
QUESTION_STREAM = "1+1等于几？请简短回答。"


def _check_config() -> None:
    if not API_KEY or API_KEY == "sk-xxx":
        raise SystemExit("请先在脚本顶部填写 API_KEY，或设置环境变量 DASHSCOPE_API_KEY")
    if not BASE_URL:
        raise SystemExit("请填写 BASE_URL")
    if not MODEL_FAST or not MODEL_DEEP:
        raise SystemExit("请填写 MODEL_FAST / MODEL_DEEP")


def _extra_body(enable_thinking: bool) -> dict:
    return {"enable_thinking": bool(enable_thinking)}


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def test_sync(client) -> str:
    print("=" * 20 + f" 非流式测试 ({MODEL_FAST}) " + "=" * 20)
    kwargs = {
        "model": MODEL_FAST,
        "messages": [{"role": "user", "content": QUESTION_SYNC}],
        "max_tokens": 300,
        "extra_body": _extra_body(False),
    }
    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("非流式返回为空，请检查模型权限或 enable_thinking 配置")
    usage = _usage_dict(resp.usage)
    print(text)
    print(f"[OK] 非流式成功，回答长度 {len(text)} 字符")
    print(f"[OK] sync usage = {usage}\n")
    if not usage.get("total_tokens") and not usage.get("prompt_tokens"):
        raise RuntimeError("非流式未返回 usage，请检查 API 响应")
    return text


def test_stream(client) -> str:
    print("=" * 20 + f" 流式测试 ({MODEL_DEEP}) " + "=" * 20)
    kwargs = {
        "model": MODEL_DEEP,
        "messages": [{"role": "user", "content": QUESTION_STREAM}],
        "max_tokens": 200,
        "stream": True,
        # 关键：OpenAI 兼容流式默认不返回 usage，必须显式打开
        "stream_options": {"include_usage": True},
        "extra_body": _extra_body(ENABLE_THINKING),
    }

    parts: list[str] = []
    is_answering = False
    final_usage: dict = {}
    chunk_idx = 0

    for chunk in client.chat.completions.create(**kwargs):
        chunk_idx += 1
        usage = _usage_dict(getattr(chunk, "usage", None))
        if usage:
            final_usage = usage
            print(f"\n[chunk#{chunk_idx} usage] {usage}", flush=True)

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning and ENABLE_THINKING:
            if not is_answering:
                print("[思考] ", end="", flush=True)
            print(reasoning, end="", flush=True)
            continue

        content = getattr(delta, "content", None) or ""
        if content:
            if not is_answering:
                if ENABLE_THINKING:
                    print("\n[回答] ", end="", flush=True)
                is_answering = True
            parts.append(content)
            print(content, end="", flush=True)

    print()
    answer = "".join(parts).strip()
    if not answer:
        raise RuntimeError("流式返回为空，请检查模型权限或网络")
    print(f"[OK] 流式成功，回答长度 {len(answer)} 字符")
    print(f"[OK] stream final usage = {final_usage}")
    if not final_usage.get("total_tokens") and not final_usage.get("prompt_tokens"):
        raise RuntimeError(
            "流式末帧未返回 usage。请确认请求包含 "
            'stream_options={"include_usage": True}'
        )
    print("[OK] 流式 usage 字段完整\n")
    return answer


def main() -> None:
    _check_config()

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("缺少依赖，请先运行: pip install openai")

    print(f"BASE_URL   = {BASE_URL}")
    print(f"MODEL_FAST = {MODEL_FAST}")
    print(f"MODEL_DEEP = {MODEL_DEEP}")
    print(f"THINKING   = {ENABLE_THINKING}\n")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120.0)

    try:
        test_sync(client)
        test_stream(client)
        print("=" * 20 + " 全部通过 " + "=" * 20)
    except Exception as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        print("\n常见原因：")
        print("  - 401: API Key 无效或过期")
        print("  - 404: MODEL 名称与控制台不一致")
        print("  - Connection error: 网络/代理问题，或 BASE_URL 填错")
        print("  - 流式无 usage: 缺少 stream_options.include_usage=True")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
