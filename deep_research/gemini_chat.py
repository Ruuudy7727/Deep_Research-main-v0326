import json
import requests
import traceback
from typing import Tuple, Dict, Any, Iterator
from dotenv import load_dotenv
import os
import statistics
import time
from pathlib import Path

try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _HAVE_RETRY = True
except Exception:
    HTTPAdapter = None  # type: ignore
    Retry = None  # type: ignore
    _HAVE_RETRY = False

# 项目根目录（deep_research/ 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载环境变量
ENV_PATH = str(_PROJECT_ROOT / ".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)


# =============================================================================
# 复用 HTTP 连接：避免每次 LLM 调用都做新的 TCP/TLS 握手 (~100-300ms)。
# Deep 模式 10-15 次 LLM 调用累积可省 1-4s。
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
    """暴露给宿主程序复用的 Session（已启用 keep-alive + retry）。"""
    return _HTTP_SESSION


# 分场景超时：planner / router 应该秒级返回；只有 final report / 多模态会更久。
GEMINI_TIMEOUT_FAST = float(os.getenv("GEMINI_TIMEOUT_FAST", "45"))
GEMINI_TIMEOUT_LONG = float(os.getenv("GEMINI_TIMEOUT_LONG", "120"))

# --- 标准模型配置 ---
MIDEA_API_KEY = os.getenv("MIDEA_API_KEY", "")
MIDEA_AIGC_USER = os.getenv("MIDEA_AIGC_USER", "user")
GEMINI_URL_SYNC = "https://aimpapi.midea.com/t-aigc/mip-chat-app/gemini/official/standard/sync/v1/chat/completions"
GEMINI_AIMP_BIZ_ID = os.getenv("GEMINI_AIMP_BIZ_ID", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "user")

# --- [新增] RPO 最终报告专用配置 ---
MIDEA_API_KEY_RPO = os.getenv("MIDEA_API_KEY_RPO", "")
GEMINI_AIMP_BIZ_ID_RPO = os.getenv("GEMINI_AIMP_BIZ_ID_RPO", "")
GEMINI_MODEL_RPO = os.getenv("GEMINI_MODEL_RPO", "")
GEMINI_URL_SYNC_RPO = "https://aimpapi.midea.com/t-aigc/mip-chat-app/gemini/official/standard/stream/v2/chat/completions"

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_URL = os.getenv("QWEN_URL", "https://aimpapi.midea.com/t-aigc/aimp-qwen3-32b/v1/chat/completions")
QWEN_MODEL = os.getenv("QWEN_MODEL", "/model/qwen3-32b")


def gemini_chat_once(user_text: str, system_instruction: str, temperature: float = 0.3, max_tokens: int = 4096) -> Tuple[str, Dict[str, Any]]:
    """
    标准 Gemini 非流式接口
    """
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
            proxies={"http": None, "https": None}
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return text, data.get("usageMetadata", {})
    except Exception as e:
        print(f"Gemini Sync Error: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", {}


def gemini_chat_once_rpo(user_text: str, system_instruction: str, temperature: float = 0.3, max_tokens: int = 8192) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    RPO 专用通道（适配 Stream/v2 接口）- 实现真正的流式输出 (Yield)
    """
    # 1. 基础校验
    if not MIDEA_API_KEY_RPO or not GEMINI_AIMP_BIZ_ID_RPO or not GEMINI_MODEL_RPO:
        error_msg = "[Config Error] RPO 环境变量 (MIDEA_API_KEY_RPO, GEMINI_AIMP_BIZ_ID_RPO, GEMINI_MODEL_RPO) 未正确设置"
        print(error_msg)
        yield error_msg, {}
        return

    # 2. 组装 Headers (参考文档)
    headers = {
        "Authorization": f"Bearer {MIDEA_API_KEY_RPO}",
        "Aimp-Biz-Id": GEMINI_AIMP_BIZ_ID_RPO, # 文档要求: gemini-3-pro-preview
        "AIGC-USER": MIDEA_AIGC_USER,          # 文档要求: 4A账号
        "Content-Type": "application/json",    # 文档要求: application/json
    }

    # 3. 组装 Body (严格参考文档，移除 "stream": True)
    body = {
        "model": GEMINI_MODEL_RPO, # 文档固定为: gemini-3-pro-preview
        "contents": [
            {
                "role": "user", 
                "parts": [{"text": user_text}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            # 注意：文档中未要求在此处传 "stream": True，绝对不要加
        }
    }

    # 4. 发起请求 (stream=True 保持开启，用于接收 SSE 流)
    try:
        resp = _HTTP_SESSION.post(
            GEMINI_URL_SYNC_RPO,
            headers=headers,
            json=body,
            timeout=GEMINI_TIMEOUT_LONG,
            stream=True,
            proxies={"http": None, "https": None}
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_msg = f"RPO 请求失败: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
             error_msg += f" (Status: {e.response.status_code}, Body: {e.response.text})"
        yield error_msg, {}
        return

    # 5. 解析流式响应 (Yield)
    final_text = ""
    usage_metadata = {}

    for line in resp.iter_lines():
        if not line:
            continue
        
        try:
            decoded_line = line.decode('utf-8').strip()
            
            # 匹配 SSE 格式: "data: {...}"
            if decoded_line.startswith("data: "):
                content_str = decoded_line[6:] # 去除前缀 "data: "
                
                # 过滤结束标识
                if content_str == "[DONE]":
                    break
                    
                try:
                    chunk = json.loads(content_str)
                    
                    # A. 提取文本 (candidates -> content -> parts -> text)
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text_fragment = parts[0].get("text", "")
                            if text_fragment:
                                final_text += text_fragment
                                # 实时 Yield 当前累积的文本
                                yield final_text, usage_metadata
                    
                    # B. 提取 Token 消耗 (通常在流的更新中或最后)
                    if "usageMetadata" in chunk:
                        usage_metadata = chunk["usageMetadata"]
                        # 如果有更新 usage，也 yield 一次
                        yield final_text, usage_metadata
                        
                except json.JSONDecodeError:
                    print(f"[RPO Warning] JSON 解析错误, 数据片段: {content_str[:50]}...")
                    continue
        except Exception as e:
             print(f"[RPO Stream Error] Line processing failed: {e}")
             continue
    
    # 确保最后一次 yield 包含完整的 usage
    yield final_text, usage_metadata


def qwen_chat_once(user_text: str, system_instruction: str = "", temperature: float = 0, max_tokens: int = 4096, enable_thinking: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Qwen3 非流式接口调用
    :param enable_thinking: 是否开启 Qwen3 的思考模式 (默认为 True)
    """
    if not QWEN_API_KEY:
        print("[Warning] QWEN_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }

    # 构造 messages
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
        "chat_template_kwargs": {"enable_thinking": enable_thinking}
    }

    try:
        resp = _HTTP_SESSION.post(
            QWEN_URL,
            headers=headers,
            json=body,
            timeout=GEMINI_TIMEOUT_FAST,
            proxies={"http": None, "https": None}
        )
        resp.raise_for_status()
        
        data = resp.json()
        
        choices = data.get("choices", [])
        if not choices:
            return "", {}
            
        message = choices[0].get("message", {})
        content = message.get("content", "")
        
        # 思考内容处理 (如果需要可以取消注释)
        # reasoning = message.get("reasoning_content", "")
        
        usage = data.get("usage", {})
        return content, usage

    except Exception as e:
        print(f"Qwen API call failed: {e}")
        return "", {}


# 假设你将提供的代码保存为了 gemini_chat.py
# from gemini_chat import gemini_chat_once_rpo
import time
import sys

def test_stream():
    print("--- 开始测试 RPO 流式输出 ---")
    
    # 构造测试输入
    user_text = "请写一首关于春天的五言绝句，并逐句解释。"
    system_instruction = "你是中国古诗词专家。"
    
    # 调用函数，获取生成器
    # 注意：这里函数返回的是一个 iterator，不会立即执行网络请求，直到开始遍历
    stream_generator = gemini_chat_once_rpo(
        user_text=user_text,
        system_instruction=system_instruction
    )
    
    start_time = time.time()
    last_text_len = 0
    
    try:
        # 遍历生成器
        for current_text, usage in stream_generator:
            # 计算这一帧新增了多少字符
            new_chars = current_text[last_text_len:]
            
            # 模拟打字机效果打印出来
            # flush=True 确保立即显示，不经过缓存
            sys.stdout.write(new_chars)
            sys.stdout.flush()
            
            last_text_len = len(current_text)
            
            # 检查是否有报错信息返回 (根据代码逻辑，报错也是 yield 出来的)
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
