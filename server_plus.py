#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
server_plus.py  —  科宝 Cobot 前端服务
=======================================================
基于 FastAPI + SSE + 自定义 HTML 前端，提供专业级 BMS/EMS 诊断 UI。
复用 server.py 中的后端 Agent 逻辑，仅替换 Gradio 前端为自定义 HTML。

启动:
    cd <PROJECT_ROOT>
    python server_plus.py          # 默认 0.0.0.0:50222
"""

import logging
import os
import sys
import json
import asyncio
import time
import datetime
import requests
import base64
import mimetypes
import re
import tempfile
import html as html_module

try:
    from markdown_it import MarkdownIt
except ImportError:
    MarkdownIt = None  # type: ignore
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# 0. 全局共享状态
# ==========================================
SHARED_STATE: Dict[str, Any] = {
    "timeline_html": "",
    "logs": [],
    "final_report": "",
    "is_running": False,
    "stream_buffer": "",
    "steps_data": [],
    "chart_url": None,
    "raw_dataframe": [],
    "query_params": {},
    "research_brief": "",
    "research_notes": "",
    "draft_report": "",
    "agent_thoughts": "",
    "last_pre_brief_cases": "",
    "last_db_raw_results": [],
    # --- Plus 新增 ---
    "structured_metrics": {},
    "diagnosis_tags": [],
    "diagnosis_summary": "",
    "diagnosis_highlights": [],
    "suggestions": [],
    "chat_history": [],
    "current_mode": "fast",
    "workspace_name": "默认工作空间",
    "session_memory": True,
    # --- 图文一体（RAG 图片索引）---
    # 每项: {"url": "/kb_images/doc-XXX/HASH.jpg", "rel": "images/doc-XXX/HASH.jpg",
    #        "caption": "<chunk 内容片段>", "source": "h1/.../xxx.md", "score": 0.xx}
    "retrieved_images": [],
    # 本次报告中"实际被注入到 LLM 的"图片（已严格按 1..N 编号，与 prompt 「图N」对齐）
    # 每项: {"idx": 1, "url": ..., "rel": ..., "caption": ..., "source": ...}
    "answer_images": [],
    # --- 6 类前端布局支持（Plus v2）---
    # 由 supervisor / clarify_with_user 写入；前端据此切换展示卡片。
    # 取值：direct / kb_retrieval / station_device_td / alerting / troubleshooting / deep_research
    "task_type": "direct",
    # 知识库证据明细（图文一体右侧证据面板用）。
    # 每项：{"idx": 1, "source": "...md", "score": 0.92, "chunk_text": "...",
    #        "image_paths": ["images/doc-XXX/HASH.jpg"], "kind": "text|image"}
    "evidence_chunks": [],
    # 告警明细行（alarm 卡片表格 + 摘要计算用）。
    # 每项：原 alarm_event 行裁剪后的 dict（start_time/end_time/station_code/bmu_code/cell_id/severity/average_severity/summary_cn 等）。
    "alarm_rows": [],
    # 告警概要：{"total":N, "cells_involved":N, "avg_severity":F, "max_severity":F, "top_keywords":[(kw,count),...]}
    "alarm_summary": {},
    # 通用 SQL 表（station_device_td 卡片底部「原始数据表」用）。
    # {"columns": ["ts","voltage","soc",...], "rows": [[...], ...], "title": "..."}
    "sql_table": {"columns": [], "rows": [], "title": ""},
    # station_device_td 执行后的完整 SQL 文本列表（前端 SQL 面板展示）
    "executed_sqls": [],
    # 意图澄清候选场景（need_clarification 时展示给用户选择）
    "clarify_candidates": [],
}

MEMORY = MemorySaver()
CURRENT_THREAD_ID = f"web_{int(time.time())}"

# ==========================================
# 1. 环境配置
# ==========================================
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"), override=False)

os.environ["MIDEA_API_KEY"] = os.getenv("MIDEA_API_KEY", "")
MIDEA_API_KEY = os.environ["MIDEA_API_KEY"]
GEMINI_URL_SYNC = os.getenv("GEMINI_URL_SYNC", "https://aimpapi.midea.com/t-aigc/mip-chat-app/gemini/official/standard/sync/v1/chat/completions")
GEMINI_AIMP_BIZ_ID = os.getenv("GEMINI_AIMP_BIZ_ID", "gemini-2.5-flash")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MIDEA_AIGC_USER = os.getenv("MIDEA_AIGC_USER", "user")
UI_VERBOSE_DB_LOG = os.getenv("UI_VERBOSE_DB_LOG", "1").strip().lower() in {"1", "true", "yes", "y", "on"}

SERVER_PLUS_PORT = int(os.getenv("SERVER_PLUS_PORT", "50221"))
# Uvicorn 访问日志：探针大量 HEAD / 时设为 0 可关闭 access log，保留其它 INFO（如启动）
_SERVER_PLUS_ACCESS_LOG = os.getenv("SERVER_PLUS_ACCESS_LOG", "1").strip().lower() in {
    "1", "true", "yes", "y", "on",
}


class _Suppress405AccessLog(logging.Filter):
    """屏蔽 uvicorn access 中带 405 的行（例如仅有 GET 而无 HEAD 时的探针）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if " 405 " in msg or "405 Method Not Allowed" in msg:
            return False
        return True


def _configure_uvicorn_access_logging() -> None:
    if os.getenv("SERVER_PLUS_SUPPRESS_ACCESS_405", "1").strip().lower() in {
        "1", "true", "yes", "y", "on", "",
    }:
        logging.getLogger("uvicorn.access").addFilter(_Suppress405AccessLog())

# --- 知识库图片索引 ---
# RAG 知识库图片根目录（与 step1_build_konwledge.py / step2.5_json2chroma.py 输出一致）
KB_IMAGES_DIR = PROJECT_ROOT / "rag_data" / "all" / "images"
KB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
# 单条任务最多沉淀多少张相关图片，用于注入答案
MAX_RETRIEVED_IMAGES = int(os.getenv("MAX_RETRIEVED_IMAGES", "8"))
# 注入到答案 Markdown 的图片标题
RELATED_IMAGES_HEADING = os.getenv("RELATED_IMAGES_HEADING", "📎 相关图示")

# --- 多模态（图文一体推理）配置 ---
# 报告阶段最多向 Gemini 多模态调用塞多少张图（控制请求体大小 + 推理稳定性）
MAX_LLM_IMAGES = int(os.getenv("MAX_LLM_IMAGES", "4"))
# 单张图最大字节数（超过则跳过，避免 base64 后请求体爆掉）
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(1_500_000)))  # ~1.5 MB
# 多模态调用前给模型加的 system 后缀（指导模型如何利用图片）
MULTIMODAL_SYSTEM_HINT = (
    "\n\n[多模态补充指令]\n"
    "你将额外收到 {n} 张来自知识库的图片，它们已按出现顺序编号为「图1」、「图2」…「图{n}」。\n"
    "写作要求：\n"
    "(1) 先看图，再写文；\n"
    "(2) 当某段结论确实参考了某张图时，请在那段文字里**显式写出**对应编号，例如"
    "「如图1所示，电压曲线呈现阶跃……」「图2 中红框区域为产气位置」。\n"
    "    宿主程序会在你写下「图N」的段落后**自动渲染**该缩略图，方便读者对照；\n"
    "(3) 若图中有清晰的元件、结构、流程、数值等可识别信息，请在结论中精确引用；\n"
    "(4) 仅当图与问题确实相关时才引用，避免硬贴；不相关的图可以不提；\n"
    "(5) **不要**自己写 markdown 图片语法（`![](...)`）或粘贴图片 URL —— "
    "宿主程序会按你写的「图N」自动就近内联，并在末尾以「📎 相关图示」列出未被引用的图。"
)


def encode_image_to_base64(image_path: str) -> Optional[Dict[str, str]]:
    try:
        if not image_path or image_path.startswith("http"):
            return None
        if not os.path.exists(image_path):
            return None
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return {"mimeType": mime_type, "data": encoded}
    except Exception as e:
        print(f"[Image Encode Error] {e}", flush=True)
        return None


def _chart_public_url(chart_path: Optional[str]) -> Optional[str]:
    if not chart_path:
        return None
    return f"/figure/{os.path.basename(chart_path)}"


def _resolve_chart_file_path(chart_path: Optional[str]) -> Optional[str]:
    """Resolve chart_output / chart_url to a readable local file under figure/."""
    if not chart_path:
        return None
    p = str(chart_path).strip()
    if p.startswith("http://") or p.startswith("https://"):
        bn = os.path.basename(p.split("?", 1)[0])
        for base in (PROJECT_ROOT / "figure", Path.cwd() / "figure"):
            cand = base / bn
            if cand.is_file():
                return str(cand)
        return None
    if os.path.isfile(p):
        return p
    bn = os.path.basename(p)
    for base in (PROJECT_ROOT / "figure", Path.cwd() / "figure"):
        cand = base / bn
        if cand.is_file():
            return str(cand)
    return None


def _chart_image_data_uri(chart_path: Optional[str]) -> Optional[str]:
    path = _resolve_chart_file_path(chart_path)
    if not path:
        return None
    try:
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{b64}"
    except OSError as e:
        print(f"[Report HTML] chart embed failed: {e}", flush=True)
        return None


_LEADING_MD_IMG_RE = re.compile(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*\n*", re.MULTILINE)


def _strip_duplicate_leading_chart_md(md_text: str, chart_path: Optional[str]) -> str:
    """Remove leading ![...](...) lines that point at the same chart file (avoid double image in HTML)."""
    if not chart_path:
        return md_text
    bn = os.path.basename(str(chart_path).split("?", 1)[0])
    s = (md_text or "").lstrip("\n\r \t")
    while True:
        m = _LEADING_MD_IMG_RE.match(s)
        if not m:
            break
        url = (m.group(1) or "").strip().strip('"').strip("'")
        url_bn = os.path.basename(url.split("?", 1)[0])
        if url_bn == bn or url.endswith(bn) or f"/figure/{bn}" in url.replace(" ", ""):
            s = s[m.end() :].lstrip()
        else:
            break
    return s


def _markdown_to_html_fragment(md_text: str) -> str:
    if MarkdownIt is None:
        return f"<pre>{html_module.escape(md_text)}</pre>"
    md = MarkdownIt("commonmark", {"html": False, "linkify": False, "breaks": True})
    return md.render(md_text or "")


_REPORT_HTML_STYLE = """
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "PingFang SC", sans-serif;
  line-height: 1.65; color: #1e293b; max-width: 900px; margin: 24px auto; padding: 0 16px 48px; }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; color: #0f172a; }
.meta { color: #64748b; font-size: 0.875rem; margin-bottom: 1.5rem; }
.report-chart { margin: 1.25rem 0; text-align: center; }
.report-chart img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.report-body { margin-top: 1.5rem; }
.report-body img { max-width: 100%; height: auto; }
.report-body pre { background: #0f172a; color: #e2e8f0; padding: 12px 16px; border-radius: 8px; overflow-x: auto; font-size: 0.875rem; }
.report-body code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
.report-body blockquote { border-left: 4px solid #3b82f6; margin: 12px 0; padding: 4px 16px; color: #475569; background: #f8fafc; }
"""


def _build_report_html(report_md: str, chart_path: Optional[str], timestamp: str) -> str:
    """Single-file HTML: embedded chart (base64) + markdown body."""
    body_md = _strip_duplicate_leading_chart_md(report_md or "", chart_path)
    chart_uri = _chart_image_data_uri(chart_path)
    chart_block = ""
    if chart_uri:
        chart_block = (
            f'<div class="report-chart"><img src="{chart_uri}" alt="Chart" /></div>'
        )
    inner = _markdown_to_html_fragment(body_md)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>诊断报告</title>
<style>{_REPORT_HTML_STYLE}</style>
</head>
<body>
<h1>诊断报告</h1>
<p class="meta">生成时间：{html_module.escape(timestamp)}</p>
{chart_block}
<div class="report-body">{inner}</div>
</body>
</html>
"""


def _compose_report_with_chart(report_text: str, chart_path: Optional[str]) -> str:
    report_text = report_text or ""
    chart_url = _chart_public_url(chart_path)
    if not chart_url:
        return report_text
    chart_md = f"![Chart]({chart_url})"
    if chart_md in report_text or "<img " in report_text:
        return report_text
    task_type = (SHARED_STATE.get("task_type") or "direct").strip().lower()
    if task_type == "station_device_td" and chart_path:
        return report_text
    return f"{chart_md}\n\n{report_text}" if report_text else chart_md


def _img_caption_line(img: Dict[str, Any], idx: int) -> str:
    """为某张图生成一行"图 N ｜ 来源: ..."形式的小注释。"""
    source = str(img.get("source_label") or img.get("source") or "").strip()
    bits = [f"_图 {idx}_"]
    if source:
        bits.append(f"来源: `{source}`")
    return " ｜ ".join(bits)


def _build_related_images_md(
    images: Optional[List[Dict[str, Any]]],
    *,
    skip_idxs: Optional[set] = None,
) -> str:
    """把图片列表渲染成"📎 相关图示"段落。

    每张图按它自身携带的 `idx` 编号显示「图 N」，与 LLM prompt 「图1」「图2」对齐。
    如果某张图的 idx 在 `skip_idxs` 内（已经被就近内联到正文里），则跳过避免重复。
    """
    if not images:
        return ""
    skip = skip_idxs or set()
    items: List[str] = []
    for fallback_i, img in enumerate(images, 1):
        if not isinstance(img, dict):
            continue
        url = img.get("url")
        if not url:
            continue
        idx = int(img.get("idx") or fallback_i)
        if idx in skip:
            continue
        caption = (img.get("caption") or "").strip()
        block = [f"![图 {idx}]({url})"]
        block.append(f"> {_img_caption_line(img, idx)}")
        if caption:
            block.append(f"> 上下文: {caption}")
        items.append("\n".join(block))
    if not items:
        return ""
    return f"\n\n---\n\n## {RELATED_IMAGES_HEADING}\n\n" + "\n\n".join(items) + "\n"


def _report_already_has_image(report_text: str, url: str) -> bool:
    if not report_text or not url:
        return False
    return (url in report_text)


# 匹配「图1」「图 1」「图10」「图 12」等 LLM 在正文中可能写出的引用编号。
# 不匹配「图书」「图谱」「图标」等 false positive：要求「图」后必须紧跟可选空白 + 数字。
_FIG_REF_RE = re.compile(r"图\s*(\d{1,2})(?!\d)")


def _inline_referenced_images(
    report_text: str,
    images: List[Dict[str, Any]],
) -> Tuple[str, set]:
    """在正文里就近渲染被 LLM 引用过的图。

    - 在每张图（按 idx）首次出现「图N」的段落后插入 markdown 缩略图；
    - 已经手动内联（report 里已含相同 url）的图不再重复；
    - 返回 (新正文, 已被内联的 idx 集合)，供调用方决定末尾还要不要再列。
    """
    if not report_text or not images:
        return report_text or "", set()

    by_idx: Dict[int, Dict[str, Any]] = {}
    for fallback_i, img in enumerate(images, 1):
        if not isinstance(img, dict) or not img.get("url"):
            continue
        idx = int(img.get("idx") or fallback_i)
        if idx not in by_idx:
            by_idx[idx] = img

    if not by_idx:
        return report_text, set()

    inlined: set = set()
    found: List[Tuple[int, int]] = []  # (insertion_pos, idx)
    seen_idxs: set = set()
    for m in _FIG_REF_RE.finditer(report_text):
        idx = int(m.group(1))
        if idx in seen_idxs or idx not in by_idx:
            continue
        url = by_idx[idx].get("url", "")
        if url and _report_already_has_image(report_text, url):
            seen_idxs.add(idx)
            inlined.add(idx)
            continue
        # 在该匹配所在段落（双换行分段）的末尾插入缩略图
        para_end = report_text.find("\n\n", m.end())
        if para_end == -1:
            para_end = len(report_text)
        found.append((para_end, idx))
        seen_idxs.add(idx)

    if not found:
        return report_text, inlined

    # 从后往前插，避免位置偏移
    out = report_text
    for pos, idx in sorted(found, key=lambda x: x[0], reverse=True):
        img = by_idx[idx]
        url = img["url"]
        snippet = (
            f"\n\n![图 {idx}]({url})\n"
            f"> {_img_caption_line(img, idx)}\n"
        )
        out = out[:pos] + snippet + out[pos:]
        inlined.add(idx)
    return out, inlined


# 仅这两类前端布局会内联 RAG 图片到 Markdown 正文中；
# station_device_td / alerting / troubleshooting / direct 都在前端有独立面板，避免双份显示。
_IMAGE_INLINE_TASK_TYPES = {"kb_retrieval", "deep_research"}


def _compose_full_report(
    report_text: str,
    chart_path: Optional[str] = None,
    images: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """统一组装最终 Markdown：报告正文 + 图表 + 就近内联图 + 末尾相关图示。

    优先级：
      1) 优先使用 SHARED_STATE['answer_images'] —— 即 LLM 本次"实际收到"的有序图，
         编号与 prompt 中「图1」「图2」… 严格 1:1 对齐；
      2) 回退到 `images` 参数 / SHARED_STATE['retrieved_images']（无多模态时）。

    流程：
      - chart 仍置于报告头；
      - LLM 正文里写过「图N」的，自动在该段后内联缩略图；
      - 末尾"📎 相关图示"只列**未被内联**的剩余图，避免重复展示。

    Plus v2：仅 kb_retrieval / deep_research 这两类 task_type 才会内联图片到正文，
    其它 task_type（station_device_td / alerting / troubleshooting / direct）由前端独立面板渲染，避免重复展示。
    """
    composed = _compose_report_with_chart(report_text or "", chart_path)

    task_type = (SHARED_STATE.get("task_type") or "direct").strip().lower()
    if task_type and task_type not in _IMAGE_INLINE_TASK_TYPES:
        return composed

    used = list(SHARED_STATE.get("answer_images") or [])
    if not used:
        if images is None:
            images = SHARED_STATE.get("retrieved_images") or []
        # retrieved_images 没有 idx 字段，给一个 fallback：按列表序号补
        used = []
        for i, img in enumerate(images or [], 1):
            if isinstance(img, dict):
                d = dict(img)
                d.setdefault("idx", i)
                used.append(d)
    if not used:
        return composed

    composed, inlined_idxs = _inline_referenced_images(composed, used)

    pending: List[Dict[str, Any]] = []
    for img in used:
        if not isinstance(img, dict):
            continue
        idx = int(img.get("idx") or 0)
        if idx in inlined_idxs:
            continue
        if _report_already_has_image(composed, img.get("url", "")):
            continue
        pending.append(img)

    extra = _build_related_images_md(pending)
    if not extra:
        return composed
    return f"{composed}{extra}"


# --- 本地 HTTP 封装 ---
def gemini_chat_once_http(
    user_text: str,
    system_instruction: str,
    images: Optional[List[str]] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    if not MIDEA_API_KEY:
        return "Error: No API Key", {}
    headers = {
        "Authorization": f"Bearer {MIDEA_API_KEY}",
        "Aimp-Biz-Id": GEMINI_AIMP_BIZ_ID,
        "AIGC-USER": MIDEA_AIGC_USER,
        "Content-Type": "application/json",
    }
    content_parts = []
    if user_text:
        content_parts.append({"text": user_text})
    if images:
        for img_path in images:
            img_data = encode_image_to_base64(img_path)
            if img_data:
                content_parts.append({"inlineData": img_data})
    body = {
        "model": GEMINI_MODEL,
        "contents": [{"role": "user", "parts": content_parts}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    try:
        resp = requests.post(GEMINI_URL_SYNC, headers=headers, json=body, timeout=180)
        if 200 <= resp.status_code < 300:
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return text, {}
        return f"Error: {resp.status_code} - {resp.text}", {}
    except Exception as e:
        return f"Exception: {str(e)}", {}


# ==========================================
# 1.4 多模态注入（让 Gemini 真正"看见"RAG 命中图片）
# ----------------------------------------------------------------------
# `server_initial.py` 留下了 `gemini_chat_once_http(images=...)` 这条多模态
# 通道，但从未实际调用。这里把它接通到"报告生成阶段"，让 Gemini 能基于
# RAG 命中图（标准件框图、流程图、原理示意等）进行"图文联合推理"。
# ==========================================
# 命中即被认为是"最终答案/报告生成阶段"的关键字（覆盖 deep + fast + draft 三路）
_FINAL_ANSWER_PHASE_KEYWORDS = (
    # research_agent_full.py final_report_generation
    "诊断报告", "Research Brief", "研究简报", "diagnosis report",
    # utils.py draft_report_prompt / scope draft
    "结构良好的回答", "well-structured answer",
    "初步分析草稿", "precision-focused answer",
    # single_agent_supervisor.solve_simple_task
    "资深技术顾问", "基于以上信息，生成一份精炼的诊断报告",
)


def _is_final_answer_phase(system_instruction: str, user_text: str = "") -> bool:
    """根据 system prompt / user prompt 判断当前 LLM 调用是否在"最终答案阶段"。

    报告/答案阶段 → 才会注入图片做多模态推理；
    其他阶段（澄清、规划、子任务）→ 不注入，避免 token / 成本爆炸。
    """
    blob = f"{system_instruction or ''}\n{user_text or ''}"
    return any(k in blob for k in _FINAL_ANSWER_PHASE_KEYWORDS)


# 多模态会强制把流式 RPO 降级成同步 sync，首字节会从 ~1s 拉到 ~10-30s。
# Fast 模式用户对 TTFB 最敏感，默认仅在 deep 模式启用图文联合推理。
# 设 MULTIMODAL_FAST_MODE_ENABLED=1 可强制 fast 模式也带图。
MULTIMODAL_FAST_MODE_ENABLED = os.getenv("MULTIMODAL_FAST_MODE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
# 没图就不必经过短路逻辑；图片数量门槛——单图通常意义不大反而拖慢。
MULTIMODAL_MIN_IMAGES = max(1, int(os.getenv("MULTIMODAL_MIN_IMAGES", "1")))


def _multimodal_allowed_for_current_mode() -> bool:
    """fast 模式默认禁用多模态短路（保住流式首字节）。"""
    if MULTIMODAL_FAST_MODE_ENABLED:
        return True
    mode = str(SHARED_STATE.get("current_mode", "fast") or "fast").lower()
    return mode == "deep"


def _build_kb_image_paths_for_llm() -> Tuple[List[str], List[Dict[str, Any]]]:
    """从 SHARED_STATE 取 top-N RAG 命中图。

    返回:
      paths : 用于 multimodal API 的本地绝对路径列表；
      picked: 与 paths 同序的元数据列表，每项含 1-based `idx`、url、rel、caption、source。
              `idx` 与 prompt 中「图N」一一对齐，便于报告组装时还原引用。
    """
    images_state = list(SHARED_STATE.get("retrieved_images") or [])
    if not images_state:
        return [], []
    paths: List[str] = []
    picked: List[Dict[str, Any]] = []
    for img in images_state[:MAX_LLM_IMAGES * 2]:  # 多取一倍，过滤后再 cap
        if not isinstance(img, dict):
            continue
        rel = (img.get("rel") or "").lstrip("./").lstrip("/")
        rel_in_kb = rel[len("images/"):] if rel.startswith("images/") else rel
        local = KB_IMAGES_DIR / rel_in_kb
        if not local.is_file():
            continue
        try:
            size = local.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > MAX_IMAGE_BYTES:
            print(
                f"[RAG-IMG] skip (size {size}B out of [1, {MAX_IMAGE_BYTES}B]): {local.name}",
                flush=True,
            )
            continue
        paths.append(str(local))
        picked.append({
            "idx": len(paths),  # 1-based, 与 prompt 「图N」对齐
            "url": img.get("url"),
            "rel": img.get("rel"),
            "caption": img.get("caption", ""),
            "source": img.get("source", ""),
        })
        if len(paths) >= MAX_LLM_IMAGES:
            break
    return paths, picked


def _maybe_call_multimodal(
    user_text: str,
    system_instruction: str,
    *,
    label: str = "sync",
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """如果当前阶段判定需要多模态 + 有可用图片，则发起 multimodal 调用并返回结果。

    返回 None 表示"无需多模态 / 没有可用图片"，调用方应走原有文本路径。
    副作用：将"本次实际注入给 LLM 的图"写入 SHARED_STATE['answer_images']，
            供 _compose_full_report 后续按相同编号在正文/末尾渲染。
    """
    if not _is_final_answer_phase(system_instruction, user_text):
        return None
    if not _multimodal_allowed_for_current_mode():
        return None
    image_paths, picked = _build_kb_image_paths_for_llm()
    if not image_paths or len(image_paths) < MULTIMODAL_MIN_IMAGES:
        return None
    SHARED_STATE["answer_images"] = picked
    augmented_sys = (system_instruction or "") + MULTIMODAL_SYSTEM_HINT.format(
        n=len(image_paths)
    )
    print(
        f"[RAG-IMG] multimodal call ({label}): +{len(image_paths)} image(s) "
        f"(idx 1..{len(image_paths)})",
        flush=True,
    )
    text, usage = gemini_chat_once_http(
        user_text,
        augmented_sys,
        images=image_paths,
    )
    return text, usage


# --- 适配器挂载 ---
try:
    import deep_research.gemini_chat as original_module

    _original_rpo_func = original_module.gemini_chat_once_rpo

    def _adapter_gemini_chat_rpo(user_text, system_instruction, **kwargs):
        # ── 多模态短路：报告阶段且有 RAG 命中图，则牺牲流式改走 sync multimodal ──
        # 流式 RPO 接口不支持 images；为了"图文联合推理"，在最关键的最终答案
        # 调用上短路成 sync 多模态。文本仍写入 stream_buffer，UI 体感是"一次性出文"。
        mm = _maybe_call_multimodal(user_text, system_instruction, label="rpo")
        if mm is not None:
            text, usage = mm
            SHARED_STATE["stream_buffer"] = text
            return text, usage

        stream_generator = _original_rpo_func(
            user_text=user_text, system_instruction=system_instruction, **kwargs
        )
        final_text = ""
        final_usage: Dict[str, Any] = {}
        try:
            for chunk in stream_generator:
                if isinstance(chunk, tuple):
                    if len(chunk) >= 1:
                        final_text = chunk[0]
                    if len(chunk) >= 2:
                        final_usage = chunk[1]
                elif isinstance(chunk, str):
                    final_text = chunk
                else:
                    final_text = str(chunk)
                SHARED_STATE["stream_buffer"] = final_text
                time.sleep(0.001)
        except Exception as e:
            print(f"[Adapter Stream Error]: {e}", flush=True)
            final_text += f"\n[System Error during stream: {str(e)}]"
        return final_text, final_usage

    original_module.gemini_chat_once_rpo = _adapter_gemini_chat_rpo
    print("[Patch] RPO adapter mounted (multimodal-aware)", flush=True)
except ImportError:
    print("[Init] deep_research.gemini_chat not found", flush=True)


def _gemini_chat_sync_lc(messages: List[BaseMessage]) -> AIMessage:
    sys_parts = [m.content for m in messages if isinstance(m, SystemMessage)]
    user_parts = [str(m.content) for m in messages if not isinstance(m, SystemMessage)]
    sys_text = "\n".join(sys_parts)
    user_text = "\n".join(user_parts)

    # ── 报告阶段 + 有 RAG 命中图 → 走多模态 ──
    mm = _maybe_call_multimodal(user_text, sys_text, label="sync_lc")
    if mm is not None:
        return AIMessage(content=mm[0])

    text, _ = gemini_chat_once_http(user_text, sys_text)
    return AIMessage(content=text)


async def _gemini_chat_async_lc(messages: List[BaseMessage]) -> AIMessage:
    return await asyncio.to_thread(_gemini_chat_sync_lc, messages)


gemini_chat_runnable = RunnableLambda(func=_gemini_chat_sync_lc, afunc=_gemini_chat_async_lc)

import deep_research.utils as dr_utils

try:
    dr_utils.set_models(gemini_chat_runnable)
except Exception as e:
    print(f"[Init] Model injection failed: {e}", flush=True)


# ==========================================
# 1.5 RAG 图片索引适配器
# ----------------------------------------------------------------------
# `deep_research/utils.py::_lc_retrieve_hybrid_async` 默认只把 `source/score/type`
# 透传到上层，导致 chunk 中的 `image_paths`（在 step1/step2.5 阶段已写入 Chroma
# metadata）在向 LLM 输出时被丢弃。这里在不改动 deep_research 包的前提下
# monkey-patch 该函数，让它把 image_paths 等字段一并保留；并包装
# `unified_local_search`，把每轮检索命中的图片沉淀到 SHARED_STATE，便于：
#   1) 在最终报告里以 Markdown 形式注入相关图示；
#   2) 通过 SSE state 事件推送给前端实时展示。
# ==========================================
def _norm_image_paths_field(raw: Any) -> List[str]:
    """Chroma metadata 里 image_paths 可能是 ';' 拼接的字符串、列表或 None。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [s.strip() for s in raw.split(";") if s and s.strip()]
        return items
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _kb_image_url(rel_path: str) -> str:
    """`images/doc-XXX/HASH.jpg` -> `/kb_images/doc-XXX/HASH.jpg`"""
    rel = (rel_path or "").lstrip("./").lstrip("/")
    if rel.startswith("images/"):
        rel = rel[len("images/"):]
    return f"/kb_images/{rel}"


def _kb_image_exists(rel_path: str) -> bool:
    """检查图片文件是否真的落地（避免给前端塞 404 链接）。"""
    rel = (rel_path or "").lstrip("./").lstrip("/")
    if rel.startswith("images/"):
        rel = rel[len("images/"):]
    return (KB_IMAGES_DIR / rel).is_file()


def _to_relative_source_path(raw_source: Any) -> str:
    """把证据来源路径规范成相对路径，避免前端展示绝对路径。"""
    s = str(raw_source or "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    p = Path(s)
    if p.is_absolute():
        try:
            return p.relative_to(PROJECT_ROOT).as_posix()
        except Exception:
            # 若绝对路径不在 PROJECT_ROOT 下，尽量截取出项目内可读段落
            marker = f"/{PROJECT_ROOT.name}/"
            idx = s.find(marker)
            if idx >= 0:
                return s[idx + len(marker):].lstrip("/")
            # 最后兜底：只返回文件名，避免泄露机器绝对路径
            return p.name
    return s.lstrip("./")


def _to_source_display_name(raw_source: Any) -> str:
    """前端来源显示名：仅保留最后文档名（兼容 / \\ ／ 分隔符）。"""
    s = str(raw_source or "").strip()
    if not s:
        return ""
    parts = [p for p in re.split(r"[\\/／]+", s) if p]
    return parts[-1] if parts else s


# =============================================================================
# 图片单独召回（图文双路检索）
# -----------------------------------------------------------------------------
# 文本召回归文本，图片召回归图片：用同一条 query 在向量库里做大 k 召回，
# 然后从结果里**只挑带 image_paths 的 chunk**，作为"语义最相关的图片证据"。
# 这样不会再出现"段落相邻但语义不相关"的图。
# =============================================================================
# 每次最多挑出多少张"语义最相关"的图片证据
IMAGE_RECALL_TOPK = int(os.getenv("IMAGE_RECALL_TOPK", "2"))
# 用多大的候选池来扫描（只看带 image_paths 的 chunk，所以池要足够大才能命中）
IMAGE_RECALL_POOL_K = int(os.getenv("IMAGE_RECALL_POOL_K", "30"))


def _image_recall_for_query_sync(query: str) -> List[Dict[str, Any]]:
    """在向量库里按 query 单独召回"带图 chunk"，返回 top-N 图片元数据列表。

    注意：monkey-patch 的 `_wrapped_vss` 会按 (query, k) 缓存调用结果，所以同 query
    多次调用只会真正打一次 embedding API。
    """
    if not query or not str(query).strip():
        return []
    vdb = getattr(dr_utils, "vectordb_instance", None)
    if vdb is None:
        return []
    try:
        raw = vdb.similarity_search_with_score(query, k=IMAGE_RECALL_POOL_K)
    except Exception as e:
        print(f"[Image-Recall] vss failed: {e}", flush=True)
        return []
    out: List[Dict[str, Any]] = []
    for doc, dist in raw or []:
        meta = dict(getattr(doc, "metadata", None) or {})
        rels = _norm_image_paths_field(meta.get("image_paths"))
        if not rels:
            continue
        content = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
        caption = content[:160] + ("..." if len(content) > 160 else "")
        source_full = _to_relative_source_path(meta.get("file_path") or meta.get("source") or "")
        source_label = _to_source_display_name(source_full)
        sim = 1.0 - float(dist)
        for rel in rels:
            if not _kb_image_exists(rel):
                continue
            out.append({
                "rel": rel,
                "caption": caption,
                "source": source_full,
                "source_label": source_label,
                "score": sim,
            })
        if len(out) >= IMAGE_RECALL_TOPK:
            break
    return out[:IMAGE_RECALL_TOPK]


async def _image_recall_for_query_async(query: str) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_image_recall_for_query_sync, query)


def _add_images_to_bucket(items: List[Dict[str, Any]]) -> int:
    """把 `_image_recall_for_query_*` 给出的图片合并进 SHARED_STATE['retrieved_images']。

    去重以 url 为键；超出 `MAX_RETRIEVED_IMAGES` 时停止追加。
    """
    if not items:
        return 0
    bucket: List[Dict[str, Any]] = list(SHARED_STATE.get("retrieved_images") or [])
    seen = {it.get("url") for it in bucket if isinstance(it, dict)}
    added = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        rel = str(it.get("rel") or "").strip()
        if not rel or not _kb_image_exists(rel):
            continue
        url = _kb_image_url(rel)
        if url in seen:
            continue
        seen.add(url)
        src_full = str(it.get("source") or "").strip()
        src_label = str(it.get("source_label") or _to_source_display_name(src_full)).strip()
        bucket.append({
            "url": url,
            "rel": rel,
            "caption": it.get("caption", ""),
            "source": src_full,
            "source_label": src_label,
            "score": float(it.get("score") or 0.0),
        })
        if src_full:
            print(f"[RAG-SOURCE] image source: '{src_full}' -> '{src_label}'", flush=True)
        added += 1
        if len(bucket) >= MAX_RETRIEVED_IMAGES:
            break
    if added:
        SHARED_STATE["retrieved_images"] = bucket
    return added


def _record_retrieved_images_from_results(results: List[Dict[str, Any]]) -> int:
    """从 `unified_local_search` 命中结果中提取 image_paths 写入 SHARED_STATE。

    去重以 url 为键，按命中顺序保留首次出现的 score / caption / source。
    返回新加入的图片数量。
    """
    if not isinstance(results, list) or not results:
        return 0
    bucket: List[Dict[str, Any]] = list(SHARED_STATE.get("retrieved_images") or [])
    seen = {item.get("url") for item in bucket if isinstance(item, dict)}
    added = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") or {}
        rels = _norm_image_paths_field(meta.get("image_paths"))
        if not rels:
            # 命中 chunk 自身没图就跳过；"语义相关的图"由 _image_recall_for_query_* 单独召回补齐。
            continue
        content = str(item.get("content") or "").strip().replace("\n", " ")
        caption = content[:160] + ("..." if len(content) > 160 else "")
        source_full = _to_relative_source_path(meta.get("file_path") or meta.get("source") or "")
        source_label = _to_source_display_name(source_full)
        score = float(meta.get("score") or 0.0)
        for rel in rels:
            if not _kb_image_exists(rel):
                continue
            url = _kb_image_url(rel)
            if url in seen:
                continue
            seen.add(url)
            bucket.append({
                "url": url,
                "rel": rel,
                "caption": caption,
                "source": source_full,
                "source_label": source_label,
                "score": score,
            })
            if source_full:
                print(f"[RAG-SOURCE] image source: '{source_full}' -> '{source_label}'", flush=True)
            added += 1
            if len(bucket) >= MAX_RETRIEVED_IMAGES:
                break
        if len(bucket) >= MAX_RETRIEVED_IMAGES:
            break
    if added:
        SHARED_STATE["retrieved_images"] = bucket
    return added


MAX_EVIDENCE_CHUNKS = int(os.getenv("MAX_EVIDENCE_CHUNKS", "12"))


def _record_evidence_chunks_from_results(query: str, results: List[Dict[str, Any]]) -> int:
    """从 unified_local_search 命中结果中提取文本证据片段写入 SHARED_STATE。

    与 `_record_retrieved_images_from_results` 是兄弟函数：图片走 retrieved_images，
    文本（含可能附带的图片缩略图）走 evidence_chunks。
    去重以 (source, content_hash) 为键，按命中顺序保留首次出现的 score / chunk_text。
    """
    if not isinstance(results, list) or not results:
        return 0
    bucket: List[Dict[str, Any]] = list(SHARED_STATE.get("evidence_chunks") or [])
    seen = {(item.get("source"), item.get("content_hash")) for item in bucket if isinstance(item, dict)}
    added = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") or {}
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        chunk_text = content if len(content) <= 800 else content[:800] + "..."
        source = _to_relative_source_path(meta.get("file_path") or meta.get("source") or "unknown")
        source_label = _to_source_display_name(source) or "unknown"
        score = float(meta.get("score") or 0.0)
        # 右侧"证据面板"里每条 chunk 的缩略图严格使用该 chunk 自挂的 image_paths（保证关联性），
        # 不再用邻近/同文档兜底。
        rels = _norm_image_paths_field(meta.get("image_paths"))
        rels_existing = [r for r in rels if _kb_image_exists(r)]
        urls = [_kb_image_url(r) for r in rels_existing]
        kind = "image" if (urls and len(content) < 80) else "text"
        content_hash = hash(content[:200])
        key = (source, content_hash)
        if key in seen:
            continue
        seen.add(key)
        bucket.append({
            "idx": len(bucket) + 1,
            "source": source,
            "source_label": source_label,
            "score": round(score, 4),
            "chunk_text": chunk_text,
            "image_paths": rels_existing,
            "image_urls": urls,
            "kind": kind,
            "query": str(query or "")[:120],
            "content_hash": content_hash,
        })
        if source:
            print(f"[RAG-SOURCE] evidence source: '{source}' -> '{source_label}'", flush=True)
        added += 1
        if len(bucket) >= MAX_EVIDENCE_CHUNKS:
            break
    if added:
        # 重新按 score 倒序，再编号 idx
        bucket.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        for i, it in enumerate(bucket):
            it["idx"] = i + 1
        SHARED_STATE["evidence_chunks"] = bucket
    return added


def _install_rag_image_capture_patch() -> None:
    """安装 monkey-patch：保留 chunk metadata + 捕获 image_paths。

    优化：原实现会在每次混合检索后**同步重跑**一次 similarity_search_with_score
    以拿回完整 metadata（含 image_paths），即每次 RAG 查询要打 2 次嵌入 API。
    这里改为：包装 vectordb.similarity_search_with_score 做"单次缓存"——
    第一次调用走 API 并缓存到 vectordb 实例上；patch 路径上的第二次调用直接命中缓存。
    """
    if getattr(dr_utils, "_kb_image_patch_installed", False):
        return

    # ---- 1. 保留 _lc_retrieve_hybrid_async 的全量 metadata ----
    original_hybrid = getattr(dr_utils, "_lc_retrieve_hybrid_async", None)
    if original_hybrid is not None:
        def _ensure_vss_cache(vectordb) -> None:
            if vectordb is None or getattr(vectordb, "_vss_cache_wrapped", False):
                return
            _orig_vss = vectordb.similarity_search_with_score

            def _wrapped_vss(q, k=4, **kw):
                key = (str(q), int(k))
                cached = getattr(vectordb, "_vss_last_call", None)
                if cached and cached[0] == key:
                    return cached[1]
                res = _orig_vss(q, k=k, **kw)
                vectordb._vss_last_call = (key, res)
                return res

            vectordb.similarity_search_with_score = _wrapped_vss
            vectordb._vss_cache_wrapped = True

        async def _patched_hybrid(query, vectordb, top_k, is_plan: bool = False):
            _ensure_vss_cache(vectordb)
            results = await original_hybrid(query, vectordb=vectordb, top_k=top_k, is_plan=is_plan)
            if not results:
                return results
            try:
                # 命中缓存（不再发起第二次嵌入查询），从原始 doc.metadata 拿 image_paths 等字段
                cached = getattr(vectordb, "_vss_last_call", None)
                docs_with_score = cached[1] if cached else []
                meta_by_content = {}
                for doc, _ in docs_with_score or []:
                    meta_by_content[doc.page_content] = dict(doc.metadata or {})
                for item in results:
                    extra = meta_by_content.get(item.get("content"), {})
                    if not extra:
                        continue
                    base_meta = item.get("metadata") or {}
                    for key in ("image_paths", "file_path", "full_doc_id", "_id", "tokens"):
                        if extra.get(key) is not None and key not in base_meta:
                            base_meta[key] = extra.get(key)
                    item["metadata"] = base_meta
            except Exception as patch_err:
                print(f"[RAG-IMG] metadata enrich failed: {patch_err}", flush=True)
            return results

        dr_utils._lc_retrieve_hybrid_async = _patched_hybrid

    # ---- 2. 包装 unified_local_search，捕获命中图片 + 文本证据 ----
    original_unified = getattr(dr_utils, "unified_local_search", None)
    if original_unified is not None:
        async def _patched_unified(query: str, top_k: int = 3, **kwargs):
            results = await original_unified(query, top_k=top_k, **kwargs)
            try:
                added_img = _record_retrieved_images_from_results(results or [])
                added_chunks = _record_evidence_chunks_from_results(query, results or [])
                # 图片单独召回：从向量库中按 query 直接挑"语义最相关的带图 chunk"，
                # 不依赖文本召回 top-k 是否恰好命中带图段落。
                added_img2 = 0
                try:
                    extra = await _image_recall_for_query_async(query)
                    added_img2 = _add_images_to_bucket(extra)
                except Exception as ir_err:
                    print(f"[RAG-CAPTURE] image-recall failed: {ir_err}", flush=True)
                if added_img or added_img2 or added_chunks:
                    print(
                        f"[RAG-CAPTURE] +{added_img} image(s) (hit), +{added_img2} image(s) (semantic), "
                        f"+{added_chunks} chunk(s) for query='{str(query)[:40]}'",
                        flush=True,
                    )
            except Exception as cap_err:
                print(f"[RAG-CAPTURE] capture failed: {cap_err}", flush=True)
            return results

        dr_utils.unified_local_search = _patched_unified

    # ---- 3. 包装 local_db._retrieve_hybrid_sync，覆盖深度模式 pre_brief_retrieval ----
    try:
        from deep_research import local_db as _local_db_mod
        original_sync = getattr(_local_db_mod, "_retrieve_hybrid_sync", None)
        if original_sync is not None and not getattr(_local_db_mod, "_evidence_patch_installed", False):
            def _patched_sync(query: str, top_k: int = 3, **kwargs):
                results = original_sync(query, top_k=top_k, **kwargs)
                try:
                    added_img = _record_retrieved_images_from_results(results or [])
                    added_chunks = _record_evidence_chunks_from_results(query, results or [])
                    # 同 _patched_unified：在深度模式下也跑一次"图片单独召回"。
                    added_img2 = 0
                    try:
                        extra = _image_recall_for_query_sync(query)
                        added_img2 = _add_images_to_bucket(extra)
                    except Exception as ir_err:
                        print(f"[RAG-CAPTURE-SYNC] image-recall failed: {ir_err}", flush=True)
                    if added_img or added_img2 or added_chunks:
                        print(
                            f"[RAG-CAPTURE-SYNC] +{added_img} image(s) (hit), +{added_img2} image(s) (semantic), "
                            f"+{added_chunks} chunk(s) for query='{str(query)[:40]}'",
                            flush=True,
                        )
                except Exception as cap_err:
                    print(f"[RAG-CAPTURE-SYNC] capture failed: {cap_err}", flush=True)
                return results

            _local_db_mod._retrieve_hybrid_sync = _patched_sync
            _local_db_mod._evidence_patch_installed = True
    except Exception as patch_err:
        print(f"[Patch] local_db evidence capture failed: {patch_err}", flush=True)

    dr_utils._kb_image_patch_installed = True
    print("[Patch] RAG image-capture installed (image_paths -> SHARED_STATE).", flush=True)


_install_rag_image_capture_patch()


from deep_research.research_agent_full import deep_researcher_builder
from deep_research.single_agent_supervisor import single_agent, build_single_supervisor_graph

try:
    FAST_AGENT = build_single_supervisor_graph(checkpointer=MEMORY, enable_history=True)
    print("[Init] Fast-mode agent ready with MemorySaver", flush=True)
except Exception as _e:
    print(f"[Init] Fast-mode fallback: {_e}", flush=True)
    FAST_AGENT = single_agent


# ==========================================
# 2. 核心业务逻辑
# ==========================================

def format_node_name(name):
    mapping = {
        "clarify_with_user": "意图澄清",
        "write_draft_report": "初步分析/计划",
        "pre_brief_retrieval": "深度检索",
        "solve_simple_task": "快速响应",
        "supervisor_subgraph": "多智能体协作",
        "final_report_generation": "报告生成",
        "retrieve_battery_node": "数据库诊断",
        "generate_chart_node": "绘制图表",
        "supervisor": "监督调度",
    }
    return mapping.get(name, name)


def find_recursive(data, key):
    if isinstance(data, dict):
        if key in data and data[key]:
            return data[key]
        for v in data.values():
            res = find_recursive(v, key)
            if res:
                return res
    return None


def _truncate_text(text: Any, max_len: int = 140) -> str:
    s = str(text or "").replace("\n", " ").strip()
    return s if len(s) <= max_len else s[:max_len] + "..."


def _compact_sql(sql: str) -> str:
    return _truncate_text(" ".join(str(sql or "").split()), max_len=180)


def _extract_plan_tables(plans: Any) -> List[str]:
    if not isinstance(plans, list):
        return []
    out: List[str] = []
    for p in plans:
        if isinstance(p, dict):
            t = str(p.get("table", "")).strip()
            if t:
                out.append(t)
    return out


def _build_prior_turns(chat_history: Optional[List[Dict[str, Any]]], max_pairs: int = 5) -> List[str]:
    if not chat_history:
        return []
    items = [m for m in chat_history if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    if not items:
        return []
    turns: List[Tuple[str, str]] = []
    i = 0
    while i < len(items):
        m = items[i]
        if m.get("role") == "user":
            user_text = str(m.get("content", "") or "").strip()
            ai_parts: List[str] = []
            j = i + 1
            while j < len(items) and items[j].get("role") == "assistant":
                content = str(items[j].get("content", "") or "").strip()
                if content and "Thinking..." not in content and "分析完成" not in content:
                    ai_parts.append(content)
                j += 1
            ai_text = "\n".join(ai_parts).strip()
            if user_text and ai_text:
                turns.append((user_text, ai_text))
            i = j if j > i else i + 1
        else:
            i += 1
    turns = turns[-max_pairs:]
    result: List[str] = []
    for u, a in turns:
        u_trim = u if len(u) <= 600 else u[:600] + "..."
        a_trim = a if len(a) <= 1500 else a[:1500] + "...(truncated)"
        result.append(f"User: {u_trim}\nAI: {a_trim}")
    return result


# =============================================================================
# Plus v2: 5 类前端布局所需的结构化字段提取
# =============================================================================
_ALARM_TABLE_KEYS = {"alarm_event", "alarm_events"}
_ALARM_ROW_FIELDS = (
    "id", "start_time", "end_time", "ts",
    "station_code", "bms_code", "bmu_code", "pack_code", "cell_id",
    "severity", "average_severity", "max_severity",
    "summary_cn", "summary", "alarm_type", "alarm_code",
    "duration", "trigger_count",
)

# 把告警 summary_cn 切成可分类的标签（充电/放电/静置 + 电压/温度/容量等）
_ALARM_KEYWORD_BUCKETS = [
    ("充电工况异常", ("充电", "充电中", "充电时", "充电过程")),
    ("放电工况异常", ("放电", "放电中", "放电时", "放电过程")),
    ("静置工况", ("静置",)),
    ("电压异常", ("电压", "压差", "压低", "压高", "过压", "欠压")),
    ("温度异常", ("温度", "过温", "高温", "低温", "温差")),
    ("电流异常", ("电流", "过流")),
    ("容量异常", ("容量", "SOC", "soc", "SOH", "soh")),
    ("内阻异常", ("内阻", "DCR", "阻抗", "阻值")),
    ("一致性异常", ("一致性", "离散", "均衡", "偏差")),
]


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _build_alarm_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    cells = sorted({str(r.get("cell_id") or "").strip() for r in rows if r.get("cell_id")})
    sev_vals: List[float] = []
    avg_sev_vals: List[float] = []
    for r in rows:
        s = _coerce_float(r.get("severity"))
        if s is not None:
            sev_vals.append(s)
        a = _coerce_float(r.get("average_severity"))
        if a is not None:
            avg_sev_vals.append(a)
    summaries = [str(r.get("summary_cn") or r.get("summary") or "") for r in rows]
    bucket_counts: Dict[str, int] = {label: 0 for label, _ in _ALARM_KEYWORD_BUCKETS}
    for s in summaries:
        for label, kws in _ALARM_KEYWORD_BUCKETS:
            if any(k in s for k in kws):
                bucket_counts[label] += 1
    top_keywords = sorted(
        [(k, v) for k, v in bucket_counts.items() if v > 0],
        key=lambda x: x[1], reverse=True,
    )
    max_sev = max(sev_vals) if sev_vals else (max(avg_sev_vals) if avg_sev_vals else 0.0)
    avg_sev = (sum(avg_sev_vals) / len(avg_sev_vals)) if avg_sev_vals else (
        (sum(sev_vals) / len(sev_vals)) if sev_vals else 0.0
    )
    if max_sev >= 4:
        risk_label, risk_color = "高严重度告警", "red"
    elif max_sev >= 2:
        risk_label, risk_color = "中严重度告警", "orange"
    elif max_sev > 0:
        risk_label, risk_color = "低严重度告警", "blue"
    else:
        risk_label, risk_color = "无严重度数据", "blue"
    return {
        "total": len(rows),
        "cells_involved": len(cells),
        "cells_sample": cells[:6],
        "avg_severity": round(avg_sev, 2),
        "max_severity": round(max_sev, 2),
        "top_keywords": top_keywords,
        "risk_label": risk_label,
        "risk_color": risk_color,
    }


def _build_sql_table(rows: List[Dict[str, Any]], max_rows: int = 200) -> Dict[str, Any]:
    valid = [r for r in rows if isinstance(r, dict) and not r.get("error")]
    if not valid:
        return {"columns": [], "rows": [], "title": ""}
    # 列顺序：以第一行为基准，但优先把时间列放到最前
    keys: List[str] = []
    for r in valid[:5]:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    priority = ("ts", "start_time", "end_time", "time", "cell_id", "bmu_code", "station_code")
    columns = [k for k in priority if k in keys] + [k for k in keys if k not in priority]
    out_rows: List[List[Any]] = []
    for r in valid[:max_rows]:
        out_rows.append([r.get(c) for c in columns])
    return {
        "columns": columns,
        "rows": out_rows,
        "title": "",
        "total": len(valid),
        "shown": len(out_rows),
    }


def _extract_structured_data(output: dict):
    """从 Agent 输出中提取结构化数据，填充 SHARED_STATE 的 Plus 字段。

    扩展 v2：除原有 metrics/diagnosis_tags/diagnosis_summary 外，新增：
      - SHARED_STATE["task_type"]   ← state.task_type
      - SHARED_STATE["alarm_rows"]  ← raw_db_results 中的告警行（按 alarm_event 表）
      - SHARED_STATE["alarm_summary"] ← 总数 / 涉及电芯 / 平均严重度 / 主要异常类型
      - SHARED_STATE["sql_table"]   ← 通用表格（station_device_td 卡片底部展示）
    """
    # task_type 透传
    tt = find_recursive(output, "task_type")
    if isinstance(tt, str) and tt.strip():
        SHARED_STATE["task_type"] = tt.strip()

    db_sqls = find_recursive(output, "db_executed_sqls")
    if isinstance(db_sqls, list):
        SHARED_STATE["executed_sqls"] = [str(s) for s in db_sqls if isinstance(s, str) and s.strip()]

    db_evidence = find_recursive(output, "db_evidence_bundle")
    if isinstance(db_evidence, dict):
        suff = db_evidence.get("evidence_sufficiency")
        conf = db_evidence.get("confidence")
        concl = db_evidence.get("diagnosis_conclusion", "")
        SHARED_STATE["diagnosis_summary"] = str(concl)[:500] if concl else ""

        tags = []
        if suff and str(suff).lower() in ("sufficient", "high"):
            tags.append({"text": "数据充足", "color": "green"})
        if conf and float(conf) > 0.8:
            tags.append({"text": "高置信度", "color": "green"})
        elif conf and float(conf) > 0.5:
            tags.append({"text": "中置信度", "color": "orange"})
        SHARED_STATE["diagnosis_tags"] = tags

    raw_rows = find_recursive(output, "raw_db_results")
    if raw_rows and isinstance(raw_rows, list):
        # 1) metrics: 收集所有非 null 数值 → 求均值
        from collections import defaultdict
        accum: dict[str, list] = defaultdict(list)
        _METRIC_KEYS = ("soc", "soh", "voltage", "temperature", "current", "temp",
                        "cell_avg_vol", "cell_avg_temp", "tmax", "tmin", "vmax", "vmin")
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            for key in _METRIC_KEYS:
                val = row.get(key)
                if val is None:
                    val = row.get(key.upper())
                if val is not None:
                    try:
                        accum[key.lower()].append(float(val))
                    except (ValueError, TypeError):
                        pass
            if row.get("alarm_count") is not None:
                try:
                    accum["alarm_count"].append(float(row["alarm_count"]))
                except (ValueError, TypeError):
                    pass
        metrics = {}
        for k, vals in accum.items():
            if vals:
                metrics[k] = round(sum(vals) / len(vals), 3)
        if "temp" in metrics and "temperature" not in metrics:
            metrics["temperature"] = metrics.pop("temp")
        if metrics:
            SHARED_STATE["structured_metrics"] = metrics

        # 2) 通用 SQL 表（station_device_td 用）
        sql_table = _build_sql_table(raw_rows)
        if sql_table.get("columns"):
            SHARED_STATE["sql_table"] = sql_table

        # 3) 按 db_route 归一化数据库三场景 task_type
        db_route = find_recursive(output, "db_route") or ""
        if isinstance(db_route, str):
            route_l = db_route.lower()
            if route_l in {"station_device_td", "alerting", "troubleshooting"}:
                SHARED_STATE["task_type"] = route_l

        # 4) 告警明细 + 概要：按 db_route 或表名识别
        is_alarm_route = (
            (isinstance(db_route, str) and db_route.lower() == "alerting")
            or any(
                isinstance(r, dict) and str(r.get("table_name") or "").lower() in _ALARM_TABLE_KEYS
                for r in raw_rows
            )
        )
        if is_alarm_route:
            alarm_rows: List[Dict[str, Any]] = []
            for r in raw_rows:
                if not isinstance(r, dict) or r.get("error"):
                    continue
                slim = {k: r.get(k) for k in _ALARM_ROW_FIELDS if k in r}
                if not slim:
                    # 表名不是 alarm_event 但落在 alerting route：保留全部小行
                    slim = dict(r)
                alarm_rows.append(slim)
            if alarm_rows:
                # 按 average_severity > severity > start_time 倒序，方便前端默认展示
                def _sort_key(x: Dict[str, Any]):
                    return (
                        -(_coerce_float(x.get("average_severity")) or 0.0),
                        -(_coerce_float(x.get("severity")) or 0.0),
                        str(x.get("start_time") or ""),
                    )
                alarm_rows.sort(key=_sort_key)
                SHARED_STATE["alarm_rows"] = alarm_rows[:200]
                SHARED_STATE["alarm_summary"] = _build_alarm_summary(alarm_rows)
                # 同时把 task_type 校正为 alerting（兜底，防止 supervisor 没标对）
                SHARED_STATE["task_type"] = "alerting"


async def background_graph_runner(
    question: str,
    deep_mode: bool = False,
    prior_turns: Optional[List[str]] = None,
):
    global CURRENT_THREAD_ID
    try:
        SHARED_STATE["is_running"] = True
        mode_label = "Deep Research" if deep_mode else "Fast"
        SHARED_STATE["logs"].append(f"[Start - {mode_label}] {datetime.datetime.now().strftime('%H:%M:%S')}\n")

        if prior_turns:
            SHARED_STATE["logs"].append(
                f"[Context] injected {len(prior_turns)} prior turns (thread={CURRENT_THREAD_ID})\n"
            )

        dr_utils.set_models(gemini_chat_runnable)
        history_list: List[str] = list(prior_turns or [])

        if deep_mode:
            try:
                graph = deep_researcher_builder(llm=gemini_chat_runnable, checkpointer=MEMORY)
                if hasattr(graph, "compile") and not hasattr(graph, "astream_events"):
                    graph = graph.compile()
            except Exception:
                graph = deep_researcher_builder().compile()
            inputs = {
                "messages": [HumanMessage(content=question)],
                "user_request": question,
            }
        else:
            graph = FAST_AGENT
            inputs = {
                "messages": [HumanMessage(content=question)],
                "user_request": question,
                "raw_notes": [],
                "notes": [],
                "supervisor_messages": [],
            }

        config = {"configurable": {"thread_id": CURRENT_THREAD_ID}}

        prior_state = None
        try:
            prior_state = graph.get_state(config)
        except Exception:
            pass
        prior_values = (getattr(prior_state, "values", None) or {}) if prior_state else {}

        if history_list:
            if not prior_values.get("history"):
                inputs["history"] = history_list

        saved_brief = SHARED_STATE.get("last_pre_brief_cases", "") or ""
        if saved_brief and not prior_values.get("pre_brief_cases"):
            inputs["pre_brief_cases"] = saved_brief

        saved_db_rows = SHARED_STATE.get("last_db_raw_results", []) or []
        if isinstance(saved_db_rows, list) and saved_db_rows and not prior_values.get("db_raw_results"):
            inputs["db_raw_results"] = saved_db_rows

        node_start_times: Dict[str, float] = {}
        tracked_nodes = {
            "clarify_with_user", "write_draft_report", "pre_brief_retrieval",
            "solve_simple_task", "supervisor_subgraph", "final_report_generation",
            "generate_chart_node", "retrieve_battery_node",
        }

        async for event in graph.astream_events(inputs, config=config, version="v1"):
            kind, name, data = event.get("event"), event.get("name"), event.get("data", {})

            if kind == "on_chain_start" and name in tracked_nodes:
                node_start_times[name] = time.perf_counter()
                existing = next((s for s in SHARED_STATE["steps_data"] if s["name"] == name), None)
                if not existing:
                    SHARED_STATE["steps_data"].append(
                        {"name": name, "status": "running", "detail": "", "duration": None}
                    )
                else:
                    existing["status"] = "running"
                SHARED_STATE["logs"].append(f"\nNode: {format_node_name(name)}")

            elif kind == "on_chain_end":
                if name in node_start_times:
                    duration = time.perf_counter() - node_start_times.pop(name)
                    for step in SHARED_STATE["steps_data"]:
                        if step["name"] == name:
                            step["status"] = "completed"
                            step["duration"] = duration
                    SHARED_STATE["logs"].append(
                        f"\n✅ Node Done: {format_node_name(name)} ({duration:.2f}s)"
                    )

                if isinstance(data.get("output"), dict):
                    output = data["output"]

                    chart = find_recursive(output, "chart_output")
                    if chart:
                        SHARED_STATE["chart_url"] = chart
                        SHARED_STATE["logs"].append(f"\n📊 Chart generated: {os.path.basename(chart)}")

                    report = find_recursive(output, "final_report")
                    if report:
                        SHARED_STATE["final_report"] = _compose_full_report(
                            report,
                            SHARED_STATE["chart_url"],
                            SHARED_STATE.get("retrieved_images") or [],
                        )
                        n_imgs = len(SHARED_STATE.get("retrieved_images") or [])
                        SHARED_STATE["logs"].append(
                            f"\n📄 Final report updated ({len(report)} chars, +{n_imgs} kb image(s))"
                        )

                    _extract_structured_data(output)

                    # --- Detailed DB logging (same as server.py) ---
                    db_route = find_recursive(output, "db_route")
                    db_plans = find_recursive(output, "db_query_plans")
                    db_sqls = find_recursive(output, "db_executed_sqls")
                    db_warnings = find_recursive(output, "db_plan_sanitizer_warnings")
                    db_evidence = find_recursive(output, "db_evidence_bundle")
                    db_llm_traces = find_recursive(output, "db_llm_traces")
                    db_related = bool(
                        db_route or db_plans or db_sqls or db_warnings
                        or (isinstance(db_llm_traces, dict) and db_llm_traces)
                    )

                    if db_related:
                        if isinstance(db_route, str):
                            SHARED_STATE["logs"].append(f"\n🧭 DB Route: {db_route}")

                        plan_tables = _extract_plan_tables(db_plans)
                        if plan_tables:
                            uniq_tables = sorted(set(plan_tables))
                            SHARED_STATE["logs"].append(
                                f"\n🗂 DB Plans: {len(plan_tables)} query plan(s) | tables: {', '.join(uniq_tables)}"
                            )

                        if isinstance(db_sqls, list) and db_sqls:
                            for i, sql in enumerate(db_sqls[:3], 1):
                                SHARED_STATE["logs"].append(
                                    f"\n🧾 SQL[{i}/{len(db_sqls)}]: {_compact_sql(str(sql))}"
                                )
                            if len(db_sqls) > 3:
                                SHARED_STATE["logs"].append(
                                    f"\n🧾 SQL: ... (+{len(db_sqls) - 3} more queries)"
                                )

                        if isinstance(db_warnings, list) and db_warnings:
                            SHARED_STATE["logs"].append(
                                f"\n🧼 Sanitizer Warnings: {len(db_warnings)} plan(s) adjusted"
                            )
                            for warn in db_warnings[:3]:
                                if not isinstance(warn, dict):
                                    continue
                                idx = warn.get("plan_index", "?")
                                w_items = warn.get("warnings", [])
                                if isinstance(w_items, list) and w_items:
                                    SHARED_STATE["logs"].append(
                                        f"\n   - plan#{idx}: {_truncate_text('; '.join(map(str, w_items)), 200)}"
                                    )

                        if isinstance(db_evidence, dict):
                            suff = db_evidence.get("evidence_sufficiency")
                            conf = db_evidence.get("confidence")
                            concl = db_evidence.get("diagnosis_conclusion")
                            if suff is not None or conf is not None or concl:
                                SHARED_STATE["logs"].append(
                                    f"\n📦 Evidence Bundle:"
                                    f"\n   sufficiency = {suff}"
                                    f"\n   confidence  = {conf}"
                                    f"\n   conclusion  = {_truncate_text(concl, 200)}"
                                )

                        if isinstance(db_llm_traces, dict) and db_llm_traces:
                            SHARED_STATE["logs"].append(
                                f"\n🤖 DB LLM Traces ({len(db_llm_traces)} steps):"
                            )
                            for k in sorted(db_llm_traces.keys()):
                                block = db_llm_traces.get(k)
                                if not isinstance(block, dict):
                                    continue
                                raw_r = block.get("raw_response") or block.get("partial_raw") or ""
                                if raw_r:
                                    SHARED_STATE["logs"].append(
                                        f"\n   [{k}] {_truncate_text(raw_r, 600)}"
                                    )
                                if block.get("error"):
                                    SHARED_STATE["logs"].append(
                                        f"\n   [{k}] ERROR: {block.get('error')}"
                                    )

                    raw_rows = find_recursive(output, "raw_db_results")
                    if raw_rows and isinstance(raw_rows, list):
                        formatted_rows = []
                        for row in raw_rows:
                            if isinstance(row, dict):
                                formatted_rows.append(
                                    [str(row.get("cell_id", "")), str(row.get("summary_cn", ""))]
                                )
                            else:
                                formatted_rows.append([str(row), ""])
                        SHARED_STATE["raw_dataframe"] = formatted_rows
                        SHARED_STATE["last_db_raw_results"] = raw_rows
                        SHARED_STATE["logs"].append(
                            f"\n🔢 Retrieved {len(raw_rows)} DB records"
                        )

                    q_params = find_recursive(output, "db_query_params")
                    if q_params and isinstance(q_params, dict):
                        SHARED_STATE["query_params"] = q_params
                        SHARED_STATE["logs"].append(
                            f"\n🔍 Query Params: {json.dumps(q_params, ensure_ascii=False)}"
                        )
                        cands = q_params.get("clarify_candidates")
                        if cands and isinstance(cands, list):
                            SHARED_STATE["clarify_candidates"] = cands

                    r_brief = find_recursive(output, "research_brief")
                    if r_brief and isinstance(r_brief, str):
                        SHARED_STATE["research_brief"] = r_brief
                        SHARED_STATE["logs"].append(
                            f"\n📋 Research Brief created ({len(r_brief)} chars)"
                        )

                    r_notes = find_recursive(output, "notes")
                    if r_notes and isinstance(r_notes, list):
                        notes_text = "\n\n".join([f"- {n}" for n in r_notes if isinstance(n, str)])
                        if notes_text:
                            SHARED_STATE["research_notes"] = notes_text
                            SHARED_STATE["logs"].append(
                                f"\n📝 Collected {len(r_notes)} research notes"
                            )

                    d_report = find_recursive(output, "draft_report")
                    if d_report and isinstance(d_report, str):
                        SHARED_STATE["draft_report"] = d_report
                        SHARED_STATE["logs"].append(
                            f"\n📝 Draft report updated ({len(d_report)} chars)"
                        )

        try:
            final_state = graph.get_state(config)
            final_vals = (getattr(final_state, "values", None) or {}) if final_state else {}
            pbc = final_vals.get("pre_brief_cases", "") or ""
            if pbc:
                SHARED_STATE["last_pre_brief_cases"] = pbc
            raw_rows = final_vals.get("db_raw_results", []) or []
            if isinstance(raw_rows, list) and raw_rows:
                SHARED_STATE["last_db_raw_results"] = raw_rows
        except Exception:
            pass

    except Exception as e:
        import traceback
        SHARED_STATE["logs"].append(f"\nError: {str(e)}\n{traceback.format_exc()}")
        print(f"Error: {e}", flush=True)
    finally:
        SHARED_STATE["is_running"] = False


# ==========================================
# 3. FastAPI 应用
# ==========================================
app = FastAPI(title="科宝 Cobot")

os.makedirs(PROJECT_ROOT / "figure", exist_ok=True)
os.makedirs(PROJECT_ROOT / "reports", exist_ok=True)
app.mount("/figure", StaticFiles(directory=str(PROJECT_ROOT / "figure")), name="figure")
app.mount("/reports", StaticFiles(directory=str(PROJECT_ROOT / "reports")), name="reports")
# 暴露 RAG 知识库图片：浏览器 `<img src="/kb_images/doc-XXX/HASH.jpg">` 即可加载
app.mount("/kb_images", StaticFiles(directory=str(KB_IMAGES_DIR)), name="kb_images")

from deep_research import funasr_service as _funasr


@app.get("/api/asr/status")
async def api_asr_status():
    return {
        "enabled": _funasr.is_enabled(),
        "ffmpeg": _funasr.ffmpeg_available(),
        "model_loaded": _funasr.model_loaded(),
        "model": os.getenv("FUNASR_MODEL", "paraformer-zh"),
        "last_error": _funasr.last_load_error(),
    }


@app.post("/api/asr/transcribe")
async def api_asr_transcribe(audio: UploadFile = File(...)):
    if not _funasr.is_enabled():
        raise HTTPException(
            status_code=503,
            detail="FunASR 未启用或未安装。请在 thinkdepth 环境执行: pip install funasr modelscope",
        )
    raw = await audio.read()
    limit = _funasr.max_upload_bytes()
    if len(raw) > limit:
        mb = limit // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"音频过大，上限约 {mb} MB")

    suffix = Path(audio.filename or "upload.webm").suffix or ".webm"
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        text = await asyncio.to_thread(_funasr.transcribe_file, tmp_path, suffix)
        return {"text": text or ""}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


class ChatRequest(BaseModel):
    message: str
    mode: str = "fast"


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = PROJECT_ROOT / "templates" / "chat_plus.html"
    return HTMLResponse(content=html_path.read_text("utf-8"))


@app.head("/")
async def index_head():
    """健康检查/负载均衡常用 HEAD /；无此路由时 FastAPI 会对 HEAD 返回 405 并刷屏 INFO。"""
    html_path = PROJECT_ROOT / "templates" / "chat_plus.html"
    body = html_path.read_bytes()
    return Response(
        content=b"",
        headers={
            "content-length": str(len(body)),
            "content-type": "text/html; charset=utf-8",
        },
    )


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if SHARED_STATE.get("is_running", False):
        return {"status": "busy", "message": "上一条分析仍在执行中，请等待完成后再发送。"}

    SHARED_STATE["chat_history"].append({"role": "user", "content": req.message, "ts": time.time()})

    SHARED_STATE["logs"] = []
    SHARED_STATE["steps_data"] = []
    SHARED_STATE["final_report"] = ""
    SHARED_STATE["stream_buffer"] = ""
    SHARED_STATE["chart_url"] = None
    SHARED_STATE["raw_dataframe"] = []
    SHARED_STATE["query_params"] = {}
    SHARED_STATE["structured_metrics"] = {}
    SHARED_STATE["diagnosis_tags"] = []
    SHARED_STATE["diagnosis_summary"] = ""
    SHARED_STATE["diagnosis_highlights"] = []
    SHARED_STATE["suggestions"] = []
    SHARED_STATE["retrieved_images"] = []
    SHARED_STATE["answer_images"] = []
    # --- Plus v2: 6 类布局相关字段每轮重置 ---
    SHARED_STATE["task_type"] = "direct"
    SHARED_STATE["evidence_chunks"] = []
    SHARED_STATE["alarm_rows"] = []
    SHARED_STATE["alarm_summary"] = {}
    SHARED_STATE["sql_table"] = {"columns": [], "rows": [], "title": ""}
    SHARED_STATE["executed_sqls"] = []
    SHARED_STATE["clarify_candidates"] = []
    SHARED_STATE["current_mode"] = req.mode

    prior_turns = _build_prior_turns(SHARED_STATE["chat_history"], max_pairs=5)
    deep_mode = req.mode == "deep"

    asyncio.create_task(background_graph_runner(req.message, deep_mode=deep_mode, prior_turns=prior_turns))

    return {"status": "started", "mode": req.mode}


@app.get("/api/stream")
async def api_stream():
    async def event_generator():
        last_report = ""
        last_log_len = 0
        last_steps_hash = ""
        tick = 0
        response_saved = False

        while True:
            is_running = SHARED_STATE.get("is_running", False)

            # 流式片段也带上图表/RAG 图片，最大化"边写边出图"的体感
            report = SHARED_STATE.get("stream_buffer") or SHARED_STATE.get("final_report", "")
            report = _compose_full_report(
                report,
                SHARED_STATE.get("chart_url"),
                SHARED_STATE.get("retrieved_images") or [],
            )

            if report != last_report:
                last_report = report
                yield f"event: report\ndata: {json.dumps({'html': report}, ensure_ascii=False)}\n\n"

            logs = SHARED_STATE.get("logs", [])
            if len(logs) > last_log_len:
                new_logs = logs[last_log_len:]
                last_log_len = len(logs)
                yield f"event: log\ndata: {json.dumps({'entries': new_logs}, ensure_ascii=False)}\n\n"

            steps = SHARED_STATE.get("steps_data", [])
            steps_hash = json.dumps(steps, ensure_ascii=False, default=str)
            if steps_hash != last_steps_hash:
                last_steps_hash = steps_hash
                yield f"event: timeline\ndata: {json.dumps({'steps': steps}, ensure_ascii=False)}\n\n"

            metrics = SHARED_STATE.get("structured_metrics", {})
            chart_url = _chart_public_url(SHARED_STATE.get("chart_url"))
            diag_tags = SHARED_STATE.get("diagnosis_tags", [])
            diag_summary = SHARED_STATE.get("diagnosis_summary", "")
            kb_images = SHARED_STATE.get("retrieved_images") or []
            task_type = SHARED_STATE.get("task_type") or "direct"
            evidence_chunks = SHARED_STATE.get("evidence_chunks") or []
            alarm_rows = SHARED_STATE.get("alarm_rows") or []
            alarm_summary = SHARED_STATE.get("alarm_summary") or {}
            sql_table = SHARED_STATE.get("sql_table") or {"columns": [], "rows": [], "title": ""}
            executed_sqls = SHARED_STATE.get("executed_sqls") or []
            clarify_candidates = SHARED_STATE.get("clarify_candidates") or []

            if tick % 10 == 0:
                state_payload = {
                    "is_running": is_running,
                    "metrics": metrics,
                    "chart_url": chart_url,
                    "diagnosis_tags": diag_tags,
                    "diagnosis_summary": diag_summary,
                    "mode": SHARED_STATE.get("current_mode", "fast"),
                    "kb_images": kb_images,
                    # --- Plus v2: 6 类前端布局所需 ---
                    "task_type": task_type,
                    "evidence_chunks": evidence_chunks,
                    "alarm_rows": alarm_rows,
                    "alarm_summary": alarm_summary,
                    "sql_table": sql_table,
                    "executed_sqls": executed_sqls,
                    "clarify_candidates": clarify_candidates,
                }
                yield f"event: state\ndata: {json.dumps(state_payload, ensure_ascii=False, default=str)}\n\n"

            if not is_running and tick > 5:
                # Always send a final state snapshot so the frontend
                # receives any data populated after the last periodic state event.
                final_metrics = SHARED_STATE.get("structured_metrics", {})
                final_chart = _chart_public_url(SHARED_STATE.get("chart_url"))
                final_sql_table = SHARED_STATE.get("sql_table") or {"columns": [], "rows": [], "title": ""}
                final_sqls = SHARED_STATE.get("executed_sqls") or []
                final_state = {
                    "is_running": False,
                    "metrics": final_metrics,
                    "chart_url": final_chart,
                    "diagnosis_tags": SHARED_STATE.get("diagnosis_tags", []),
                    "diagnosis_summary": SHARED_STATE.get("diagnosis_summary", ""),
                    "mode": SHARED_STATE.get("current_mode", "fast"),
                    "kb_images": SHARED_STATE.get("retrieved_images") or [],
                    "task_type": SHARED_STATE.get("task_type") or "direct",
                    "evidence_chunks": SHARED_STATE.get("evidence_chunks") or [],
                    "alarm_rows": SHARED_STATE.get("alarm_rows") or [],
                    "alarm_summary": SHARED_STATE.get("alarm_summary") or {},
                    "sql_table": final_sql_table,
                    "executed_sqls": final_sqls,
                    "clarify_candidates": SHARED_STATE.get("clarify_candidates") or [],
                }
                yield f"event: state\ndata: {json.dumps(final_state, ensure_ascii=False, default=str)}\n\n"

                if report and not response_saved:
                    response_saved = True
                    SHARED_STATE["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": report,
                            "kb_images": list(SHARED_STATE.get("retrieved_images") or []),
                            "ts": time.time(),
                        }
                    )
                yield f"event: complete\ndata: {json.dumps({'status': 'done'})}\n\n"
                break

            tick += 1
            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/reset")
async def api_reset():
    global CURRENT_THREAD_ID, MEMORY

    old_thread = CURRENT_THREAD_ID
    CURRENT_THREAD_ID = f"web_{int(time.time())}"

    try:
        if hasattr(MEMORY, "clear"):
            MEMORY.clear()
    except Exception:
        pass
    MEMORY = MemorySaver()

    for key in list(SHARED_STATE.keys()):
        if isinstance(SHARED_STATE[key], list):
            SHARED_STATE[key] = []
        elif isinstance(SHARED_STATE[key], dict):
            SHARED_STATE[key] = {}
        elif isinstance(SHARED_STATE[key], bool):
            SHARED_STATE[key] = False if key != "session_memory" else True
        elif isinstance(SHARED_STATE[key], str):
            SHARED_STATE[key] = "" if key != "workspace_name" else "默认工作空间"
        else:
            SHARED_STATE[key] = None

    return {"status": "reset", "thread_id": CURRENT_THREAD_ID}


@app.get("/api/history")
async def api_history():
    return {"history": SHARED_STATE.get("chat_history", [])}


@app.get("/api/examples")
async def api_examples():
    examples_path = PROJECT_ROOT / "example_questions.json"
    if not examples_path.exists():
        return {"questions": []}
    return {"questions": json.loads(examples_path.read_text("utf-8"))}


@app.get("/api/download")
async def api_download():
    report = SHARED_STATE.get("final_report", "")
    if not report:
        return {"error": "暂无报告可下载"}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Report_{ts_file}.html"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    filepath = reports_dir / filename
    chart_path = SHARED_STATE.get("chart_url")
    content = _build_report_html(report, chart_path, timestamp)
    filepath.write_text(content, encoding="utf-8")
    return FileResponse(str(filepath), filename=filename, media_type="text/html")


# ==========================================
# 4. 启动
# ==========================================
if __name__ == "__main__":
    host, port = "0.0.0.0", SERVER_PLUS_PORT
    print(f"科宝 Cobot starting: http://localhost:{port}", flush=True)
    _configure_uvicorn_access_logging()
    uvicorn.run(app, host=host, port=port, access_log=_SERVER_PLUS_ACCESS_LOG)
