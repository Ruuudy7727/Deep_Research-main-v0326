#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Research Utilities and Tools.

本模块在原有基础上增加了第二个本地数据库目录 CHROMA_PERSIST_PLAN_DIR 的支持：
- 读取 CHROMA_PERSIST_DIR 与 CHROMA_PERSIST_PLAN_DIR 两个持久化目录
- 使用同一套 CHROMA_COLLECTION_NAME、EMBED_BASE_URL、EMBED_MODEL 配置
- 为两个库分别构建 BM25 索引
- 在统一检索 unified_local_search 中融合两个库的检索结果

其余接口与行为保持兼容。
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Dict, Callable
import json
import time
import asyncio
import importlib
import traceback
import glob

from typing_extensions import Annotated, List, Literal

from deep_research.embedding_client import build_embedding_client, get_embedding_settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 尽量早加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
except Exception:
    pass

# LangChain 基础依赖
try:
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.tools import tool, InjectedToolArg
except Exception as e:
    raise RuntimeError("缺少 langchain-core 相关依赖，请先安装: pip install langchain-core") from e

# Summary 结构：优先使用项目内定义；若不可用，提供一个最小兜底定义
try:
    from deep_research.state_research import Summary  # 需具备字段: summary, key_excerpts
except Exception:
    try:
        from pydantic import BaseModel
    except Exception:
        class _SummaryFallback:
            def __init__(self, summary: str = "", key_excerpts: str = ""):
                self.summary = summary
                self.key_excerpts = key_excerpts
        Summary = _SummaryFallback
    else:
        class Summary(BaseModel):
            summary: str = ""
            key_excerpts: str = ""

# Tavily：如果没有 API Key，使用 Dummy 客户端，避免导入即报错
try:
    from tavily import TavilyClient as _RealTavilyClient
    _have_tavily_pkg = True
except Exception:
    _have_tavily_pkg = False

class _DummyTavilyClient:
    def search(self, query: str, max_results: int = 3, include_raw_content: bool = True, topic: str = "general"):
        return {"query": query, "results": []}

def _build_tavily_client():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
    except Exception:
        pass
    api_key = os.getenv("TAVILY_API_KEY", "")
    if _have_tavily_pkg and api_key:
        try:
            return _RealTavilyClient(api_key=api_key)
        except Exception:
            return _DummyTavilyClient()
    return _DummyTavilyClient()

def get_today_str() -> str:
    try:
        return datetime.now().strftime("%a %b %-d, %Y")
    except Exception:
        return datetime.now().strftime("%a %b %d, %Y")

def get_current_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

# ===== CONFIGURATION =====
# 惰性初始化 + 可注入
_summarization_model: Optional[Any] = None
_writer_model: Optional[Any] = None

# 工具审计配置
_TOOL_AUDIT_ENABLED: bool = False
_TOOL_AUDIT_LOG_PATH: str = os.getenv("TOOL_AUDIT_LOG_PATH", "/tmp/agent_tool_calls.log")

def set_tool_audit(enabled: bool = True, path: Optional[str] = None):
    global _TOOL_AUDIT_ENABLED, _TOOL_AUDIT_LOG_PATH
    _TOOL_AUDIT_ENABLED = enabled
    if path:
        _TOOL_AUDIT_LOG_PATH = path

def _record_tool_audit(event: str, name: str, args: Optional[dict] = None,
                       result: Optional[str] = None, error: Optional[Exception] = None):
    if not _TOOL_AUDIT_ENABLED:
        return
    ts = datetime.now().isoformat()
    entry = {
        "ts": ts,
        "event": event,  # "start" / "end" / "error"
        "tool": name,
        "args": args,
        "error": str(error) if error else None,
        "result_preview": (result[:500] if isinstance(result, str) else None),
    }
    try:
        with open(_TOOL_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    try:
        print(f"[TOOL_AUDIT] {event} {name} args={json.dumps(args, ensure_ascii=False) if args else '{}'}", flush=True)
    except Exception:
        pass

def _build_default_model(max_tokens: Optional[int] = None):
    from langchain_core.runnables import RunnableLambda
    def _dummy_invoke(msgs: List[BaseMessage]) -> AIMessage:
        return AIMessage(content="未配置LLM：请使用 deep_research.utils.set_models(...) 注入自定义模型")
    return RunnableLambda(_dummy_invoke)

def set_models(summarizer: Any, writer: Optional[Any] = None):
    global _summarization_model, _writer_model
    _summarization_model = summarizer
    _writer_model = writer or summarizer

def get_summarization_model():
    global _summarization_model
    if _summarization_model is None:
        _summarization_model = _build_default_model()
    return _summarization_model

def get_writer_model():
    global _writer_model
    if _writer_model is None:
        _writer_model = _build_default_model(max_tokens=32000)
    return _writer_model

tavily_client = _build_tavily_client()
MAX_CONTEXT_LENGTH = 250000

# ===== SEARCH FUNCTIONS =====

def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> List[dict]:
    search_docs = []
    for query in search_queries:
        try:
            result = tavily_client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic
            )
        except Exception:
            result = {"query": query, "results": []}
        search_docs.append(result)
    return search_docs

def summarize_webpage_content(webpage_content: str) -> str:
    try:
        model = get_summarization_model()
        if hasattr(model, "with_structured_output"):
            structured_model = model.with_structured_output(Summary)
            summary_obj = structured_model.invoke([
                HumanMessage(content=f"Summarize the following content with key excerpts.\n\nDate: {get_today_str()}\n\nContent:\n{webpage_content}")
            ])
            s = getattr(summary_obj, "summary", "")
            k = getattr(summary_obj, "key_excerpts", "")
            formatted_summary = (
                f"<summary>\n{s}\n</summary>\n\n"
                f"<key_excerpts>\n{k}\n</key_excerpts>"
            )
            return formatted_summary

        out = model.invoke([
            HumanMessage(content=f"Please summarize the following webpage content and extract key excerpts.\n\nDate: {get_today_str()}\n\nContent:\n{webpage_content}")
        ])
        if isinstance(out, AIMessage):
            text = out.content
        else:
            text = str(out)
        formatted_summary = (
            f"<summary>\n{text}\n</summary>\n\n"
            f"<key_excerpts>\n\n</key_excerpts>"
        )
        return formatted_summary

    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    unique_results = {}
    for response in search_results:
        for result in response.get('results', []):
            url = result.get('url')
            if not url:
                continue
            if url not in unique_results:
                unique_results[url] = result
    return unique_results

def process_search_results(unique_results: dict) -> dict:
    summarized_results = {}
    for url, result in unique_results.items():
        try:
            if not result.get("raw_content"):
                content = result.get('content', "")
            else:
                content = summarize_webpage_content(result['raw_content'][:MAX_CONTEXT_LENGTH])
        except Exception:
            content = result.get('content', "")

        summarized_results[url] = {
            'title': result.get('title', url),
            'content': content
        }
    return summarized_results

def format_search_output(summarized_results: dict) -> str:
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"
    return formatted_output

# ===== RESEARCH TOOLS 与 直调函数 =====

def refine_draft_report_direct(research_brief: str, findings: str, draft_report: str) -> str:
    from deep_research.prompts import report_generation_with_draft_insight_prompt
    draft_report_prompt = report_generation_with_draft_insight_prompt.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str()
    )
    model = get_writer_model()
    out = model.invoke([HumanMessage(content=draft_report_prompt)])
    if isinstance(out, AIMessage):
        return out.content
    return str(out)

@tool(parse_docstring=True)
def refine_draft_report(research_brief: Annotated[str, InjectedToolArg],
                        findings: Annotated[str, InjectedToolArg],
                        draft_report: Annotated[str, InjectedToolArg]):
    """Refine draft report.

    Synthesizes all research findings into a comprehensive draft report.

    Args:
        research_brief (str): User's research request.
        findings (str): Collected research findings for the user request.
        draft_report (str): Draft report based on the findings and user request.

    Returns:
        str: Refined draft report.
    """
    _record_tool_audit(
        event="start",
        name="refine_draft_report",
        args={
            "brief_len": len(research_brief or ""),
            "findings_len": len(findings or ""),
            "draft_len": len(draft_report or "")
        }
    )
    try:
        result = refine_draft_report_direct(research_brief, findings, draft_report)
        _record_tool_audit(event="end", name="refine_draft_report", args=None, result=result)
        return result
    except Exception as e:
        _record_tool_audit(event="error", name="refine_draft_report", args=None, error=e)
        raise

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query (str): A single search query to execute.
        max_results (int): Maximum number of results to return.
        topic (Literal["general", "news", "finance"]): Topic to filter results.

    Returns:
        str: Formatted string of search results with summaries.
    """
    _record_tool_audit(
        event="start",
        name="tavily_search",
        args={"query": query, "max_results": max_results, "topic": topic}
    )
    try:
        search_results = tavily_search_multiple(
            [query],
            max_results=max_results,
            topic=topic,
            include_raw_content=True,
        )
        unique_results = deduplicate_search_results(search_results)
        summarized_results = process_search_results(unique_results)
        formatted = format_search_output(summarized_results)
        _record_tool_audit(event="end", name="tavily_search", args=None, result=formatted)
        return formatted
    except Exception as e:
        _record_tool_audit(event="error", name="tavily_search", args=None, error=e)
        raise

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.

    Args:
        reflection (str): Your detailed reflection on research progress, findings, gaps, and next steps.

    Returns:
        str: Confirmation that reflection was recorded for decision-making.
    """
    _record_tool_audit(event="start", name="think_tool", args={"reflection_len": len(reflection or "")})
    try:
        result = f"Reflection recorded: {reflection}"
        _record_tool_audit(event="end", name="think_tool", args=None, result=result)
        return result
    except Exception as e:
        _record_tool_audit(event="error", name="think_tool", args=None, error=e)
        raise

# ==============================================================================
# --- ✨ 2.2 本地检索具体实现（Chroma + BM25） + PLAN 目录支持 ✨ ---
# ==============================================================================

RETRIEVAL_DEPS_OK = True
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None
    RETRIEVAL_DEPS_OK = False
    print("⚠️ 缺少依赖 rank_bm25。混合检索的BM25部分将不可用。请运行: pip install rank-bm25")

Chroma = None
_chroma_import_error = None
try:
    from langchain_chroma import Chroma as _LCChroma
    Chroma = _LCChroma
except Exception as e1:
    try:
        from langchain_community.vectorstores import Chroma as _LCChroma
        Chroma = _LCChroma
    except Exception as e2:
        _chroma_import_error = (e1, e2)
        RETRIEVAL_DEPS_OK = False
        print("⚠️ 缺少依赖 langchain-chroma 或 langchain_community.vectorstores.Chroma。请运行: pip install langchain-chroma 或升级 langchain-community")

if build_embedding_client is None or get_embedding_settings is None:
    RETRIEVAL_DEPS_OK = False
    print("⚠️ 缺少在线 embedding 客户端依赖。请检查 deep_research.embedding_client 与 openai 包。")

try:
    from langchain_core.documents import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    import chromadb as _chromadb_mod
    _CHROMADB_AVAILABLE = True
except Exception:
    _chromadb_mod = None
    _CHROMADB_AVAILABLE = False

RAG_INIT_ENTRYPOINT = os.getenv("RAG_INIT_ENTRYPOINT", "")
RAG_UNIFIED_SEARCH = os.getenv("RAG_UNIFIED_SEARCH", "")

# 主库
vectordb_instance: Optional["Chroma"] = None
bm25_index: Optional["BM25Okapi"] = None
bm25_docs: List["Document"] = []
bm25_doc_ids: List[str] = []
bm25_ready: bool = False

# 计划库（第二目录）
vectordb_plan_instance: Optional["Chroma"] = None
bm25_index_plan: Optional["BM25Okapi"] = None
bm25_docs_plan: List["Document"] = []
bm25_doc_ids_plan: List[str] = []
bm25_ready_plan: bool = False

BM25_ALPHA = 0.6
FINAL_TOP_K = 3

def zh_tokenize(text: str) -> List[str]:
    try:
        import jieba
        return [w.strip() for w in jieba.cut(text, cut_all=False) if w and w.strip()]
    except (ImportError, AttributeError):
        text = text.strip()
        if not text:
            return []
        if len(text) <= 2:
            return list(text)
        return [text[i : i + 2] for i in range(len(text) - 1)]

def build_doc_id(meta: Dict[str, Any], fallback_id: Optional[str] = None) -> str:
    if not meta: return fallback_id or ""
    keys_to_try = ["_id", "chunk_id", "id", "source_id"]
    for key in keys_to_try:
        if key in meta and meta[key]:
            return str(meta[key])
    return fallback_id or ""

def minmax_norm(values: Dict[str, float]) -> Dict[str, float]:
    if not values: return {}
    vs = list(values.values())
    vmin, vmax = min(vs), max(vs)
    if vmax == vmin: return {k: 0.5 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}

def _detect_legacy_sqlite(persist_dir: str) -> bool:
    try:
        return os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))
    except Exception:
        return False

def _init_chroma_vectorstore_for_dir(persist_dir: str, collection_name: str, embeddings: Any) -> Optional["Chroma"]:
    """按目录初始化 Chroma，兼容新旧架构"""
    if not RETRIEVAL_DEPS_OK or not Chroma or not build_embedding_client:
        print("❌ [Chroma] 依赖不满足，无法初始化。请安装: langchain-chroma、rank-bm25、openai")
        return None
    try:
        print(f"⌛ [Chroma] 正在连接持久化向量库...")
        print(f"   - 目录: {persist_dir}")
        print(f"   - 集合: {collection_name}")
        if not os.path.exists(persist_dir):
            print(f"❌ [Chroma] 错误: 持久化目录 '{persist_dir}' 不存在。请检查您的 .env 配置。")
            return None

        if _CHROMADB_AVAILABLE and hasattr(_chromadb_mod, "PersistentClient"):
            try:
                client = _chromadb_mod.PersistentClient(path=persist_dir)
                vs = Chroma(collection_name=collection_name, client=client, embedding_function=embeddings)
                print("✅ [Chroma] 连接成功（新架构 PersistentClient）。")
                return vs
            except Exception as ne:
                print(f"⚠️ [Chroma] 新架构连接失败，将尝试旧架构回退。具体: {ne}")

        chroma_db_impl = os.getenv("CHROMA_DB_IMPL", "").strip()
        if not chroma_db_impl and _detect_legacy_sqlite(persist_dir):
            chroma_db_impl = "sqlite"
            os.environ["CHROMA_DB_IMPL"] = "sqlite"
            print("💡 [Chroma] 检测到 legacy sqlite 库，已在本次运行中临时设置 CHROMA_DB_IMPL=sqlite。")

        vs = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print("✅ [Chroma] 连接成功（旧架构 persist_directory）。")
        return vs

    except Exception as e:
        print(f"❌ [Chroma] 初始化失败: {e}")
        traceback.print_exc()
        return None

def _init_chroma_vectorstore_sync() -> Optional["Chroma"]:
    """保持兼容的主库初始化（使用 CHROMA_PERSIST_DIR）"""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
    except Exception:
        pass

    if not RETRIEVAL_DEPS_OK or not build_embedding_client:
        print("❌ [Chroma] 依赖不满足，无法初始化。")
        return None

    emb_settings = get_embedding_settings()
    emb_base_url = emb_settings["base_url"]
    emb_model = emb_settings["model"]
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "raptor_kb")

    print(f"   - Embedding Base URL: {emb_base_url}")
    print(f"   - 嵌入模型: {emb_model}")

    embeddings = build_embedding_client()
    return _init_chroma_vectorstore_for_dir(persist_dir, collection_name, embeddings)

def _build_bm25_from_chroma_sync(chroma_vs: "Chroma", is_plan: bool = False):
    """从指定 Chroma 构建 BM25 索引。is_plan=True 时写入 plan 变量。"""
    global bm25_index, bm25_docs, bm25_doc_ids, bm25_ready
    global bm25_index_plan, bm25_docs_plan, bm25_doc_ids_plan, bm25_ready_plan

    if not RETRIEVAL_DEPS_OK or not BM25Okapi:
        print("⚠️ [BM25] 依赖缺失或不可用，跳过BM25索引构建。")
        return
    try:
        if not hasattr(chroma_vs, "_collection") or chroma_vs._collection is None:
            print("⚠️ [BM25] 未能访问底层 Chroma 集合，跳过BM25索引构建。")
            if is_plan:
                bm25_ready_plan = False
            else:
                bm25_ready = False
            return

        collection = chroma_vs._collection
        count = collection.count()
        if count == 0:
            print("⚠️ [BM25] Chroma集合为空，无法构建BM25索引。")
            if is_plan:
                bm25_ready_plan = False
            else:
                bm25_ready = False
            return

        print(f"⌛ [BM25] 开始从Chroma构建索引... (文档数: {count}) [is_plan={is_plan}]")

        data = collection.get()
        docs_raw, metas_raw, ids_raw = data.get("documents", []), data.get("metadatas", []), data.get("ids", [])

        if is_plan:
            bm25_docs_plan.clear()
            bm25_doc_ids_plan.clear()
        else:
            bm25_docs.clear()
            bm25_doc_ids.clear()

        token_corpus = []

        for i, text in enumerate(docs_raw or []):
            if text is None:
                continue
            meta = metas_raw[i] if metas_raw and i < len(metas_raw) else {}
            doc = Document(page_content=text, metadata=meta)
            fallback_id = str(ids_raw[i]) if ids_raw and i < len(ids_raw) else f"doc_{i}"
            did = build_doc_id(doc.metadata, fallback_id=fallback_id)

            if is_plan:
                bm25_docs_plan.append(doc)
                bm25_doc_ids_plan.append(did)
            else:
                bm25_docs.append(doc)
                bm25_doc_ids.append(did)

            token_corpus.append(zh_tokenize(text))

        if not token_corpus:
            raise ValueError("分词后语料库为空。")

        index = BM25Okapi(token_corpus)
        if is_plan:
            bm25_index_plan = index
            bm25_ready_plan = True
        else:
            bm25_index = index
            bm25_ready = True

        print("✅ [BM25] 索引构建完成。")

    except Exception as e:
        if is_plan:
            bm25_ready_plan = False
        else:
            bm25_ready = False
        print(f"❌ [BM25] 构建索引时发生严重错误: {e}")
        traceback.print_exc()

async def _lc_retrieve_hybrid_async(query: str, vectordb: "Chroma", top_k: int, is_plan: bool = False) -> List[Dict]:
    """内置的异步混合检索实现（支持 main/plan 两套BM25）。"""
    if not RETRIEVAL_DEPS_OK or vectordb is None:
        return []

    emb_results = await asyncio.to_thread(
        lambda: vectordb.similarity_search_with_score(query, k=top_k * 5)
    )
    emb_scores_raw, emb_docs_map = {}, {}
    for doc, dist in emb_results or []:
        did = build_doc_id(doc.metadata, fallback_id=f"emb_{hash(doc.page_content)}")
        emb_docs_map[did] = doc
        emb_scores_raw[did] = 1.0 - float(dist)
    emb_scores_norm = minmax_norm(emb_scores_raw)

    if is_plan:
        bm_ready = bm25_ready_plan
        bm_index = bm25_index_plan
        bm_ids = bm25_doc_ids_plan
        bm_docs_list = bm25_docs_plan
    else:
        bm_ready = bm25_ready
        bm_index = bm25_index
        bm_ids = bm25_doc_ids
        bm_docs_list = bm25_docs

    bm25_scores_raw = {}
    if bm_ready and bm_index:
        q_tokens = zh_tokenize(query)
        all_scores = bm_index.get_scores(q_tokens)
        top_n_indices = sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)[:top_k*5]
        for idx, score in top_n_indices:
            if score > 0 and idx < len(bm_ids):
                bm25_scores_raw[bm_ids[idx]] = float(score)
    bm25_scores_norm = minmax_norm(bm25_scores_raw)

    all_ids = set(emb_scores_norm.keys()) | set(bm25_scores_norm.keys())
    fused_list = []
    bm25_id_to_doc = {did: doc for did, doc in zip(bm_ids, bm_docs_list)}

    for did in all_ids:
        combined_score = (BM25_ALPHA * bm25_scores_norm.get(did, 0.0)) + ((1.0 - BM25_ALPHA) * emb_scores_norm.get(did, 0.0))
        doc = emb_docs_map.get(did) or bm25_id_to_doc.get(did)
        if doc:
            fused_list.append((combined_score, doc))

    fused_sorted = sorted(fused_list, key=lambda x: x[0], reverse=True)
    final_items = fused_sorted[:top_k]

    sources = []
    for s, d in final_items:
        base_meta = dict((d.metadata or {}))
        # 保留底层检索返回的完整 metadata（尤其 image_paths/file_path/full_doc_id），
        # 只覆盖统一字段，避免上游链路丢失图文证据关联。
        base_meta["source"] = base_meta.get("source", "?")
        base_meta["score"] = float(s)
        base_meta["type"] = "hybrid_plan" if is_plan else "hybrid_main"
        sources.append({
            "content": d.page_content,
            "metadata": base_meta,
        })
    return sources

def initialize_all_retrievers() -> None:
    """
    初始化本地RAG检索资源。
    [优化] 实现单例模式：如果已初始化，直接返回。
    [优化] 仅初始化主库，跳过 Plan 库以节省时间。
    """
    global vectordb_instance, vectordb_plan_instance, bm25_ready

    # 1. 单例检查：如果主库已经加载，直接跳过
    if vectordb_instance is not None and bm25_ready:
        # print("⚡ [RAG] 检索器已在内存中，跳过重复初始化。") 
        return

    if RAG_INIT_ENTRYPOINT:
        print(f"🚀 [RAG] 检测到外部初始化入口: '{RAG_INIT_ENTRYPOINT}'")
        try:
            init_fn = _load_callable_from_entrypoint(RAG_INIT_ENTRYPOINT)
            init_fn()
            print("✅ [RAG] 外部检索器初始化完成。")
        except Exception as e:
            print(f"⚠️ [RAG] 外部检索器初始化失败: {e}")
        return

    print("ℹ️ [RAG] 正在初始化本地检索资源...")

    # 读取配置
    emb_settings = get_embedding_settings()
    emb_base_url = emb_settings["base_url"]
    emb_model = emb_settings["model"]
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "raptor_kb")
    persist_dir_main = os.getenv("CHROMA_PERSIST_DIR", "")
    # persist_dir_plan = os.getenv("CHROMA_PERSIST_PLAN_DIR", "") # [优化] 暂时忽略 Plan 库

    print(f"   - Embedding Base URL: {emb_base_url}")
    print(f"   - 嵌入模型: {emb_model}")
    print(f"   - 主库目录: {persist_dir_main}")

    if not RETRIEVAL_DEPS_OK or not build_embedding_client or not Chroma:
        print("❌ [RAG] 依赖不满足，无法初始化本地检索。")
        return

    embeddings = build_embedding_client()

    # 2. 初始化主库
    if persist_dir_main:
        vectordb_instance = _init_chroma_vectorstore_for_dir(persist_dir_main, collection_name, embeddings)
        if vectordb_instance:
            _build_bm25_from_chroma_sync(vectordb_instance, is_plan=False)
        else:
            print("❌ [RAG] 主库初始化失败。")
    else:
        print("⚠️ [RAG] 未配置 CHROMA_PERSIST_DIR。")

    # [优化] 注释掉 Plan 库的加载逻辑，解决 "这里构建rag_output_plan又是为什么" 的问题
    # if persist_dir_plan and os.path.exists(persist_dir_plan):
    #     vectordb_plan_instance = _init_chroma_vectorstore_for_dir(persist_dir_plan, collection_name, embeddings)
    #     if vectordb_plan_instance:
    #         _build_bm25_from_chroma_sync(vectordb_plan_instance, is_plan=True)
    # else:
    #     pass 

async def unified_local_search(query: str, top_k: int = 3, **kwargs) -> List[Dict]:
    """
    统一的本地检索入口：仅使用主库检索。
    [优化] 增加自动初始化检查。
    """
    # [优化] 自动初始化：防止调用者忘记调用 init_local_retrievers
    if vectordb_instance is None:
        initialize_all_retrievers()

    raw_results = []
    
    # 1. 优先尝试外部自定义检索函数
    if RAG_UNIFIED_SEARCH:
        # ... (保留原外部检索逻辑) ...
        try:
            search_fn = _load_callable_from_entrypoint(RAG_UNIFIED_SEARCH)
            if asyncio.iscoroutinefunction(search_fn):
                raw_results = await search_fn(query=query, top_k=top_k, **kwargs)
            else:
                raw_results = await asyncio.to_thread(search_fn, query=query, top_k=top_k, **kwargs)
        except Exception as e:
            print(f"⚠️ [RAG] 调用外部检索失败: {e}")
            return []
            
    # 2. 内置路径：仅检索主库
    else:
        if vectordb_instance:
            print(f"🔍 [RAG] 使用主库混合检索: '{query}'")
            raw_results = await _lc_retrieve_hybrid_async(query, vectordb=vectordb_instance, top_k=top_k, is_plan=False)
            if not raw_results:
                raw_results = []
        else:
            print("ℹ️ [RAG] 主库未初始化或不可用。")
            raw_results = []

    # 3. 标准化分数并截取
    normalized = []
    if isinstance(raw_results, list) and raw_results:
        score_map = {i: float(item.get("metadata", {}).get("score", 0.0)) for i, item in enumerate(raw_results)}
        score_norm = minmax_norm(score_map)
        
        fused = []
        for i, item in enumerate(raw_results):
            meta = item.get("metadata", {}) or {}
            meta["score"] = float(score_norm.get(i, 0.0))
            item["metadata"] = meta
            fused.append(item)
            
        fused_sorted = sorted(fused, key=lambda x: x.get("metadata", {}).get("score", 0.0), reverse=True)
        normalized = fused_sorted[:top_k]
    
    print(f"✅ [RAG] 本地检索完成，返回 {len(normalized)} 条结果。")
    return normalized



# Local KB chunk 摘要：默认走"截断"以避免每条命中再发一次 LLM round-trip。
# 设 LOCAL_SEARCH_LLM_SUMMARY=1 可强制启用 LLM 摘要（成本更高、延迟更大）。
_LOCAL_SEARCH_LLM_SUMMARY = os.getenv("LOCAL_SEARCH_LLM_SUMMARY", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
_LOCAL_SEARCH_TRUNCATE_LEN = int(os.getenv("LOCAL_SEARCH_TRUNCATE_LEN", "1000"))


def _summarize_text_content(text: str) -> str:
    if not text:
        return ""
    if not _LOCAL_SEARCH_LLM_SUMMARY:
        s = text.strip()
        if len(s) <= _LOCAL_SEARCH_TRUNCATE_LEN:
            return s
        return s[:_LOCAL_SEARCH_TRUNCATE_LEN] + "...[truncated]"
    try:
        model = get_summarization_model()
        prompt = f"Summarize the following text with key excerpts.\n\nDate: {get_today_str()}\n\nContent:\n{text[:MAX_CONTEXT_LENGTH]}"
        if hasattr(model, "with_structured_output"):
            structured_model = model.with_structured_output(Summary)
            summary_obj = structured_model.invoke([HumanMessage(content=prompt)])
            s = getattr(summary_obj, "summary", "")
            k = getattr(summary_obj, "key_excerpts", "")
            return f"<summary>\n{s}\n</summary>\n\n<key_excerpts>\n{k}\n</key_excerpts>"
        out = model.invoke([HumanMessage(content=prompt)])
        text_out = out.content if isinstance(out, AIMessage) else str(out)
        return f"<summary>\n{text_out}\n</summary>\n\n<key_excerpts>\n\n</key_excerpts>"
    except Exception as e:
        print(f"Failed to summarize text: {e}")
        return text[:1000] + "..." if len(text) > 1000 else text

def _format_local_search_output(results: List[Dict]) -> str:
    if not results:
        return "No local search results. Ensure the local vector store is initialized and populated."
    formatted_output = "Local search results: \n\n"
    for i, item in enumerate(results, 1):
        meta = item.get("metadata", {}) or {}
        source = meta.get("source") or meta.get("path") or meta.get("file") or "local"
        title = meta.get("title") or os.path.basename(str(source))
        summary = _summarize_text_content(item.get("content", ""))
        formatted_output += f"\n\n--- SOURCE {i}: {title} ---\n"
        formatted_output += f"Source: {source}\nScore: {meta.get('score', 'N/A')}\nType: {meta.get('type', 'hybrid')}\n\n"
        formatted_output += f"SUMMARY:\n{summary}\n\n"
        formatted_output += "-" * 80 + "\n"
    return formatted_output

# ==============================================================================
# --- ✨ 2.3 本地数据库工具（初始化、入库、检索、状态） ✨ ---
# ==============================================================================

def _load_callable_from_entrypoint(entry: str) -> Callable:
    module_path, func_name = entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)
    return func

@tool(parse_docstring=True)
def init_local_retrievers() -> str:
    """Initialize local RAG retrievers (Chroma + BM25).

    Safe to call multiple times (Singleton pattern).

    Returns:
        str: JSON string describing initialization status.
    """
    _record_tool_audit(event="start", name="init_local_retrievers", args={})
    try:
        # 懒加载：如果已加载则直接返回，如果不为 None 说明已经初始化过
        initialize_all_retrievers()
        status = {
            "main_chroma_ready": vectordb_instance is not None,
            "main_bm25_ready": bm25_ready,
            "plan_chroma_ready": False, # Plan库已被禁用
        }
        msg = json.dumps(status, ensure_ascii=False)
        _record_tool_audit(event="end", name="init_local_retrievers", args=None, result=msg)
        return msg
    except Exception as e:
        _record_tool_audit(event="error", name="init_local_retrievers", args=None, error=e)
        raise


@tool(parse_docstring=True)
def ingest_local_documents(
    directory: str,
    pattern: str = "*.txt",
    max_files: int = 1000
) -> str:
    """Ingest local documents into Chroma vectorstore (main DB).

    Args:
        directory (str): Directory path containing files to ingest.
        pattern (str): Glob pattern for files (e.g., '*.md', '*.txt').
        max_files (int): Maximum number of files to ingest.

    Returns:
        str: Status message with number of files ingested and BM25 rebuild result.
    """
    _record_tool_audit(event="start", name="ingest_local_documents", args={"directory": directory, "pattern": pattern, "max_files": max_files})
    try:
        if not RETRIEVAL_DEPS_OK or not Chroma or not build_embedding_client:
            msg = "Dependencies missing: please install langchain-chroma (or langchain-community), chromadb, rank-bm25, jieba, openai."
            _record_tool_audit(event="end", name="ingest_local_documents", args=None, result=msg)
            return msg

        global vectordb_instance
        if vectordb_instance is None:
            vectordb_instance = _init_chroma_vectorstore_sync()
            if vectordb_instance is None:
                msg = "Failed to initialize Chroma vectorstore. Check CHROMA_PERSIST_DIR."
                _record_tool_audit(event="end", name="ingest_local_documents", args=None, result=msg)
                return msg

        files = sorted(glob.glob(os.path.join(directory, pattern)))
        if not files:
            msg = f"No files matched pattern '{pattern}' in '{directory}'."
            _record_tool_audit(event="end", name="ingest_local_documents", args=None, result=msg)
            return msg

        files = files[:max_files]
        texts, metadatas, ids = [], [], []
        for i, fpath in enumerate(files):
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    content = fh.read()
                if not content.strip():
                    continue
                texts.append(content)
                metadatas.append({"source": fpath, "path": fpath, "title": os.path.basename(fpath)})
                ids.append(f"doc_{hash(fpath)}_{i}")
            except Exception as fe:
                print(f"⚠️ [Ingest] 读取文件失败: {fpath} -> {fe}")

        if not texts:
            msg = "No non-empty files to ingest."
            _record_tool_audit(event="end", name="ingest_local_documents", args=None, result=msg)
            return msg

        vectordb_instance.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        try:
            vectordb_instance.persist()
        except Exception:
            pass

        _build_bm25_from_chroma_sync(vectordb_instance, is_plan=False)

        msg = f"Ingested {len(texts)} documents into Chroma (main) and rebuilt BM25."
        _record_tool_audit(event="end", name="ingest_local_documents", args=None, result=msg)
        return msg

    except Exception as e:
        _record_tool_audit(event="error", name="ingest_local_documents", args=None, error=e)
        raise

@tool(parse_docstring=True)
def local_search(
    query: str,
    top_k: Annotated[int, InjectedToolArg] = 3
) -> str:
    """Search local Chroma+BM25 knowledge bases and summarize results.

    If the vector store is not initialized, this tool will attempt to initialize it automatically.

    Args:
        query: Search query string.
        top_k: Maximum number of hybrid results to return. Defaults to 5.

    Returns:
        str: Formatted string of local search results with summaries.
    """
    _record_tool_audit(event="start", name="local_search", args={"query": query, "top_k": top_k})
    try:
        # 自动初始化检查 (单例模式)
        if vectordb_instance is None:
            initialize_all_retrievers()
            
        if vectordb_instance is None:
            formatted = "Local vectorstore not initialized. Please check configuration."
            _record_tool_audit(event="end", name="local_search", args=None, result=formatted)
            return formatted

        results = asyncio.run(unified_local_search(query=query, top_k=top_k))
        formatted = _format_local_search_output(results)
        _record_tool_audit(event="end", name="local_search", args=None, result=formatted)
        return formatted
    except Exception as e:
        _record_tool_audit(event="error", name="local_search", args=None, error=e)
        raise

@tool(parse_docstring=True)
def local_retrieval_status() -> str:
    """Report the status of local retrievers (Chroma + BM25 for main and plan).

    Returns:
        str: JSON string describing dependency and readiness status.
    """
    _record_tool_audit(event="start", name="local_retrieval_status", args={})
    try:
        status = {
            "deps_ok": RETRIEVAL_DEPS_OK,
            "main_chroma_ready": vectordb_instance is not None,
            "main_bm25_ready": bm25_ready,
            "plan_chroma_ready": vectordb_plan_instance is not None,
            "plan_bm25_ready": bm25_ready_plan,
        }
        msg = json.dumps(status, ensure_ascii=False)
        _record_tool_audit(event="end", name="local_retrieval_status", args=None, result=msg)
        return msg
    except Exception as e:
        _record_tool_audit(event="error", name="local_retrieval_status", args=None, error=e)
        raise

def local_search_direct(query: str, top_k: int = 5) -> List[Dict]:
    if vectordb_instance is None and vectordb_plan_instance is None:
        initialize_all_retrievers()
    if vectordb_instance is None and vectordb_plan_instance is None:
        return []
    return asyncio.run(unified_local_search(query=query, top_k=top_k))
