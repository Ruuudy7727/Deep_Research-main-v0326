#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import traceback
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 尽早加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
except Exception:
    pass

# 兼容导入 Chroma 向量库（新包优先，旧包回退）
Chroma = None
try:
    from langchain_chroma import Chroma as _LCChroma
    Chroma = _LCChroma
except Exception:
    try:
        from langchain_community.vectorstores import Chroma as _LCChroma
        Chroma = _LCChroma
    except Exception:
        Chroma = None

# 检测 chromadb 新架构（>=0.5）
try:
    import chromadb as _chromadb_mod
    _CHROMADB_AVAILABLE = True
except Exception:
    _chromadb_mod = None
    _CHROMADB_AVAILABLE = False

# Embeddings（在线 API）
try:
    from deep_research.embedding_client import build_embedding_client, get_embedding_settings
except Exception:
    build_embedding_client = None
    get_embedding_settings = None

# Document 兜底
try:
    from langchain_core.documents import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# BM25（可选）
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

# 简单中文分词
def zh_tokenize(text: str) -> List[str]:
    try:
        import jieba
        return [w.strip() for w in jieba.cut(text, cut_all=False) if w and w.strip()]
    except Exception:
        t = (text or "").strip()
        if not t:
            return []
        if len(t) <= 2:
            return list(t)
        return [t[i:i+2] for i in range(len(t) - 1)]

def build_doc_id(meta: Dict[str, Any], fallback_id: Optional[str] = None) -> str:
    if not meta:
        return fallback_id or ""
    for k in ["_id", "chunk_id", "id", "source_id"]:
        if k in meta and meta[k]:
            return str(meta[k])
    return fallback_id or ""

def minmax_norm(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    vs = list(values.values())
    vmin, vmax = min(vs), max(vs)
    if vmax == vmin:
        return {k: 0.5 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}

def _detect_legacy_sqlite(persist_dir: str) -> bool:
    try:
        return os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))
    except Exception:
        return False

def _resolve_chroma_dir(path_candidate: str) -> Optional[str]:
    """
    解析可用的 Chroma 持久化目录：
    - 如果 path_candidate 存在并且包含 chroma.sqlite3，则直接使用。
    - 如果 path_candidate 不存在且以 /chroma_kb 结尾，则尝试其上一层目录（父目录若含 chroma.sqlite3 则使用父目录）。
    - 如果 path_candidate 存在但不含 chroma.sqlite3，尝试 path_candidate/chroma_kb。
    - 否则返回 None。
    """
    try:
        if not path_candidate:
            return None
        pc = path_candidate.strip()
        if os.path.isdir(pc) and os.path.exists(os.path.join(pc, "chroma.sqlite3")):
            return pc
        # endswith chroma_kb 且不存在：尝试父目录
        if (not os.path.exists(pc)) and pc.endswith(os.sep + "chroma_kb"):
            parent = os.path.dirname(pc)
            if os.path.isdir(parent) and os.path.exists(os.path.join(parent, "chroma.sqlite3")):
                print(f"💡 [local_db] {pc} 不存在，已自动回退到父目录: {parent}")
                return parent
        # 尝试子目录 chroma_kb
        cand = os.path.join(pc, "chroma_kb")
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "chroma.sqlite3")):
            return cand
        return None
    except Exception:
        return None

# 全局状态
_vectordb: Optional["Chroma"] = None
_bm25_index: Optional["BM25Okapi"] = None
_bm25_docs: List["Document"] = []
_bm25_doc_ids: List[str] = []
_bm25_ready: bool = False

# 融合权重与TopK
_BM25_ALPHA = 0.5
_FINAL_TOP_K = 3

def _pick_existing_collection(client, prefer_name: Optional[str]) -> Optional[str]:
    """
    在 chromadb 新架构下，从已有集合里选一个：
    - 若 prefer_name 在列表中，使用它；
    - 否则如果有集合，选择第一个；
    - 若无集合则返回 None（后续会创建空集合，但这常意味着路径/集合名不匹配）。
    """
    try:
        cols = client.list_collections()
        names = [c.name for c in cols] if cols else []
        if prefer_name and prefer_name in names:
            return prefer_name
        if names:
            if prefer_name and prefer_name not in names:
                print(f"💡 [local_db] 未找到集合 '{prefer_name}'，将使用已有集合: {names[0]}")
            return names[0]
        print("⚠️ [local_db] 当前路径未发现已存在的集合，将创建空集合（可能不是你预期的目录或集合名）。")
        return prefer_name or "default"
    except Exception as e:
        print(f"⚠️ [local_db] 列出集合失败：{e}")
        return prefer_name or "default"

def _init_chroma(persist_dir_override: Optional[str] = None) -> Optional["Chroma"]:
    """
    初始化/连接 Chroma 向量库，兼容新旧 chromadb 架构。
    优先使用 persist_dir_override，否则使用 CHROMA_PERSIST_PLAN_DIR；
    若 CHROMA_PERSIST_PLAN_DIR 为空，回退到 CHROMA_PERSIST_DIR。
    """
    if Chroma is None or build_embedding_client is None or get_embedding_settings is None:
        print("❌ [local_db] 依赖不满足：缺少 Chroma 或在线 embedding 客户端。请检查 langchain-chroma 与 openai 依赖。")
        return None
    try:
        # 再次加载 .env，确保环境变量可用
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
        except Exception:
            pass

        emb_settings = get_embedding_settings()
        emb_base_url = emb_settings["base_url"]
        emb_model = emb_settings["model"]

        # 计划专用路径优先
        plan_env = os.getenv("CHROMA_PERSIST_PLAN_DIR", "").strip()
        fallback_env = os.getenv("CHROMA_PERSIST_DIR", "").strip()
        chosen_raw = (persist_dir_override or "").strip() or plan_env or fallback_env

        if not chosen_raw:
            print("❌ [local_db] 未设置 CHROMA_PERSIST_PLAN_DIR/CHROMA_PERSIST_DIR，且未传入 persist_dir 覆盖参数。")
            return None

        # 自动解析真实可用的目录
        resolved_dir = _resolve_chroma_dir(chosen_raw)
        if not resolved_dir:
            print(f"❌ [local_db] 无法在如下路径解析到有效的 Chroma 目录：{chosen_raw}")
            print("    提示：目录下应存在 chroma.sqlite3；若你把 sqlite 放在根目录，请将环境变量指向根目录。")
            return None

        prefer_collection = os.getenv("CHROMA_COLLECTION_NAME", "").strip() or None

        print(f"⌛ [local_db] 连接本地知识库...")
        print(f"   - 目录: {resolved_dir}")
        print(f"   - 偏好集合: {prefer_collection or '(自动选择)'}")
        print(f"   - Embedding Base URL: {emb_base_url}")
        print(f"   - 嵌入模型: {emb_model}")

        embeddings = build_embedding_client()

        # 新架构（chromadb >= 0.5）
        if _CHROMADB_AVAILABLE and hasattr(_chromadb_mod, "PersistentClient"):
            try:
                client = _chromadb_mod.PersistentClient(path=resolved_dir)
                chosen_collection = _pick_existing_collection(client, prefer_collection)
                vs = Chroma(collection_name=chosen_collection, client=client, embedding_function=embeddings)
                print(f"✅ [local_db] 连接成功（新架构 PersistentClient），集合: {chosen_collection}")
                return vs
            except Exception as ne:
                print(f"⚠️ [local_db] 新架构连接失败，将回退旧架构：{ne}")

        # 旧架构（chromadb 0.4.x）
        chroma_db_impl = os.getenv("CHROMA_DB_IMPL", "").strip()
        if not chroma_db_impl and _detect_legacy_sqlite(resolved_dir):
            os.environ["CHROMA_DB_IMPL"] = "sqlite"
            print("💡 [local_db] 检测到 legacy sqlite，已设置 CHROMA_DB_IMPL=sqlite (仅当前进程)")

        # 旧架构无法列出已有集合名，只能依赖传入的 collection_name。
        collection_name = prefer_collection or "default"
        vs = Chroma(collection_name=collection_name, persist_directory=resolved_dir, embedding_function=embeddings)
        print(f"✅ [local_db] 连接成功（旧架构 persist_directory），集合: {collection_name}")
        return vs

    except Exception as e:
        print(f"❌ [local_db] 初始化失败：{e}")
        traceback.print_exc()
        return None

def _build_bm25(vs: "Chroma") -> None:
    """
    从 Chroma 中读取所有文档，构建 BM25。若不可用则安全跳过。
    """
    global _bm25_index, _bm25_docs, _bm25_doc_ids, _bm25_ready
    _bm25_index = None
    _bm25_docs = []
    _bm25_doc_ids = []
    _bm25_ready = False

    if BM25Okapi is None:
        print("⚠️ [local_db] 未安装 rank_bm25，跳过 BM25 构建。")
        return
    try:
        if not hasattr(vs, "_collection") or vs._collection is None:
            print("⚠️ [local_db] 访问不到底层 Chroma 集合，跳过 BM25 构建。")
            return
        collection = vs._collection
        count = collection.count()
        if count == 0:
            print("⚠️ [local_db] 向量库为空，无法构建 BM25。")
            return

        print(f"⌛ [local_db] 正在构建 BM25（文档数：{count}）...")
        data = collection.get()
        docs_raw, metas_raw, ids_raw = data.get("documents", []), data.get("metadatas", []), data.get("ids", [])

        token_corpus: List[List[str]] = []
        for i, text in enumerate(docs_raw or []):
            if text is None:
                continue
            meta = metas_raw[i] if metas_raw and i < len(metas_raw) else {}
            doc = Document(page_content=text, metadata=meta or {})
            _bm25_docs.append(doc)
            fallback_id = str(ids_raw[i]) if ids_raw and i < len(ids_raw) else f"doc_{i}"
            _bm25_doc_ids.append(build_doc_id(doc.metadata, fallback_id=fallback_id))
            token_corpus.append(zh_tokenize(text))

        if not token_corpus:
            print("⚠️ [local_db] 语料为空，跳过 BM25 构建。")
            return

        _bm25_index = BM25Okapi(token_corpus)
        _bm25_ready = True
        print("✅ [local_db] BM25 构建完成。")
    except Exception as e:
        print(f"❌ [local_db] BM25 构建失败：{e}")
        traceback.print_exc()
        _bm25_ready = False

def init_local_kb(persist_dir: Optional[str] = None) -> str:
    """
    初始化本地知识库（专用 plan 库）。
    [优化] 实现单例模式，避免重复加载。
    """
    global _vectordb, _bm25_ready
    
    # 1. 单例检查：如果全局变量已有值，且没有强制指定新路径，则复用
    if _vectordb is not None:
        # 如果你想支持动态切换目录，这里需要更复杂的逻辑；
        # 但通常 plan 库路径是固定的，直接返回即可。
        return "Local KB (plan) already initialized (Cached)."

    _vectordb = _init_chroma(persist_dir_override=persist_dir)
    if _vectordb is None:
        return "Failed: local KB (plan) not initialized."
        
    _build_bm25(_vectordb)
    
    if _bm25_ready:
        return "Local KB (plan) initialized with BM25."
    return "Local KB (plan) initialized (BM25 unavailable or skipped)."

def _summarize_text_content(text: str, max_len: int = 800) -> str:
    """
    轻量级摘要：安全截断，避免依赖外部 LLM。
    """
    if not text:
        return ""
    s = text.strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "...[truncated]"

def _retrieve_hybrid_sync(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    同步混合检索：向量检索 + BM25 融合（若 BM25 可用）。
    返回标准化结果列表：[{content, metadata:{source,title,score,type}}]
    """
    results: List[Tuple[float, Document]] = []
    if _vectordb is None:
        return []

    # 1) 向量检索
    try:
        emb_raw = _vectordb.similarity_search_with_score(query, k=max(1, top_k * 5))
        emb_scores_raw: Dict[str, float] = {}
        emb_docs_map: Dict[str, Document] = {}
        for doc, dist in emb_raw or []:
            did = build_doc_id(doc.metadata, fallback_id=f"emb_{hash(doc.page_content)}")
            emb_docs_map[did] = doc
            emb_scores_raw[did] = 1.0 - float(dist)  # 距离->相似度
        emb_scores = minmax_norm(emb_scores_raw)
    except Exception as e:
        print(f"⚠️ [local_db] 向量检索失败：{e}")
        emb_scores, emb_docs_map = {}, {}

    # 2) BM25
    bm25_scores = {}
    if _bm25_ready and _bm25_index is not None:
        try:
            q_tokens = zh_tokenize(query)
            all_scores = _bm25_index.get_scores(q_tokens)
            idx_scores = sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)[:max(1, top_k * 5)]
            for idx, sc in idx_scores:
                if sc > 0 and idx < len(_bm25_doc_ids):
                    bm25_scores[_bm25_doc_ids[idx]] = float(sc)
            bm25_scores = minmax_norm(bm25_scores)
        except Exception as e:
            print(f"⚠️ [local_db] BM25 检索失败：{e}")
            bm25_scores = {}

    # 3) 融合
    all_ids = set(emb_scores.keys()) | set(bm25_scores.keys())
    bm25_id_to_doc = {did: doc for did, doc in zip(_bm25_doc_ids, _bm25_docs)}
    fused: List[Tuple[float, Document]] = []
    for did in all_ids:
        score = (_BM25_ALPHA * bm25_scores.get(did, 0.0)) + ((1.0 - _BM25_ALPHA) * emb_scores.get(did, 0.0))
        doc = emb_docs_map.get(did) or bm25_id_to_doc.get(did)
        if doc:
            fused.append((score, doc))

    fused_sorted = sorted(fused, key=lambda x: x[0], reverse=True)[:max(1, top_k)]
    out: List[Dict[str, Any]] = []
    for s, d in fused_sorted:
        meta = d.metadata or {}
        out.append({
            "content": d.page_content,
            "metadata": {
                "source": meta.get("source") or meta.get("path") or meta.get("file") or "local",
                "title": meta.get("title") or os.path.basename(str(meta.get("source") or "")) or "local",
                "score": float(s),
                "type": "hybrid"
            }
        })
    return out

def _format_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No local search results (plan KB)."
    msg = "Local search results (plan KB): \n\n"
    for i, item in enumerate(results, 1):
        meta = item.get("metadata", {}) or {}
        title = meta.get("title") or "local"
        source = meta.get("source") or "local"
        score = meta.get("score", "N/A")
        summary = _summarize_text_content(item.get("content", ""))
        msg += f"--- SOURCE {i}: {title} ---\n"
        msg += f"Source: {source}\nScore: {score}\n\n"
        msg += f"SUMMARY:\n{summary}\n"
        msg += "-" * 80 + "\n"
    return msg

def search_local_kb(query: str, top_k: int = 3) -> str:
    """
    查询本地 plan 知识库。如尚未初始化，会尝试自动初始化。
    """
    # input() # 这里的 input() 最好删掉，会阻塞自动化脚本
    global _vectordb
    if _vectordb is None:
        _ = init_local_kb() # 第一次调用初始化，后续复用
        
    if _vectordb is None:
        return "Local KB (plan) not initialized. Please check CHROMA_PERSIST_PLAN_DIR and dependencies."
    try:
        results = _retrieve_hybrid_sync(query=query, top_k=top_k)
        return _format_results(results)
    except Exception as e:
        print(f"❌ [local_db] 检索失败：{e}")
        traceback.print_exc()
        return "Local KB (plan) search failed."

def local_kb_status() -> str:
    """
    返回本地 plan 知识库状态。
    """
    status = {
        "chroma_ready": _vectordb is not None,
        "bm25_ready": _bm25_ready
    }
    return json.dumps(status, ensure_ascii=False)
