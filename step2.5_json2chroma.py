#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import time
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
# 额外引入一个错误类型，用于处理数据库集合不存在的情况
from chromadb.errors import NotFoundError

from deep_research.embedding_client import build_embedding_client, get_embedding_settings

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent

# =========================
# 配置区（按需修改）
# =========================
KV_JSON_PATH = str(_PROJECT_ROOT / "rag_data" / "all" / "kv_store_text_chunks.json")

# Chroma 持久化目录与集合名（集合名建议与线上服务一致）
CHROMA_DIR = str(_PROJECT_ROOT / "rag_data" / "all")
CHROMA_COLLECTION_NAME = "raptor_kb"

# 在线 Embedding 配置
EMBED_SETTINGS = get_embedding_settings()
EMBED_MODEL = str(EMBED_SETTINGS["model"])

# 在线 embedding 服务对请求频率有限制，默认使用顺序限速。
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
EMBED_BATCH_INTERVAL = float(os.getenv("EMBED_BATCH_INTERVAL", "1.0"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "8"))
EMBED_RETRY_BASE_DELAY = float(os.getenv("EMBED_RETRY_BASE_DELAY", "2.0"))

# =========================
# 代码区（无需修改）
# =========================

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_kv_json(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("kv_store_text_chunks.json 应为字典形式：{id: {content: ... , ...}}")
    return data

def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in meta.items():
        if k == "content":
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Chroma metadata 仅支持标量，这里将列表压平为分号分隔字符串
            out[k] = ";".join(str(item) for item in v if item is not None)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out

def build_payloads_from_data(data: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    返回列表: [(id, content, metadata), ...]
    仅收集存在 content 且非空的条目。
    """
    items: List[Tuple[str, str, Dict[str, Any]]] = []
    skipped = 0
    for _id, obj in data.items():
        if not isinstance(obj, dict):
            skipped += 1
            continue
        content = obj.get("content", "")
        if not isinstance(content, str) or not content.strip():
            skipped += 1
            continue
        meta = sanitize_metadata(obj)
        items.append((str(_id), content, meta))
    if skipped:
        logging.warning(f"有 {skipped} 条记录被跳过（content 缺失或无效）。")
    logging.info(f"有效条目数: {len(items)}")
    return items

def embed_texts(emb_model: Any, payloads: List[Tuple[str, str, Dict[str, Any]]]) -> List[Tuple[str, str, Dict[str, Any], List[float]]]:
    """
    顺序、限速地对文本生成嵌入，避免在线服务 429 限流。
    """
    total = len(payloads)
    logging.info(
        f"开始顺序生成嵌入，共 {total} 条，batch_size={BATCH_SIZE}，"
        f"batch_interval={EMBED_BATCH_INTERVAL}s..."
    )

    results: List[Tuple[str, str, Dict[str, Any], List[float]]] = []
    completed_count = 0

    for start_index in range(0, total, BATCH_SIZE):
        batch_payloads = payloads[start_index:start_index + BATCH_SIZE]
        batch_texts = [p[1] for p in batch_payloads]

        last_error = None
        batch_embeddings = None
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                batch_embeddings = emb_model.embed_documents(batch_texts)
                break
            except Exception as e:
                last_error = e
                wait_seconds = EMBED_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logging.warning(
                    f"处理从索引 {start_index} 开始的批次失败，第 {attempt}/{EMBED_MAX_RETRIES} 次重试，"
                    f"{wait_seconds:.1f}s 后继续。错误: {e}"
                )
                time.sleep(wait_seconds)

        if batch_embeddings is None:
            raise RuntimeError(f"处理从索引 {start_index} 开始的批次失败，已达到最大重试次数。最后错误: {last_error}")

        if len(batch_embeddings) != len(batch_payloads):
            raise RuntimeError(
                f"处理从索引 {start_index} 开始的批次返回数量不匹配："
                f"期望 {len(batch_payloads)}，实际 {len(batch_embeddings)}"
            )

        for payload, vec in zip(batch_payloads, batch_embeddings):
            _id, content, meta = payload
            if not vec:
                raise RuntimeError(f"ID: {_id} 返回空嵌入向量，已中断写入 Chroma。")
            results.append((_id, content, meta, vec))

        completed_count += len(batch_payloads)
        logging.info(f"嵌入进度: {completed_count}/{total}")
        if completed_count < total and EMBED_BATCH_INTERVAL > 0:
            time.sleep(EMBED_BATCH_INTERVAL)

    logging.info("嵌入生成完毕。")
    return results

def write_to_chroma(chroma_dir: str, collection_name: str, records: List[Tuple[str, str, Dict[str, Any], List[float]]]):
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        client.delete_collection(name=collection_name)
        logging.info(f"已删除已存在的集合: {collection_name}")
    except NotFoundError:
        logging.info(f"集合 {collection_name} 不存在，将直接创建。")
    except Exception as e:
        logging.warning(f"删除集合时发生意外错误: {e}，将尝试继续创建。")

    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    logging.info(f"已创建集合: {collection_name}")

    total_written = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        valid_batch = [r for r in batch if r[3]]
        if not valid_batch:
            continue

        ids = [r[0] for r in valid_batch]
        docs = [r[1] for r in valid_batch]
        metas = [r[2] for r in valid_batch]
        embs = [r[3] for r in valid_batch]

        try:
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            total_written += len(valid_batch)
            logging.info(f"已写入 {total_written}/{len(records)}")
        except Exception as e:
            logging.error(f"在批次 {i//BATCH_SIZE + 1} 写入 ChromaDB 时失败: {e}")

    logging.info(f"✅ 完成写入，最终成功写入 {collection.count()} 条。")

def main():
    logging.info("🚀 开始基于 kv_store_text_chunks.json 构建 Chroma 向量库...")
    data = load_kv_json(KV_JSON_PATH)
    payloads = build_payloads_from_data(data)
    if not payloads:
        logging.error("未发现有效的 content，程序结束。")
        return

    try:
        embeddings = build_embedding_client()
        logging.info(f"已连接在线 Embedding 服务: {EMBED_SETTINGS['base_url']}，嵌入模型: {EMBED_MODEL}")
        test_vec = embeddings.embed_query("embedding connectivity test")
        if not test_vec:
            logging.error("在线 Embedding 服务连通性测试返回空向量，程序结束。")
            return
        logging.info(f"在线 Embedding 服务连通性测试成功，向量维度: {len(test_vec)}")
    except Exception as e:
        logging.error(f"初始化在线 Embedding 客户端失败: {e}")
        return

    try:
        records = embed_texts(embeddings, payloads)
        write_to_chroma(CHROMA_DIR, CHROMA_COLLECTION_NAME, records)
    except Exception as e:
        logging.error(f"构建 Chroma 向量库失败: {e}")
        return

    logging.info("🎉 全部完成！")

if __name__ == "__main__":
    main()
