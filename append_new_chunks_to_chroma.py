#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
append_new_chunks_to_chroma.py — 一次性增量入库脚本

用途
----
在 step1_build_konwledge.py 重新生成 `kv_store_text_chunks.json` 之后，
只把"新增"或"被更新"的 chunks 计算 embedding 并 upsert 到现有 Chroma `raptor_kb`。
**不会**删除/重建整个集合，**不会**对已有 6000+ 旧 chunks 重新 embedding。

工作机制
--------
1) 读旧 KV (来自 _append_backup/kv_store_text_chunks.json.bak.*)
2) 读新 KV (来自 rag_data/all/kv_store_text_chunks.json)
3) 取 diff:
     - 新增: 仅在新 KV 出现的 chunk_id
     - 变更: 同 chunk_id 但 content 变了（极少见，理论上 chunk_id = md5(content)，所以不会出现）
     - 删除: 仅在旧 KV 出现 → 默认不动 Chroma，仅打印警告
4) 仅对"新增 + 变更" chunks:
     a) 用现有 embedding client 计算向量（限速 + 重试）
     b) 用 chromadb.PersistentClient 连接现有库
     c) collection.upsert(ids, documents, metadatas, embeddings)
5) 简短自检：随机抽 1 条，确认 image_paths 字段被正确写入

注意
----
- 必须先跑过 step1（产出新 KV），否则脚本会判定无新增直接退出。
- 必须有最近的备份（或显式传 --baseline-kv），否则无法 diff。

使用
----
    python append_new_chunks_to_chroma.py                  # 自动找最近备份
    python append_new_chunks_to_chroma.py --dry-run        # 仅打印 diff，不写库
    python append_new_chunks_to_chroma.py --baseline-kv <path>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from deep_research.embedding_client import build_embedding_client, get_embedding_settings


# ---- 配置 ----
KV_PATH_NEW = PROJECT_ROOT / "rag_data" / "all" / "kv_store_text_chunks.json"
CHROMA_DIR = PROJECT_ROOT / "rag_data" / "all"
COLLECTION_NAME = "raptor_kb"
BACKUP_DIR = PROJECT_ROOT / "_append_backup"

# embedding 限速参数（与 step2.5_json2chroma.py 保持一致）
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
EMBED_BATCH_INTERVAL = float(os.getenv("EMBED_BATCH_INTERVAL", "1.0"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "8"))
EMBED_RETRY_BASE_DELAY = float(os.getenv("EMBED_RETRY_BASE_DELAY", "2.0"))


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _find_latest_backup_kv() -> Path:
    """从 _append_backup/ 目录里找最近一份 kv_store_text_chunks.json.bak.*"""
    if not BACKUP_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到备份目录 {BACKUP_DIR}。请先备份原 kv_store_text_chunks.json。"
        )
    candidates = sorted(BACKUP_DIR.glob("kv_store_text_chunks.json.bak.*"))
    if not candidates:
        raise FileNotFoundError(
            f"{BACKUP_DIR} 下没有 kv_store_text_chunks.json.bak.*，无法做 diff。"
        )
    return candidates[-1]


def _load_kv(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path} 内容不是 dict")
    return d


def _diff_kv(
    old: Dict[str, Dict[str, Any]],
    new: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str]]:
    """返回 (added_ids, changed_ids, removed_ids)。

    `changed`: 同 id 但 content 不同。理论上 chunk_id = md5(content) 不会出现，
                 但保留兜底逻辑以防 step1 切分逻辑改动。
    """
    old_ids = set(old.keys())
    new_ids = set(new.keys())
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    changed: List[str] = []
    for cid in sorted(new_ids & old_ids):
        if (new[cid].get("content") or "") != (old[cid].get("content") or ""):
            changed.append(cid)
    return added, changed, removed


def _summarize_added(
    added_ids: List[str],
    new_kv: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """按 doc_id 聚合，统计 chunk 数 / 带图 chunk 数 / 涉及图片数。"""
    by_doc: Dict[str, Dict[str, Any]] = {}
    img_total = 0
    for cid in added_ids:
        item = new_kv[cid]
        doc_id = item.get("full_doc_id") or "?"
        d = by_doc.setdefault(
            doc_id,
            {"file_path": item.get("file_path", "?"), "chunks": 0, "img_chunks": 0, "imgs": 0},
        )
        d["chunks"] += 1
        imgs = item.get("image_paths") or []
        if imgs:
            d["img_chunks"] += 1
            d["imgs"] += len(imgs)
            img_total += len(imgs)
    return {"by_doc": by_doc, "img_total": img_total}


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """与 step2.5_json2chroma.py::sanitize_metadata 完全一致。

    Chroma metadata 仅支持标量；list/tuple/set → ';' 拼接，dict → JSON 串。
    image_paths 在这里会被序列化为 ';' 拼接的字符串，server_plus.py 已能解析。
    """
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k == "content":
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple, set)):
            out[k] = ";".join(str(item) for item in v if item is not None)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def _build_payloads(
    ids: List[str],
    new_kv: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    payloads: List[Tuple[str, str, Dict[str, Any]]] = []
    for cid in ids:
        obj = new_kv[cid]
        content = obj.get("content", "")
        if not isinstance(content, str) or not content.strip():
            logging.warning(f"  -> 跳过空 content: {cid}")
            continue
        payloads.append((cid, content, _sanitize_metadata(obj)))
    return payloads


def _embed_payloads(
    emb_model: Any,
    payloads: List[Tuple[str, str, Dict[str, Any]]],
) -> List[Tuple[str, str, Dict[str, Any], List[float]]]:
    """顺序 + 限速生成 embedding。与 step2.5_json2chroma.py 同款实现。"""
    total = len(payloads)
    logging.info(
        f"开始顺序生成 embedding，共 {total} 条，batch_size={BATCH_SIZE}，"
        f"interval={EMBED_BATCH_INTERVAL}s ..."
    )
    results: List[Tuple[str, str, Dict[str, Any], List[float]]] = []
    completed = 0
    for start in range(0, total, BATCH_SIZE):
        batch = payloads[start:start + BATCH_SIZE]
        texts = [p[1] for p in batch]
        last_err = None
        vecs = None
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                vecs = emb_model.embed_documents(texts)
                break
            except Exception as e:
                last_err = e
                wait = EMBED_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logging.warning(
                    f"  batch starting at {start} failed "
                    f"(attempt {attempt}/{EMBED_MAX_RETRIES}), retry in {wait:.1f}s: {e}"
                )
                time.sleep(wait)
        if vecs is None:
            raise RuntimeError(f"batch starting at {start} 多次失败: {last_err}")
        if len(vecs) != len(batch):
            raise RuntimeError(
                f"batch starting at {start}: 期望 {len(batch)} 条向量，实际 {len(vecs)}"
            )
        for (cid, content, meta), vec in zip(batch, vecs):
            if not vec:
                raise RuntimeError(f"id={cid} 返回空向量，已中断")
            results.append((cid, content, meta, vec))
        completed += len(batch)
        logging.info(f"embedding 进度: {completed}/{total}")
        if completed < total and EMBED_BATCH_INTERVAL > 0:
            time.sleep(EMBED_BATCH_INTERVAL)
    return results


def _upsert_to_chroma(
    records: List[Tuple[str, str, Dict[str, Any], List[float]]],
) -> int:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        raise RuntimeError(
            f"集合 {COLLECTION_NAME!r} 不存在；请确认你没有意外删过集合。"
        )

    before = collection.count()
    written = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        valid = [r for r in batch if r[3]]
        if not valid:
            continue
        ids = [r[0] for r in valid]
        docs = [r[1] for r in valid]
        metas = [r[2] for r in valid]
        embs = [r[3] for r in valid]
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        written += len(valid)
        logging.info(f"upsert: {written}/{len(records)}")
    after = collection.count()
    logging.info(f"集合 {COLLECTION_NAME!r}: {before} → {after} (净增 {after - before})")
    return after


def _smoke_check(
    sample_ids: List[str],
    expected_doc_id_prefix: str = "doc-",
) -> None:
    """随机抽几条新 chunk，验证它们已入库且 image_paths 等关键字段保留。"""
    if not sample_ids:
        return
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_collection(name=COLLECTION_NAME)
    sample = col.get(ids=sample_ids[:5], include=["metadatas", "documents"])
    print("\n=== 自检：新 chunk 在 Chroma 中的状态 ===")
    for cid, meta, doc in zip(sample["ids"], sample["metadatas"], sample["documents"]):
        img = meta.get("image_paths") or "(none)"
        print(
            f"  • {cid}\n"
            f"    file_path: {meta.get('file_path')!r}\n"
            f"    image_paths: {img}\n"
            f"    content_head: {(doc or '')[:80].replace(chr(10), ' ')}..."
        )


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-kv", type=str, default="",
                        help="旧 kv_store_text_chunks.json 路径；不传则自动找最近备份")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印 diff，不调用 embedding 也不写库")
    args = parser.parse_args()

    baseline = Path(args.baseline_kv) if args.baseline_kv else _find_latest_backup_kv()
    logging.info(f"基线 KV (旧): {baseline}")
    logging.info(f"目标 KV (新): {KV_PATH_NEW}")

    old_kv = _load_kv(baseline)
    new_kv = _load_kv(KV_PATH_NEW)

    added, changed, removed = _diff_kv(old_kv, new_kv)
    logging.info(
        f"diff: 新增 {len(added)} 条，变更 {len(changed)} 条，"
        f"消失 {len(removed)} 条 (旧 {len(old_kv)} → 新 {len(new_kv)})"
    )

    if removed:
        logging.warning(
            f"⚠ 有 {len(removed)} 条旧 chunk 在新 KV 中消失，但**不会**从 Chroma 删除。"
            "  这些 ID 仍会留在向量库里。如需清理请手动 collection.delete(ids=...)."
        )
        for cid in removed[:5]:
            logging.warning(f"   - removed: {cid}")
        if len(removed) > 5:
            logging.warning(f"   ... +{len(removed) - 5} more")

    todo = added + changed
    if not todo:
        logging.info("✅ 没有新增/变更 chunks，无需 upsert。")
        return

    summary = _summarize_added(added, new_kv)
    print("\n=== 新增 chunks 按 doc 聚合 ===")
    for doc_id, info in summary["by_doc"].items():
        print(
            f"  • {doc_id} | {info['file_path']}"
            f" | {info['chunks']} chunks ({info['img_chunks']} 含图, {info['imgs']} 张图)"
        )
    print(f"  合计图片引用: {summary['img_total']}\n")

    if args.dry_run:
        logging.info("[dry-run] 已退出，未调用 embedding，未写 Chroma。")
        return

    payloads = _build_payloads(todo, new_kv)
    if not payloads:
        logging.warning("过滤后无可入库 payloads，退出。")
        return

    settings = get_embedding_settings()
    logging.info(
        f"embedding endpoint: {settings.get('base_url')!r}  model: {settings.get('model')!r}"
    )
    emb = build_embedding_client()
    test_vec = emb.embed_query("connectivity test")
    if not test_vec:
        raise RuntimeError("embedding 服务连通性测试返回空向量。")
    logging.info(f"embedding 服务 OK，dim={len(test_vec)}")

    records = _embed_payloads(emb, payloads)
    _upsert_to_chroma(records)
    _smoke_check(sample_ids=[r[0] for r in records])
    logging.info("🎉 增量追加完成。")


if __name__ == "__main__":
    main()
