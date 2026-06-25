"""Run the full ablation experiment matrix against server_plus.py.

Usage:
  # 1) Start server with default (full) config, then:
  python eval/run_ablation_suite.py --base-url http://127.0.0.1:50221 --variants full

  # 2) For each ablation variant, restart server with env from eval/ablation_variants.json:
  set ABLATION_DISABLE_BM25=1 && set BM25_ALPHA=0 && python server_plus.py
  python eval/run_ablation_suite.py --variants wo_bm25

  # 3) Compare all results:
  python eval/compare_ablation_results.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


DATASET_MAP = {
    "seed": Path("eval/datasets/server_plus_seed.jsonl"),
    "sql": Path("eval/datasets/from_docs_sql_batch.jsonl"),
    "rag": Path("eval/datasets/from_docs_rag_direct_batch.jsonl"),
    "deep": Path("eval/datasets/from_docs_deep_batch.jsonl"),
    "seed_kb": Path("eval/datasets/ablation_seed_kb.jsonl"),
    "seed_db": Path("eval/datasets/ablation_seed_db.jsonl"),
    "image_subset": Path("eval/datasets/ablation_image_subset.jsonl"),
    "tot_mixed": Path("eval/datasets/ablation_tot_mixed.jsonl"),
}

STREAM_TIMEOUT = {
    "seed": 900,
    "sql": 900,
    "rag": 600,
    "deep": 1500,
    "seed_kb": 600,
    "seed_db": 900,
    "image_subset": 900,
    "tot_mixed": 900,
}


def load_variants() -> Dict[str, Any]:
    return json.loads(Path("eval/ablation_variants.json").read_text(encoding="utf-8"))


def server_ready(base_url: str, timeout: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/", method="GET")
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except Exception:
            return False


def run_dataset(
    base_url: str,
    variant: str,
    dataset_key: str,
    out_root: Path,
    *,
    limit: int | None,
    force_mode: str | None,
) -> int:
    dataset_path = DATASET_MAP[dataset_key]
    out_dir = out_root / variant / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "eval/run_frontend_api_eval.py",
        "--base-url",
        base_url,
        "--dataset",
        str(dataset_path),
        "--out-dir",
        str(out_dir),
        "--variant",
        variant,
        "--stream-timeout",
        str(STREAM_TIMEOUT.get(dataset_key, 600)),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if force_mode:
        cmd.extend(["--force-mode", force_mode])

    print(f"\n=== [{variant}] {dataset_key} -> {out_dir} ===", flush=True)
    proc = subprocess.run(cmd, check=False)
    meta = {
        "variant": variant,
        "dataset": dataset_key,
        "dataset_path": str(dataset_path),
        "out_dir": str(out_dir),
        "force_mode": force_mode,
        "returncode": proc.returncode,
        "finished_at_unix": time.time(),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return proc.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:50221")
    p.add_argument("--out-root", type=Path, default=Path("eval_outputs"))
    p.add_argument(
        "--variants",
        default="full,wo_routing,wo_bm25,wo_schema,wo_images,wo_tot",
        help="Comma-separated variant names from eval/ablation_variants.json",
    )
    p.add_argument("--limit", type=int, default=None, help="Smoke-test limit per dataset")
    p.add_argument("--skip-health-check", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    variants_cfg = load_variants()
    selected = [v.strip() for v in args.variants.split(",") if v.strip()]

    if not args.skip_health_check and not server_ready(args.base_url):
        print(f"[error] Server not reachable at {args.base_url}", flush=True)
        print("Start server_plus.py first. See eval/ABLATION_SERVER.md", flush=True)
        return 1

    rc = 0
    for variant in selected:
        cfg = variants_cfg.get(variant)
        if not cfg:
            print(f"[skip] Unknown variant: {variant}")
            continue
        env_hint = cfg.get("env") or {}
        if env_hint:
            print(f"[info] Expected server env for {variant}: {env_hint}")
        force_mode = (cfg.get("eval_overrides") or {}).get("force_mode")
        for ds in cfg.get("datasets") or []:
            if ds not in DATASET_MAP:
                print(f"[skip] Unknown dataset key: {ds}")
                continue
            cur = run_dataset(
                args.base_url,
                variant,
                ds,
                args.out_root,
                limit=args.limit,
                force_mode=force_mode,
            )
            if cur != 0:
                rc = cur
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
