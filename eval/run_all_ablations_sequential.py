"""Run all ablation datasets sequentially against one server (no parallel contention)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


JOBS = [
    ("wo_routing", "seed", ["--force-mode", "deep"], 1500),
    ("wo_bm25", "rag", [], 600),
    ("wo_bm25", "seed_kb", [], 600),
    ("wo_schema", "sql", [], 900),
    ("wo_schema", "seed_db", [], 900),
    ("wo_images", "image_subset", [], 900),
    ("wo_tot", "tot_mixed", [], 900),
]

DATASET = {
    "seed": "eval/datasets/server_plus_seed.jsonl",
    "rag": "eval/datasets/from_docs_rag_direct_batch.jsonl",
    "seed_kb": "eval/datasets/ablation_seed_kb.jsonl",
    "sql": "eval/datasets/from_docs_sql_batch.jsonl",
    "seed_db": "eval/datasets/ablation_seed_db.jsonl",
    "image_subset": "eval/datasets/ablation_image_subset.jsonl",
    "tot_mixed": "eval/datasets/ablation_tot_mixed.jsonl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="https://aiops-pre.szclou.com:50221")
    p.add_argument("--out-root", type=Path, default=Path("eval_outputs"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--delay-seconds", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rc = 0
    for variant, ds, extra, stream_timeout in JOBS:
        out_dir = args.out_root / variant / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "eval/run_frontend_api_eval.py",
            "--base-url",
            args.base_url,
            "--dataset",
            DATASET[ds],
            "--out-dir",
            str(out_dir),
            "--variant",
            variant,
            "--stream-timeout",
            str(stream_timeout),
            *extra,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        print(f"\n=== sequential: {variant}/{ds} ===", flush=True)
        cur = subprocess.call(cmd)
        if cur != 0:
            rc = cur
        meta = {
            "variant": variant,
            "dataset": ds,
            "base_url": args.base_url,
            "returncode": cur,
            "note": "Client-side eval only unless local server started with ABLATION_* env.",
        }
        (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(args.delay_seconds)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
