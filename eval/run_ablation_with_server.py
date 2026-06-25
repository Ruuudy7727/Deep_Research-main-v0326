"""Start server_plus with ablation env, run one dataset eval, then stop server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional

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


def wait_for_server(base_url: str, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base_url.rstrip("/") + "/", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--dataset-key", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:50221")
    p.add_argument("--out-root", type=Path, default=Path("eval_outputs"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--server-start-timeout", type=float, default=180.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    variants = json.loads(Path("eval/ablation_variants.json").read_text(encoding="utf-8"))
    cfg = variants.get(args.variant)
    if not cfg:
        raise SystemExit(f"Unknown variant: {args.variant}")

    env = os.environ.copy()
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})

    server = subprocess.Popen(
        [sys.executable, "server_plus.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if not wait_for_server(args.base_url, timeout=args.server_start_timeout):
            print("[error] Server did not become ready", flush=True)
            return 1

        cmd = [
            sys.executable,
            "eval/run_ablation_suite.py",
            "--base-url",
            args.base_url,
            "--out-root",
            str(args.out_root),
            "--variants",
            args.variant,
            "--skip-health-check",
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        # run_ablation_suite runs all datasets for variant; call eval runner directly instead
        dataset_path = DATASET_MAP[args.dataset_key]
        out_dir = args.out_root / args.variant / args.dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)
        force_mode = (cfg.get("eval_overrides") or {}).get("force_mode")
        cmd = [
            sys.executable,
            "eval/run_frontend_api_eval.py",
            "--base-url",
            args.base_url,
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(out_dir),
            "--variant",
            args.variant,
            "--stream-timeout",
            str(STREAM_TIMEOUT.get(args.dataset_key, 600)),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if force_mode:
            cmd.extend(["--force-mode", force_mode])

        print(f"[run] {' '.join(cmd)}", flush=True)
        return subprocess.call(cmd, env=env)
    finally:
        if server.poll() is None:
            if sys.platform == "win32":
                server.terminate()
            else:
                server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
