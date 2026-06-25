"""Run one verified V2 ablation variant against remote server_plus.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "datasets" / "v2_ablation_40.jsonl"
VARIANT_SUBSETS = {
    "full": {"routing", "sql", "retrieval", "deep"},
    "wo_routing": {"routing"},
    "wo_bm25": {"retrieval"},
    "wo_schema": {"sql"},
    "wo_images": {"deep"},
    "wo_multi_agent": {"deep"},
}


def url(base: str, route: str) -> str:
    return urllib.parse.urljoin(base.rstrip("/") + "/", route.lstrip("/"))


def get_json(base: str, route: str, timeout: float = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url(base, route), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def post_json(base: str, route: str, payload: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
    req = urllib.request.Request(
        url(base, route),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def post_empty(base: str, route: str, timeout: float = 30) -> dict[str, Any]:
    req = urllib.request.Request(url(base, route), data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
        return json.loads(text) if text.strip() else {}


def parse_sse(lines: Iterable[bytes]) -> Iterable[dict[str, Any]]:
    event = "message"
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data:
                raw_data = "\n".join(data)
                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError:
                    payload = {"raw": raw_data}
                yield {"event": event, "data": payload}
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())


def stream(base: str, timeout: float) -> Iterable[dict[str, Any]]:
    req = urllib.request.Request(
        url(base, "/api/stream"),
        headers={"Accept": "text/event-stream"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        yield from parse_sse(response)


def load_dataset(variant: str) -> list[dict[str, Any]]:
    subsets = VARIANT_SUBSETS[variant]
    rows = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["subset"] in subsets:
                rows.append(row)
    return rows


def local_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def infer_action(state: dict[str, Any], requested_mode: str) -> str:
    if state.get("clarify_candidates"):
        return "CLARIFY"
    trace = state.get("ablation_trace") or {}
    if trace.get("execution_path") == "deep" or requested_mode == "deep":
        return "DEEP"
    sqls = state.get("executed_sqls") or []
    chart = state.get("chart_url")
    evidence = state.get("evidence_chunks") or []
    if sqls and chart:
        return "DATABASE_CHART"
    if sqls:
        return "DATABASE"
    if chart:
        return "CHART"
    if evidence:
        return "RETRIEVE"
    return "DIRECT"


def run_item(base: str, item: dict[str, Any], status: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.time()
    result = {**item, "service_status": status, "started_at": started, "events": [], "status": "unknown"}
    post_empty(base, "/api/reset")
    chat = post_json(base, "/api/chat", {"message": item["question"], "mode": item["mode"]})
    result["chat_response"] = chat
    if chat.get("status") != "started":
        result["status"] = chat.get("status", "chat_error")
        return result
    if chat.get("config_fingerprint") != status["config_fingerprint"]:
        raise RuntimeError("Service fingerprint changed between status check and /api/chat")

    report = ""
    final_state: dict[str, Any] = {}
    complete: dict[str, Any] = {}
    for event in stream(base, timeout):
        result["events"].append(event)
        if event["event"] == "report":
            report = str(event["data"].get("html") or "")
        elif event["event"] == "state":
            final_state = event["data"]
        elif event["event"] == "complete":
            complete = event["data"]
            break

    result["elapsed_wall_seconds"] = time.time() - started
    result["report"] = report
    result["final_state"] = final_state
    result["complete"] = complete
    result["ablation_trace"] = final_state.get("ablation_trace") or {}
    result["observed_task_type"] = final_state.get("task_type")
    result["observed_action"] = infer_action(final_state, item["mode"])
    result["executed_sqls"] = final_state.get("executed_sqls") or []
    result["sql_table"] = final_state.get("sql_table") or {}
    result["sql_row_count"] = len((result["sql_table"] or {}).get("rows") or [])
    result["evidence_chunks"] = final_state.get("evidence_chunks") or []
    result["kb_images"] = final_state.get("kb_images") or []
    result["answer_elapsed_seconds"] = (
        final_state.get("answer_elapsed_seconds") or complete.get("answer_elapsed_seconds")
    )
    result["status"] = "done" if complete.get("status") == "done" and report.strip() else "incomplete"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://aiops-pre.szclou.com:50221")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_SUBSETS))
    parser.add_argument("--experiment-id", default="paper_v2")
    parser.add_argument("--root", type=Path, default=ROOT / "eval_outputs" / "v2")
    parser.add_argument("--stream-timeout", type=float, default=1200)
    parser.add_argument("--allow-commit-mismatch", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from deep_research.ablation_config import AblationConfig

    expected = AblationConfig.for_variant(args.variant)
    service = get_json(args.base_url, "/api/eval/status")
    if service.get("protocol_version") != 2:
        raise SystemExit("Remote service does not expose ablation protocol V2.")
    if service.get("variant") != args.variant or service.get("config_fingerprint") != expected.fingerprint:
        raise SystemExit(
            f"Remote configuration mismatch: expected {args.variant}/{expected.fingerprint}, "
            f"got {service.get('variant')}/{service.get('config_fingerprint')}"
        )
    commit = local_commit()
    if not args.allow_commit_mismatch and commit != "unknown" and service.get("git_commit") != commit:
        raise SystemExit(
            f"Git commit mismatch: local={commit}, remote={service.get('git_commit')}. "
            "Sync Git or use --allow-commit-mismatch only for an intentional test."
        )

    out = args.root / args.experiment_id / args.variant
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = {
        "experiment_id": args.experiment_id,
        "variant": args.variant,
        "dataset": str(DATASET),
        "service_status": service,
        "local_git_commit": commit,
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["service_status"]["config_fingerprint"] != service["config_fingerprint"]:
            raise SystemExit("Refusing to resume into a directory with a different fingerprint.")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    results_path = out / "results.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["id"]] = row

    rows = load_dataset(args.variant)
    if args.limit:
        rows = rows[: args.limit]
    for index, item in enumerate(rows, 1):
        if existing.get(item["id"], {}).get("status") == "done":
            print(f"[{index}/{len(rows)}] {item['id']} resume-skip")
            continue
        current = get_json(args.base_url, "/api/eval/status")
        if (
            current.get("config_fingerprint") != service["config_fingerprint"]
            or current.get("started_at") != service["started_at"]
        ):
            raise SystemExit("Remote service restarted or changed configuration during the experiment.")
        print(f"[{index}/{len(rows)}] {item['id']} {args.variant}", flush=True)
        try:
            existing[item["id"]] = run_item(
                args.base_url, item, service, timeout=args.stream_timeout
            )
        except Exception as exc:
            existing[item["id"]] = {**item, "status": "error", "error": repr(exc)}
        ordered = [existing[row["id"]] for row in rows if row["id"] in existing]
        results_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in ordered),
            encoding="utf-8",
        )
        time.sleep(0.5)
    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
