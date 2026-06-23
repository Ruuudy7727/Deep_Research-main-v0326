"""Batch evaluation runner for the server_plus.py frontend API.

The service accepts one request at a time:
  POST /api/chat     {"message": "...", "mode": "fast|deep"}
  GET  /api/stream   text/event-stream

This script keeps the implementation dependency-free so it can run both on a
local workstation and on the remote Jupyter environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonDict = Dict[str, Any]


@dataclass
class SseEvent:
    event: str
    data: JsonDict


def load_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def url_join(base_url: str, route: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", route.lstrip("/"))


def post_json(base_url: str, route: str, payload: JsonDict, timeout: float) -> JsonDict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url_join(base_url, route),
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return json.loads(text)


def post_empty(base_url: str, route: str, timeout: float) -> JsonDict:
    req = urllib.request.Request(url_join(base_url, route), data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return json.loads(text) if text.strip() else {}


def parse_sse_blocks(lines: Iterable[bytes]) -> Iterable[Tuple[str, str]]:
    event_name = "message"
    data_lines: List[str] = []

    for raw in lines:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    if data_lines:
        yield event_name, "\n".join(data_lines)


def stream_events(base_url: str, route: str, timeout: float) -> Iterable[SseEvent]:
    req = urllib.request.Request(
        url_join(base_url, route),
        headers={"Accept": "text/event-stream"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for event_name, data_text in parse_sse_blocks(resp):
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                data = {"raw": data_text}
            yield SseEvent(event=event_name, data=data)
            if event_name == "complete":
                break


def compact_text(value: Any, max_chars: int = 600) -> str:
    text = str(value or "")
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def infer_action(state: JsonDict, mode: str) -> str:
    if state.get("clarify_candidates"):
        return "CLARIFY"
    if (mode or "").lower() == "deep":
        return "DEEP"

    has_sql = bool(state.get("executed_sqls"))
    has_chart = bool(state.get("chart_url"))
    has_retrieval = bool(state.get("evidence_chunks") or state.get("kb_images"))

    if has_sql and has_chart:
        return "DATABASE_CHART"
    if has_chart:
        return "CHART"
    if has_sql:
        return "DATABASE"
    if has_retrieval:
        return "RETRIEVE"
    return "DIRECT"


def extract_tables_from_sql(sqls: List[str]) -> List[str]:
    tables: List[str] = []
    for sql in sqls:
        tokens = str(sql).replace("\n", " ").split()
        for idx, tok in enumerate(tokens[:-1]):
            if tok.lower() in {"from", "join"}:
                table = tokens[idx + 1].strip("`\"[],;")
                if table and table not in tables:
                    tables.append(table)
    return tables


def run_one(
    base_url: str,
    item: JsonDict,
    *,
    request_timeout: float,
    stream_timeout: float,
    reset_between: bool,
) -> JsonDict:
    if reset_between:
        try:
            post_empty(base_url, "/api/reset", timeout=request_timeout)
        except Exception as exc:  # noqa: BLE001 - keep eval robust
            print(f"[warn] reset failed for {item.get('id')}: {exc}")

    question = str(item.get("question") or "").strip()
    mode = str(item.get("mode") or "fast").strip().lower()
    payload: JsonDict = {"message": question, "mode": mode}
    if item.get("clarify_route"):
        payload["clarify_route"] = item["clarify_route"]

    started_at = time.time()
    result: JsonDict = {
        "id": item.get("id"),
        "category": item.get("category"),
        "question": question,
        "mode": mode,
        "gold_task_type": item.get("gold_task_type"),
        "gold_action": item.get("gold_action"),
        "gold_tables": item.get("gold_tables") or [],
        "started_at_unix": started_at,
        "status": "unknown",
        "events_seen": [],
        "errors": [],
    }

    try:
        chat_resp = post_json(base_url, "/api/chat", payload, timeout=request_timeout)
        result["chat_response"] = chat_resp
    except Exception as exc:  # noqa: BLE001
        result["status"] = "chat_error"
        result["errors"].append(str(exc))
        result["wall_elapsed_seconds"] = time.time() - started_at
        return result

    if result["chat_response"].get("status") == "busy":
        result["status"] = "busy"
        result["wall_elapsed_seconds"] = time.time() - started_at
        return result

    last_report = ""
    logs: List[str] = []
    timeline: JsonDict = {}
    final_state: JsonDict = {}
    complete_payload: JsonDict = {}

    try:
        for ev in stream_events(base_url, "/api/stream", timeout=stream_timeout):
            result["events_seen"].append(ev.event)
            if ev.event == "report":
                last_report = str(ev.data.get("html") or "")
            elif ev.event == "log":
                entries = ev.data.get("entries") or []
                logs.extend(str(x) for x in entries)
            elif ev.event == "timeline":
                timeline = ev.data
            elif ev.event == "state":
                final_state = ev.data
            elif ev.event == "complete":
                complete_payload = ev.data
                break
    except Exception as exc:  # noqa: BLE001
        result["status"] = "stream_error"
        result["errors"].append(str(exc))

    result["wall_elapsed_seconds"] = time.time() - started_at
    result["report"] = last_report
    result["report_preview"] = compact_text(last_report)
    result["logs"] = logs
    result["timeline"] = timeline
    result["final_state"] = final_state
    result["complete"] = complete_payload

    task_type = final_state.get("task_type")
    executed_sqls = final_state.get("executed_sqls") or []
    sql_table = final_state.get("sql_table") or {}
    alarm_rows = final_state.get("alarm_rows") or []
    evidence_chunks = final_state.get("evidence_chunks") or []
    kb_images = final_state.get("kb_images") or []
    answer_elapsed = (
        final_state.get("answer_elapsed_seconds")
        or complete_payload.get("answer_elapsed_seconds")
        or timeline.get("elapsed_seconds")
    )

    result["status"] = "done" if complete_payload.get("status") == "done" else result["status"]
    result["observed_task_type"] = task_type
    result["observed_action"] = infer_action(final_state, mode)
    result["observed_tables"] = extract_tables_from_sql([str(x) for x in executed_sqls])
    result["executed_sql_count"] = len(executed_sqls)
    result["sql_row_count"] = len(sql_table.get("rows") or [])
    result["alarm_row_count"] = len(alarm_rows)
    result["evidence_chunk_count"] = len(evidence_chunks)
    result["kb_image_count"] = len(kb_images)
    result["has_chart"] = bool(final_state.get("chart_url"))
    result["answer_elapsed_seconds"] = answer_elapsed
    result["task_type_correct"] = (
        task_type == item.get("gold_task_type") if item.get("gold_task_type") else None
    )
    result["action_correct"] = (
        result["observed_action"] == item.get("gold_action") if item.get("gold_action") else None
    )
    gold_tables = set(item.get("gold_tables") or [])
    result["table_hit"] = (
        bool(gold_tables.intersection(result["observed_tables"])) if gold_tables else None
    )
    return result


def mean(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def build_metrics(results: List[JsonDict]) -> JsonDict:
    total = len(results)
    done = [r for r in results if r.get("status") == "done"]
    labeled_task = [r for r in results if r.get("task_type_correct") is not None]
    labeled_action = [r for r in results if r.get("action_correct") is not None]
    table_labeled = [r for r in results if r.get("table_hit") is not None]
    db_expected = [
        r
        for r in results
        if str(r.get("gold_action") or "").upper() in {"DATABASE", "DATABASE_CHART"}
    ]

    by_category: JsonDict = {}
    for r in results:
        cat = str(r.get("category") or "uncategorized")
        cur = by_category.setdefault(cat, {"total": 0, "done": 0, "latencies": []})
        cur["total"] += 1
        if r.get("status") == "done":
            cur["done"] += 1
        if isinstance(r.get("answer_elapsed_seconds"), (int, float)):
            cur["latencies"].append(float(r["answer_elapsed_seconds"]))

    for cur in by_category.values():
        cur["completion_rate"] = cur["done"] / cur["total"] if cur["total"] else None
        cur["mean_answer_elapsed_seconds"] = mean(cur.pop("latencies"))

    return {
        "total": total,
        "done": len(done),
        "completion_rate": len(done) / total if total else None,
        "task_type_accuracy": (
            sum(1 for r in labeled_task if r.get("task_type_correct")) / len(labeled_task)
            if labeled_task
            else None
        ),
        "action_accuracy": (
            sum(1 for r in labeled_action if r.get("action_correct")) / len(labeled_action)
            if labeled_action
            else None
        ),
        "table_hit_rate": (
            sum(1 for r in table_labeled if r.get("table_hit")) / len(table_labeled)
            if table_labeled
            else None
        ),
        "db_sql_presence_rate": (
            sum(1 for r in db_expected if r.get("executed_sql_count", 0) > 0)
            / len(db_expected)
            if db_expected
            else None
        ),
        "mean_answer_elapsed_seconds": mean(
            [r.get("answer_elapsed_seconds") for r in done]  # type: ignore[list-item]
        ),
        "mean_wall_elapsed_seconds": mean(
            [r.get("wall_elapsed_seconds") for r in results]  # type: ignore[list-item]
        ),
        "mean_evidence_chunks": mean(
            [r.get("evidence_chunk_count") for r in done]  # type: ignore[list-item]
        ),
        "mean_kb_images": mean([r.get("kb_image_count") for r in done]),  # type: ignore[list-item]
        "chart_generation_count": sum(1 for r in done if r.get("has_chart")),
        "by_category": by_category,
    }


def write_summary_csv(path: Path, results: List[JsonDict]) -> None:
    fields = [
        "id",
        "category",
        "mode",
        "status",
        "gold_task_type",
        "observed_task_type",
        "task_type_correct",
        "gold_action",
        "observed_action",
        "action_correct",
        "executed_sql_count",
        "sql_row_count",
        "alarm_row_count",
        "evidence_chunk_count",
        "kb_image_count",
        "has_chart",
        "answer_elapsed_seconds",
        "wall_elapsed_seconds",
        "table_hit",
        "observed_tables",
        "question",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            if isinstance(row.get("observed_tables"), list):
                row["observed_tables"] = ";".join(row["observed_tables"])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:50221")
    parser.add_argument("--dataset", type=Path, default=Path("eval/datasets/server_plus_seed.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_outputs/server_plus_seed"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--stream-timeout", type=float, default=600.0)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument(
        "--reset-between",
        action="store_true",
        help="Reset server conversation state before each question. This is the default.",
    )
    parser.add_argument(
        "--no-reset-between",
        dest="reset_between",
        action="store_false",
        help="Keep conversation history across questions.",
    )
    parser.set_defaults(reset_between=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: List[JsonDict] = []

    for idx, item in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {item.get('id')} mode={item.get('mode')} ...", flush=True)
        result = run_one(
            args.base_url,
            item,
            request_timeout=args.request_timeout,
            stream_timeout=args.stream_timeout,
            reset_between=args.reset_between,
        )
        results.append(result)
        write_jsonl(args.out_dir / "results.jsonl", results)
        status = result.get("status")
        task = result.get("observed_task_type")
        action = result.get("observed_action")
        elapsed = result.get("answer_elapsed_seconds")
        print(f"  -> {status} task={task} action={action} elapsed={elapsed}", flush=True)
        if idx < len(rows) and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    metrics = build_metrics(results)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_summary_csv(args.out_dir / "summary.csv", results)

    print(f"\nWrote: {args.out_dir / 'results.jsonl'}")
    print(f"Wrote: {args.out_dir / 'summary.csv'}")
    print(f"Wrote: {args.out_dir / 'metrics.json'}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
