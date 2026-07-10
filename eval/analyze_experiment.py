"""Analyze verified V2 ablation runs and optionally perform blinded LLM judging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAIRS = {
    "wo_routing": "routing",
    "wo_bm25": "retrieval",
    "wo_schema": "sql",
    "wo_images": "deep",
    "wo_multi_agent": "deep",
}


def load_env_file(path: Path) -> None:
    """Load a simple .env file without requiring python-dotenv."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(rows: list[dict[str, Any]], predicate) -> float | None:
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else None


def avg(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    return mean(clean) if clean else None


def table_names(sqls: list[str]) -> set[str]:
    names = set()
    for sql in sqls:
        names.update(re.findall(r"(?i)\b(?:from|join)\s+[`\"]?([\w.]+)", str(sql)))
    return names


def row_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if r.get("status") == "done"]
    routing = [r for r in done if r.get("gold_action")]
    sql_rows = [r for r in done if r.get("subset") == "sql"]
    rag_rows = [r for r in done if r.get("subset") == "retrieval"]
    deep_rows = [r for r in done if r.get("subset") == "deep"]

    def sql_table_hit(row: dict[str, Any]) -> bool:
        return bool(set(row.get("gold_tables") or []) & table_names(row.get("executed_sqls") or []))

    def sql_term_hit(row: dict[str, Any]) -> bool:
        blob = " ".join(row.get("executed_sqls") or []).lower()
        terms = [str(x).lower() for x in row.get("gold_sql_terms") or []]
        return bool(terms) and all(term in blob for term in terms)

    def result_consistent(row: dict[str, Any]) -> bool:
        lo = row.get("expected_row_count_min")
        hi = row.get("expected_row_count_max")
        if not isinstance(lo, int) or not isinstance(hi, int):
            return False
        count = int(row.get("sql_row_count") or 0)
        return bool(row.get("executed_sqls")) and lo <= count <= hi

    def keyword_hit(row: dict[str, Any]) -> bool:
        blob = json.dumps(row.get("evidence_chunks") or [], ensure_ascii=False).lower()
        terms = [str(x).lower() for x in row.get("gold_keywords") or []]
        return bool(terms) and any(term in blob for term in terms)

    def reciprocal_rank(row: dict[str, Any]) -> float:
        terms = [str(x).lower() for x in row.get("gold_keywords") or []]
        for index, chunk in enumerate(row.get("evidence_chunks") or [], 1):
            blob = json.dumps(chunk, ensure_ascii=False).lower()
            if any(term in blob for term in terms):
                return 1.0 / index
        return 0.0

    return {
        "total": len(rows),
        "done": len(done),
        "completion_rate": len(done) / len(rows) if rows else None,
        "action_accuracy": rate(routing, lambda r: r.get("observed_action") == r.get("gold_action")),
        "task_type_accuracy": rate(
            routing, lambda r: r.get("observed_task_type") == r.get("gold_task_type")
        ),
        "mean_latency_seconds": avg([r.get("answer_elapsed_seconds") for r in done]),
        "sql_generation_rate": rate(sql_rows, lambda r: bool(r.get("executed_sqls"))),
        "sql_select_only_rate": rate(
            sql_rows,
            lambda r: bool(r.get("executed_sqls"))
            and all(re.match(r"(?is)^\s*select\b", x or "") for x in r.get("executed_sqls") or []),
        ),
        "sql_table_hit_rate": rate(sql_rows, sql_table_hit),
        "sql_term_hit_rate": rate(sql_rows, sql_term_hit),
        "sql_result_consistency": rate(sql_rows, result_consistent),
        "retrieval_keyword_recall": rate(rag_rows, keyword_hit),
        "retrieval_mrr": avg([reciprocal_rank(r) for r in rag_rows]),
        "mean_evidence_chunks": avg([len(r.get("evidence_chunks") or []) for r in rag_rows]),
        "visual_evidence_rate": rate(
            [r for r in deep_rows if r.get("requires_image")],
            lambda r: int((r.get("ablation_trace") or {}).get("injected_image_count") or 0) > 0,
        ),
    }


def sql_specialized_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if r.get("status") == "done"]
    sql_rows = [r for r in done if r.get("subset") == "sql"]

    def table_hit(row: dict[str, Any]) -> bool:
        return bool(set(row.get("gold_tables") or []) & table_names(row.get("executed_sqls") or []))

    def term_hit(row: dict[str, Any]) -> bool:
        blob = " ".join(row.get("executed_sqls") or []).lower()
        terms = [str(x).lower() for x in row.get("gold_sql_terms") or []]
        return bool(terms) and all(term in blob for term in terms)

    return {
        "total": len(rows),
        "done": len(done),
        "completion_rate": len(done) / len(rows) if rows else None,
        "clarification_rate": rate(sql_rows, lambda r: r.get("observed_action") == "CLARIFY"),
        "action_accuracy": rate(
            sql_rows, lambda r: r.get("observed_action") == r.get("gold_action")
        ),
        "task_type_accuracy": rate(
            sql_rows, lambda r: r.get("observed_task_type") == r.get("gold_task_type")
        ),
        "sql_generation_rate": rate(sql_rows, lambda r: bool(r.get("executed_sqls"))),
        "sql_table_hit_rate": rate(sql_rows, table_hit),
        "sql_term_hit_rate": rate(sql_rows, term_hit),
        "sql_ready_success_rate": rate(
            sql_rows,
            lambda r: bool(r.get("executed_sqls")) and table_hit(r) and term_hit(r),
        ),
        "sql_nonempty_result_rate": rate(
            sql_rows,
            lambda r: bool(r.get("executed_sqls")) and int(r.get("sql_row_count") or 0) > 0,
        ),
        "mean_latency_seconds": avg([r.get("answer_elapsed_seconds") for r in sql_rows]),
    }


def analyze_sql_specialized(exp: Path, analysis: Path) -> int:
    variants = ["full", "wo_schema"]
    data = {v: load_jsonl(exp / v / "results.jsonl") for v in variants}
    if not data["full"]:
        raise SystemExit(f"Missing full results under {exp}")

    validation = {v: validate_trace(v, rows) if rows else ["missing results"] for v, rows in data.items()}
    full_ids = {r["id"] for r in data["full"]}
    schema_ids = {r["id"] for r in data["wo_schema"]}
    if data["wo_schema"] and full_ids != schema_ids:
        validation["wo_schema"].append(
            f"paired IDs mismatch: missing={sorted(full_ids-schema_ids)}, "
            f"extra={sorted(schema_ids-full_ids)}"
        )
    for variant, rows in data.items():
        fingerprints = {
            (r.get("service_status") or {}).get("config_fingerprint")
            for r in rows
            if r.get("service_status")
        }
        if rows and len(fingerprints) != 1:
            validation[variant].append("mixed service fingerprints")

    metrics = {v: sql_specialized_metrics(rows) for v, rows in data.items() if rows}
    (analysis / "sql_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis / "sql_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (analysis / "sql_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["variant", *next(iter(metrics.values())).keys()]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant, values in metrics.items():
            writer.writerow({"variant": variant, **values})

    full_by_id = {r["id"]: r for r in data["full"]}
    comparisons = []
    for row_id in sorted(full_ids | schema_ids):
        full = full_by_id.get(row_id, {})
        ablated = next((r for r in data["wo_schema"] if r.get("id") == row_id), {})
        comparisons.append({
            "id": row_id,
            "question": full.get("question") or ablated.get("question"),
            "gold_action": full.get("gold_action") or ablated.get("gold_action"),
            "full": {
                "status": full.get("status"),
                "observed_action": full.get("observed_action"),
                "observed_task_type": full.get("observed_task_type"),
                "executed_sqls": full.get("executed_sqls") or [],
                "sql_row_count": full.get("sql_row_count"),
                "latency_seconds": full.get("answer_elapsed_seconds"),
            },
            "wo_schema": {
                "status": ablated.get("status"),
                "observed_action": ablated.get("observed_action"),
                "observed_task_type": ablated.get("observed_task_type"),
                "executed_sqls": ablated.get("executed_sqls") or [],
                "sql_row_count": ablated.get("sql_row_count"),
                "latency_seconds": ablated.get("answer_elapsed_seconds"),
            },
        })
    (analysis / "sql_item_comparison.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in comparisons),
        encoding="utf-8",
    )

    clarify_cases = [
        {
            "variant": variant,
            "id": row.get("id"),
            "question": row.get("question"),
            "observed_task_type": row.get("observed_task_type"),
        }
        for variant, rows in data.items()
        for row in rows
        if row.get("observed_action") == "CLARIFY"
    ]
    failures = [
        {
            "variant": variant,
            "id": row.get("id"),
            "status": row.get("status"),
            "observed_action": row.get("observed_action"),
            "error": row.get("error"),
        }
        for variant, rows in data.items()
        for row in rows
        if row.get("status") != "done"
        or not row.get("executed_sqls")
        or row.get("observed_action") == "CLARIFY"
    ]
    (analysis / "sql_clarification_cases.json").write_text(
        json.dumps(clarify_cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis / "sql_failure_cases.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table = [
        "# SQL Specialized Ablation Study",
        "",
        "| Variant | Completion | Clarify | Action Acc. | SQL Generation | Table Hit | Term Hit | SQL Ready Success | Non-empty (info only) | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, values in metrics.items():
        pct = lambda x: "—" if x is None else f"{100*x:.1f}%"
        latency = (
            "—" if values["mean_latency_seconds"] is None
            else f"{values['mean_latency_seconds']:.2f}"
        )
        table.append(
            f"| {variant} | {pct(values['completion_rate'])} | "
            f"{pct(values['clarification_rate'])} | {pct(values['action_accuracy'])} | "
            f"{pct(values['sql_generation_rate'])} | {pct(values['sql_table_hit_rate'])} | "
            f"{pct(values['sql_term_hit_rate'])} | "
            f"{pct(values['sql_ready_success_rate'])} | "
            f"{pct(values['sql_nonempty_result_rate'])} | {latency} |"
        )
    (analysis / "sql_ablation_table.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    print(f"Wrote SQL specialized analysis to {analysis}")
    if any(validation.values()):
        print("WARNING: sql_validation.json contains issues; do not use the table until resolved.")
    return 0


def validate_trace(variant: str, rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for row in rows:
        if row.get("status") != "done":
            continue
        trace = row.get("ablation_trace") or {}
        if trace.get("variant") != variant:
            errors.append(f"{row['id']}: trace variant={trace.get('variant')}")
        if variant == "wo_bm25" and int(trace.get("bm25_call_count") or 0) != 0:
            errors.append(f"{row['id']}: BM25 was called")
        if variant == "wo_images" and int(trace.get("injected_image_count") or 0) != 0:
            errors.append(f"{row['id']}: image was injected")
        if variant == "wo_images" and int(trace.get("retrieved_image_count") or 0) != 0:
            errors.append(f"{row['id']}: image metadata reached shared state")
        if variant == "wo_images" and row.get("kb_images"):
            errors.append(f"{row['id']}: kb_images is not empty")
        if variant == "wo_images":
            for chunk in row.get("evidence_chunks") or []:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("image_paths") or chunk.get("image_urls") or chunk.get("images"):
                    errors.append(f"{row['id']}: image metadata remains in evidence chunks")
                    break
        if variant == "wo_schema" and int(trace.get("sanitizer_call_count") or 0) != 0:
            errors.append(f"{row['id']}: sanitizer was called")
        if variant == "wo_multi_agent" and int(trace.get("researcher_call_count") or 0) != 0:
            errors.append(f"{row['id']}: researcher was called")
        if variant == "wo_multi_agent" and "supervisor_subgraph" in (trace.get("executed_nodes") or []):
            errors.append(f"{row['id']}: supervisor_subgraph was executed")
        if variant == "wo_routing" and trace.get("execution_path") != "deep":
            errors.append(f"{row['id']}: execution path was not deep")
    return errors


JUDGE_METRICS = (
    "correctness",
    "completeness",
    "traceability",
    "actionability",
    "uncertainty",
)


def parse_judge_json(raw: str) -> dict[str, Any]:
    """Parse judge output, including common fenced or comma-damaged JSON."""
    cleaned = str(raw or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Judge returned no JSON object: {cleaned[:300]}")
    candidate = cleaned[start : end + 1]
    candidate = (
        candidate.replace("“", '"')
        .replace("”", '"')
        .replace("：", ":")
        .replace("，", ",")
    )
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Fallback for otherwise regular output with missing commas.
        parsed = {}
        for side in ("A", "B"):
            block_match = re.search(
                rf'["\']?{side}["\']?\s*:\s*\{{(.*?)\}}',
                candidate,
                flags=re.I | re.S,
            )
            if not block_match:
                raise
            block = block_match.group(1)
            scores: dict[str, int] = {}
            for metric in JUDGE_METRICS:
                score_match = re.search(
                    rf'["\']?{metric}["\']?\s*:\s*([1-5](?:\.0+)?)',
                    block,
                    flags=re.I,
                )
                if not score_match:
                    raise
                scores[metric] = int(float(score_match.group(1)))
            parsed[side] = scores
        preferred_match = re.search(
            r'["\']?preferred["\']?\s*:\s*["\']?(A|B|tie)["\']?',
            candidate,
            flags=re.I,
        )
        reason_match = re.search(
            r'["\']?reason["\']?\s*:\s*["\']([^"\']*)',
            candidate,
            flags=re.I | re.S,
        )
        parsed["preferred"] = preferred_match.group(1) if preferred_match else "tie"
        parsed["reason"] = reason_match.group(1).strip() if reason_match else ""

    for side in ("A", "B"):
        scores = parsed.get(side)
        if isinstance(scores, list) and len(scores) == len(JUDGE_METRICS):
            scores = dict(zip(JUDGE_METRICS, scores))
        if not isinstance(scores, dict):
            raise ValueError(f"Judge output missing score block {side}")
        normalized: dict[str, int] = {}
        for metric in JUDGE_METRICS:
            value = scores.get(metric)
            if not isinstance(value, (int, float)) or not 1 <= float(value) <= 5:
                raise ValueError(f"Invalid {side}.{metric} score: {value!r}")
            normalized[metric] = int(round(float(value)))
        parsed[side] = normalized
    preferred = str(parsed.get("preferred") or "tie").strip()
    parsed["preferred"] = preferred if preferred in {"A", "B", "tie"} else "tie"
    parsed["reason"] = str(parsed.get("reason") or "")[:300]
    return parsed


def judge_call(prompt: str, max_attempts: int = 3) -> dict[str, Any]:
    endpoint = os.getenv(
        "GEMINI_URL_SYNC",
        "https://aimpapi.midea.com/t-aigc/mip-chat-app/gemini/official/standard/sync/v1/chat/completions",
    )
    api_key = os.getenv("MIDEA_API_KEY", "")
    if not api_key:
        raise RuntimeError("MIDEA_API_KEY is required for --llm-judge")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt += (
                "\n\nThe previous response could not be parsed. Return one complete JSON "
                "object only. Use double quotes and commas between every field."
            )
        body = {
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "contents": [{"role": "user", "parts": [{"text": attempt_prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Aimp-Biz-Id": os.getenv("GEMINI_AIMP_BIZ_ID", "gemini-2.5-flash"),
                "AIGC-USER": os.getenv("MIDEA_AIGC_USER", "user"),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(str(part.get("text") or "") for part in parts)
            return parse_judge_json(text)
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(1.0)
    raise RuntimeError(f"Judge failed after {max_attempts} attempts: {last_error}")


def build_judge_prompt(question: str, a: str, b: str) -> str:
    return f"""你是储能电池故障诊断论文的盲评专家。不要猜测答案来自哪个系统。
请分别对回答A和回答B按1-5分评分：correctness、completeness、traceability、
actionability、uncertainty。只返回JSON：
{{"A":{{"correctness":1,"completeness":1,"traceability":1,"actionability":1,"uncertainty":1}},
"B":{{"correctness":1,"completeness":1,"traceability":1,"actionability":1,"uncertainty":1}},
"preferred":"A|B|tie","reason":"不超过80字"}}

问题：{question}

回答A：
{a[:12000]}

回答B：
{b[:12000]}"""


def build_judge_prompt(question: str, a: str, b: str) -> str:
    """Build a compact, low-variance judge prompt."""
    return f"""You are conducting a blinded evaluation of answers for a battery
fault-diagnosis research paper. Do not infer which system produced either answer.

Score Answer A and Answer B from 1 to 5 in this exact order:
correctness, completeness, traceability, actionability, uncertainty awareness.

Return exactly one compact JSON object on one line, with no markdown or explanation:
{{"A":[1,1,1,1,1],"B":[1,1,1,1,1],"preferred":"A"}}

The preferred value must be "A", "B", or "tie".

Question:
{question}

Answer A:
{a[:12000]}

Answer B:
{b[:12000]}"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "eval_outputs" / "v2")
    parser.add_argument("--experiment-id", default="paper_v2")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument(
        "--sql-specialized",
        action="store_true",
        help="Analyze only the paired full and wo_schema SQL-specialized runs.",
    )
    args = parser.parse_args()

    load_env_file(ROOT / ".env")

    exp = args.root / args.experiment_id
    analysis = exp / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    if args.sql_specialized:
        return analyze_sql_specialized(exp, analysis)
    variants = ["full", *PAIRS]
    data = {v: load_jsonl(exp / v / "results.jsonl") for v in variants}
    if not data["full"]:
        raise SystemExit(f"Missing full results under {exp}")

    validation: dict[str, list[str]] = {}
    for variant in variants:
        if not data[variant]:
            validation[variant] = ["missing results"]
            continue
        validation[variant] = validate_trace(variant, data[variant])
        fingerprints = {
            (r.get("service_status") or {}).get("config_fingerprint")
            for r in data[variant]
            if r.get("service_status")
        }
        if len(fingerprints) != 1:
            validation[variant].append("mixed service fingerprints")

    full_by_id = {r["id"]: r for r in data["full"]}
    for variant, subset in PAIRS.items():
        expected = {r["id"] for r in data["full"] if r.get("subset") == subset}
        observed = {r["id"] for r in data[variant]}
        if data[variant] and expected != observed:
            validation[variant].append(
                f"paired IDs mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
            )
    full_deep = [r for r in data["full"] if r.get("subset") == "deep" and r.get("status") == "done"]
    required_image_full = [r for r in full_deep if r.get("requires_image")]
    if data["wo_images"] and required_image_full and not any(
        int((r.get("ablation_trace") or {}).get("injected_image_count") or 0) > 0
        for r in required_image_full
    ):
        validation["wo_images"].append(
            "full baseline injected no image on any requires_image sample; image contribution is not measurable"
        )
    if data["wo_multi_agent"] and full_deep and not any(
        int((r.get("ablation_trace") or {}).get("researcher_call_count") or 0) > 0
        or "ConductResearch" in ((r.get("ablation_trace") or {}).get("executed_nodes") or [])
        for r in full_deep
    ):
        validation["wo_multi_agent"].append(
            "full baseline has no verified ConductResearch dispatch; multi-agent contribution is not measurable"
        )

    metrics = {variant: row_metrics(rows) for variant, rows in data.items() if rows}
    paired_metrics: dict[str, Any] = {}
    for variant, subset in PAIRS.items():
        full_subset = [r for r in data["full"] if r.get("subset") == subset]
        paired_metrics[variant] = {
            "subset": subset,
            "full": row_metrics(full_subset),
            "ablated": row_metrics(data[variant]) if data[variant] else None,
        }
    (analysis / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis / "paired_metrics.json").write_text(
        json.dumps(paired_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (analysis / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["variant", *next(iter(metrics.values())).keys()]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant, values in metrics.items():
            writer.writerow({"variant": variant, **values})

    previous_judges = {
        row.get("pair_id"): row
        for row in load_jsonl(analysis / "blind_judge.jsonl")
        if row.get("pair_id") and isinstance(row.get("verdict"), dict)
    }
    judge_rows = []
    review_rows = []
    total_judge_pairs = sum(len(data[variant]) for variant in PAIRS)
    judge_pair_index = 0
    for variant, subset in PAIRS.items():
        for ablated in data[variant]:
            judge_pair_index += 1
            full = full_by_id.get(ablated["id"])
            if not full:
                print(
                    f"[judge {judge_pair_index}/{total_judge_pairs}] "
                    f"{variant}:{ablated['id']} skipped: missing full answer",
                    flush=True,
                )
                continue
            swapped = int(hashlib.sha256(f"{variant}:{ablated['id']}".encode()).hexdigest(), 16) % 2 == 0
            left, right = (ablated, full) if swapped else (full, ablated)
            package = {
                "pair_id": f"{variant}:{ablated['id']}",
                "question": ablated["question"],
                "answer_A": left.get("report", ""),
                "answer_B": right.get("report", ""),
            }
            previous = previous_judges.get(package["pair_id"])
            if previous:
                package["verdict"] = previous["verdict"]
                package["mapping"] = previous.get("mapping") or {
                    "A": left["id"] + (":ablated" if swapped else ":full"),
                    "B": right["id"] + (":full" if swapped else ":ablated"),
                }
                print(
                    f"[judge {judge_pair_index}/{total_judge_pairs}] "
                    f"{package['pair_id']} reused",
                    flush=True,
                )
            if args.llm_judge and left.get("report") and right.get("report"):
                if "verdict" not in package:
                    print(
                        f"[judge {judge_pair_index}/{total_judge_pairs}] "
                        f"{package['pair_id']} calling API...",
                        flush=True,
                    )
                    try:
                        verdict = judge_call(
                            build_judge_prompt(
                                package["question"], package["answer_A"], package["answer_B"]
                            )
                        )
                        package["verdict"] = verdict
                        package["mapping"] = {
                            "A": left["id"] + (":ablated" if swapped else ":full"),
                            "B": right["id"] + (":full" if swapped else ":ablated"),
                        }
                        print(
                            f"[judge {judge_pair_index}/{total_judge_pairs}] "
                            f"{package['pair_id']} success "
                            f"(preferred={verdict.get('preferred')})",
                            flush=True,
                        )
                    except Exception as exc:
                        package["judge_error"] = repr(exc)
                        print(
                            f"[judge {judge_pair_index}/{total_judge_pairs}] "
                            f"{package['pair_id']} failed: {exc}",
                            flush=True,
                        )
                verdict = package.get("verdict")
                if isinstance(verdict, dict):
                    scores = [
                        value
                        for side in ("A", "B")
                        for value in (verdict.get(side) or {}).values()
                        if isinstance(value, (int, float))
                    ]
                    low = bool(scores and min(scores) < 3)
                    a_mean = avg(list((verdict.get("A") or {}).values()))
                    b_mean = avg(list((verdict.get("B") or {}).values()))
                    disagreement = (
                        a_mean is not None
                        and b_mean is not None
                        and abs(a_mean - b_mean) > 1.5
                    )
                    sample = random.Random(package["pair_id"]).random() < 0.2
                    if low or disagreement or sample:
                        review_rows.append({
                            **package,
                            "review_reason": {
                                "low_score": low,
                                "large_difference": disagreement,
                                "random_20pct": sample,
                            },
                        })
            judge_rows.append(package)
            # Checkpoint after every pair so interruption does not lose completed calls.
            (analysis / "blind_judge.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in judge_rows),
                encoding="utf-8",
            )
            (analysis / "manual_review.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review_rows),
                encoding="utf-8",
            )

    (analysis / "blind_judge.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in judge_rows), encoding="utf-8"
    )
    (analysis / "manual_review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review_rows), encoding="utf-8"
    )

    table = [
        "# Table III. Ablation Study",
        "",
        "| Variant | Completion | Action Acc. | SQL Table Hit | SQL Result | Retrieval Recall | MRR | Visual Evidence | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table_rows = [("full_all", metrics["full"])]
    for variant, pair in paired_metrics.items():
        table_rows.append((f"full/{pair['subset']}", pair["full"]))
        if pair["ablated"] is not None:
            table_rows.append((variant, pair["ablated"]))
    for variant, m in table_rows:
        fmt = lambda x: "—" if x is None else f"{100*x:.1f}%"
        latency = "—" if m["mean_latency_seconds"] is None else f"{m['mean_latency_seconds']:.2f}"
        table.append(
            f"| {variant} | {fmt(m['completion_rate'])} | {fmt(m['action_accuracy'])} | "
            f"{fmt(m['sql_table_hit_rate'])} | {fmt(m['sql_result_consistency'])} | "
            f"{fmt(m['retrieval_keyword_recall'])} | {fmt(m['retrieval_mrr'])} | "
            f"{fmt(m['visual_evidence_rate'])} | {latency} |"
        )
    (analysis / "table_iii.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    failures = [
        {"variant": v, "id": r.get("id"), "status": r.get("status"), "error": r.get("error")}
        for v, rows in data.items()
        for r in rows
        if r.get("status") != "done"
    ]
    (analysis / "failure_cases.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote analysis to {analysis}")
    if any(validation.values()):
        print("WARNING: validation.json contains issues; do not use the table until resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
