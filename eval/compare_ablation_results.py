"""Compare ablation eval runs against Ours-Full baseline.

Reads summary.csv / metrics.json from eval_outputs and produces:
  - ablation_comparison.json
  - failure_modes.json
  - paper/table_iii.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonDict = Dict[str, Any]

DATASET_FILES = {
    "seed": "eval/datasets/server_plus_seed.jsonl",
    "sql": "eval/datasets/from_docs_sql_batch.jsonl",
    "rag": "eval/datasets/from_docs_rag_direct_batch.jsonl",
    "deep": "eval/datasets/from_docs_deep_batch.jsonl",
    "seed_kb": "eval/datasets/ablation_seed_kb.jsonl",
    "seed_db": "eval/datasets/ablation_seed_db.jsonl",
    "image_subset": "eval/datasets/ablation_image_subset.jsonl",
    "tot_mixed": "eval/datasets/ablation_tot_mixed.jsonl",
}

ABLATION_VARIANTS = ["wo_routing", "wo_bm25", "wo_schema", "wo_images", "wo_tot"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> List[JsonDict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pct(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{100.0 * value:.1f}%"


def sec(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}s"


def num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def delta_str(full: Optional[float], abl: Optional[float], *, lower_is_better: bool = False) -> str:
    if full is None or abl is None:
        return "—"
    diff = abl - full
    if abs(diff) < 1e-9:
        return "0"
    improved = diff < 0 if lower_is_better else diff > 0
    sign = "+" if diff > 0 else ""
    arrow = "↑" if (diff > 0 and not lower_is_better) or (diff < 0 and lower_is_better) else "↓"
    if not improved:
        arrow = "↓" if arrow == "↑" else "↑"
    return f"{sign}{diff:.3f} ({arrow})"


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def classify_failure(row: JsonDict) -> str:
    if row.get("status") != "done":
        return "incomplete"
    gold = str(row.get("gold_action") or "").upper()
    observed = str(row.get("observed_action") or "").upper()
    if gold in {"DATABASE", "DATABASE_CHART"}:
        if observed == "CLARIFY":
            return "clarify_instead_of_db"
        if observed == "DIRECT" and int(row.get("executed_sql_count") or 0) == 0:
            return "no_sql"
        if row.get("table_hit") in (False, "False", "false", 0, "0"):
            return "wrong_table"
    if gold == "RETRIEVE":
        if observed == "DIRECT" and int(row.get("evidence_chunk_count") or 0) == 0:
            return "no_retrieval"
    if gold == "DATABASE_CHART" and row.get("has_chart") in (False, "False", "false", 0, "0"):
        return "missing_chart"
    if row.get("action_correct") in (True, "True", "true", 1, "1"):
        return "success"
    return "routing_error"


def extended_metrics(rows: List[JsonDict]) -> JsonDict:
    latencies = [
        float(r["answer_elapsed_seconds"])
        for r in rows
        if r.get("status") == "done" and r.get("answer_elapsed_seconds") not in (None, "")
    ]
    failures: Dict[str, int] = {}
    for r in rows:
        mode = classify_failure(r)
        failures[mode] = failures.get(mode, 0) + 1
    return {
        "p50_answer_elapsed_seconds": percentile(latencies, 0.5),
        "p95_answer_elapsed_seconds": percentile(latencies, 0.95),
        "failure_modes": failures,
        "retrieve_trigger_rate": _rate(
            rows,
            lambda r: str(r.get("observed_action") or "").upper() == "RETRIEVE",
            lambda r: str(r.get("gold_action") or "").upper() == "RETRIEVE",
        ),
        "clarify_rate": _rate(rows, lambda r: str(r.get("observed_action") or "").upper() == "CLARIFY"),
        "strict_db_success_rate": _rate(
            rows,
            lambda r: str(r.get("gold_action") or "").upper() in {"DATABASE", "DATABASE_CHART"}
            and int(r.get("executed_sql_count") or 0) > 0
            and r.get("table_hit") in (True, "True", "true", 1, "1"),
            lambda r: str(r.get("gold_action") or "").upper() in {"DATABASE", "DATABASE_CHART"},
        ),
    }


def _rate(
    rows: List[JsonDict],
    numer_pred,
    denom_pred=None,
) -> Optional[float]:
    denom_rows = [r for r in rows if denom_pred(r)] if denom_pred else rows
    if not denom_rows:
        return None
    num = sum(1 for r in denom_rows if numer_pred(r))
    return num / len(denom_rows)


def load_gold_metrics(results_rows: List[JsonDict], gold_dir: Path) -> JsonDict:
    out: JsonDict = {}
    rag_gold = gold_dir / "rag_gold.json"
    sql_gold = gold_dir / "sql_gold.json"
    image_gold = gold_dir / "image_gold.json"

    if rag_gold.exists():
        gold = {g["id"]: g for g in load_json(rag_gold)}
        hits = 0
        total = 0
        for row in results_rows:
            g = gold.get(str(row.get("id")))
            if not g:
                continue
            total += 1
            chunks = ((row.get("final_state") or {}).get("evidence_chunks") or [])
            text_blob = json.dumps(chunks, ensure_ascii=False).lower()
            keywords = [str(k).lower() for k in g.get("gold_chunk_keywords") or []]
            if keywords and any(k in text_blob for k in keywords):
                hits += 1
            elif int(row.get("evidence_chunk_count") or 0) > 0:
                hits += 1
        out["keyword_recall_proxy"] = hits / total if total else None

    if sql_gold.exists():
        gold = {g["id"]: g for g in load_json(sql_gold)}
        row_match = 0
        total = 0
        for row in results_rows:
            g = gold.get(str(row.get("id")))
            if not g:
                continue
            total += 1
            count = int(row.get("sql_row_count") or 0)
            lo = int(g.get("expected_row_count_min", 0))
            hi = int(g.get("expected_row_count_max", 10**9))
            if lo <= count <= hi and int(row.get("executed_sql_count") or 0) > 0:
                row_match += 1
        out["result_consistency_proxy"] = row_match / total if total else None

    if image_gold.exists():
        gold = {g["id"]: g for g in load_json(image_gold)}
        hits = 0
        total = 0
        for row in results_rows:
            g = gold.get(str(row.get("id")))
            if not g:
                continue
            total += 1
            if int(row.get("kb_image_count") or 0) > 0:
                hits += 1
        out["visual_evidence_recall_proxy"] = hits / total if total else None

    return out


def compare_pair(full_metrics: JsonDict, abl_metrics: JsonDict) -> JsonDict:
    keys = [
        "completion_rate",
        "task_type_accuracy",
        "action_accuracy",
        "table_hit_rate",
        "db_sql_presence_rate",
        "mean_answer_elapsed_seconds",
        "mean_evidence_chunks",
        "mean_kb_images",
    ]
    delta: JsonDict = {}
    for k in keys:
        delta[k] = None
        fv, av = full_metrics.get(k), abl_metrics.get(k)
        if isinstance(fv, (int, float)) and isinstance(av, (int, float)):
            delta[k] = av - fv
    return delta


def build_table_iii(comparisons: JsonDict) -> str:
    lines = [
        "# Table III: Ablation Study on Critical Modules (Automated Metrics)",
        "",
        "| Variant | Dataset (n) | Action Acc | Table Hit | SQL Exec | Evid. Chunks | KB Images | Latency |",
        "|---------|-------------|------------|-----------|----------|--------------|-----------|---------|",
    ]
    full = comparisons.get("full", {})
    for ds_name, payload in sorted(full.items()):
        m = payload.get("metrics", {})
        lines.append(
            "| Ours-Full | "
            f"{ds_name} ({m.get('total', '—')}) | "
            f"{pct(m.get('action_accuracy'))} | "
            f"{pct(m.get('table_hit_rate'))} | "
            f"{pct(m.get('db_sql_presence_rate'))} | "
            f"{num(m.get('mean_evidence_chunks'))} | "
            f"{num(m.get('mean_kb_images'))} | "
            f"{sec(m.get('mean_answer_elapsed_seconds'))} |"
        )

    label_map = {
        "wo_routing": "w/o Routing",
        "wo_bm25": "w/o BM25",
        "wo_schema": "w/o Schema",
        "wo_images": "w/o Images",
        "wo_tot": "w/o ToT",
    }
    for variant in ABLATION_VARIANTS:
        vdata = comparisons.get(variant, {})
        for ds_name, payload in sorted(vdata.items()):
            m = payload.get("metrics", {})
            d = payload.get("delta_vs_full", {})
            lines.append(
                f"| {label_map.get(variant, variant)} | "
                f"{ds_name} ({m.get('total', '—')}) | "
                f"{pct(m.get('action_accuracy'))} ({delta_str(full.get(ds_name, {}).get('metrics', {}).get('action_accuracy'), m.get('action_accuracy'))}) | "
                f"{pct(m.get('table_hit_rate'))} ({delta_str(full.get(ds_name, {}).get('metrics', {}).get('table_hit_rate'), m.get('table_hit_rate'))}) | "
                f"{pct(m.get('db_sql_presence_rate'))} ({delta_str(full.get(ds_name, {}).get('metrics', {}).get('db_sql_presence_rate'), m.get('db_sql_presence_rate'))}) | "
                f"{num(m.get('mean_evidence_chunks'))} ({delta_str(full.get(ds_name, {}).get('metrics', {}).get('mean_evidence_chunks'), m.get('mean_evidence_chunks'))}) | "
                f"{num(m.get('mean_kb_images'))} ({delta_str(full.get(ds_name, {}).get('metrics', {}).get('mean_kb_images'), m.get('mean_kb_images'))}) | "
                f"{sec(m.get('mean_answer_elapsed_seconds'))} ({delta_str(full.get(ds_name, {}).get('metrics', {}).get('mean_answer_elapsed_seconds'), m.get('mean_answer_elapsed_seconds'), lower_is_better=True)}) |"
            )
    lines.extend(["", "## Notes", "- Parentheses show delta vs Ours-Full on the same dataset.", "- Expert scores deferred to Phase 2.", ""])
    return "\n".join(lines)


def pick_fail_cases(all_results: JsonDict, limit: int = 3) -> List[JsonDict]:
    cases: List[JsonDict] = []
    for variant, ds_map in all_results.items():
        for ds_name, payload in ds_map.items():
            rows = payload.get("results") if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                mode = classify_failure(row)
                if mode == "success":
                    continue
                cases.append(
                    {
                        "variant": variant,
                        "dataset": ds_name,
                        "id": row.get("id"),
                        "failure_mode": mode,
                        "question": row.get("question"),
                        "gold_action": row.get("gold_action"),
                        "observed_action": row.get("observed_action"),
                        "executed_sql_count": row.get("executed_sql_count"),
                        "evidence_chunk_count": row.get("evidence_chunk_count"),
                        "kb_image_count": row.get("kb_image_count"),
                    }
                )
                if len(cases) >= limit:
                    return cases
    return cases


def load_results_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gather_variant(root: Path, variant: str, datasets: Iterable[str], gold_dir: Path) -> JsonDict:
    out: JsonDict = {}
    for ds in datasets:
        run_dir = root / variant / ds
        metrics_path = run_dir / "metrics.json"
        results_path = run_dir / "results.jsonl"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        results = load_results_jsonl(results_path)
        ext = extended_metrics(load_csv_rows(run_dir / "summary.csv") or [])
        metrics.update(ext)
        metrics.update(load_gold_metrics(results, gold_dir))
        out[ds] = {"metrics": metrics, "results": results}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("eval_outputs"))
    p.add_argument("--gold-dir", type=Path, default=Path("eval/datasets/gold"))
    p.add_argument("--out-dir", type=Path, default=Path("eval_outputs/analysis"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    variants_cfg = load_json(Path("eval/ablation_variants.json"))
    all_ds = set()
    for cfg in variants_cfg.values():
        all_ds.update(cfg.get("datasets") or [])

    full_data = gather_variant(args.root, "full", all_ds, args.gold_dir)
    comparisons: JsonDict = {"full": {ds: {"metrics": p["metrics"]} for ds, p in full_data.items()}}

    for variant in ABLATION_VARIANTS:
        ds_list = variants_cfg.get(variant, {}).get("datasets") or []
        vdata = gather_variant(args.root, variant, ds_list, args.gold_dir)
        comparisons[variant] = {}
        for ds, payload in vdata.items():
            delta = compare_pair(full_data.get(ds, {}).get("metrics", {}), payload["metrics"])
            comparisons[variant][ds] = {"metrics": payload["metrics"], "delta_vs_full": delta}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ablation_comparison.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fail_cases = pick_fail_cases(
        {**{"full": full_data}, **{v: gather_variant(args.root, v, variants_cfg[v]["datasets"], args.gold_dir) for v in ABLATION_VARIANTS}},
        limit=5,
    )
    (args.out_dir / "failure_modes.json").write_text(
        json.dumps(fail_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paper_dir = Path("eval/paper")
    paper_dir.mkdir(parents=True, exist_ok=True)
    table_md = build_table_iii(comparisons)
    (paper_dir / "table_iii.md").write_text(table_md, encoding="utf-8")

    fail_md = ["# §4.10 Failure Mode Examples", ""]
    for i, case in enumerate(fail_cases[:3], start=1):
        fail_md.extend(
            [
                f"## Case {i}: {case.get('failure_mode')} ({case.get('variant')} / {case.get('id')})",
                "",
                f"- **Question:** {case.get('question')}",
                f"- **Gold action:** {case.get('gold_action')}",
                f"- **Observed action:** {case.get('observed_action')}",
                f"- **SQL count:** {case.get('executed_sql_count')}",
                f"- **Evidence chunks:** {case.get('evidence_chunk_count')}",
                f"- **KB images:** {case.get('kb_image_count')}",
                "",
            ]
        )
    (paper_dir / "section_4_10_failures.md").write_text("\n".join(fail_md), encoding="utf-8")

    print(f"Wrote {args.out_dir / 'ablation_comparison.json'}")
    print(f"Wrote {paper_dir / 'table_iii.md'}")
    print(f"Wrote {paper_dir / 'section_4_10_failures.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
