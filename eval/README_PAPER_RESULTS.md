# Paper Experiment Result Tables

This workflow turns evaluation outputs and human annotations into traceable
paper tables. It does not fabricate missing values: unmeasured metrics are
printed as `TBD`.

## 1. Complete the datasets

- Keep the existing `eval/datasets/v2_ablation_40.jsonl` as the historical
  ablation seed set.
- Expand the task suite toward 120-200 tasks with these `subset` values:
  `routing`, `sql`, `retrieval`, `deep`, and `end_to_end`.
- Required gold fields:
  - routing: `gold_task_type`, `gold_action`
  - sql: `gold_tables`, `gold_sql_terms`, optional row-count bounds
  - retrieval: `gold_evidence_ids` or `gold_keywords`
  - deep/end_to_end: reference evidence and manual scores in the CSV template

## 2. Fill dataset statistics

Copy `eval/datasets/paper_dataset_stats.template.json` to a real file, then
replace every `TODO` with anonymized counts from the structured logs and
knowledge base.

Recommended fields to report in the paper: data time span, alarm-event count,
telemetry field count, diagnostic-result count, knowledge-base document count,
chunk count, linked image count, historical-case count, and task split.

## 3. Add manual scores

Copy `eval/datasets/paper_manual_scores.template.csv` to a real file and score
each answer with:

- `correctness`, `completeness`, `traceability`, `actionability`: 1-5
- `root_cause_correct`: 0 or 1
- `unsupported_claim_rate`: 0-1, lower is better

Use blinded scoring for final numbers. For reviewer consistency, double-score
at least 20% of diagnostic cases.

## 4. Run experiments

Use the existing runner for each system variant. Example:

```powershell
python eval/run_experiment.py `
  --base-url https://aiops-pre.szclou.com:50221 `
  --experiment-id paper_v2_expanded `
  --variant full `
  --dataset eval/datasets/YOUR_EXPANDED_TASKS.jsonl
```

For baseline variants that are not implemented as server ablations yet
(`pure_llm`, `rag_only`, `tool_only`, `dense`, `bm25`), store compatible
`results.jsonl` files under:

```text
eval_outputs/v2/paper_v2_expanded/<variant>/results.jsonl
```

The rows should keep the same task IDs and include `status`, `report`,
`answer_elapsed_seconds`, optional `llm_usage` (token totals from the server),
and any available evidence/SQL fields.

## 5. Generate paper tables

```powershell
python eval/paper_tables.py `
  --root eval_outputs/v2 `
  --experiment-id paper_v2_expanded `
  --sql-experiment-id paper_v2_sql20 `
  --dataset-stats eval/datasets/YOUR_DATASET_STATS.json `
  --manual-scores eval/datasets/YOUR_MANUAL_SCORES.csv `
  --out eval_outputs/v2/paper_tables_expanded
```

Outputs:

- `paper_experiment_results.md`
- `paper_experiment_results.docx`
- one Markdown file per table
- `summary.json`

Only use the generated paper tables after the relevant validation files under
the experiment analysis directory are empty or explicitly explained in the
paper.
