# Server Plus API Evaluation

This folder contains a lightweight evaluation harness for the `server_plus.py`
frontend service. It submits questions through `/api/chat`, consumes the SSE
stream from `/api/stream`, and writes structured outputs for paper tables.

## Quick Start

Run against a local service:

```bash
python eval/run_frontend_api_eval.py \
  --base-url http://127.0.0.1:50221 \
  --dataset eval/datasets/server_plus_seed.jsonl \
  --out-dir eval_outputs/local_seed
```

Run against the pre-release service:

```bash
python eval/run_frontend_api_eval.py \
  --base-url https://aiops-pre.szclou.com:50221 \
  --dataset eval/datasets/server_plus_seed.jsonl \
  --out-dir eval_outputs/pre_seed
```

Use `--limit N` for a smoke test. The service is single-run oriented, so this
runner executes requests sequentially. By default, the runner calls
`POST /api/reset` before each question to avoid prior chat history leaking into
the next sample. Use `--no-reset-between` only when evaluating multi-turn
behavior.

## Dataset Format

Each line is a JSON object:

```json
{
  "id": "alerting_001",
  "question": "pack-7 在 2026-04-18 当天有哪些告警？",
  "mode": "fast",
  "gold_task_type": "alerting",
  "gold_action": "DATABASE",
  "gold_tables": ["alarm_event"]
}
```

Common labels:

- `mode`: `fast` or `deep`
- `gold_task_type`: `direct`, `kb_retrieval`, `station_device_td`,
  `alerting`, `troubleshooting`, `deep_research`
- `gold_action`: `DIRECT`, `RETRIEVE`, `DATABASE`, `CHART`,
  `DATABASE_CHART`, `DEEP`, or `CLARIFY`

## Outputs

The runner writes:

- `results.jsonl`: one full record per question, including final state fields.
- `summary.csv`: compact table for spreadsheet analysis.
- `metrics.json`: aggregate metrics such as completion rate, task-type
  accuracy, action accuracy, SQL presence, and latency.

Useful fields collected from SSE:

- `task_type`
- `executed_sqls`
- `sql_table`
- `alarm_rows`
- `evidence_chunks`
- `kb_images`
- `chart_url`
- `timeline.steps`
- `answer_elapsed_seconds`
- `report`

These are enough for routing accuracy, database-query evaluation, retrieval
evidence statistics, chart-generation statistics, and end-to-end latency tables.
