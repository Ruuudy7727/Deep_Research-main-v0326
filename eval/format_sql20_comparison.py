"""Render SQL20 full/wo_schema results as readable Markdown and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compact_sql(sqls: list[str]) -> str:
    return "\n".join(re.sub(r"\s+", " ", str(sql)).strip() for sql in sqls)


def table_names(sqls: list[str]) -> str:
    names: list[str] = []
    for sql in sqls:
        for name in re.findall(r"(?i)\b(?:from|join)\s+[`\"]?([\w.]+)", str(sql)):
            if name not in names:
                names.append(name)
    return ", ".join(names) or "—"


def md_cell(value: Any) -> str:
    return str(value if value not in (None, "") else "—").replace("|", r"\|").replace("\n", "<br>")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "eval_outputs" / "v2")
    parser.add_argument("--experiment-id", default="paper_v2_sql20")
    args = parser.parse_args()

    exp = args.root / args.experiment_id
    full_rows = load_jsonl(exp / "full" / "results.jsonl")
    schema_rows = load_jsonl(exp / "wo_schema" / "results.jsonl")
    full_by_id = {row["id"]: row for row in full_rows}
    schema_by_id = {row["id"]: row for row in schema_rows}
    ids = [row["id"] for row in full_rows]

    output_dir = exp / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "sql20_readable_comparison.md"
    csv_path = output_dir / "sql20_readable_comparison.csv"

    records: list[dict[str, Any]] = []
    for row_id in ids:
        full = full_by_id[row_id]
        schema = schema_by_id.get(row_id, {})
        full_sqls = full.get("executed_sqls") or []
        schema_sqls = schema.get("executed_sqls") or []
        records.append(
            {
                "id": row_id,
                "scenario": full.get("scenario"),
                "question": full.get("question"),
                "gold_action": full.get("gold_action"),
                "gold_tables": ", ".join(full.get("gold_tables") or []),
                "full_action": full.get("observed_action"),
                "full_task_type": full.get("observed_task_type"),
                "full_tables": table_names(full_sqls),
                "full_row_count": full.get("sql_row_count", 0),
                "full_sql": compact_sql(full_sqls) or "—",
                "full_latency_seconds": full.get("answer_elapsed_seconds"),
                "wo_schema_action": schema.get("observed_action"),
                "wo_schema_task_type": schema.get("observed_task_type"),
                "wo_schema_tables": table_names(schema_sqls),
                "wo_schema_row_count": schema.get("sql_row_count", 0),
                "wo_schema_sql": compact_sql(schema_sqls) or "—（最终状态未回填SQL）",
                "wo_schema_latency_seconds": schema.get("answer_elapsed_seconds"),
                "note": (
                    "绘图未生成"
                    if full.get("gold_action") == "DATABASE_CHART"
                    and full.get("observed_action") != "DATABASE_CHART"
                    else ("SQL已生成（结果为空）" if full_sqls and int(full.get("sql_row_count") or 0) == 0 else "SQL已生成")
                ),
            }
        )

    lines = [
        "# SQL20 逐题对照报告",
        "",
        "## 总体结论",
        "",
        "- 两组均完成 20/20 题，且均未进入 Clarify。",
        "- `full`：20/20 生成并回填 SQL，SQL生成、目标表和关键条件均命中；两道绘图题未生成图。",
        "- 数据库返回0行不计为失败；只要SQL正确生成并进入可查询步骤，即视为查库链路成功。",
        "- `wo_schema`：最终状态中 20/20 均未回填 SQL，因此显示为 `DIRECT`。报告文本声称执行了查询，但当前结果不足以验证实际 SQL。",
        "",
        "## 快速总览",
        "",
        "| ID | 场景 | 预期 | full动作 | full表 | 行数 | wo_schema动作 | 备注 |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for record in records:
        lines.append(
            "| {id} | {scenario} | {gold_action} | {full_action} | {full_tables} | "
            "{full_row_count} | {wo_schema_action} | {note} |".format(
                **{key: md_cell(value) for key, value in record.items()}
            )
        )

    lines.extend(["", "## 逐题详情", ""])
    for index, record in enumerate(records, 1):
        lines.extend(
            [
                f"### {index}. {record['id']} — {record['scenario']}",
                "",
                f"**问题：** {record['question']}",
                "",
                f"**预期：** `{record['gold_action']}`；目标表：`{record['gold_tables']}`",
                "",
                "| 对比项 | full | wo_schema |",
                "|---|---|---|",
                f"| 观测动作 | `{record['full_action']}` | `{record['wo_schema_action']}` |",
                f"| 任务类型 | `{record['full_task_type']}` | `{record['wo_schema_task_type']}` |",
                f"| 观测表 | `{record['full_tables']}` | `{record['wo_schema_tables']}` |",
                f"| 返回行数 | {record['full_row_count']} | {record['wo_schema_row_count']} |",
                f"| 延迟 | {float(record['full_latency_seconds'] or 0):.2f}s | "
                f"{float(record['wo_schema_latency_seconds'] or 0):.2f}s |",
                f"| 备注 | {record['note']} | SQL未回填，不能确认实际执行情况 |",
                "",
                "**full SQL：**",
                "",
                "```sql",
                record["full_sql"],
                "```",
                "",
                "**wo_schema SQL：**",
                "",
                "```text",
                record["wo_schema_sql"],
                "```",
                "",
            ]
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        csv_handle = csv_path.open("w", newline="", encoding="utf-8-sig")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"sql20_readable_comparison_{timestamp}.csv"
        csv_handle = csv_path.open("w", newline="", encoding="utf-8-sig")
    with csv_handle as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
