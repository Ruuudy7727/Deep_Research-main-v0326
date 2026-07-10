# SQL20 查库专项实验

本实验只比较 `full` 和 `wo_schema`。远程服务器需要由实验执行者手动切换；
本地脚本只会检查当前远程变体并发送测试问题，不会启动、停止或切换远程服务。

## 1. 文件

- 数据集：`eval/datasets/v2_sql_20_no_clarify.jsonl`
- 实验客户端：`eval/run_experiment.py`
- PowerShell 包装脚本：`eval/run_sql20_client.ps1`
- 分析脚本：`eval/analyze_experiment.py --sql-specialized`
- 默认输出目录：`eval_outputs/v2/paper_v2_sql20`

原始 `eval/datasets/v2_ablation_40.jsonl` 不会被修改。

## 2. 远程服务器：启动 full

在远程服务器项目目录执行：

```bash
cd /work/v0511/Deep_Research-main-v0326
git status --short
python eval/serve_variant.py --variant full
```

如果服务由 supervisor、systemd 或其他进程管理器托管，请用原有方式停止旧实例，
再令其以环境变量 `ABLATION_VARIANT=full` 启动 `server_plus.py`。

检查状态：

```bash
curl -s https://aiops-pre.szclou.com:50221/api/eval/status
```

返回结果中的 `variant` 必须是 `full`。

## 3. 本地：运行 full

先运行1题冒烟：

```powershell
powershell -ExecutionPolicy Bypass -File eval/run_sql20_client.ps1 `
  -Variant full `
  -ExperimentId paper_v2_sql20 `
  -Limit 1
```

确认正常后运行全部20题。脚本会复用已完成的题目，因此不需要删除冒烟结果：

```powershell
powershell -ExecutionPolicy Bypass -File eval/run_sql20_client.ps1 `
  -Variant full `
  -ExperimentId paper_v2_sql20
```

也可直接调用 Python：

```powershell
python eval/run_experiment.py `
  --base-url https://aiops-pre.szclou.com:50221 `
  --variant full `
  --experiment-id paper_v2_sql20 `
  --dataset eval/datasets/v2_sql_20_no_clarify.jsonl
```

## 4. 远程服务器：在 50201 启动 wo_schema

保留 50221 端口的 `full` 实例，在远程服务器另开一个终端，通过
`server_50201.py` 在 50201 端口启动 `wo_schema`：

```bash
cd /work/v0511/Deep_Research-main-v0326
python server_50201.py --variant wo_schema
```

检查 50201 服务：

```bash
curl -s https://aiops.szclou.com:50201/api/eval/status
```

返回结果中的 `variant` 必须是 `wo_schema`。不要在状态仍为 `full` 时运行
`wo_schema` 客户端；客户端也会主动拒绝这种混跑。

## 5. 本地：运行 wo_schema

```powershell
powershell -ExecutionPolicy Bypass -File eval/run_sql20_client.ps1 `
  -Variant wo_schema `
  -BaseUrl https://aiops.szclou.com:50201/ `
  -ExperimentId paper_v2_sql20
```

或：

```powershell
python eval/run_experiment.py `
  --base-url https://aiops.szclou.com:50201/ `
  --variant wo_schema `
  --experiment-id paper_v2_sql20 `
  --dataset eval/datasets/v2_sql_20_no_clarify.jsonl
```

## 6. 生成专项分析

两组均完成后执行：

```powershell
python eval/analyze_experiment.py `
  --root eval_outputs/v2 `
  --experiment-id paper_v2_sql20 `
  --sql-specialized
```

主要输出：

- `analysis/sql_metrics.json`
- `analysis/sql_metrics.csv`
- `analysis/sql_ablation_table.md`
- `analysis/sql_item_comparison.jsonl`
- `analysis/sql_clarification_cases.json`
- `analysis/sql_failure_cases.json`
- `analysis/sql_validation.json`

只有当 `sql_validation.json` 中两组均为空数组时，才使用论文表格。

## 7. 注意事项

- `full` 与 `wo_schema` 必须使用同一个 `ExperimentId`，分析脚本才能按题目ID配对。
- 不要复用旧的 `paper_v2` 实验目录。
- 运行器会记录数据集 SHA-256，并拒绝在同一变体目录中混入不同版本的数据集。
- 若远程提交与本地提交不同，优先同步代码；只有明确知道差异与本实验无关时，
  才使用包装脚本的 `-AllowCommitMismatch`。
- 两道绘图题会检查 `DATABASE_CHART`。如果系统只执行SQL但没有生成图，
  它们会记为动作未命中，但不代表进入了澄清阶段。
