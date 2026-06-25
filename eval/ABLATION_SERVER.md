# Ablation Experiments — Server Runbook

本目录包含论文 **Table III 消融实验** 所需的全部代码与数据标注。  
**不在本机跑实验**；同步到服务器后按下列步骤执行。

## 目录结构

```text
eval/
  ablation_variants.json          # 5 组消融 + full 的环境变量定义
  run_ablation_suite.py           # 按 variant 批量跑评测
  run_all_ablations_sequential.py # 顺序跑全部消融（单 server、无并发 busy）
  run_ablation_with_server.py     # 自动起停 server + 单次 dataset 评测
  run_on_server.sh                # 服务器一键流程（交互式重启 env）
  compare_ablation_results.py     # 生成对比表 + Table III markdown
  datasets/
    server_plus_seed.jsonl        # 混合 9 题
    from_docs_sql_batch.jsonl     # SQL 15 题
    from_docs_rag_direct_batch.jsonl
    from_docs_deep_batch.jsonl
    ablation_*.jsonl              # 各消融专用子集
    gold/                         # SQL / RAG / 图文 gold 标注
deep_research/
  ablation_config.py              # 读取 ABLATION_* 环境变量
```

## 消融开关（环境变量）

| 变量 | 作用 |
|------|------|
| `ABLATION_DISABLE_BM25=1` | 关闭 BM25，Dense-only |
| `BM25_ALPHA=0` | 与上项配合 |
| `ABLATION_DISABLE_SCHEMA=1` | Direct Text-to-SQL（跳过 plan/sanitize） |
| `ABLATION_DISABLE_IMAGES=1` | 不回传 KB 图片 / 不做 multimodal |
| `ABLATION_DISABLE_TOT=1` | 不生成 ToT 前置规划 |
| `ABLATION_FORCE_ALWAYS_DEEP=1` | API 层全部走 Deep |
| `ABLATION_FORCE_ALWAYS_COMPLEX=1` | 图内强制 Complex 分支 |
| `ABLATION_DISABLE_MULTI_AGENT=1` | 跳过 supervisor_subgraph（可选） |

完整配置见 [`ablation_variants.json`](ablation_variants.json)。

## 服务器执行流程

### 1. 启动 Full 基线

```bash
# 终端 A：默认 env，无 ABLATION_* 
python server_plus.py

# 终端 B
python eval/run_ablation_suite.py \
  --base-url http://127.0.0.1:50221 \
  --out-root eval_outputs \
  --variants full
```

产出：`eval_outputs/full/{seed,sql,rag,deep}/metrics.json`

### 2. 逐个消融（每次改 env 后重启 server）

以 **w/o BM25** 为例：

```bash
export ABLATION_VARIANT=wo_bm25
export ABLATION_DISABLE_BM25=1
export BM25_ALPHA=0
python server_plus.py   # 重启

python eval/run_ablation_suite.py --variants wo_bm25
```

**w/o Routing** 额外说明：评测脚本会通过 `--force-mode deep` 覆盖 dataset 中的 `mode`；服务器还需 `ABLATION_FORCE_ALWAYS_DEEP=1`。

各 variant 应对数据集：

| Variant | 数据集 | 题数 |
|---------|--------|------|
| full | seed, sql, rag, deep | 51 |
| wo_routing | seed | 9 |
| wo_bm25 | rag, seed_kb | 17 |
| wo_schema | sql, seed_db | 19 |
| wo_images | image_subset | 8 |
| wo_tot | tot_mixed | 12 |

### 3. 一键顺序跑（单 server，需手动换 env）

```bash
bash eval/run_on_server.sh          # 完整
bash eval/run_on_server.sh --smoke  # 每集 1 题烟测
```

或自动起停 server（单 variant + 单 dataset）：

```bash
python eval/run_ablation_with_server.py \
  --variant wo_bm25 --dataset-key rag --limit 1
```

### 4. 生成 Table III

```bash
python eval/compare_ablation_results.py --root eval_outputs
```

产出：

- `eval_outputs/analysis/ablation_comparison.json`
- `eval/paper/table_iii.md`
- `eval/paper/section_4_10_failures.md`

## Gold 标注（Layer C 指标）

| 文件 | 用途 |
|------|------|
| `datasets/gold/sql_gold.json` | 15 题参考 SQL + 期望行数区间 → result_consistency_proxy |
| `datasets/gold/rag_gold.json` | 10 题 chunk 关键词 → keyword_recall_proxy |
| `datasets/gold/image_gold.json` | 8 题图文题 → visual_evidence_recall_proxy |

可在服务器上补全 `gold_chunk_ids` / `gold_image_filenames` 后重跑 compare。

## 输出约定

每个 run 目录包含：

```text
eval_outputs/{variant}/{dataset}/
  results.jsonl
  summary.csv
  metrics.json
  variant.json
  run_meta.json
```

`eval_outputs/` 已在 `.gitignore` 中，实验结果留在服务器，论文表从 `eval/paper/` 拷贝。

## 注意事项

1. **服务单实例**：不要并行多个 `run_frontend_api_eval.py`，否则会 `busy`。
2. **Server-side 消融必须重启 server**：pre 环境无法改 env，必须在自有服务器跑 BM25/Schema/Images/ToT。
3. **wo_routing** 可在客户端用 `--force-mode deep` 部分复现；完整效果需 server `ABLATION_FORCE_ALWAYS_DEEP=1`。
4. pilot 历史结果在 `eval_outputs/eval_outputs/`（旧路径），新实验统一写到 `eval_outputs/full/`。
