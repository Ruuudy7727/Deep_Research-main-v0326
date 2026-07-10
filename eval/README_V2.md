# Ablation Experiment V2

V2 uses `server_plus.py` remotely and runs the experiment client locally.

## Remote

```bash
cd /work/v0511/Deep_Research-main-v0326
git fetch origin
git checkout eval-harness
git pull --ff-only origin eval-harness
```

Stop the old service and start one variant:

```bash
python eval/serve_variant.py --variant full
python eval/serve_variant.py --variant wo_routing
python eval/serve_variant.py --variant wo_bm25
python eval/serve_variant.py --variant wo_schema
python eval/serve_variant.py --variant wo_images
python eval/serve_variant.py --variant wo_multi_agent
```

Check `https://aiops-pre.szclou.com:50221/api/eval/status`.

## Local

After each remote restart, run the matching variant:

```powershell
python eval/run_experiment.py `
  --base-url https://aiops-pre.szclou.com:50221 `
  --experiment-id paper_v2 `
  --variant full
```

Repeat for all variants. Add `--limit 1` for a smoke test.

To refresh only one subset after a tracing or service fix:

```powershell
python eval/run_experiment.py `
  --base-url https://aiops-pre.szclou.com:50221 `
  --experiment-id paper_v2 `
  --variant full `
  --rerun-subset deep
```

Analyze completed runs:

```powershell
python eval/analyze_experiment.py `
  --root eval_outputs/v2 `
  --experiment-id paper_v2 `
  --llm-judge
```

The local `.env` supplies model gateway credentials. Do not use the generated
`table_iii.md` while `validation.json` contains any issue.

## SQL20 专项实验

新的无澄清查库题集、`full`/`wo_schema` 远程切换命令和专项分析流程见
`eval/README_SQL20.md`。
