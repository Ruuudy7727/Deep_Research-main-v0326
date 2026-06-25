# §4.8 Ablation Study

This section is auto-generated after running on the server:

```bash
python eval/compare_ablation_results.py --root eval_outputs
```

Copy the table from [`table_iii.md`](table_iii.md) into the paper.

## Interpretation guide

| Ablation | Expected primary effect |
|----------|-------------------------|
| w/o Routing | Latency ↑ on Simple tasks; action labels shift to DEEP |
| w/o BM25 | evidence_chunks ↓, keyword_recall_proxy ↓ on RAG set |
| w/o Schema | table_hit ↓, result_consistency_proxy ↓ on SQL set |
| w/o Images | kb_images → 0, visual_evidence_recall_proxy ↓ |
| w/o ToT | action_accuracy ↓, task_type_accuracy ↓ on routing set |

**Status:** Pending server run. Placeholder until `eval_outputs/` is populated.
