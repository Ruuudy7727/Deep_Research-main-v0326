# Table III: Ablation Study on Critical Modules (Automated Metrics)

| Variant | Dataset (n) | Action Acc | Table Hit | SQL Exec | Evid. Chunks | KB Images | Latency |
|---------|-------------|------------|-----------|----------|--------------|-----------|---------|
| Ours-Full | deep (12) | 100.0% | — | — | 5.25 | 0.58 | 10.6s |
| Ours-Full | rag (15) | — | — | — | — | — | — |
| Ours-Full | seed (9) | — | — | 0.0% | — | — | — |
| Ours-Full | sql (15) | — | — | 0.0% | — | — | — |
| w/o BM25 | rag (15) | 50.0% (—) | — (—) | — (—) | 4.14 (—) | 0.50 (—) | 9.2s (—) |
| w/o BM25 | seed_kb (9) | — (—) | — (—) | 0.0% (—) | — (—) | — (—) | — (—) |
| w/o Schema | seed_db (9) | — (—) | — (—) | 0.0% (—) | — (—) | — (—) | — (—) |
| w/o Schema | sql (15) | — (—) | — (—) | 0.0% (0) | — (—) | — (—) | — (—) |
| w/o Images | image_subset (15) | 50.0% (—) | — (—) | — (—) | 4.14 (—) | 0.50 (—) | 9.2s (—) |
| w/o ToT | tot_mixed (9) | — (—) | — (—) | 0.0% (—) | — (—) | — (—) | — (—) |

## Notes
- Parentheses show delta vs Ours-Full on the same dataset.
- Expert scores deferred to Phase 2.
