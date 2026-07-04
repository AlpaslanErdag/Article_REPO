# Analysis Scripts

Statistical tests and summary tables for the SVSF paper. All scripts read the
per-question results CSV produced by the benchmark harness (app_v11.py).

## Requirements
```
pip install pandas numpy scipy matplotlib
```

## Scripts

### `mcnemar_test.py` — paired accuracy significance
McNemar's test (continuity-corrected) comparing two systems' answer accuracy on
the same questions. Accuracy is a binary per-question outcome, so McNemar is the
appropriate paired test.
```
python mcnemar_test.py history.csv --system-a SVSF --system-b Vanilla --metric EM_flex
```

### `latency_analysis.py` — retrieval efficiency (Scalability Scissors)
Mean per-system retrieval latency and speedup factors relative to the
cross-encoder reranker — the paper's core efficiency result.
```
python latency_analysis.py history.csv --reference RerankBGE
```

### `accuracy_summary.py` — main accuracy table
Mean exact-match (or F1) accuracy per system, optionally per dataset.
```
python accuracy_summary.py history.csv --metric EM_flex --by-source
```

## Column conventions
Per-question columns expected in the CSV:
- `{System}_EM_flex`, `{System}_EM`, `{System}_F1` — accuracy metrics
- `{System}_Time_ms` — retrieval latency
- `Source` — dataset name (for per-dataset breakdowns)

Systems evaluated in this paper: `Vanilla`, `SVSF`, `MMR`, `Rerank`, `RerankBGE`, `MultiQuery`.
