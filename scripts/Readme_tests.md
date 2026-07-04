# Analysis Scripts

Statistical tests and figure generation for the SVSF two-stage retrieval paper.
All scripts read the per-question results CSV produced by the benchmark harness.

## Requirements
```
pip install pandas numpy scipy matplotlib
```

## Scripts

### `mcnemar_test.py` — paired accuracy significance
McNemar's test (with continuity correction) for comparing two systems' answer
accuracy on the same questions. Accuracy is a binary per-question outcome, so
McNemar is the correct paired test.
```
python mcnemar_test.py history.csv --system-a TwoStage --system-b FullRerank --metric EM_flex
```

### `wilcoxon_recall.py` — paired recall significance
Wilcoxon signed-rank test + bootstrap 95% CI on paired recall differences.
Recall is continuous, so Wilcoxon (not McNemar) is used.
```
python wilcoxon_recall.py history.csv --system-a SVSFctx --system-b SVSF
```

### `generate_figures.py` — paper figures
Regenerates recall-by-dataset, latency (scalability scissors), and
two-stage-vs-full-rerank figures.
```
python generate_figures.py --csv history.csv --outdir figures/
# latency values can be supplied via --latency-csv (columns: system,latency_ms)
```

## Column conventions
The CSV is expected to contain per-question columns:
- `{System}_Recall` (0–1) for recall tests and figures
- `{System}_EM_flex` (0/1) for accuracy tests
- `Source` (dataset name) for per-dataset breakdowns

Systems: `Vanilla`, `SVSF`, `SVSFctx`, `TwoStage`, `FullRerank`.
