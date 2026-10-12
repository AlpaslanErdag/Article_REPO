# Query-Anchored Sequential Dense Feedback for Multi-Hop Retrieval

Code and per-question records for *"Query-Anchored Sequential Dense Feedback for Multi-Hop
Retrieval: Accuracy Limits, Segmentation Effects, and Reader Compensation."*

The paper evaluates sequential dense feedback (implemented as SVSF) in four roles —
standalone retrieval, a feedback mechanism, a deep-chain retriever, and an equal-budget
candidate generator — and reports where each boundary lies. This repository contains the
harness and the per-question records from which every table is computed, so the numbers can
be checked rather than taken on trust.

## Which file produces which table

| Paper | Produced by | Per-question records |
|---|---|---|
| Table 1, Table 2 (§5.1–5.2) | `app_v11.py` parameter sweep | `results/history_paramgrid_full.csv` |
| Table 3, Table 5 (§5.4, §5.6) | `app_v11.py` main evaluation | `results/main_1500.csv` |
| Table 4 (§5.5) | `scripts/latency_profile.py` | `results/latency_profile.json` |
| Table 6 (§5.7) | `scripts/retrieval_recall_main.py` | `results/retrieval_recall_1500.csv` |
| §5.7 diagnostic study | `scripts/retrieval_recall_diagnostic.py` | stdout only (summarised in §5.7) |
| Table 7 (§5.8) | `app_v11.py` on the deep-chain benchmark | `results/history_deepchain_hard_200.csv` |
| §5.9 CoT ablation | `scripts/cot_ablation.py` | `results/cot_ablation_v2.csv`, `results/history_deepchain_xhard_n4.csv` |
| §5.9 interaction test | `scripts/cot_interaction_test.py` | reads `cot_ablation_v2.csv` |
| §5.10 (n_docs sensitivity) | `app_v11.py` | `results/history_deepchain_xhard_n5.csv` |
| §5.11 mega-pool | `app_v11.py` | `results/history_sharedpool_mega.csv` |
| **Table 8 (§5.12)** | `scripts/candgen_megapool.py` | `results/candgen_megapool.csv` |
| §4.5 robustness checks | `scripts/robustness/` | stdout |

Summary statistics are recomputed from the record files by `scripts/accuracy_summary.py`,
`scripts/latency_analysis.py` and `scripts/mcnemar_test.py`.

## The three synthetic deep-chain variants

Three synthetic benchmarks of increasing difficulty are used in different sections. The
`Source` field in each record file identifies which one a run used.

| `Source` in records | Paper | Vanilla / SVSF / BGE (Flexible EM) |
|---|---|---|
| `DeepChainHard` | §5.8, Table 7 | 72.0 / 71.5 / 79.5 |
| `DeepChainXHard` | §5.9, §5.10 | 42.5 / 44.5 / 59.0 |
| `DeepChainMega` | §5.11 | 26.0 / 30.0 / not run |

The BGE reranker was not run in the mega-pool configuration; its columns are zero in that
record file.

## Layout

```
app_v11.py                 frozen retrieval engine: all systems compared in the paper
run_svsf_article.ipynb     notebook that drives the main evaluation
scripts/
  fast_nodes_patch.py      batched pool embedding (embed_pool_once)
  latency_profile.py       §5.5 / Table 4 — full per-query latency distribution
  retrieval_recall_main.py §5.7 / Table 6 — retrieval-level recall, main configuration
  retrieval_recall_diagnostic.py
                           §5.7 — 150-question diagnostic study, three datasets,
                           including the pseudo-relevance-feedback baselines
  cot_ablation.py          §5.9 — chain-of-thought on/off, both metrics, LLM judge
  cot_interaction_test.py  §5.9 — difference-in-differences interaction test + Holm
  candgen_megapool.py      §5.12 / Table 8 — equal-budget candidate generation
  candgen_per_question_corpora.py
                           supplementary; see note below
  accuracy_summary.py      recomputes accuracy tables from record files
  latency_analysis.py      recomputes latency tables
  mcnemar_test.py          paired significance tests
  robustness/
    prefix_ablation.py     §4.5 — query:/passage: prefix sensitivity
    encode_parity.py       §4.5 — batched vs. single-passage encoding parity
data/                      benchmarks and the distractor pool
results/                   per-question records
```

## Models and configuration

All inference is local; no closed-source API calls.

| Role | Model |
|---|---|
| Embedding | `BAAI/bge-m3` |
| Reranker (strong) | `BAAI/bge-reranker-v2-m3` |
| Reranker (classic) | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `gemma3:12b` via Ollama, 64k context, greedy decoding (temperature 0) |

Default retrieval configuration: `n_docs = 4`, `alpha = 1.0`, `beta = 0.6`, `tau = 0.2`,
semantic chunking threshold `0.3`. Similarity search is an exact brute-force cosine scan; no
approximate-nearest-neighbour index is used.

## Three caveats

1. **Generation is greedy but not bit-identical across runs.** Answer-level figures reproduce
   to within roughly one point rather than exactly. Where a number in the paper comes from a
   separate run of the same configuration, the paper says so.
2. **Latency is hardware-dependent.** All timings were measured on a single NVIDIA A100 with
   no concurrent load. The harness is released so that the measurement protocol — warm-up
   passes, CUDA synchronisation, per-query repetition, tail statistics — can be reused; the
   absolute values should not be expected to transfer.
3. **The exploratory sections use smaller context windows.** §5.1–5.3 were run earlier with
   8k–32k context as the harness evolved; §5.4 onward uses the 64k configuration. Those
   sections compare systems within a single run, so the comparison is unaffected, but their
   absolute values are not directly comparable to the confirmatory tables.

## Record files

Record files are the unmodified run logs. Each row carries the question, configuration
fields, the question's passage text, each system's generated answer, and per-system outcomes
(exact-match, flexible exact-match, F1, LLM-judge verdict, latency), so every reported metric
can be recomputed from the records.

**`results/history_negpheromone_ablation.csv`** is an exploratory run of the optional
negative-pheromone mechanism described in §3.4. That mechanism is **not enabled in any
experiment reported in the paper**; the record is kept for completeness and no reported
result depends on it.

**`scripts/candgen_per_question_corpora.py`** is the per-question-corpus variant of the
candidate-generation comparison. It was superseded by the mega-pool version reported in
§5.12, because in per-question corpora (≈11 chunks) a 50-candidate budget covers nearly the
whole corpus and candidate generation performs almost no filtering. It is included so that
the design decision behind §5.12 is visible.

## Running

```bash
pip install -r requirements.txt
ollama serve &
ollama pull gemma3:12b
```

Most scripts expect the engine to be loaded first (`app_v11.py` and
`scripts/fast_nodes_patch.py`). `scripts/cot_ablation.py` imports `app_v11` directly and runs
locally with checkpointing. `scripts/candgen_megapool.py` needs the distractor pool:

```python
import os
os.environ['SVSF_SHARED_CORPUS_PATH'] = 'data/mega_pool_5k.txt'
```

## Data

The benchmarks are public: HotpotQA (CC BY-SA 4.0), 2WikiMultihopQA (Apache 2.0) and MuSiQue
(CC BY 4.0), used through their released distributions. The synthetic deep-chain sets
constructed for §5.8–§5.11 are included under `data/`.
