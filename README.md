# SVSF: Stigmergic Vector-Space Foraging for Efficient Multi-Hop Retrieval

This repository contains the reference implementation, evaluation harness, and datasets for **SVSF (Stigmergic Vector-Space Foraging)**, a lightweight multi-hop retrieval method in which a sequence of agents forages through the embedding space, guided by a dual-attractor score that balances query attraction against a stigmergic pheromone trail accumulated from previously selected evidence.

SVSF matches single-pass dense retrieval in accuracy while retrieving **51–93× faster** than a cross-encoder reranker, with an efficiency advantage that widens as the candidate pool grows (the *Scalability Scissors*).

> Paper: *SVSF: Stigmergic Vector-Space Foraging for Efficient Multi-Hop Retrieval* (under review).

---

## Key idea

For multi-hop questions, the document holding the final answer (the *bridge document*) often shares little similarity with the original query, so single-pass retrievers miss it. SVSF addresses this without expensive reranking:

```
Score(D) = α · cos(Q₀, D)  +  β · cos(Pₜ, D)
                │                    │
        query attraction      pheromone trail (the chain)
```

Each of `N` agents picks the top-scoring document, deposits its evidence into the pheromone trail `Pₜ`, and biases the next agent toward connected evidence — following the reasoning chain in `O(N)` vectorized cosine operations, with no transformer in the retrieval loop.

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── app_v11.py                 # Flask evaluation harness (all retrieval systems + metrics)
├── run_svsf.ipynb             # Colab notebook (runs the full harness end-to-end)
├── index_v11.html             # Batch-evaluation UI (if needed)
├── data/
│   ├── deepchain_*.csv        # Self-constructed deep-chain QA sets
│   ├── mega_pool_5k.txt       # 4,942-passage stress-test pool (real + synthetic distractors)
│   └── *_queries.csv          # Query files for shared-pool / mega-pool runs
├── results/
│   └── history_*.csv          # Raw experiment logs (per-system answers, metrics, latency)
└── scripts/
    └── analysis/              # McNemar / Wilcoxon / figure-generation scripts
```

---

## Setup

Requires Python 3.10+ and a CUDA GPU (experiments used a single NVIDIA A100 on Google Colab).

```bash
pip install -r requirements.txt
# Core deps: sentence-transformers, FlagEmbedding (bge-m3), rank-bm25,
#            scikit-learn, numpy, flask, ollama (for the generator)
```

The generator (`gemma3:12b`) is served via [Ollama](https://ollama.com). The dense embedder is `BAAI/bge-m3`; the SOTA reranker baseline is `BAAI/bge-reranker-v2-m3`.

### Quick start (Colab — recommended)

Open `run_svsf.ipynb`, run all cells. The app cell launches the harness; the UI (`index_v11.html`) drives batch runs.

### Local

```bash
ollama pull gemma3:12b
python app_v11.py          # serves the harness on localhost
```

---

## Datasets

**Public benchmarks (not redistributed here — download from the original sources):**
- HotpotQA (distractor) — https://hotpotqa.github.io
- 2WikiMultihopQA — https://github.com/Alab-NII/2wikimultihop
- MuSiQue — https://github.com/StonyBrookNLP/musique

We sample 50 validation instances per benchmark (identical samples across all configurations) for the ablation grid, and 1,500 questions for the main confirmatory run.

**Self-constructed datasets (included in `data/`):**
- `deepchain_*.csv` — deep-chain multi-hop questions with controllable distractor density.
- `mega_pool_5k.txt` — a 4,942-passage pool (1,442 real passages + 3,500 topically-similar synthetic distractors) used for the bottleneck stress test.

All public benchmarks remain under their original licenses; please cite the original dataset papers when using them.

---

## Reproducing the main results

| Result | Config |
|---|---|
| Main 1,500-question run | `N=4, α=1.0, β=0.6, τ=0.2, semantic chunking thr=0.3, manifold=1.0` |
| Deep-chain stress test | same config, deep-chain set, `gemma3:12b-64k` |
| Mega-pool stress test | set `SVSF_SHARED_CORPUS_PATH=data/mega_pool_5k.txt` |
| CoT ablation | set `SVSF_NO_COT=1` |

Raw logs for every reported result are in `results/`. Analysis scripts in `scripts/analysis/` regenerate the McNemar/Wilcoxon tests and figures from these logs.

**Best configuration:** `N=4, α=1.0, β=0.6, τ=0.2, semantic chunking threshold=0.3`.

---

## Citation

```bibtex
@article{erdag_svsf,
  title   = {SVSF: Stigmergic Vector-Space Foraging for Efficient Multi-Hop Retrieval},
  author  = {Erdağ, Alpaslan},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Released under the MIT License — see [LICENSE](LICENSE).
