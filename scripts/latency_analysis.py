#!/usr/bin/env python3
"""
Retrieval-latency analysis for the SVSF paper.

Computes mean per-system retrieval latency (ms) from the results CSV and reports
speedup factors relative to the cross-encoder rerankers — the paper's core
efficiency finding (the "Scalability Scissors").

Systems: Vanilla, SVSF, MMR, Rerank, RerankBGE, MultiQuery.

Usage:
    python latency_analysis.py history.csv
    python latency_analysis.py history.csv --reference RerankBGE
"""
import argparse
import pandas as pd

SYSTEMS = ["Vanilla", "SVSF", "MMR", "Rerank", "RerankBGE", "MultiQuery"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--reference", default="RerankBGE",
                    help="system to compute speedups against (default: RerankBGE)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    means = {}
    for s in SYSTEMS:
        col = f"{s}_Time_ms"
        if col in df.columns:
            v = df[col].dropna()
            if len(v):
                means[s] = v.mean()

    if not means:
        raise SystemExit("No *_Time_ms columns found in CSV.")

    ref = args.reference
    ref_ms = means.get(ref)
    print(f"Mean retrieval latency (n={len(df)} questions)")
    print(f"{'System':<12}{'Latency (ms)':<15}{'Speedup vs ' + ref}")
    for s in SYSTEMS:
        if s in means:
            speed = f"{ref_ms / means[s]:.1f}x faster" if (ref_ms and s != ref and means[s] > 0) else ("(reference)" if s == ref else "")
            print(f"{s:<12}{means[s]:<15.2f}{speed}")


if __name__ == "__main__":
    main()
