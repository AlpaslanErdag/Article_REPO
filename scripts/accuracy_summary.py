#!/usr/bin/env python3
"""
Accuracy summary table for the SVSF paper.

Reports mean exact-match accuracy (and F1) per system, optionally broken down by
dataset (Source). This reproduces the paper's main accuracy table.

Systems: Vanilla, SVSF, MMR, Rerank, RerankBGE, MultiQuery.

Usage:
    python accuracy_summary.py history.csv
    python accuracy_summary.py history.csv --metric EM_flex --by-source
"""
import argparse
import pandas as pd

SYSTEMS = ["Vanilla", "SVSF", "MMR", "Rerank", "RerankBGE", "MultiQuery"]


def summarize(df, metric):
    row = {}
    for s in SYSTEMS:
        col = f"{s}_{metric}"
        if col in df.columns:
            v = df[col].dropna()
            if len(v):
                row[s] = 100 * v.mean()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--metric", default="EM_flex", help="EM, EM_flex, or F1")
    ap.add_argument("--by-source", action="store_true", help="break down per dataset")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    groups = [("All", df)]
    if args.by_source and "Source" in df.columns:
        for src in sorted(df["Source"].dropna().unique()):
            groups.append((src, df[df["Source"] == src]))

    header = f"{'Dataset':<20}" + "".join(f"{s:<12}" for s in SYSTEMS)
    print(f"Accuracy ({args.metric}, %)")
    print(header)
    for name, sub in groups:
        row = summarize(sub, args.metric)
        line = f"{name:<20}" + "".join(f"{row.get(s, float('nan')):<12.1f}" if s in row else f"{'-':<12}" for s in SYSTEMS)
        print(line)


if __name__ == "__main__":
    main()
