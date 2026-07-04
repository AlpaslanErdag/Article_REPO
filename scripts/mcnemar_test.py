#!/usr/bin/env python3
"""
McNemar's test for paired accuracy comparison between retrieval systems.

Compares two systems' per-question correctness (exact-match flags) on the SAME
questions and reports whether their accuracy difference is statistically significant.

Usage:
    python mcnemar_test.py history.csv --system-a SVSF --system-b Vanilla
    python mcnemar_test.py history.csv --metric EM_flex   # default metric

The CSV must contain per-question columns named "{System}_{metric}", e.g.
"SVSF_EM_flex", "Vanilla_EM_flex" (0/1 per row).
"""
import argparse
import pandas as pd
from scipy.stats import chi2


def mcnemar(a_correct, b_correct):
    """Paired McNemar with continuity correction.
    Returns (b_only, a_only, statistic, p_value)."""
    b_only = sum(1 for x, y in zip(a_correct, b_correct) if x == 0 and y == 1)
    a_only = sum(1 for x, y in zip(a_correct, b_correct) if x == 1 and y == 0)
    n = a_only + b_only
    if n == 0:
        return b_only, a_only, 0.0, 1.0
    stat = (abs(a_only - b_only) - 1) ** 2 / n
    p = 1 - chi2.cdf(stat, 1)
    return b_only, a_only, stat, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--system-a", default="TwoStage")
    ap.add_argument("--system-b", default="FullRerank")
    ap.add_argument("--metric", default="EM_flex")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    col_a = f"{args.system_a}_{args.metric}"
    col_b = f"{args.system_b}_{args.metric}"
    for c in (col_a, col_b):
        if c not in df.columns:
            raise SystemExit(f"Column not found: {c}\nAvailable: {list(df.columns)}")

    a = df[col_a].fillna(0).astype(int).tolist()
    b = df[col_b].fillna(0).astype(int).tolist()
    b_only, a_only, stat, p = mcnemar(a, b)

    acc_a = 100 * sum(a) / len(a)
    acc_b = 100 * sum(b) / len(b)
    print(f"n = {len(a)} paired questions ({args.metric})")
    print(f"  {args.system_a}: {acc_a:.1f}%   {args.system_b}: {acc_b:.1f}%")
    print(f"  {args.system_a}-only correct: {a_only}   {args.system_b}-only correct: {b_only}")
    print(f"  McNemar chi2 = {stat:.3f},  p = {p:.4f}")
    print(f"  Verdict: {'SIGNIFICANT difference (p<0.05)' if p < 0.05 else 'no significant difference (equivalent)'}")


if __name__ == "__main__":
    main()
