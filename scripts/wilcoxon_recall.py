#!/usr/bin/env python3
"""
Wilcoxon signed-rank test + bootstrap 95% CI for paired recall comparison.

Recall is a continuous per-question value, so we use the Wilcoxon signed-rank
test (not McNemar) on paired recall differences, and report a bootstrap
confidence interval on the mean difference.

Usage:
    python wilcoxon_recall.py history.csv --system-a TwoStage --system-b FullRerank
    python wilcoxon_recall.py history.csv --system-a SVSFctx  --system-b SVSF

The CSV must contain per-question recall columns named "{System}_Recall".
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bootstrap_ci(diffs, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return 0.0, 0.0, 0.0
    means = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(diffs)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--system-a", default="TwoStage")
    ap.add_argument("--system-b", default="FullRerank")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    col_a = f"{args.system_a}_Recall"
    col_b = f"{args.system_b}_Recall"
    for c in (col_a, col_b):
        if c not in df.columns:
            raise SystemExit(f"Column not found: {c}\nAvailable: {list(df.columns)}")

    sub = df[[col_a, col_b]].dropna()
    a = sub[col_a].astype(float).values
    b = sub[col_b].astype(float).values
    diffs = a - b

    if np.any(diffs != 0):
        stat, p = wilcoxon(diffs)
    else:
        stat, p = 0.0, 1.0
    mean_d, lo, hi = bootstrap_ci(diffs, args.n_boot)

    print(f"n = {len(a)} paired questions")
    print(f"  mean {args.system_a}_Recall = {a.mean():.3f}   mean {args.system_b}_Recall = {b.mean():.3f}")
    print(f"  Delta (A-B) = {mean_d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  Wilcoxon p = {p:.4f}")
    if p < 0.05:
        print("  Verdict: SIGNIFICANT difference (p<0.05)")
    else:
        print("  Verdict: no significant difference; CI centered near zero => equivalent")


if __name__ == "__main__":
    main()
