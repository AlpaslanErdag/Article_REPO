#!/usr/bin/env python3
"""
Generate the paper's figures from a results CSV (or from summary numbers).

Produces:
  fig_recall_by_dataset.png   - recall per system across datasets
  fig_latency_scissors.png    - mega-pool latency (scalability scissors)
  fig_twostage_vs_full.png    - reranker load vs resulting recall

Usage:
    # From a per-question CSV (computes means automatically):
    python generate_figures.py --csv history.csv --outdir figures/

    # Latency figure uses fixed mega-pool measurements (edit LATENCY dict below
    # or pass --latency-csv with columns: system,latency_ms).
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Academic palette
TWO = "#E63946"; FULL = "#13294B"; SVSF = "#457B9D"; VAN = "#A8A8A8"; CTX = "#2A9D8F"; INK = "#1A2433"

SYSTEMS = ["Vanilla", "SVSF", "SVSFctx", "TwoStage", "FullRerank"]
LABELS = {"Vanilla": "Vanilla", "SVSF": "SVSF", "SVSFctx": "SVSF+ctx",
          "TwoStage": "TwoStage", "FullRerank": "FullRerank"}
COLORS = {"Vanilla": VAN, "SVSF": SVSF, "SVSFctx": CTX, "TwoStage": TWO, "FullRerank": FULL}

# Default mega-pool latency (ms) — replace with your measured values or --latency-csv
LATENCY = {"SVSF gather": 135.9, "TwoStage (8)": 200.8,
           "FullRerank (50)": 303.2, "FullRerank (200)": 948.6}


def fig_recall_by_dataset(df, outdir):
    """Grouped bar chart: recall per system, per dataset + combined."""
    if "Source" not in df.columns:
        print("[skip] recall-by-dataset: no Source column"); return
    datasets = sorted(df["Source"].dropna().unique().tolist())
    groups = datasets + ["Combined"]
    data = {s: [] for s in SYSTEMS}
    for g in groups:
        sub = df if g == "Combined" else df[df["Source"] == g]
        for s in SYSTEMS:
            col = f"{s}_Recall"
            data[s].append(sub[col].dropna().mean() if col in sub else np.nan)

    x = np.arange(len(groups)); w = 0.16
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, s in enumerate(SYSTEMS):
        ax.bar(x + (i - 2) * w, data[s], w, label=LABELS[s], color=COLORS[s],
               zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Recall@4", fontsize=13, fontweight="bold")
    ax.set_title("Candidate-Set Recall across Benchmarks", fontsize=14, fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
    ax.legend(fontsize=10, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", alpha=0.25, zorder=0); ax.set_ylim(0, 1.08)
    plt.tight_layout()
    p = os.path.join(outdir, "fig_recall_by_dataset.png")
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    print("[ok]", p)


def fig_latency_scissors(latency, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(latency.keys()); vals = list(latency.values())
    cols = [SVSF, TWO, FULL, "#6B7A99"][:len(names)]
    bars = ax.bar(names, vals, color=cols, zorder=3, width=0.6, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Retrieval latency (ms)", fontsize=13, fontweight="bold")
    ax.set_title("Mega-Pool Latency (Scalability Scissors)", fontsize=14, fontweight="bold", color=INK)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.015, f"{v:.0f}",
                ha="center", fontweight="bold", fontsize=11)
    plt.tight_layout()
    p = os.path.join(outdir, "fig_latency_scissors.png")
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    print("[ok]", p)


def fig_twostage_vs_full(df, outdir, k_gather=8, k_full=50):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax1.bar(["TwoStage", "FullRerank"], [k_gather, k_full], color=[TWO, FULL],
            zorder=3, width=0.55, edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("Candidates reranked", fontsize=12, fontweight="bold")
    ax1.set_title("Reranker workload", fontsize=13, fontweight="bold", color=INK)
    ax1.grid(axis="y", alpha=0.25, zorder=0)
    for i, v in enumerate([k_gather, k_full]):
        ax1.text(i, v + 1, str(v), ha="center", fontweight="bold", fontsize=13)

    ts = df["TwoStage_Recall"].dropna().mean() if "TwoStage_Recall" in df else np.nan
    fr = df["FullRerank_Recall"].dropna().mean() if "FullRerank_Recall" in df else np.nan
    ax2.bar(["TwoStage", "FullRerank"], [ts, fr], color=[TWO, FULL],
            zorder=3, width=0.55, edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("Recall@4", fontsize=12, fontweight="bold")
    ax2.set_title("Resulting recall", fontsize=13, fontweight="bold", color=INK)
    ax2.set_ylim(0, 1.0); ax2.grid(axis="y", alpha=0.25, zorder=0)
    for i, v in enumerate([ts, fr]):
        if not np.isnan(v):
            ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=13)
    plt.tight_layout()
    p = os.path.join(outdir, "fig_twostage_vs_full.png")
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    print("[ok]", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="per-question results CSV")
    ap.add_argument("--latency-csv", help="optional CSV with columns: system,latency_ms")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    latency = LATENCY
    if args.latency_csv:
        ldf = pd.read_csv(args.latency_csv)
        latency = dict(zip(ldf["system"], ldf["latency_ms"]))
    fig_latency_scissors(latency, args.outdir)

    if args.csv:
        df = pd.read_csv(args.csv)
        fig_recall_by_dataset(df, args.outdir)
        fig_twostage_vs_full(df, args.outdir)
    else:
        print("[info] no --csv given; only latency figure generated")


if __name__ == "__main__":
    main()
