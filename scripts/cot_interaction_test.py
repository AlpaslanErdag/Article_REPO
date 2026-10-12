# PAPER: §5.9 — CoT x retriever interaction test (DiD + Holm)
#!/usr/bin/env python3
"""
CoT x RETRIEVER ETKILESIM TESTI (difference-in-differences)
===========================================================
NEDEN BU TEST GEREKLI:
  §5.9 su an sunu soyluyor: "SVSF'te CoT dususu anlamli (p=0.0008), BGE'de anlamli
  degil (p=1.00), dolayisiyla CoT'un faydasi retrieval kalitesiyle ters orantili."
  Bu cikarim ISTATISTIKSEL OLARAK GECERSIZ. Klasik hata:
  **anlamli ile anlamsiz arasindaki fark, otomatik olarak anlamli degildir.**

  Dogru test, dususlerin BIRBIRINDEN farkli olup olmadigini soran bir etkilesim testi:

      DiD(A,B) = (A_cot - A_nocot) - (B_cot - B_nocot)

  Bu, soru duzeyinde eslesmis veriyle bootstrap edilebilir (ayni sorular her iki
  kosulda ve her iki sistemde gorulmus).

GIRDI: cot_ablation_v2.csv  (scriptB_local_cot.py ciktisi)
       kolonlar: qidx, cond(cot|nocot), system, strict, flex, judge, notfound, ms

CIKTI:
  1. Her sistem icin CoT dususu + eslesmis bootstrap %95 GA
  2. Tum sistem ciftleri icin DiD + bootstrap GA + p (iki yonlu)
  3. Holm-Bonferroni duzeltmesi (DiD ailesi icin)
  4. BGE'yi referans alan asil test: "BGE'nin dususu digerlerinden anlamli az mi?"
  5. Retrieval kalitesi ile CoT dususu arasindaki iliski (Spearman)

KULLANIM: python cot_interaction_test.py [--metric flex|strict|judge] [--boot 10000]
"""
import argparse
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IN = "cot_ablation_v2.csv"
# §5.9'da raporlanan CoT-ACIK Flex EM degerleri (retrieval kalitesi vekili olarak)
COT_ON_QUALITY = {"RerankBGE": 59.0, "MultiQuery": 55.5, "Rerank": 52.5,
                  "SVSF": 51.5, "Vanilla": 51.0, "MMR": 39.0}


def load(metric):
    try:
        df = pd.read_csv(IN)
    except FileNotFoundError:
        sys.exit(f"HATA: {IN} bulunamadi. scriptB_local_cot.py ciktisini bu klasore koy.")
    need = {"qidx", "cond", "system", metric}
    if not need.issubset(df.columns):
        sys.exit(f"HATA: eksik kolon. gereken={need}, mevcut={set(df.columns)}")
    # sistem x kosul -> soru-indeksli seri
    piv = {}
    for s in sorted(df.system.unique()):
        on = df[(df.system == s) & (df.cond == "cot")].set_index("qidx")[metric]
        off = df[(df.system == s) & (df.cond == "nocot")].set_index("qidx")[metric]
        idx = on.index.intersection(off.index)
        if len(idx) == 0:
            continue
        piv[s] = (on.loc[idx].astype(float), off.loc[idx].astype(float))
    return piv


def paired_boot(vals, nboot, rng):
    """Eslesmis bootstrap: soru duzeyinde yeniden ornekleme."""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    means = np.empty(nboot)
    for b in range(nboot):
        means[b] = v[rng.integers(0, n, n)].mean()
    return means


def boot_p(means):
    """Iki yonlu bootstrap p: sifirin dagilimin neresinde oldugu."""
    below = float((means <= 0).mean())
    return float(min(1.0, 2 * min(below, 1 - below)))


def holm(pairs):
    """pairs: [(etiket, p), ...] -> [(etiket, p, p_adj, karar)]"""
    order = sorted(pairs, key=lambda x: x[1])
    m = len(order)
    out, running = [], 0.0
    for i, (lab, p) in enumerate(order):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out.append((lab, p, adj, "ANLAMLI" if adj < 0.05 else "fark yok"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="flex", choices=["flex", "strict", "judge"])
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(20260901)

    piv = load(a.metric)
    systems = list(piv.keys())
    n_q = len(next(iter(piv.values()))[0])
    print(f"{'='*86}\nCoT x RETRIEVER ETKILESIM TESTI  |  metrik={a.metric}  |  "
          f"n={n_q} soru  |  bootstrap={a.boot}\n{'='*86}")

    # ── 1) Sistem basina CoT dususu (eslesmis) ──
    print("\n[1] CoT dususu (CoT_acik - CoT_kapali), eslesmis bootstrap %95 GA")
    print(f"  {'Sistem':<14}{'CoT ON':<9}{'CoT OFF':<10}{'dusus':<9}{'%95 GA':<22}{'p'}")
    drops = {}
    for s in systems:
        on, off = piv[s]
        d = (on - off).values
        means = paired_boot(d, a.boot, rng)
        lo, hi = np.percentile(means, [2.5, 97.5])
        p = boot_p(means)
        drops[s] = d
        print(f"  {s:<14}{100*on.mean():<9.1f}{100*off.mean():<10.1f}"
              f"{100*d.mean():<9.1f}[{100*lo:+.1f}, {100*hi:+.1f}]{'':<7}{p:.4f}")

    # ── 2) Tum ciftler icin DiD ──
    print(f"\n[2] ETKILESIM (DiD): dususler birbirinden farkli mi?")
    print(f"    DiD(A,B) = (A_cot - A_nocot) - (B_cot - B_nocot);  pozitif = A daha cok kaybediyor")
    print(f"  {'A vs B':<28}{'DiD':<9}{'%95 GA':<24}{'p'}")
    did_ps = []
    for A, B in itertools.combinations(systems, 2):
        d = drops[A] - drops[B]          # ayni sorular -> eslesmis
        means = paired_boot(d, a.boot, rng)
        lo, hi = np.percentile(means, [2.5, 97.5])
        p = boot_p(means)
        did_ps.append((f"{A} vs {B}", p))
        star = " *" if p < 0.05 else ""
        print(f"  {A+' vs '+B:<28}{100*d.mean():<+9.1f}[{100*lo:+.1f}, {100*hi:+.1f}]{'':<9}{p:.4f}{star}")

    # ── 3) Holm duzeltmesi ──
    print(f"\n[3] Holm-Bonferroni duzeltmesi ({len(did_ps)} DiD karsilastirmasi)")
    print(f"  {'Karsilastirma':<28}{'ham p':<10}{'Holm p':<10}{'karar'}")
    for lab, p, adj, dec in holm(did_ps):
        print(f"  {lab:<28}{p:<10.4f}{adj:<10.4f}{dec}")

    # ── 4) ASIL TEST: BGE referansli ──
    ref = "RerankBGE"
    if ref in systems:
        print(f"\n[4] ASIL TEST — {ref} (en guclu retriever) referans")
        print(f"    Hipotez: guclu retriever'da CoT dususu digerlerinden ANLAMLI DAHA AZ")
        fam = []
        for s in systems:
            if s == ref:
                continue
            d = drops[s] - drops[ref]
            means = paired_boot(d, a.boot, rng)
            lo, hi = np.percentile(means, [2.5, 97.5])
            fam.append((f"{s} - {ref}", boot_p(means), 100*d.mean(), 100*lo, 100*hi))
        print(f"  {'Sistem - BGE':<24}{'fazla dusus':<14}{'%95 GA':<24}{'ham p':<10}{'Holm p':<10}{'karar'}")
        hp = {l: (adj, dec) for l, _, adj, dec in holm([(f[0], f[1]) for f in fam])}
        for lab, p, m, lo, hi in fam:
            adj, dec = hp[lab]
            print(f"  {lab:<24}{m:<+14.1f}[{lo:+.1f}, {hi:+.1f}]{'':<9}{p:<10.4f}{adj:<10.4f}{dec}")
        n_sig = sum(1 for lab, _, _, _, _ in fam if hp[lab][1] == "ANLAMLI")
        print(f"\n  -> {n_sig}/{len(fam)} sistemde dusus BGE'den anlamli olarak fazla.")
        if n_sig == 0:
            print("     UYARI: hicbiri anlamli degilse '§5.9 iddiasi ETKILESIM DUZEYINDE")
            print("     DESTEKLENMIYOR' demek zorundayiz; bulguyu betimsel olarak yazariz.")

    # ── 5) Retrieval kalitesi ile dusus iliskisi ──
    q = [(COT_ON_QUALITY[s], 100*drops[s].mean()) for s in systems if s in COT_ON_QUALITY]
    if len(q) >= 4:
        x, y = zip(*q)
        rho, pv = spearmanr(x, y)
        print(f"\n[5] Retrieval kalitesi (CoT-acik Flex EM) ile CoT dususu iliskisi")
        print(f"    Spearman rho = {rho:+.3f}, p = {pv:.4f}  (n={len(q)} sistem)")
        print(f"    Negatif rho = 'kalite yuksekse dusus az' (§5.9'un iddiasi)")
        print(f"    NOT: n={len(q)} sistem cok kucuk; bu betimsel bir egilim, kanit degil.")

    print(f"\n{'#'*86}")
    print("MAKALEYE NASIL YAZILACAK")
    print("  * [4]'te anlamli cikanlar varsa: '§5.9'un iddiasi etkilesim testiyle dogrulandi'")
    print("    ve hangi ciftlerde dogrulandigi acikca yazilir.")
    print("  * Hicbiri anlamli degilse: iddia BETIMSEL hale getirilir \u2014")
    print("    'CoT dususu en guclu retriever'da nominal olarak en kucuktur, ancak")
    print("     dususler arasindaki fark bu ornek buyuklugunde anlamli degildir.'")
    print("  * Her iki durumda da [5]'in n=6 sistemle betimsel oldugu belirtilir.")


if __name__ == "__main__":
    main()
