# PAPER: §5.7 / Table 6 — retrieval-level recall, main 1,500-question configuration
"""
SCRIPT C — ANA 1,500-SORULUK KURULUMDA RETRIEVAL-LEVEL RECALL
Sira (COLAB): DATASETS -> TWO STAGES -> FAST NODES -> BU
  (NOT: yerelde app_v11.py ile KOSMAZ; gold_passages yalnizca Colab'daki
   TWO STAGES surumunde uretiliyor. app_v11.py'de bu alan yok.)

NEDEN BU SCRIPT VAR (hakem itirazi):
  "§5.7 retrieval-level recall degerli; fakat 150 soru/dataset uzerinde ve
   per-question corpora ile yapilmis. Ana 1,500-question setup uzerinde neden
   evidence recall yok?"
Cevap: bu kosu, MAKALENIN ANA DEGERLENDIRME KURULUMUNDA ayni metrikleri uretir
  -> §5.7 "diagnostic" olmaktan cikip ana kurulumda da desteklenmis olur.

TASARIM — makalenin ana kurulumuyla BIREBIR ayni:
  * build_mixed_validation_set(n_sample=500, sources=[HotpotQA, 2Wiki, MuSiQue])
    -> 3 x 500 = 1,500 soru (makalenin ana deneyiyle ayni cagri)
  * Ayni chunking: semantic, thr=SEM_THR (Paper 1 varsayilani 0.3)
  * gold isaretleme: mark_gold_nodes(nodes, gold_passages, overlap_threshold=0.6)
    -> bir gold birden cok chunk'a bolunse bile dogru eslesir
  * n_docs = 4 (makalenin calisma noktasi) + siki butceler k=2,3
  * Sistemler: Vanilla, MMR, MiniLM, BGE, SVSF + PRF-Rocchio/iter/seq + SVSF-noQ0
    (PRF'ler C2 talebini ana kurulumda da karsilar)

METRIKLER (§5.7 ile ayni tanimlar):
  SF-Recall@k       : gold chunk'larin kaci top-k'da
  Full-chain@k      : sorunun TUM gold'lari top-k'da mi (ikili)
  (Bridge@k YOK: hop sirasi yalnizca MuSiQue'de mevcut; ana kurulum uc veri
   setinin karisimi oldugu icin burada raporlanmaz. §5.7'deki gerekce gecerli.)

CIKTI: ekrana tablo + retrieval_recall_1500.csv (soru bazli, yeniden analiz icin)
CHECKPOINT: her 25 soruda CSV'ye yazar; kesilirse --resume ile devam eder.
"""
import os, csv, time
import numpy as np
from scipy.stats import wilcoxon

N_PER_SOURCE = 500                 # 3 x 500 = 1500 (makalenin ana kurulumu)
SOURCES      = ["HotpotQA", "2WikiMultihopQA", "MuSiQue"]
KS           = (2, 3, 4)
KMAX         = max(KS)
SEM_THR      = 0.3
ALPHA, BETA  = 1.0, 0.6
PRF_M, PRF_A, PRF_B, PRF_ITERS = 3, 1.0, 0.5, 3
OUT          = "retrieval_recall_1500.csv"
SEED         = 20260826

FIELDS = ["qi", "source", "system"] + [f"sf@{k}" for k in KS] + [f"fc@{k}" for k in KS]


def _n(x):
    return x / (np.linalg.norm(x) + 1e-9)


# ── retriever'lar: indeks listesi dondurur (iz normalize; asimetri yok) ──
def rank_vanilla(qv, U, k):
    return list(np.argsort(U @ _n(qv))[::-1][:k])

def rank_prf_rocchio(qv, U, k, m=PRF_M, a=PRF_A, b=PRF_B):
    qn = _n(qv); top = np.argsort(U @ qn)[::-1][:m]
    return list(np.argsort(U @ _n(a*qn + b*_n(U[top].mean(axis=0))))[::-1][:k])

def rank_prf_iter(qv, U, k, m=PRF_M, a=PRF_A, b=PRF_B, it=PRF_ITERS):
    qn = _n(qv); q = qn
    for _ in range(it):
        top = np.argsort(U @ q)[::-1][:m]
        q = _n(a*qn + b*_n(U[top].mean(axis=0)))
    return list(np.argsort(U @ q)[::-1][:k])

def rank_prf_seq(qv, U, k, a=PRF_A, b=PRF_B):
    qn = _n(qv); acc = np.zeros_like(qn); c = 0; out = []; taken = set()
    for _ in range(k):
        q = _n(a*qn + b*_n(acc/c)) if c else qn
        s = U @ q
        for i in taken: s[i] = -1e9
        i = int(np.argmax(s)); taken.add(i); out.append(i)
        acc = acc + U[i]; c += 1
    return out

def rank_svsf_generic(qv, U, k, q0_in_trail=True, alpha=ALPHA, beta=BETA):
    qn = _n(qv); sq = U @ qn
    acc = np.zeros_like(qn); c = 0; out = []; taken = set()
    for _ in range(k):
        s = alpha*sq + (beta*(U @ _n(acc/c)) if c else 0.0)
        s = np.array(s, dtype=float) if not np.isscalar(s) else alpha*sq.copy()
        for i in taken: s[i] = -1e9
        i = int(np.argmax(s)); taken.add(i); out.append(i)
        acc = acc + ((qn + U[i])*0.5 if q0_in_trail else U[i]); c += 1
    return out


def main():
    print(f"[1/3] {N_PER_SOURCE} soru/kaynak x {len(SOURCES)} kaynak yukleniyor...")
    import random as _r
    _r.seed(SEED)
    data = build_mixed_validation_set(n_sample=N_PER_SOURCE, sources=SOURCES)
    print(f"      toplam {len(data)} soru")
    have_gold = sum(1 for d in data if d.get("gold_passages"))
    print(f"      gold_passages olan: {have_gold}/{len(data)}")
    if have_gold == 0:
        print("!!! gold_passages yok. Colab'daki TWO STAGES surumunu kullandiginizdan")
        print("    emin olun (app_v11.py'de bu alan bulunmuyor).")
        return

    refiner_mini = globals().get("cross_encoder")
    refiner_bge = globals().get("cross_encoder_bge")

    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((int(row["qi"]), row["system"]))
        print(f"[2/3] checkpoint: {len(done)} kayit var, atlanacak")
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="", encoding="utf-8"); w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new: w.writeheader()

    print("[3/3] kosu basliyor...\n")
    t0 = time.time(); used = 0; skipped = 0
    for qi, item in enumerate(data):
        gp = item.get("gold_passages") or []
        if not gp:
            skipped += 1; continue
        try:
            nodes = [ChunkNode(f"P_{i}", t) for i, t in
                     enumerate(semantic_chunking(item["corpus"], similarity_threshold=SEM_THR))]
            if len(nodes) <= KMAX:
                skipped += 1; continue
            mark_gold_nodes(nodes, gp, overlap_threshold=0.6)
            gold = {i for i, n in enumerate(nodes) if getattr(n, "is_gold", False)}
            if not gold:
                skipped += 1; continue
            embs = np.vstack([n.text_embedding for n in nodes])
            U = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
            qv = model.encode("query: " + item["query"])

            ranked = {
                "Vanilla":      rank_vanilla(qv, U, KMAX),
                "PRF-Rocchio":  rank_prf_rocchio(qv, U, KMAX),
                "PRF-iter":     rank_prf_iter(qv, U, KMAX),
                "PRF-seq":      rank_prf_seq(qv, U, KMAX),
                "SVSF (pub.)":  rank_svsf_generic(qv, U, KMAX, True),
                "SVSF-noQ0":    rank_svsf_generic(qv, U, KMAX, False),
            }
            ranked["MMR"] = [nodes.index(x) for x in mmr_search_step(qv, embs, nodes, KMAX)]
            if refiner_mini:
                ranked["Rerank-MiniLM"] = [nodes.index(x) for x in
                    rerank_search_step(qv, item["query"], embs, nodes, KMAX, refiner_mini)]
            if refiner_bge:
                ranked["Rerank-BGE"] = [nodes.index(x) for x in
                    rerank_search_step(qv, item["query"], embs, nodes, KMAX, refiner_bge)]

            for sysname, r in ranked.items():
                if (qi, sysname) in done:
                    continue
                row = {"qi": qi, "source": item.get("source", "?"), "system": sysname}
                for k in KS:
                    sel = set(r[:k])
                    row[f"sf@{k}"] = len(sel & gold) / len(gold)
                    row[f"fc@{k}"] = 1.0 if gold.issubset(sel) else 0.0
                w.writerow(row)
            used += 1
            if used % 25 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  {qi+1}/{len(data)} (islenen {used})  {el/60:.1f} dk  |  "
                      f"tahmini kalan {el/used*(len(data)-qi-1)/60:.1f} dk")
        except KeyboardInterrupt:
            print("\nDurduruldu; tekrar calistirinca devam eder."); break
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  !! soru {qi}: {str(e)[:120]}")
    fh.close()
    print(f"\nislenen {used} soru | atlanan {skipped}")
    report()


def report():
    import pandas as pd
    if not os.path.exists(OUT):
        print("Sonuc dosyasi yok."); return
    df = pd.read_csv(OUT)
    nq = df.qi.nunique()
    print(f"\n{'='*92}\nANA KURULUM RETRIEVAL RECALL — {nq} soru "
          f"({', '.join(f'{s}:{df[df.system==df.system.iloc[0]].source.eq(s).sum()}' for s in df.source.unique())})\n{'='*92}")
    order = ["Vanilla", "MMR", "Rerank-MiniLM", "Rerank-BGE",
             "PRF-Rocchio", "PRF-iter", "PRF-seq", "SVSF (pub.)", "SVSF-noQ0"]
    systems = [s for s in order if s in set(df.system)]

    for pref, lbl in (("sf", "SF-Recall"), ("fc", "Full-chain recall")):
        print(f"\n  ── {lbl} (tum kaynaklar) ──")
        print(f"  {'Method':<15}" + "".join(f"@{k}{'':<8}" for k in KS))
        for s in systems:
            sub = df[df.system == s]
            print(f"  {s:<15}" + "".join(f"{sub[f'{pref}@{k}'].mean():<10.3f}" for k in KS))
        # anlamlilik: SVSF (pub.) referans, her k icin
        if "SVSF (pub.)" in systems:
            for k in KS:
                base = df[df.system == "SVSF (pub.)"].set_index("qi")[f"{pref}@{k}"]
                print(f"    -- @{k}, SVSF(pub.) referans --")
                for s in systems:
                    if s == "SVSF (pub.)": continue
                    o = df[df.system == s].set_index("qi")[f"{pref}@{k}"]
                    idx = base.index.intersection(o.index)
                    d = (o.loc[idx] - base.loc[idx]).values
                    try:
                        _, p = wilcoxon(d) if np.any(d != 0) else (0, 1.0)
                    except Exception:
                        p = 1.0
                    print(f"      {s:<15}{d.mean():+.3f}  p={p:.4f} "
                          f"{'ANLAMLI' if p < 0.05 else 'fark yok'}")

    print(f"\n  ── Kaynak bazli SF-Recall@4 ──")
    print(f"  {'Method':<15}" + "".join(f"{s[:12]:<14}" for s in df.source.unique()))
    for s in systems:
        row = ""
        for src in df.source.unique():
            sub = df[(df.system == s) & (df.source == src)]
            row += f"{sub['sf@4'].mean():<14.3f}" if len(sub) else f"{'-':<14}"
        print(f"  {s:<15}{row}")


try:
    build_mixed_validation_set; ChunkNode; semantic_chunking; mark_gold_nodes; model
except NameError as e:
    print(f"!!! Gerekli fonksiyon yok ({e}). TWO STAGES ve FAST NODES hucrelerini calistirin.")
else:
    main()
