# PAPER: supplementary — superseded by §5.12 (see README)
"""
ESIT-BUTCELI CANDIDATE-GENERATION KARSILASTIRMASI  (Paper 1, §8)
=================================================================
Sira (COLAB): DATASETS -> TWO STAGES -> FAST NODES -> BU
  (gold_passages yalnizca Colab'daki TWO STAGES surumunde uretiliyor.)

NEDEN BU KOSU:
  §8, sequential dense feedback'in "ucuz aday ureticisi" olarak da avantaj
  saglamadigini iddia ediyor. Bu iddia su an nitel; hakem hakli olarak nicel kanit
  istiyor. Bu script o kaniti uretir.

KRITIK TASARIM KARARLARI (hepsi bilincli):

  1. ESIT ADAY BUTCESI. Iki gatherer de k_gather=8 aday toplar, ayni reranker
     ayni sayida adayi skorlar. Bu olmadan karsilastirma anlamsiz \u2014 Paper 2'nin
     hatasi tam buydu (SVSF gather-8 vs dense top-50 kiyaslanmisti).

  2. GATHERER = PAPER 1'IN SVSF'i (svsf_search_step), SVSF-P2 DEGIL. §8'in iddiasi
     bu makalenin yontemi hakkinda; entity-novelty kanalli varyant baska bir calismanin
     konusu.

  3. OZEL RERANK FONKSIYONU. app_v11.rerank_search_step icerideki dense top-50'yi
     sabit aliyor ve verilen aday kumesini siralayamiyor. Esit-butce icin
     rerank_given() yazildi: verilen adaylari skorlar, top-n dondurur.

  4. ADAY-KUMESI RECALL AYRI RAPORLANIR. Asil mekanistik soru su: SVSF'in topladigi
     8 aday, dense top-8'de OLMAYAN gold iceriyor mu? Bu, yeniden siralamadan ONCE
     olculur; rerank sonrasi fark bunun turevi.

  5. TAVAN REFERANSI. FullRerank(50) da kosulur \u2014 esit butce olmayan ama ust sinir
     gosteren referans olarak.

CIKTI: ekrana tablo + candgen_equal_budget.csv (soru bazli)
CHECKPOINT: her 25 soruda yazar; kesilirse kaldigi yerden devam eder.
"""
import os, csv, time
import numpy as np
from scipy.stats import wilcoxon

N_PER_SOURCE = 250                  # 3 x 250 = 750 (Paper 2 ile ayni)
SOURCES      = ["HotpotQA", "2WikiMultihopQA", "MuSiQue"]
K_GATHER     = 8                    # esit aday butcesi
K_FULL       = 50                   # tavan referansi
N_OUT        = 4                    # reader'a giden pasaj sayisi (Paper 1 varsayilani)
SEM_THR      = 0.3
ALPHA, BETA, TAU = 1.0, 0.6, 0.2
OUT          = "candgen_equal_budget.csv"
SEED         = 20260901

FIELDS = ["qi","source","n_gold","n_chunks",
          "cand_dense8","cand_svsf8","cand_full50",
          "novel_svsf","novel_dense",
          "fin_dense8","fin_svsf8","fin_full50",
          "ms_gather_dense","ms_gather_svsf","ms_rr8","ms_rr50"]


def _n(x): return x / (np.linalg.norm(x) + 1e-9)


def rerank_given(cand_idx, query_text, nodes, cross_model, n_out):
    """Verilen aday kumesini skorlar ve top-n dondurur (app_v11'inki sabit 50 aliyor)."""
    if len(cand_idx) <= n_out:
        return list(cand_idx)
    pairs = [[query_text, nodes[i].text] for i in cand_idx]
    scores = cross_model.predict(pairs)
    order = np.argsort(scores)[::-1][:n_out]
    return [cand_idx[i] for i in order]


def main():
    import random as _r
    _r.seed(SEED)
    print(f"[1/3] {N_PER_SOURCE} soru/kaynak x {len(SOURCES)} = "
          f"{N_PER_SOURCE*len(SOURCES)} soru yukleniyor...")
    data = build_mixed_validation_set(n_sample=N_PER_SOURCE, sources=SOURCES)
    have = sum(1 for d in data if d.get("gold_passages"))
    print(f"      toplam {len(data)} | gold_passages olan {have}")
    if have == 0:
        print("!!! gold_passages yok \u2014 Colab'daki TWO STAGES surumunu kullanin.")
        return
    bge = globals().get("cross_encoder_bge")
    if bge is None:
        print("!!! cross_encoder_bge yok."); return

    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8") as f:
            done = {int(r["qi"]) for r in csv.DictReader(f)}
        print(f"[2/3] checkpoint: {len(done)} soru atlanacak")
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="", encoding="utf-8"); w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new: w.writeheader()

    print("[3/3] kosu basliyor...\n")
    t0 = time.time(); used = 0; skipped = 0
    from rank_bm25 import BM25Okapi
    import torch
    cuda = torch.cuda.is_available()

    for qi, item in enumerate(data):
        if qi in done: continue
        gp = item.get("gold_passages") or []
        if not gp: skipped += 1; continue
        try:
            nodes = [ChunkNode(f"P_{i}", t) for i, t in
                     enumerate(semantic_chunking(item["corpus"], similarity_threshold=SEM_THR))]
            if len(nodes) <= K_GATHER: skipped += 1; continue
            mark_gold_nodes(nodes, gp, overlap_threshold=0.6)
            gold = {i for i, n in enumerate(nodes) if getattr(n, "is_gold", False)}
            if not gold: skipped += 1; continue
            embs = np.vstack([n.text_embedding for n in nodes])
            qv = model.encode("query: " + item["query"])
            unit = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
            idx_of = {id(n): i for i, n in enumerate(nodes)}

            # ---- GATHER: dense top-8 ----
            if cuda: torch.cuda.synchronize()
            t = time.perf_counter()
            dense_rank = list(np.argsort(unit @ _n(qv))[::-1])
            d8 = [int(i) for i in dense_rank[:K_GATHER]]
            if cuda: torch.cuda.synchronize()
            ms_gd = (time.perf_counter()-t)*1000

            # ---- GATHER: SVSF-8 (Paper 1'in yontemi) ----
            raw = BM25Okapi([n.text.lower().split() for n in nodes]).get_scores(
                    item["query"].lower().split())
            mx, mn = float(raw.max()), float(raw.min())
            bm = (raw - mn) / (mx - mn if mx > mn else 1.0)
            if cuda: torch.cuda.synchronize()
            t = time.perf_counter()
            sv = svsf_search_step(qv, embs, nodes, bm, ALPHA, BETA, K_GATHER, TAU, False, 1.0)
            sv = sv[0] if isinstance(sv, tuple) else sv
            s8 = [idx_of[id(x)] for x in sv]
            if cuda: torch.cuda.synchronize()
            ms_gs = (time.perf_counter()-t)*1000

            f50 = [int(i) for i in dense_rank[:K_FULL]]

            # ---- ASSERT: esit butce ----
            assert len(d8) == len(s8) == K_GATHER, f"butce esit degil: {len(d8)} vs {len(s8)}"

            # ---- RERANK (ayni model, ayni cikti boyutu) ----
            if cuda: torch.cuda.synchronize()
            t = time.perf_counter(); fd = rerank_given(d8, item["query"], nodes, bge, N_OUT)
            fs = rerank_given(s8, item["query"], nodes, bge, N_OUT)
            if cuda: torch.cuda.synchronize()
            ms_rr8 = (time.perf_counter()-t)*1000/2
            if cuda: torch.cuda.synchronize()
            t = time.perf_counter(); ff = rerank_given(f50, item["query"], nodes, bge, N_OUT)
            if cuda: torch.cuda.synchronize()
            ms_rr50 = (time.perf_counter()-t)*1000

            G = gold; ng = len(G)
            row = dict(qi=qi, source=item.get("source","?"), n_gold=ng, n_chunks=len(nodes),
                       cand_dense8=len(G & set(d8))/ng,
                       cand_svsf8=len(G & set(s8))/ng,
                       cand_full50=len(G & set(f50))/ng,
                       novel_svsf=len(G & (set(s8)-set(d8))),   # SVSF'in getirdigi, dense'te yok
                       novel_dense=len(G & (set(d8)-set(s8))),  # dense'in getirdigi, SVSF'te yok
                       fin_dense8=len(G & set(fd))/ng,
                       fin_svsf8=len(G & set(fs))/ng,
                       fin_full50=len(G & set(ff))/ng,
                       ms_gather_dense=round(ms_gd,3), ms_gather_svsf=round(ms_gs,3),
                       ms_rr8=round(ms_rr8,3), ms_rr50=round(ms_rr50,3))
            w.writerow(row); used += 1
            if used % 25 == 0:
                fh.flush(); el=time.time()-t0
                print(f"  {qi+1}/{len(data)} (islenen {used})  {el/60:.1f} dk  |  "
                      f"kalan ~{el/used*(len(data)-qi-1)/60:.1f} dk")
        except KeyboardInterrupt:
            print("\nDurduruldu; tekrar calistirinca devam eder."); break
        except Exception as e:
            skipped += 1
            if skipped <= 5: print(f"  !! soru {qi}: {str(e)[:110]}")
    fh.close()
    print(f"\nislenen {used} | atlanan {skipped}")
    report()


def boot(d, nb=10000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    d = np.asarray(d, float); n = len(d)
    m = np.array([d[rng.integers(0,n,n)].mean() for _ in range(nb)])
    lo, hi = np.percentile(m,[2.5,97.5])
    below = float((m<=0).mean())
    return d.mean(), lo, hi, float(min(1.0, 2*min(below,1-below)))


def report():
    import pandas as pd
    if not os.path.exists(OUT): print("Sonuc dosyasi yok."); return
    df = pd.read_csv(OUT)
    print(f"\n{'='*88}\nESIT-BUTCELI CANDIDATE GENERATION  |  n={len(df)} soru  |  "
          f"k_gather={K_GATHER}, n_out={N_OUT}\n{'='*88}")
    print(f"  kaynak dagilimi: {df.source.value_counts().to_dict()}")
    print(f"  ort. gold/soru: {df.n_gold.mean():.2f} | ort. chunk/soru: {df.n_chunks.mean():.1f}\n")

    print(f"  {'Gatherer':<22}{'Butce':<8}{'Aday recall':<14}{'Rerank sonrasi':<17}{'Gather ms':<12}{'Rerank ms':<11}")
    rows=[("Dense top-8","8","cand_dense8","fin_dense8","ms_gather_dense","ms_rr8"),
          ("SVSF gather-8","8","cand_svsf8","fin_svsf8","ms_gather_svsf","ms_rr8"),
          ("Dense top-50 (tavan)","50","cand_full50","fin_full50","ms_gather_dense","ms_rr50")]
    for nm,b,c,f,gm,rm in rows:
        print(f"  {nm:<22}{b:<8}{df[c].mean():<14.3f}{df[f].mean():<17.3f}"
              f"{df[gm].median():<12.2f}{df[rm].median():<11.2f}")

    print(f"\n  --- ESIT BUTCEDE ESLI KARSILASTIRMA (SVSF-8 eksi Dense-8) ---")
    for lbl,a,b in [("Aday-kumesi recall","cand_svsf8","cand_dense8"),
                    ("Rerank sonrasi recall","fin_svsf8","fin_dense8")]:
        d=(df[a]-df[b]).values
        m,lo,hi,p=boot(d)
        try: _,pw = wilcoxon(d) if np.any(d!=0) else (0,1.0)
        except Exception: pw=1.0
        print(f"  {lbl:<24}\u0394 = {m:+.4f}  %95 GA [{lo:+.4f}, {hi:+.4f}]  "
              f"bootstrap p={p:.4f}  Wilcoxon p={pw:.4f}  "
              f"{'ANLAMLI' if min(p,pw)<0.05 else 'fark yok'}")

    print(f"\n  --- MEKANIZMA: gatherer birbirinin kacirdigini buluyor mu? ---")
    ns, nd = df.novel_svsf.sum(), df.novel_dense.sum()
    qs = (df.novel_svsf>0).sum(); qd = (df.novel_dense>0).sum()
    print(f"    SVSF-8'de olup dense-8'de olmayan gold : {ns} (toplam), {qs} soruda (%{100*qs/len(df):.1f})")
    print(f"    dense-8'de olup SVSF-8'de olmayan gold : {nd} (toplam), {qd} soruda (%{100*qd/len(df):.1f})")
    print(f"    net: {ns-nd:+d} gold")

    print(f"\n  --- GECIKME ---")
    tot_d = df.ms_gather_dense.median()+df.ms_rr8.median()
    tot_s = df.ms_gather_svsf.median()+df.ms_rr8.median()
    tot_f = df.ms_gather_dense.median()+df.ms_rr50.median()
    print(f"    Dense-8  -> rerank : {tot_d:.2f} ms")
    print(f"    SVSF-8   -> rerank : {tot_s:.2f} ms  ({tot_s-tot_d:+.2f} ms)")
    print(f"    Dense-50 -> rerank : {tot_f:.2f} ms")

    print(f"\n  --- KAYNAK BAZLI aday recall ---")
    print(f"  {'Kaynak':<20}{'Dense-8':<12}{'SVSF-8':<12}{'\u0394'}")
    for src in df.source.unique():
        sub=df[df.source==src]
        print(f"  {src:<20}{sub.cand_dense8.mean():<12.3f}{sub.cand_svsf8.mean():<12.3f}"
              f"{sub.cand_svsf8.mean()-sub.cand_dense8.mean():+.3f}")

    print(f"\n{'#'*88}")
    print("MAKALEYE (§8): 'Aday-kumesi recall' satiri asil kanittir \u2014 esit butcede SVSF'in")
    print("  topladigi kume, dense top-k'da olmayan gold iceriyor mu? Fark anlamsizsa,")
    print("  gatherer'in ek bir sey getirmedigi nicel olarak gosterilmis olur.")


try:
    build_mixed_validation_set; ChunkNode; semantic_chunking; mark_gold_nodes; model; svsf_search_step
except NameError as e:
    print(f"!!! Gerekli fonksiyon yok ({e}). TWO STAGES ve FAST NODES hucrelerini calistirin.")
else:
    main()
