# PAPER: §5.12 / Table 8 — equal-budget candidate generation, mega-pool regime
"""
ESIT-BUTCELI CANDIDATE GENERATION \u2014 MEGA-POOL REJIMI  (Paper 1, §8)
===================================================================
Sira (COLAB): DATASETS -> TWO STAGES -> FAST NODES -> BU

NEDEN BU IKINCI KOSU GEREKLI:
  Soru-basina korpus kosusunda (candgen_equal_budget.py) esit butcede fark bulunmadi.
  Ama o ortamda soru basina ~11 chunk var, yani dense top-50 = TUM KORPUS ve
  "aday uretimi" hicbir sey filtrelemiyor. Iki-asamali mimarinin savunulabilecegi
  tek rejim BUYUK HAVUZ; o yuzden iddia orada test edilmeli.

  Burada soru-basina korpus + mega-pool distractor'lari BIRLIKTE tek havuz olur.
  8 aday toplamak artik gercek bir filtreleme (binlerce pasajdan 8).

MEGA-POOL HAKKINDA (koddan dogrulandi):
  megapool_latency.py, mega-pool'un GOLD ICERMEDIGINI soyluyor \u2014 saf distractor
  kaynagi. Bu yuzden burada mega-pool'u TEK BASINA kullanamayiz; sorunun kendi
  gold pasajlarini iceren korpusunu distractor havuzuyla BIRLESTIRIYORUZ.
  Gold her zaman havuzda mevcut, ama binlerce distractor arasinda.

KURULUM:
  MEGA_POOL_PATH ortam degiskeni ya da varsayilan yol mega-pool .txt'yi gostermeli.
  Distractor'lar BIR KEZ embed edilir ve tum sorular icin yeniden kullanilir
  (yoksa kosu saatlerce surer).

KARSILASTIRMA (onceki script ile ayni mantik):
  Dense top-8   -> BGE rerank -> top-4
  SVSF gather-8 -> BGE rerank -> top-4      [esit butce]
  Dense top-50  -> BGE rerank -> top-4      [tavan referansi; artik gercek bir tavan]

CIKTI: candgen_megapool.csv + ekrana tablo. Checkpoint'li.
"""
import os, csv, time
import numpy as np
from scipy.stats import wilcoxon

MEGA_POOL_PATH = os.environ.get("SVSF_SHARED_CORPUS_PATH", "data/mega_pool_5k.txt")
N_PER_SOURCE = 100                  # 3 x 100 = 300 soru (mega-pool agir, daha az soru)
SOURCES      = ["HotpotQA", "2WikiMultihopQA", "MuSiQue"]
K_GATHER     = 8
K_FULL       = 50
N_OUT        = 4
SEM_THR      = 0.3
ALPHA, BETA, TAU = 1.0, 0.6, 0.2
MAX_DISTRACTORS = 5000              # havuz buyuklugu tavani
OUT          = "candgen_megapool.csv"
SEED         = 20260901

FIELDS = ["qi","source","n_gold","pool_size",
          "cand_dense8","cand_svsf8","cand_full50",
          "novel_svsf","novel_dense",
          "fin_dense8","fin_svsf8","fin_full50",
          "ms_gather_dense","ms_gather_svsf","ms_rr8","ms_rr50"]


def _n(x): return x / (np.linalg.norm(x) + 1e-9)


def rerank_given(cand_idx, query_text, texts, cross_model, n_out):
    if len(cand_idx) <= n_out:
        return list(cand_idx)
    pairs = [[query_text, texts[i]] for i in cand_idx]
    sc = cross_model.predict(pairs)
    order = np.argsort(sc)[::-1][:n_out]
    return [cand_idx[i] for i in order]


def svsf_on_matrix(qv, U, texts, bm, k, alpha=ALPHA, beta=BETA):
    """
    Paper 1'in SVSF kurali, dogrudan matris uzerinde (mega-pool'da ChunkNode listesi
    kurmak bellek acisindan pahali). app_v11.svsf_search_step ile ayni skorlama:
      score = alpha*cos(Q0,D) + beta*cos(P,D),  P += 0.2*0.5*(qn + v)
    """
    qn = _n(qv)
    sims_q = U @ qn
    P = np.zeros_like(qn)
    out, taken = [], set()
    for step in range(k):
        if np.any(P):
            score = alpha * sims_q + beta * (U @ _n(P))
        else:
            score = alpha * sims_q.copy()
        for i in taken:
            score[i] = -1e9
        i = int(np.argmax(score))
        taken.add(i); out.append(i)
        P = P + 0.2 * 0.5 * (qn + U[i])
    return out


def load_distractors():
    if not os.path.exists(MEGA_POOL_PATH):
        print(f"!!! mega-pool bulunamadi: {MEGA_POOL_PATH}")
        print("    SVSF_SHARED_CORPUS_PATH ortam degiskenini ayarlayin, ornegin:")
        print("    os.environ['SVSF_SHARED_CORPUS_PATH'] = '/content/mega_pool_5k.txt'")
        return None, None
    raw = open(MEGA_POOL_PATH, encoding="utf-8", errors="replace").read()
    paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 80]
    paras = paras[:MAX_DISTRACTORS]
    print(f"      mega-pool: {len(paras)} distractor pasaj")
    print("      distractor embedding'leri hesaplaniyor (bir kez)...")
    _, D = embed_pool_once(paras, batch_size=16)
    return paras, D


def main():
    import random as _r
    _r.seed(SEED)
    print(f"[1/4] mega-pool yukleniyor: {MEGA_POOL_PATH}")
    dtexts, Demb = load_distractors()
    if dtexts is None: return
    Dunit = Demb / (np.linalg.norm(Demb, axis=1, keepdims=True) + 1e-9)

    print(f"[2/4] {N_PER_SOURCE} soru/kaynak yukleniyor...")
    data = build_mixed_validation_set(n_sample=N_PER_SOURCE, sources=SOURCES)
    print(f"      {len(data)} soru | gold_passages olan: "
          f"{sum(1 for d in data if d.get('gold_passages'))}")
    bge = globals().get("cross_encoder_bge")
    if bge is None: print("!!! cross_encoder_bge yok."); return

    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8") as f:
            done = {int(r["qi"]) for r in csv.DictReader(f)}
        print(f"[3/4] checkpoint: {len(done)} soru atlanacak")
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="", encoding="utf-8"); w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new: w.writeheader()

    print(f"[4/4] kosu basliyor (havuz \u2248 {len(dtexts)} + soru korpusu)\n")
    from rank_bm25 import BM25Okapi
    import torch
    cuda = torch.cuda.is_available()
    t0 = time.time(); used = 0; skipped = 0

    for qi, item in enumerate(data):
        if qi in done: continue
        gp = item.get("gold_passages") or []
        if not gp: skipped += 1; continue
        try:
            # sorunun kendi korpusu -> chunk + gold isaretleme
            nodes = [ChunkNode(f"P_{i}", t) for i, t in
                     enumerate(semantic_chunking(item["corpus"], similarity_threshold=SEM_THR))]
            if not nodes: skipped += 1; continue
            mark_gold_nodes(nodes, gp, overlap_threshold=0.6)
            gold_local = [i for i, n in enumerate(nodes) if getattr(n, "is_gold", False)]
            if not gold_local: skipped += 1; continue

            qtexts = [n.text for n in nodes]
            Qemb = np.vstack([n.text_embedding for n in nodes])
            # BIRLESIK HAVUZ: soru korpusu (basta) + distractor'lar
            texts = qtexts + dtexts
            U = np.vstack([Qemb / (np.linalg.norm(Qemb, axis=1, keepdims=True) + 1e-9), Dunit])
            G = set(gold_local)          # indeksler basta oldugu icin degismiyor
            ng = len(G); pool = len(texts)

            qv = model.encode("query: " + item["query"])

            # ---- GATHER: dense ----
            if cuda: torch.cuda.synchronize()
            t = time.perf_counter()
            order = np.argsort(U @ _n(qv))[::-1]
            d8 = [int(i) for i in order[:K_GATHER]]
            if cuda: torch.cuda.synchronize()
            ms_gd = (time.perf_counter()-t)*1000

            # ---- GATHER: SVSF ----
            raw = BM25Okapi([x.lower().split() for x in texts]).get_scores(
                    item["query"].lower().split())
            mx, mn = float(raw.max()), float(raw.min())
            bm = (raw - mn) / (mx - mn if mx > mn else 1.0)
            if cuda: torch.cuda.synchronize()
            t = time.perf_counter()
            s8 = svsf_on_matrix(qv, U, texts, bm, K_GATHER)
            if cuda: torch.cuda.synchronize()
            ms_gs = (time.perf_counter()-t)*1000

            f50 = [int(i) for i in order[:K_FULL]]
            assert len(d8) == len(s8) == K_GATHER, "butce esit degil"

            # ---- RERANK ----
            if cuda: torch.cuda.synchronize()
            t = time.perf_counter()
            fd = rerank_given(d8, item["query"], texts, bge, N_OUT)
            fs = rerank_given(s8, item["query"], texts, bge, N_OUT)
            if cuda: torch.cuda.synchronize()
            ms_rr8 = (time.perf_counter()-t)*1000/2
            if cuda: torch.cuda.synchronize()
            t = time.perf_counter()
            ff = rerank_given(f50, item["query"], texts, bge, N_OUT)
            if cuda: torch.cuda.synchronize()
            ms_rr50 = (time.perf_counter()-t)*1000

            w.writerow(dict(qi=qi, source=item.get("source","?"), n_gold=ng, pool_size=pool,
                cand_dense8=len(G & set(d8))/ng, cand_svsf8=len(G & set(s8))/ng,
                cand_full50=len(G & set(f50))/ng,
                novel_svsf=len(G & (set(s8)-set(d8))), novel_dense=len(G & (set(d8)-set(s8))),
                fin_dense8=len(G & set(fd))/ng, fin_svsf8=len(G & set(fs))/ng,
                fin_full50=len(G & set(ff))/ng,
                ms_gather_dense=round(ms_gd,3), ms_gather_svsf=round(ms_gs,3),
                ms_rr8=round(ms_rr8,3), ms_rr50=round(ms_rr50,3)))
            used += 1
            if used % 10 == 0:
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
    lo, hi = np.percentile(m,[2.5,97.5]); below=float((m<=0).mean())
    return d.mean(), lo, hi, float(min(1.0, 2*min(below,1-below)))


def report():
    import pandas as pd
    if not os.path.exists(OUT): print("Sonuc dosyasi yok."); return
    df = pd.read_csv(OUT)
    print(f"\n{'='*88}\nMEGA-POOL REJIMI \u2014 ESIT BUTCELI CANDIDATE GENERATION")
    print(f"n={len(df)} soru | k_gather={K_GATHER} | n_out={N_OUT} | "
          f"ort. havuz {df.pool_size.mean():.0f} pasaj\n{'='*88}")
    print(f"  kaynak: {df.source.value_counts().to_dict()} | ort. gold/soru {df.n_gold.mean():.2f}\n")
    print(f"  {'Gatherer':<22}{'Butce':<8}{'Aday recall':<14}{'Rerank sonrasi':<17}{'Gather ms':<12}{'Rerank ms':<11}")
    for nm,b,c,f,gm,rm in [("Dense top-8","8","cand_dense8","fin_dense8","ms_gather_dense","ms_rr8"),
                           ("SVSF gather-8","8","cand_svsf8","fin_svsf8","ms_gather_svsf","ms_rr8"),
                           ("Dense top-50","50","cand_full50","fin_full50","ms_gather_dense","ms_rr50")]:
        print(f"  {nm:<22}{b:<8}{df[c].mean():<14.3f}{df[f].mean():<17.3f}"
              f"{df[gm].median():<12.2f}{df[rm].median():<11.2f}")

    print(f"\n  --- ESLI KARSILASTIRMA (SVSF-8 eksi Dense-8) ---")
    for lbl,a,b in [("Aday-kumesi recall","cand_svsf8","cand_dense8"),
                    ("Rerank sonrasi recall","fin_svsf8","fin_dense8")]:
        d=(df[a]-df[b]).values; m,lo,hi,p=boot(d)
        try: _,pw=wilcoxon(d) if np.any(d!=0) else (0,1.0)
        except Exception: pw=1.0
        print(f"  {lbl:<24}\u0394 = {m:+.4f}  %95 GA [{lo:+.4f}, {hi:+.4f}]  "
              f"boot p={p:.4f}  Wilcoxon p={pw:.4f}  {'ANLAMLI' if min(p,pw)<0.05 else 'fark yok'}")

    ns,nd = df.novel_svsf.sum(), df.novel_dense.sum()
    qs,qd = (df.novel_svsf>0).sum(), (df.novel_dense>0).sum()
    print(f"\n  --- MEKANIZMA ---")
    print(f"    SVSF-8'de olup dense-8'de olmayan gold : {ns} ({qs} soru, %{100*qs/len(df):.1f})")
    print(f"    dense-8'de olup SVSF-8'de olmayan gold : {nd} ({qd} soru, %{100*qd/len(df):.1f})")
    print(f"    net: {ns-nd:+d}")

    td=df.ms_gather_dense.median()+df.ms_rr8.median()
    ts=df.ms_gather_svsf.median()+df.ms_rr8.median()
    tf=df.ms_gather_dense.median()+df.ms_rr50.median()
    print(f"\n  --- GECIKME ---")
    print(f"    Dense-8  -> rerank : {td:.2f} ms")
    print(f"    SVSF-8   -> rerank : {ts:.2f} ms  ({ts-td:+.2f} ms)")
    print(f"    Dense-50 -> rerank : {tf:.2f} ms")
    print(f"\n  NOT: bu rejimde dense top-50 artik TUM havuz degil "
          f"(ort. {df.pool_size.mean():.0f} pasaj), yani aday uretimi gercek bir filtreleme.")


try:
    build_mixed_validation_set; ChunkNode; semantic_chunking; mark_gold_nodes; model; embed_pool_once
except NameError as e:
    print(f"!!! Gerekli fonksiyon yok ({e}). TWO STAGES ve FAST NODES hucrelerini calistirin.")
else:
    main()
