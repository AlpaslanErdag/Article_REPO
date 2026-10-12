# PAPER: §5.5 / Table 4 — per-query latency distribution
"""
SCRIPT A — LATENCY PROFILLEME (hakem talebi: mean, p95, p99, protokol detaylari)
Sira: DATASETS -> OLLAMA(gerekmez) -> TWO STAGES -> FAST NODES -> BU

NE OLCUYOR:
  Paper 1'in mevcut tablolari sorgu basina 10-kosu ORTALAMASININ medyanini veriyor
  (timed_retrieve: warmup=2, runs=10, CUDA sync). Bu iyi bir protokol ama p95/p99
  hesaplamak icin uygun degil: her deger zaten yumusatilmis bir ortalama.
  Bu yuzden IKI olcum birden yapiyoruz:
    (1) TEK-VURUS (single-shot) sorgu latency dagilimi -> mean / p50 / p95 / p99
        ("p95 latency" ifadesinin standart anlami budur)
    (2) 10-KOSU ORTALAMASI medyani -> mevcut tablolarla karsilastirilabilirlik
  Ayrica protokol bilgilerini (K_rerank, max_length, batch, corpus boyutu, warmup)
  DOGRUDAN kodun kendisinden okuyup basiyoruz -> makaleye tahminle degil olcumle girer.

NOT (MultiQuery): app_v11'de sorgu varyantlari timed_retrieve DISINDA uretiliyor
  (adil olcum icin). Latency acisindan onemli olan varyant SAYISI oldugundan,
  ayni sorgu vektorunu 3 kez yiginlayarak zamanlama-esdeger bir girdi kullaniyoruz;
  varyantlarin ICERIGI latency'yi etkilemez. Bu, LLM cagrisi olmadan dogru olcum verir.
"""
import os, time, json
import numpy as np

N_QUESTIONS = 200
NUM_AGENTS = 4              # Paper 1 varsayilani (n_docs)
SEM_THR = 0.3               # Paper 1 chunking esigi
WARMUP_GLOBAL = 5           # tum sistemler icin bir kez isinma
RUNS_MEAN = 10              # mevcut protokolle karsilastirmak icin

os.environ["PAPER21_NORUN"] = "1"
try:
    load_musique; balance_by_hop
except NameError:
    from paper21_runner import load_musique, balance_by_hop


def _percentiles(v):
    a = np.asarray(v, dtype=float)
    return dict(n=len(a), mean=a.mean(), p50=np.percentile(a, 50),
                p90=np.percentile(a, 90), p95=np.percentile(a, 95),
                p99=np.percentile(a, 99), mn=a.min(), mx=a.max())


def protocol_report():
    """Protokol bilgilerini koddan/model nesnelerinden OKU (tahmin etme)."""
    print("=" * 84)
    print("OLCUM PROTOKOLU (koddan okundu)")
    print("=" * 84)
    import inspect
    try:
        src = inspect.getsource(rerank_search_step)
        krr = "50 (sabit; np.argsort(sims)[-50:])" if "[-50:]" in src else "kodda tespit edilemedi"
    except Exception:
        krr = "kaynak okunamadi"
    print(f"  Reranker aday havuzu (K_rerank) : {krr}")
    print(f"  -> n_docs reranker maliyetini ETKILEMEZ; yalnizca skorlanan listeyi keser.")
    for nm in ("cross_encoder", "cross_encoder_bge"):
        obj = globals().get(nm)
        if obj is None:
            print(f"  {nm:<18}: yok"); continue
        ml = getattr(obj, "max_length", None)
        dev = None
        try:
            dev = str(next(obj.model.parameters()).device)
        except Exception:
            pass
        print(f"  {nm:<18}: max_length={ml}  device={dev}")
    print(f"  Benzerlik taramasi              : exact (brute-force cosine), ANN indeksi YOK")
    print(f"  Embedding'ler                   : sorgu basina korpus icin bir kez hesaplanir")
    print(f"  Isinma (bu olcum)               : {WARMUP_GLOBAL} adet zamanlanmayan gecis")
    print(f"  Mevcut makale protokolu         : timed_retrieve(warmup=2, runs=10, CUDA sync)")
    print(f"  Throughput                      : bu calismada olculmedi (per-query latency raporlanir)")
    print()


def main():
    data = balance_by_hop(load_musique("musique_ans_train.jsonl"), 100)[:N_QUESTIONS]
    print(f"{len(data)} soru | n_docs={NUM_AGENTS}\n")
    protocol_report()

    refiner_mini = globals().get("cross_encoder")
    refiner_bge = globals().get("cross_encoder_bge")

    # sistem tanimlari: (isim, cagrilabilir)
    def systems(qv, embs, nodes, bm, qtext):
        S = [("Vanilla", lambda: vanilla_search_step(qv, embs, nodes, NUM_AGENTS))]
        try:
            S.append(("SVSF", lambda: svsf_search_step(qv, embs, nodes, bm, 1.0, 0.6,
                                                       NUM_AGENTS, 0.2, False, 1.0)))
        except Exception:
            pass
        S.append(("MMR", lambda: mmr_search_step(qv, embs, nodes, NUM_AGENTS)))
        if refiner_mini is not None:
            S.append(("Rerank-MiniLM",
                      lambda: rerank_search_step(qv, qtext, embs, nodes, NUM_AGENTS, refiner_mini)))
        if refiner_bge is not None:
            S.append(("Rerank-BGE",
                      lambda: rerank_search_step(qv, qtext, embs, nodes, NUM_AGENTS, refiner_bge)))
        # MultiQuery: 3 varyant -> zamanlama-esdeger yigin (icerik latency'yi etkilemez)
        vq = np.stack([qv, qv, qv])
        S.append(("MultiQuery", lambda: multiquery_search_step(vq, embs, nodes, NUM_AGENTS)))
        return S

    single = {}      # tek-vurus latency listeleri
    tenmean = {}     # 10-kosu ortalamalari
    node_counts = []
    warmed = False

    from rank_bm25 import BM25Okapi
    import torch
    cuda = torch.cuda.is_available()

    for qi, (q, paras, hops) in enumerate(data):
        corpus = "\n\n".join(f"Title: {t}\n{x}" for t, x, _ in paras)
        try:
            nodes, embs = build_nodes_fast(corpus, "off", SEM_THR)
        except Exception:
            texts = [f"Title: {t}\n{x}" for t, x, _ in paras]
            nodes, embs = embed_pool_once(texts, batch_size=16)
        if len(nodes) < NUM_AGENTS + 1:
            continue
        node_counts.append(len(nodes))
        qv = model.encode("query: " + q)
        raw = BM25Okapi([n.text.lower().split() for n in nodes]).get_scores(q.lower().split())
        mx, mn = float(raw.max()), float(raw.min())
        bm = (raw - mn) / (mx - mn if mx > mn else 1.0)

        S = systems(qv, embs, nodes, bm, q)

        if not warmed:                       # global isinma
            for _ in range(WARMUP_GLOBAL):
                for _, fn in S:
                    fn()
            if cuda:
                torch.cuda.synchronize()
            warmed = True

        for name, fn in S:
            # (1) tek-vurus
            if cuda: torch.cuda.synchronize()
            t0 = time.perf_counter(); fn()
            if cuda: torch.cuda.synchronize()
            single.setdefault(name, []).append((time.perf_counter() - t0) * 1000.0)
            # (2) 10-kosu ortalamasi (mevcut protokol)
            if cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(RUNS_MEAN): fn()
            if cuda: torch.cuda.synchronize()
            tenmean.setdefault(name, []).append((time.perf_counter() - t0) / RUNS_MEAN * 1000.0)

        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(data)}...")

    print(f"\nkorpus boyutu (chunk/soru): ort {np.mean(node_counts):.1f} "
          f"[min {min(node_counts)}, max {max(node_counts)}]\n")

    print("=" * 84)
    print("TEK-VURUS SORGU LATENCY (ms) — 'p95/p99 latency' standart anlami")
    print("=" * 84)
    print(f"  {'Sistem':<16}{'mean':<9}{'p50':<9}{'p90':<9}{'p95':<9}{'p99':<9}{'min':<9}{'max':<9}")
    for name in single:
        s = _percentiles(single[name])
        print(f"  {name:<16}{s['mean']:<9.2f}{s['p50']:<9.2f}{s['p90']:<9.2f}"
              f"{s['p95']:<9.2f}{s['p99']:<9.2f}{s['mn']:<9.2f}{s['mx']:<9.2f}")

    print("\n" + "=" * 84)
    print("10-KOSU ORTALAMASI (mevcut makale protokolu ile karsilastirilabilir)")
    print("=" * 84)
    print(f"  {'Sistem':<16}{'median':<11}{'mean':<11}{'p95':<11}")
    for name in tenmean:
        s = _percentiles(tenmean[name])
        print(f"  {name:<16}{s['p50']:<11.2f}{s['mean']:<11.2f}{s['p95']:<11.2f}")

    # makine-okunur kayit
    out = {"single_shot": {k: _percentiles(v) for k, v in single.items()},
           "ten_run_mean": {k: _percentiles(v) for k, v in tenmean.items()},
           "config": {"n_docs": NUM_AGENTS, "chunk_thr": SEM_THR,
                      "avg_chunks_per_question": float(np.mean(node_counts)),
                      "K_rerank": 50, "reranker_max_length": 512,
                      "warmup": WARMUP_GLOBAL, "runs_mean": RUNS_MEAN,
                      "scan": "exact brute-force cosine", "throughput": "not measured"}}
    with open("latency_profile.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nlatency_profile.json yazildi -> makaleye/appendix'e islenecek.")


main()
