# PAPER: §5.9 — chain-of-thought ablation (both metrics + LLM judge)
#!/usr/bin/env python3
"""
SCRIPT B (YEREL) — CoT ON/OFF ABLASYONU, DeepChainXHard 200 setinde
=====================================================================
NEDEN YEREL: LLM agirlikli (200 soru x sistem x 2 kosul), saatler surebilir;
Colab hareketsizlikte oturumu kapatiyor. Bu script checkpoint'li calisir:
her sorudan sonra CSV'ye yazar, kesilirse KALDIGI YERDEN devam eder.

NEDEN app_v11 IMPORT EDIYOR: makaleyle BIREBIR ayni prompt (make_prompt),
ayni chunking (semantic thr=0.3), ayni metrikler (compute_exact_match /
compute_em_flex), ayni retriever'lar kullanilmali. Yeniden yazmak sayilari
karsilastirilamaz hale getirir. app_v11.py'de `if __name__ == "__main__"`
guard'i var, dolayisiyla import Flask sunucusunu BASLATMAZ; yalnizca
modelleri yukler ve fonksiyonlari saglar.

BU KOSU NEYI COZUYOR:
  1. §5.9'un CoT-on degerleri (51.5/50.0) dogrulanamiyordu -> ayni sette uretilir
  2. Table 5 (§5.8) baska bir veri setinden geliyordu -> ayni sette yeniden uretilir
     boylece §5.8 ve §5.9 AYNI veri setinden gelir (tutarsizlik kokten cozulur)
  3. Her iki metrik (strict EM + Flexible EM) ACIKCA etiketli raporlanir
  4. MultiQuery'nin onceki loglarda %0 cikmasi teshis edilir
  5. LLM-judge calistirilir (makalede bahsedilip raporlanmayan metrik)
  6. CAPRAZ DOGRULAMA: CoT-KAPALI tarafi mevcut n=4 loguyla karsilastirilir.
     Beklenen Flex EM: Vanilla 42.5, SVSF 44.5. Tutuyorsa kosu dogrulanmis olur.

KURULUM (yerel makine):
  pip install sentence-transformers scikit-learn rank_bm25 requests numpy pandas flask python-docx PyPDF2
  ollama serve &                  # arka planda
  ollama pull gemma3:12b-64k      # loglardaki model (yoksa gemma3:12b kullan, MODEL'i degistir)

DOSYALAR (bu script ile ayni klasorde):
  app_v11.py
  deepchain_xhard_200.csv                 (veri seti)
  history_deepchain_xhard_n4.csv          (capraz dogrulama icin, opsiyonel)

KULLANIM:
  python scriptB_local_cot.py                    # tam kosu (200 soru)
  python scriptB_local_cot.py --limit 10         # once 10 soru ile dene
  python scriptB_local_cot.py --report-only      # mevcut checkpoint'ten rapor uret
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np

# ── AYARLAR ──────────────────────────────────────────────────────────────
MODEL        = "gemma3:12b-64k"      # loglardaki model; yoksa "gemma3:12b"
NUM_AGENTS   = 4                     # loglarla ayni (Agents=4)
ALPHA, BETA  = 1.0, 0.6
TAU          = 0.2                   # Threshold
SEM_THR      = 0.3                   # Semantic (thr=0.3)
DATASET      = "deepchain_xhard_200.csv"
REF_LOG      = "history_deepchain_xhard_n4.csv"
OUT          = "cot_ablation_v2.csv"   # v1 gecersiz: kendi prompt'umuz kullanilmisti
RUN_JUDGE    = True                  # LLM-judge calistir (E3)
JUDGE_MODEL  = MODEL

FIELDS = ["qidx", "cond", "system", "pred", "gold", "strict", "flex", "judge", "notfound", "ms"]


# ── app_v11'i import et (modeller yuklenir) ──────────────────────────────
def load_engine():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists("app_v11.py"):
        sys.exit("HATA: app_v11.py bu klasorde bulunamadi.")
    print("[1/4] app_v11 import ediliyor (modeller yuklenecek, biraz surebilir)...")
    import app_v11 as E
    need = ["model", "ChunkNode", "semantic_chunking", "vanilla_search_step", "svsf_search_step",
            "mmr_search_step", "rerank_search_step", "multiquery_search_step",
            "make_prompt", "extract_answer", "query_llm",
            "compute_exact_match", "compute_em_flex"]
    missing = [n for n in need if not hasattr(E, n)]
    if missing:
        sys.exit(f"HATA: app_v11 icinde eksik: {missing}")
    print(f"      OK. cihaz={getattr(E,'_dev','?')} | "
          f"MiniLM={'var' if getattr(E,'cross_encoder',None) else 'yok'} | "
          f"BGE={'var' if getattr(E,'cross_encoder_bge',None) else 'yok'}")
    return E


# ── CoT ACIK/KAPALI: app_v11'in KENDI mekanizmasi kullanilir ─────────────
# app_v11.make_prompt(), SVSF_NO_COT ortam degiskenine bakar:
#   SVSF_NO_COT=0 -> CoT prompt'u (<thinking> + <answer>)
#   SVSF_NO_COT=1 -> CoT'suz ablasyon prompt'u (yalnizca <answer>)
# Her iki durumda cevap extract_answer() ile <answer> etiketinden ayiklanir.
# ONEMLI: kendi prompt'umuzu YAZMIYORUZ; aksi halde orijinal kosuyla
# karsilastirilamaz hale gelir (ilk denememizde bu hata yapildi).
def set_cot(on: bool):
    os.environ["SVSF_NO_COT"] = "0" if on else "1"


def build_nodes(E, corpus):
    """
    Makaledeki chunking hattinin BIREBIR aynisi (bkz. app_v11._get_shared_pool):
      semantic_chunking -> STRING listesi dondurur
      ChunkNode(id, txt) -> "passage: " prefix'i ile embedding uretir
      embedding alan adi .text_embedding'dir (.embedding DEGIL)
    """
    raw_texts = E.semantic_chunking(corpus, similarity_threshold=SEM_THR)
    nodes = [E.ChunkNode(f"P_{i}", txt) for i, txt in enumerate(raw_texts)]
    if not nodes:
        return [], None
    embs = np.vstack([n.text_embedding for n in nodes])
    return nodes, embs


def retrieve_all(E, qv, qtext, embs, nodes):
    """Makalenin alti sistemi. bm25 SVSF icin gerekli."""
    from rank_bm25 import BM25Okapi
    raw = BM25Okapi([n.text.lower().split() for n in nodes]).get_scores(qtext.lower().split())
    mx, mn = float(np.max(raw)), float(np.min(raw))
    bm = (raw - mn) / (mx - mn if mx > mn else 1.0)

    out = {}
    t = time.perf_counter(); out["Vanilla"] = (E.vanilla_search_step(qv, embs, nodes, NUM_AGENTS),
                                               (time.perf_counter()-t)*1000)
    t = time.perf_counter()
    sv = E.svsf_search_step(qv, embs, nodes, bm, ALPHA, BETA, NUM_AGENTS, TAU, False, 1.0)
    out["SVSF"] = (sv[0] if isinstance(sv, tuple) else sv, (time.perf_counter()-t)*1000)
    t = time.perf_counter(); out["MMR"] = (E.mmr_search_step(qv, embs, nodes, NUM_AGENTS),
                                           (time.perf_counter()-t)*1000)
    if getattr(E, "cross_encoder", None):
        t = time.perf_counter()
        out["Rerank"] = (E.rerank_search_step(qv, qtext, embs, nodes, NUM_AGENTS, E.cross_encoder),
                         (time.perf_counter()-t)*1000)
    if getattr(E, "cross_encoder_bge", None):
        t = time.perf_counter()
        out["RerankBGE"] = (E.rerank_search_step(qv, qtext, embs, nodes, NUM_AGENTS, E.cross_encoder_bge),
                            (time.perf_counter()-t)*1000)
    # MultiQuery: onceki loglarda %0 cikti -> varyant uretimini burada TESHIS ediyoruz
    try:
        variants = [qtext]
        vr = E.query_llm(MODEL, f"Rewrite the question in 2 different ways, one per line, "
                                f"no numbering, no explanation.\n\nQuestion: {qtext}")
        for ln in (vr or "").splitlines():
            ln = ln.strip(" -*\t")
            if ln and ln.lower() != qtext.lower() and len(ln) > 10:
                variants.append(ln)
            if len(variants) >= 3:
                break
        vembs = np.stack([E.model.encode("query: " + v) for v in variants])
        t = time.perf_counter()
        out["MultiQuery"] = (E.multiquery_search_step(vembs, embs, nodes, NUM_AGENTS),
                             (time.perf_counter()-t)*1000)
        out["_mq_variants"] = len(variants)
    except Exception as e:
        out["_mq_error"] = str(e)[:120]
    return out


def load_dataset():
    import pandas as pd
    df = pd.read_csv(DATASET)
    cols = {c.lower(): c for c in df.columns}
    q, a, c = cols.get("query"), cols.get("suggested_answer"), cols.get("corpus")
    if not all([q, a, c]):
        sys.exit(f"HATA: {DATASET} icinde query/suggested_answer/corpus yok. Kolonlar: {list(df.columns)}")
    return [(str(r[q]), str(r[a]), str(r[c])) for _, r in df.iterrows()]


def done_keys():
    if not os.path.exists(OUT):
        return set()
    seen = set()
    with open(OUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add((int(row["qidx"]), row["cond"], row["system"]))
    return seen


def report():
    import pandas as pd
    if not os.path.exists(OUT):
        sys.exit("Sonuc dosyasi yok.")
    df = pd.read_csv(OUT)
    print(f"\n{'='*84}\nCoT ABLASYONU — DeepChainXHard, n_docs={NUM_AGENTS}, model={MODEL}\n{'='*84}")
    nq = df.qidx.nunique()
    print(f"islenen soru: {nq}\n")
    for metric, lbl in (("strict", "Strict EM"), ("flex", "Flexible EM")):
        print(f"  ── {lbl} (%) ──")
        print(f"  {'Sistem':<13}{'CoT ACIK':<11}{'CoT KAPALI':<12}{'dusus':<9}{'McNemar p':<11}")
        for s in sorted(df.system.unique()):
            on = df[(df.system == s) & (df.cond == "cot")].set_index("qidx")[metric]
            off = df[(df.system == s) & (df.cond == "nocot")].set_index("qidx")[metric]
            idx = on.index.intersection(off.index)
            if len(idx) == 0:
                continue
            a, b = on.loc[idx].values, off.loc[idx].values
            n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
            if n01 + n10:
                from scipy.stats import binomtest
                p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
            else:
                p = 1.0
            print(f"  {s:<13}{100*a.mean():<11.1f}{100*b.mean():<12.1f}"
                  f"{100*(a.mean()-b.mean()):<9.1f}{p:<11.4f}")
        print()
    if "judge" in df.columns and df.judge.notna().any():
        print("  ── LLM-judge dogruluk (%) ──")
        for s in sorted(df.system.unique()):
            for c in ("cot", "nocot"):
                sub = df[(df.system == s) & (df.cond == c)].judge.dropna()
                if len(sub):
                    print(f"    {s:<13}{c:<7}{100*sub.mean():.1f}")
    print("  ── NOT FOUND orani (%) ──")
    for s in sorted(df.system.unique()):
        r = [f"{c}={100*df[(df.system==s)&(df.cond==c)].notfound.mean():.1f}" for c in ("cot","nocot")]
        print(f"    {s:<13}" + "  ".join(r))

    # ── CAPRAZ DOGRULAMA ──
    if os.path.exists(REF_LOG):
        print(f"\n  ── CAPRAZ DOGRULAMA ({REF_LOG}, CoT-KAPALI referansi) ──")
        ref = pd.read_csv(REF_LOG)
        for s in ("Vanilla", "SVSF"):
            col = f"{s}_EM_flex"
            if col not in ref.columns:
                continue
            exp = 100 * pd.to_numeric(ref[col], errors="coerce").mean()
            got = df[(df.system == s) & (df.cond == "nocot")].flex
            if len(got) == 0:
                continue
            got = 100 * got.mean()
            d = abs(exp - got)
            print(f"    {s:<10} beklenen(log) {exp:.1f}  |  bu kosu {got:.1f}  |  fark {d:.1f}"
                  f"  {'✓ TUTUYOR' if d <= 3.0 else '✗ SAPMA VAR'}")
        print("    (tam sayi eslesmesi beklenmez: LLM sicakligi/surum farki; ±3 puan makul)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="ilk N soru (deneme icin)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        report(); return

    E = load_engine()
    print("[2/4] veri seti yukleniyor...")
    data = load_dataset()
    if args.limit:
        data = data[:args.limit]
    print(f"      {len(data)} soru")

    seen = done_keys()
    print(f"[3/4] checkpoint: {len(seen)} kayit zaten var, atlanacak")
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    print("[4/4] kosu basliyor. Ctrl+C ile durdurabilirsin; tekrar calistirinca devam eder.\n")
    t0 = time.time()
    mq_diag = {"ok": 0, "err": 0, "variants": []}

    for qi, (q, gold, corpus) in enumerate(data):
        try:
            nodes, embs = build_nodes(E, corpus)
            if embs is None or len(nodes) < NUM_AGENTS:
                continue
            qv = E.model.encode("query: " + q)
            R = retrieve_all(E, qv, q, embs, nodes)
            if "_mq_variants" in R:
                mq_diag["ok"] += 1; mq_diag["variants"].append(R.pop("_mq_variants"))
            if "_mq_error" in R:
                mq_diag["err"] += 1; R.pop("_mq_error")

            for sysname, (sel, ms) in R.items():
                for cond, cot in (("cot", True), ("nocot", False)):
                    if (qi, cond, sysname) in seen:
                        continue
                    if not sel:
                        continue
                    set_cot(cot)                      # orijinal bayrak
                    prompt = E.make_prompt(sel, q)     # prompt'u app_v11 uretir
                    raw = E.query_llm(MODEL, prompt)
                    pred = E.extract_answer(raw)       # her iki kosulda ayni ayiklama
                    nf = 1 if pred.upper().startswith("NOT FOUND") else 0
                    jv = ""
                    if RUN_JUDGE and hasattr(E, "llm_judge"):
                        try:
                            # app_v11.llm_judge -> 1 / 0 / None (None = judge uygulanamadi)
                            jr = E.llm_judge(JUDGE_MODEL, q, pred, gold)
                            jv = "" if jr is None else int(jr)
                        except Exception:
                            jv = ""
                    w.writerow({"qidx": qi, "cond": cond, "system": sysname, "pred": pred[:200],
                                "gold": gold[:120],
                                "strict": E.compute_exact_match(pred, gold),
                                "flex": E.compute_em_flex(pred, gold),
                                "judge": jv, "notfound": nf, "ms": round(ms, 3)})
            fh.flush()
        except KeyboardInterrupt:
            print("\nDurduruldu. Tekrar calistirinca kaldigi yerden devam eder.")
            break
        except Exception as e:
            print(f"  !! soru {qi} hata: {str(e)[:140]}")
            continue

        if (qi + 1) % 5 == 0:
            el = time.time() - t0
            rate = el / (qi + 1)
            print(f"  {qi+1}/{len(data)}  gecen {el/60:.1f} dk  |  "
                  f"tahmini kalan {rate*(len(data)-qi-1)/60:.1f} dk")

    fh.close()
    print(f"\nMultiQuery teshisi: basarili={mq_diag['ok']} hatali={mq_diag['err']}"
          + (f" | ort varyant={np.mean(mq_diag['variants']):.1f}" if mq_diag["variants"] else ""))
    report()


if __name__ == "__main__":
    main()
