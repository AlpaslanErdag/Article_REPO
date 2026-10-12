# PAPER: §5.7 — 150-question diagnostic study incl. PRF baselines
"""
PAPER 1 — FAZ 2b: DORT VERI SETI + GERCEK HOP-TABANLI BRIDGE
Sira: DATASETS -> TWO STAGES -> FAST NODES -> BU
(paper21_runner.py dosyasi /content'te olmali — MuSiQue/MoreHopQA yukleyicileri icin)

IKI BOSLUGU KAPATIR:
  1. HotpotQA + 2WikiMultihopQA eklenir -> hakem metrikleri Paper 1'in UC ana veri setinde
     (+ MoreHopQA ek veri) raporlanabilir. TWO STAGES'teki build_mixed_validation_set()
     kullanilir (gold_passages saglar).
  2. BRIDGE metrigi duzeltilir. Onceki tanim ("query'ye cosine olarak en uzak gold")
     DONGUSELDI: Vanilla tam olarak query benzerligine gore siraladigi icin o tanimla
     bridge recall == full-chain recall oluyordu (matematiksel zorunluluk; onceki kosuda
     Vanilla'nin iki kolonu birebir aynidi). Yeni tanim: MuSiQue'de
     question_decomposition[].paragraph_support_idx ile GERCEK hop sirasi -> SON HOP gold'u.
     Bu tanim query-benzerliginden BAGIMSIZ.

DURUSTLUK KISITI (makalede belirtilecek):
  Gercek hop sirasi YALNIZCA MuSiQue'de mevcut. HotpotQA ve 2Wiki'de supporting_facts
  SIRASIZ -> "son hop" tanimlanamaz. Bu yuzden bridge metrigi yalnizca MuSiQue'de
  raporlanir; diger setlerde SF-Recall + Full-chain ile kalinir. Uydurma vekil kullanilmaz.

KULLANIM:
  VERIFY_ONLY = True  -> once paragraph_support_idx'in gercekten var oldugunu ve
                         formatini DOGRULAR (birkac ornek basar), kosu yapmaz.
  VERIFY_ONLY = False -> tam kosu.
"""
import os
import json
import random
import numpy as np
from scipy.stats import wilcoxon

VERIFY_ONLY = True          # <<< ILK CALISTIRMADA True BIRAK, cikti temizse False yap
N_QUESTIONS = 150           # veri seti basina
KS = (2, 3, 4)
KMAX = max(KS)
ALPHA, BETA = 1.0, 0.6
PRF_M, PRF_A, PRF_B, PRF_ITERS = 3, 1.0, 0.5, 3
MUSIQUE_PATH = "musique_ans_train.jsonl"
SEED = 20260826


# ══════════════ 1) MuSiQue: HOP SIRALI YUKLEYICI ══════════════
def load_musique_hopordered(path, verify_n=0):
    """
    Her soru icin: (query, paras, hops, hop_order)
      paras     : [(title, text, is_gold), ...]  (orijinal paragraphs sirasinda)
      hop_order : [paragraf_indeksi, ...] zincir sirasinda (question_decomposition'dan)
    paragraph_support_idx bulunamayan sorular ATLANIR (uydurma sira kullanilmaz).
    verify_n > 0 ise ilk verify_n ornegin yapisini basar ve bos liste dondurur.
    """
    out, skipped_no_idx, skipped_other = [], 0, 0
    shown = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            qid = str(d.get("id", ""))
            hops = None
            for h in (2, 3, 4):
                if qid.startswith(f"{h}hop"):
                    hops = h; break
            if hops is None:
                continue
            paras_raw = d.get("paragraphs", []) or []
            paras = [(p.get("title", ""), p.get("paragraph_text", ""),
                      bool(p.get("is_supporting"))) for p in paras_raw]
            qd = d.get("question_decomposition", []) or []

            if verify_n and shown < verify_n:
                print(f"\n--- ORNEK {shown+1} (id={qid}) ---")
                print(f"  paragraphs: {len(paras)} | gold(is_supporting): "
                      f"{[i for i,(_,_,g) in enumerate(paras) if g]}")
                print(f"  question_decomposition: {len(qd)} adim")
                for j, st in enumerate(qd):
                    if isinstance(st, dict):
                        print(f"    hop{j}: keys={sorted(st.keys())}")
                        print(f"           paragraph_support_idx="
                              f"{st.get('paragraph_support_idx', 'YOK')!r}")
                shown += 1
                if shown >= verify_n:
                    return []

            order = []
            ok = True
            for st in qd:
                if not isinstance(st, dict):
                    ok = False; break
                idx = st.get("paragraph_support_idx", None)
                if idx is None:
                    ok = False; break
                try:
                    idx = int(idx)
                except Exception:
                    ok = False; break
                if not (0 <= idx < len(paras)):
                    ok = False; break
                order.append(idx)
            if not ok or not order:
                skipped_no_idx += 1
                continue
            if not d.get("question") or not paras:
                skipped_other += 1
                continue
            # tutarlilik: son hop gercekten gold olarak isaretli mi?
            if not paras[order[-1]][2]:
                skipped_other += 1
                continue
            out.append((d["question"], paras, hops, order))
    print(f"[MuSiQue hop-sirali] yuklendi={len(out)}  "
          f"atlandi(idx yok/gecersiz)={skipped_no_idx}  atlandi(diger)={skipped_other}")
    return out


# ══════════════ 2) HotpotQA / 2Wiki adaptoru ══════════════
def load_from_mixed(source, n):
    """
    TWO STAGES'teki build_mixed_validation_set()'i (query, paras, hops, None) formatina cevirir.
    corpus tek string olarak '\\n\\n' ile birlesik geldiginden parcalara ayrilir; gold,
    gold_passages ile birebir string eslesmesiyle isaretlenir.
    hop sirasi YOK (supporting_facts sirasiz) -> hop_order=None, bridge raporlanmaz.
    """
    random.seed(SEED)
    items = build_mixed_validation_set(n_sample=n, sources=[source])
    out = []
    for it in items:
        parts = [p for p in it["corpus"].split("\n\n") if p.strip()]
        goldset = set(g.strip() for g in (it.get("gold_passages") or []))
        paras = []
        for p in parts:
            first = p.split("\n", 1)
            title = first[0].replace("Title: ", "").strip() if first else ""
            body = first[1] if len(first) > 1 else ""
            paras.append((title, body, p.strip() in goldset))
        if it.get("query") and paras and any(g for _, _, g in paras):
            out.append((it["query"], paras, None, None))
    print(f"[{source}] kullanilabilir soru: {len(out)}/{len(items)}")
    return out


# ══════════════ 3) RETRIEVER'LAR (iz normalize; asimetri yok) ══════════════
def _n(x):
    return x / (np.linalg.norm(x) + 1e-9)


def rank_vanilla(qv, unit, k):
    return list(np.argsort(unit @ _n(qv))[::-1][:k])


def rank_prf_rocchio(qv, unit, k, m=PRF_M, a=PRF_A, b=PRF_B):
    qn = _n(qv)
    top = np.argsort(unit @ qn)[::-1][:m]
    q2 = _n(a * qn + b * _n(unit[top].mean(axis=0)))
    return list(np.argsort(unit @ q2)[::-1][:k])


def rank_prf_iter(qv, unit, k, m=PRF_M, a=PRF_A, b=PRF_B, iters=PRF_ITERS):
    qn = _n(qv); q_cur = qn
    for _ in range(iters):
        top = np.argsort(unit @ q_cur)[::-1][:m]
        q_cur = _n(a * qn + b * _n(unit[top].mean(axis=0)))
    return list(np.argsort(unit @ q_cur)[::-1][:k])


def rank_prf_seq(qv, unit, k, a=PRF_A, b=PRF_B):
    qn = _n(qv); acc = np.zeros_like(qn); cnt = 0
    out, taken = [], set()
    for _ in range(k):
        q_cur = _n(a * qn + b * _n(acc / cnt)) if cnt else qn
        s = unit @ q_cur
        for i in taken:
            s[i] = -1e9
        i = int(np.argmax(s)); taken.add(i); out.append(i)
        acc = acc + unit[i]; cnt += 1
    return out


def rank_svsf(qv, unit, k, q0_in_trail=True, alpha=ALPHA, beta=BETA):
    qn = _n(qv); sims_q = unit @ qn
    acc = np.zeros_like(qn); cnt = 0
    out, taken = [], set()
    for _ in range(k):
        s = alpha * sims_q + (beta * (unit @ _n(acc / cnt)) if cnt else 0.0)
        s = np.array(s, dtype=float) if not np.isscalar(s) else alpha * sims_q.copy()
        for i in taken:
            s[i] = -1e9
        i = int(np.argmax(s)); taken.add(i); out.append(i)
        dep = (qn + unit[i]) * 0.5 if q0_in_trail else unit[i]
        acc = acc + dep; cnt += 1
    return out


METHODS = [
    ("Vanilla",     lambda qv, u, k: rank_vanilla(qv, u, k)),
    ("PRF-Rocchio", lambda qv, u, k: rank_prf_rocchio(qv, u, k)),
    ("PRF-iter",    lambda qv, u, k: rank_prf_iter(qv, u, k)),
    ("PRF-seq",     lambda qv, u, k: rank_prf_seq(qv, u, k)),
    ("SVSF (pub.)", lambda qv, u, k: rank_svsf(qv, u, k, True)),
    ("SVSF-noQ0",   lambda qv, u, k: rank_svsf(qv, u, k, False)),
]


def boot_ci(d, nb=3000):
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return 0.0, 0.0, 0.0
    m = [np.mean(np.random.choice(d, d.size, replace=True)) for _ in range(nb)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ══════════════ 4) KOSU ══════════════
def run(dname, data, has_hop_order):
    print(f"\n{'='*88}\n{dname} | soru-basina corpus | hop-sirali bridge: "
          f"{'VAR' if has_hop_order else 'YOK (raporlanmaz)'}\n{'='*88}")
    sf = {k: {m: [] for m, _ in METHODS} for k in KS}
    fc = {k: {m: [] for m, _ in METHODS} for k in KS}
    br = {k: {m: [] for m, _ in METHODS} for k in KS}
    ng, used = [], 0

    for qi, rec in enumerate(data[:N_QUESTIONS]):
        q, paras, hops, order = rec
        texts = [f"Title: {t}\n{x}" for t, x, _ in paras]
        gold = {i for i, (_, _, g) in enumerate(paras) if g}
        if not gold or len(texts) <= KMAX:
            continue
        nodes, embs = embed_pool_once(texts, batch_size=16)
        unit = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        qv = model.encode("query: " + q)
        ng.append(len(gold)); used += 1
        last_hop = order[-1] if has_hop_order else None

        for mname, fn in METHODS:
            ranked = fn(qv, unit, KMAX)
            for k in KS:
                sel = set(ranked[:k])
                sf[k][mname].append(len(sel & gold) / len(gold))
                fc[k][mname].append(1.0 if gold.issubset(sel) else 0.0)
                if has_hop_order:
                    br[k][mname].append(1.0 if last_hop in sel else 0.0)
        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{min(N_QUESTIONS,len(data))}...")

    print(f"  islenen soru: {used} | ort. gold/soru: {np.mean(ng):.2f}")
    blocks = [("SF-Recall", sf), ("Full-chain recall", fc)]
    if has_hop_order:
        blocks.append(("Bridge recall (GERCEK son hop)", br))

    for title, store in blocks:
        print(f"\n  ── {title} ──")
        print(f"  {'Method':<14}" + "".join(f"@{k}{'':<8}" for k in KS))
        for mname, _ in METHODS:
            print(f"  {mname:<14}" + "".join(f"{np.mean(store[k][mname]):<10.3f}" for k in KS))
        # TUM k'larda anlamlilik: Paper 1'in calisma noktasi n_docs=4, bu yuzden
        # yalnizca en siki butceye bakmak yaniltici olur (SVSF'in izi k=2'de devrede degil).
        for kx in KS:
            base = store[kx]["SVSF (pub.)"]
            print(f"    -- @{kx}, SVSF(pub.) referans --")
            for mname, _ in METHODS:
                if mname == "SVSF (pub.)":
                    continue
                d = [x - y for x, y in zip(store[kx][mname], base)]
                try:
                    _, p = wilcoxon(d) if any(v != 0 for v in d) else (0, 1.0)
                except Exception:
                    p = 1.0
                md, lo, hi = boot_ci(d)
                print(f"      {mname:<14}{md:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.4f} "
                      f"{'ANLAMLI' if p < 0.05 else 'fark yok'}")


# ── DOGRULAMA MODU ──
if VERIFY_ONLY:
    print("### VERIFY MODU — paragraph_support_idx yapisi kontrol ediliyor ###")
    if not os.path.exists(MUSIQUE_PATH):
        print(f"!!! {MUSIQUE_PATH} bulunamadi.")
    else:
        load_musique_hopordered(MUSIQUE_PATH, verify_n=3)
    print("\n### Yukaridaki ciktida 'paragraph_support_idx' gercek bir tamsayi ise")
    print("### VERIFY_ONLY = False yapip tam kosuyu baslat.")
else:
    # MuSiQue — gercek hop sirali bridge ile
    if os.path.exists(MUSIQUE_PATH):
        mq = load_musique_hopordered(MUSIQUE_PATH)
        random.Random(SEED).shuffle(mq)
        run("MuSiQue", mq, has_hop_order=True)
    # HotpotQA + 2Wiki — TWO STAGES loader'i uzerinden
    for src in ("HotpotQA", "2WikiMultihopQA"):
        try:
            data = load_from_mixed(src, N_QUESTIONS)
            if data:
                run(src, data, has_hop_order=False)
        except Exception as e:
            print(f"!!! {src} yuklenemedi: {str(e)[:200]}")

    print(f"\n{'#'*88}")
    print("MAKALEYE NOT: Bridge recall yalnizca MuSiQue'de raporlanir (gercek hop sirasi")
    print("  question_decomposition'dan gelir). HotpotQA/2Wiki'de supporting_facts SIRASIZ")
    print("  oldugu icin 'son hop' tanimlanamaz; o setlerde SF-Recall + Full-chain verilir.")
    print("  Onceki 'query'ye en uzak gold' vekili DONGUSELDI (Vanilla icin bridge==full-chain)")
    print("  ve KULLANILMAMISTIR.")
