# PAPER: §4.5 — query:/passage: prefix sensitivity
"""
PREFIX SAPMA TESTI — bge-m3 query/passage prefix asimetrisi recall'u etkiliyor mu?
Sira: app_twostage -> fast_nodes_patch -> (paper21_runner dosya) -> BU

Uc konfigurasyon, AYNI havuz, AYNI sorular, sadece Vanilla recall@8:
  CFG-1 mevcut:      sorgu="query: "+q   havuz="Title: ..."       (asimetrik — su anki kod)
  CFG-2 prefixsiz:   sorgu=q             havuz="Title: ..."       (bge-m3 resmi kullanimi)
  CFG-3 E5-tarzi:    sorgu="query: "+q   havuz="passage: Title:.."(tam E5 konvansiyonu)

Karar:
  CFG-2/CFG-3 >> CFG-1  -> asimetri GERCEK bir sapma, makaleler yeniden uretilmeli
  ucu de ~ ayni         -> bge-m3 prefixe duyarsiz, mevcut sonuclar gecerli
"""
import os
import numpy as np
from scipy.stats import wilcoxon

os.environ["PAPER21_NORUN"] = "1"
try:
    load_musique; balance_by_hop; N_PER_HOP
except NameError:
    from paper21_runner import load_musique, balance_by_hop, N_PER_HOP

N_RETRIEVE = 8
N_PER_HOP_LOCAL = 100   # 300 soru (2/3/4-hop x100)

def build_pool_texts(questions, passage_prefix=""):
    """havuz metinlerini opsiyonel prefix ile kur; gold indeks eslemesi ayni kalir."""
    seen={}; pool=[]; qgold=[]
    for q,paras,hops in questions:
        gi=[]
        for t,x,g in paras:
            key=(t,x)
            if key not in seen:
                seen[key]=len(pool)
                pool.append(f"{passage_prefix}Title: {t}\n{x}")
            if g: gi.append(seen[key])
        qgold.append(sorted(set(gi)))
    return pool,qgold

def encode_pool(texts):
    """embed_pool_once yerine dogrudan model.encode — prefix'i biz kontrol edelim.
    embed_pool_once metne kendi bir sey eklemiyorsa onu da kullanabiliriz; guvenli yol: model.encode."""
    embs = np.array([model.encode(t) for t in texts])
    return embs

def vanilla_recall(qvec, embs, gold_set, k=N_RETRIEVE):
    qn = qvec/(np.linalg.norm(qvec)+1e-9)
    norms = np.linalg.norm(embs,axis=1)+1e-9
    sims = (embs@qn)/norms
    top = set(int(i) for i in np.argsort(sims)[::-1][:k])
    return len(top & gold_set)/len(gold_set) if gold_set else 0.0

def run_cfg(name, questions, query_prefix, passage_prefix):
    pool,qgold = build_pool_texts(questions, passage_prefix)
    # havuzu bir kez encode et (bu cfg'nin passage_prefix'i ile)
    embs = encode_pool(pool)
    recs=[]
    for (q,paras,hops),gi in zip(questions,qgold):
        G=set(gi)
        if not G: continue
        qtext = query_prefix + q
        qvec = model.encode(qtext)
        recs.append(vanilla_recall(qvec, embs, G))
    return np.array(recs)

# MuSiQue (asıl şüpheli set) + istersen MoreHopQA da eklenebilir
questions = balance_by_hop(load_musique("musique_ans_train.jsonl"), N_PER_HOP_LOCAL)
print(f"MuSiQue: {len(questions)} soru\n")

cfgs = [
    ("CFG-1 mevcut (asimetrik)",   "query: ", ""),
    ("CFG-2 prefixsiz (bge-m3 resmi)", "",     ""),
    ("CFG-3 E5-tarzi (simetrik)",  "query: ", "passage: "),
]
results={}
for name,qp,pp in cfgs:
    r = run_cfg(name, questions, qp, pp)
    results[name]=r
    print(f"{name:<34} Vanilla recall@8 = {r.mean():.4f}")

print("\n--- Farklar (CFG-1 mevcut'a gore) ---")
base = results["CFG-1 mevcut (asimetrik)"]
for name,r in results.items():
    if name.startswith("CFG-1"): continue
    diff = r - base
    try: _,p = wilcoxon(diff) if np.any(diff!=0) else (0,1.0)
    except Exception: p=1.0
    print(f"{name:<34} Δ={diff.mean():+.4f}  p={p:.4f}  {'ANLAMLI' if p<0.05 else 'fark yok'}")

print("\nKARAR:")
print("  CFG-2/CFG-3, CFG-1'den ANLAMLI yuksekse -> prefix asimetrisi GERCEK sapma,")
print("     makale 1 & 2.1 mutlak sayilari yeniden uretilmeli (goreli bulgular muhtemelen ayakta).")
print("  Ucu de fark yoksa -> bge-m3 prefixe duyarsiz, mevcut sonuclar gecerli, sorun yok.")
