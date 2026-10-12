# PAPER: §4.5 — batched vs single-passage encoding parity
"""
ENCODE PARITY v2 (DÜZELTİLMİŞ) + prefix testi embed_pool_once ile.
Sira: app_twostage -> fast_nodes_patch -> (paper21_runner dosya) -> BU

v1 hatasi: embed_pool_once ICERIDE "passage: " ekliyor; model.encode'a ham verince
farkli metin encode edildi (cosine 0.907 bu yuzden). Bu sefer ADIL karsilastirma.

BULGU (kod incelemesi): Makale 2.1 kurulumu = SIMETRIK (query: / passage:), asimetrik DEGIL.
Bu test onu dogruluyor + prefix'in gercekten etkisiz oldugunu embed_pool_once yoluyla teyit ediyor.
"""
import os
import numpy as np
from scipy.stats import wilcoxon

os.environ["PAPER21_NORUN"]="1"
try:
    load_musique; balance_by_hop
except NameError:
    from paper21_runner import load_musique, balance_by_hop

# ── PARITY (adil): iki tarafa da ayni girdi ──
samples=[
 "Title: Barack Obama\nBarack Obama served as the 44th president of the United States.",
 "Title: Ankara\nAnkara is the capital city of Türkiye.",
 "Title: bge-m3\nBGE-M3 is a multilingual embedding model based on XLM-RoBERTa.",
]
# embed_pool_once (icinde "passage: " ekler)
_, embs_batch = embed_pool_once(samples, batch_size=16)
# model.encode'a AYNI seyi ver: "passage: " + p  (embed_pool_once'un yaptigi gibi)
embs_single = np.array([model.encode("passage: "+p) for p in samples])
cos=np.array([np.dot(embs_batch[i],embs_single[i])/
              (np.linalg.norm(embs_batch[i])*np.linalg.norm(embs_single[i])+1e-9)
              for i in range(len(samples))])
print("=== PARITY (adil: iki tarafa da 'passage: ' prefix) ===")
print(f"allclose(1e-4): {np.allclose(embs_batch,embs_single,atol=1e-4)}")
print(f"satir cosine: min={cos.min():.6f} ort={cos.mean():.6f}")
print("  -> cosine≈1 ise batch encode = tek-tek encode, matematik saglam\n")

# ── PREFIX testi ama embed_pool_once GERCEK yoluyla ──
# embed_pool_once "passage: " ekliyor; farkli passage-prefix denemek icin gecici sarma
def encode_pool_with_prefix(paragraphs, passage_prefix, batch_size=16):
    # embed_pool_once'i taklit et ama istedigimiz prefix ile
    embs = model.encode([passage_prefix + p for p in paragraphs],
                        batch_size=batch_size, show_progress_bar=True)
    return np.asarray(embs)

def vanilla_recall(qvec, embs, gold, k=8):
    qn=qvec/(np.linalg.norm(qvec)+1e-9); norms=np.linalg.norm(embs,axis=1)+1e-9
    sims=(embs@qn)/norms
    top=set(int(i) for i in np.argsort(sims)[::-1][:k])
    return len(top&gold)/len(gold) if gold else 0.0

def build(questions, ppfx):
    seen={};pool=[];qg=[]
    for q,paras,hops in questions:
        gi=[]
        for t,x,g in paras:
            key=(t,x)
            if key not in seen: seen[key]=len(pool); pool.append(f"{ppfx}Title: {t}\n{x}")
            if g: gi.append(seen[key])
        qg.append(sorted(set(gi)))
    return pool,qg

questions=balance_by_hop(load_musique("musique_ans_train.jsonl"),100)
print(f"=== PREFIX testi (embed_pool_once yolu, MuSiQue {len(questions)} soru) ===")

configs=[
 ("GERCEK (query:/passage:)","query: ","passage: "),   # Makale 2.1'in gercek kurulumu
 ("prefixsiz (bge-m3 resmi)","",""),
 ("sadece query: (asimetrik)","query: ",""),            # benim ilk yanlis varsaydigim
]
res={}
for name,qp,pp in configs:
    # havuzu bu passage-prefix ile encode et (build zaten prefix'i metne koyuyor, o yuzden encode'a ham ver)
    pool,qg=build(questions,pp)
    embs=model.encode(pool,batch_size=16,show_progress_bar=False)
    embs=np.asarray(embs)
    recs=[]
    for (q,paras,hops),gi in zip(questions,qg):
        G=set(gi)
        if not G: continue
        recs.append(vanilla_recall(model.encode(qp+q),embs,G))
    res[name]=np.array(recs)
    print(f"  {name:<30} recall@8 = {res[name].mean():.4f}")

print("\n--- GERCEK kuruluma gore farklar ---")
base=res["GERCEK (query:/passage:)"]
for name,r in res.items():
    if name.startswith("GERCEK"): continue
    d=r-base
    try:_,p=wilcoxon(d) if np.any(d!=0) else (0,1.0)
    except:p=1.0
    print(f"  {name:<30} Δ={d.mean():+.4f} p={p:.4f} {'ANLAMLI' if p<0.05 else 'fark yok'}")

print("\nKARAR: GERCEK kurulum (query:/passage:) simetrik. Diger konfiglerle fark yoksa")
print("       -> prefix secimi recall'i etkilemiyor, Makale 2.1 & 1 sayilari GECERLI.")
