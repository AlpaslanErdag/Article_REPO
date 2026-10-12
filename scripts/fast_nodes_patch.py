"""
BATCH-ENCODING YAMASI — GPU patlamasinin cozumu.
ChunkNode her chunk'i TEK TEK encode ediyordu (yuzlerce ayri GPU cagrisi).
Bu yama: tum chunk'lari TEK model.encode(list) cagrisiyla topluca encode eder.
app_twostage hucresinden SONRA calistir.
"""
import numpy as np

class FastNode:
    """ChunkNode ile ayni alanlar, ama embedding disaridan verilir (batch)."""
    __slots__ = ("node_id", "text", "text_embedding", "is_gold", "gold_idx", "source_title")
    def __init__(self, node_id, text, embedding, source_title=None):
        self.node_id = node_id
        self.text = text
        self.text_embedding = embedding
        self.is_gold = False
        self.gold_idx = None
        self.source_title = source_title

def _get_model():
    """model'i once bu modulde, sonra __main__ (Colab global) namespace'inde arar."""
    import sys
    g = globals()
    if "model" in g and g["model"] is not None:
        return g["model"]
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "model"):
        return main.model
    try:
        import builtins
        if hasattr(builtins, "model"):
            return builtins.model
    except Exception:
        pass
    raise NameError("embedder 'model' bulunamadi. app_twostage hucresini calistirdin mi? "
                    "Alternatif: bu modulu app'ten SONRA ayni hucrede/global'de calistir.")

def build_nodes_fast(corpus_text, contextual_mode="off", sem_thr=0.3, batch_size=64):
    """build_nodes'un batch-encoding versiyonu. 10-50x hizli, GPU-dostu."""
    import sys
    _main = sys.modules.get("__main__")
    model = _get_model()
    _sem = globals().get("semantic_chunking") or getattr(_main, "semantic_chunking", None)
    raw = _sem(corpus_text, similarity_threshold=sem_thr)
    if contextual_mode != "off":
        _ctx = globals().get("contextualize_chunks_fixed") or getattr(_main, "contextualize_chunks_fixed", None)
        raw = _ctx(raw, corpus_text, mode=contextual_mode)
    if not raw:
        return [], None
    # TEK batch cagrisi (tek tek degil!)
    embs = model.encode(["passage: " + t for t in raw],
                        batch_size=batch_size, show_progress_bar=False)
    embs = np.asarray(embs)
    nodes = [FastNode(f"C_{i}", t, embs[i]) for i, t in enumerate(raw)]
    return nodes, embs

def embed_pool_once(paragraphs, batch_size=64):
    """Paylasilan havuz icin: paragraf listesini TEK SEFERDE embed et.
    Dondurur: (nodes, embs). Sorular arasi yeniden kullanilir."""
    model = _get_model()
    embs = model.encode(["passage: " + p for p in paragraphs],
                        batch_size=batch_size, show_progress_bar=True)
    embs = np.asarray(embs)
    nodes = [FastNode(f"P_{i}", p, embs[i]) for i, p in enumerate(paragraphs)]
    return nodes, embs

print("fast_nodes yamasi hazir: build_nodes_fast / embed_pool_once / FastNode")
