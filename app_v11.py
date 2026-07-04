# ============================================================
# app_final_v11.py — SVSF Benchmark Suite (A100 Optimized, Vectorized & Strict EM)
# Port: 5010 | History: history_v11.csv + history_v11.json
# Baseline: Vanilla, MMR, Cross-Encoder, Multi-Query | UI: index_v11 + history_v11
# Metric: Substring Exact Match (SQuAD-norm) + retrieval latency | Test: McNemar/Binomial
# ============================================================

import os
import csv
import json
import logging
import re
import string
import random
import threading
import time
import io
from itertools import product
from math import prod
from datetime import datetime, timezone, timedelta

import numpy as np
import requests
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
from rank_bm25 import BM25Okapi

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    CUDA_AVAILABLE = False

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ─── 1. HARDWARE MEASUREMENT (TIMED RETRIEVE - MS ONLY) ─────────────────────────
def timed_retrieve(fn, *args, warmup=2, runs=10, **kw):
    """Wrapper to measure retrieval steps in milliseconds with absolute precision."""
    for _ in range(warmup):
        fn(*args, **kw)
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
        
    t0 = time.perf_counter()
    for _ in range(runs):
        out = fn(*args, **kw)
        
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
        
    dt_ms = (time.perf_counter() - t0) / runs * 1000.0
    return out, dt_ms

# ─── 2. STRICT EXACT MATCH AT SQUAD STANDARDS ────────────────────────────
def normalize_answer(s):
    """SQuAD-style normalization: remove articles, remove punctuation, lowercase."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))

def compute_exact_match(predicted_answer, ground_truth):
    """
    PRIMARY METRIC — strict (single-direction) substring EM.
    Basis of the paper's 'deterministic, no partial credit' claim; SHOULD NOT BE CHANGED.
    Does the ground truth appear as a contiguous sub-string (word-boundary) in the output?
    """
    if not predicted_answer or not ground_truth:
        return 0
    pred_clean = normalize_answer(predicted_answer)
    truth_clean = normalize_answer(ground_truth)
    if re.search(r'\b' + re.escape(truth_clean) + r'\b', pred_clean):
        return 1
    return 0

def compute_em_flex(predicted_answer, ground_truth):
    """
    SECONDARY — Bi-directional substring. Still DETERMINISTIC and THRESHOLD-LESS (no partial credit).
    Is the shorter string a contiguous substring of the longer one? Checks both directions.
    Recovers cases like 'The actor is John Travolta' ↔ 'John Travolta'.
    (Note: 'John Travolta' → 'John Joseph Travolta' will NOT be recovered — words are not contiguous.)
    """
    if not predicted_answer or not ground_truth:
        return 0
    p = normalize_answer(predicted_answer)
    t = normalize_answer(ground_truth)
    if not p or not t:
        return 0
    short, long = (t, p) if len(t) <= len(p) else (p, t)
    if re.search(r'\b' + re.escape(short) + r'\b', long):
        return 1
    return 0

def compute_f1(predicted_answer, ground_truth):
    """
    DIAGNOSTIC — SQuAD token-F1. Gives PARTIAL CREDIT (0.0–1.0 float).
    Prevents inflation in long/loose outputs by looking at precision and recall:
    'John Travolta' vs 'John Joseph Travolta' → 0.80 (captured),
    '11 February 1929' vs long incorrect text → 0.00 (rejected).
    NOT the primary decision metric; an extra signal to see what EM missed.
    """
    if not predicted_answer or not ground_truth:
        return 0.0
    pred_tokens = normalize_answer(predicted_answer).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = {}
    for w in truth_tokens:
        common[w] = common.get(w, 0) + 1
    num_same = 0
    for w in pred_tokens:
        if common.get(w, 0) > 0:
            num_same += 1
            common[w] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return round(2 * precision * recall / (precision + recall), 4)

# ─── 3. MODELS (BGE-M3 & CROSS-ENCODER) ────────────────────────────────────
print("[v11] Loading models...")
def _pick_device(prefer_cuda=True):
    return "cuda" if (prefer_cuda and CUDA_AVAILABLE) else "cpu"

_dev = _pick_device(True)
model = SentenceTransformer("BAAI/bge-m3", device=_dev)

try:
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=_dev)
    RERANK_AVAILABLE = True
except Exception:
    cross_encoder = None
    RERANK_AVAILABLE = False

# Powerful/updated reranker (BGE-reranker-v2-m3) — second baseline.
# Same CrossEncoder API; activation is sigmoid (scores 0-1).
try:
    from sentence_transformers import CrossEncoder as _CE2
    cross_encoder_bge = _CE2("BAAI/bge-reranker-v2-m3", device=_dev, max_length=512)
    RERANK_BGE_AVAILABLE = True
except Exception as _e:
    cross_encoder_bge = None
    RERANK_BGE_AVAILABLE = False
    logging.warning("BGE-reranker could not be loaded: %s", _e)

# ─── HISTORY CONFIGURATION ──────────────────────────────────────────────────
HISTORY_JSON_PATH = "history_v11.json"
HISTORY_CSV_PATH = "history_v11.csv"
TR_TZ = timezone(timedelta(hours=3))

CSV_HEADER = [
    "ts", "ts_pretty", "AutomationId", "RunIndex",
    "Model", "Agents", "Alpha", "Beta", "Threshold",
    "chunk_size", "chunk_overlap", "use_hetero_scout", "chunking_method", "manifold_threshold", "neg_gamma",
    "Source", "Query", "suggested_answer", "content",
    "Vanilla_Time_ms", "SVSF_Time_ms", "MMR_Time_ms", "Rerank_Time_ms", "RerankBGE_Time_ms", "MultiQuery_Time_ms",
    "Vanilla_EM", "SVSF_EM", "MMR_EM", "Rerank_EM", "RerankBGE_EM", "MultiQuery_EM",
    "Vanilla_EM_flex", "SVSF_EM_flex", "MMR_EM_flex", "Rerank_EM_flex", "RerankBGE_EM_flex", "MultiQuery_EM_flex",
    "Vanilla_F1", "SVSF_F1", "MMR_F1", "Rerank_F1", "RerankBGE_F1", "MultiQuery_F1",
    "Vanilla_Judge", "SVSF_Judge", "MMR_Judge", "Rerank_Judge", "RerankBGE_Judge", "MultiQuery_Judge",
    "Vanilla_LLM", "SVSF_LLM", "MMR_LLM", "Rerank_LLM", "RerankBGE_LLM", "MultiQuery_LLM"
]

AUTOM_LOCK = threading.Lock()
_automation_thread = None
_automation_state = {
    "running": False, "progress": 0, "total": 0, "total_runs": 0,
    "completed_runs": 0, "last_run": None, "last_error": None,
}

def _now_tr_iso(): return datetime.now(TR_TZ).isoformat(timespec="seconds")
def _now_tr_pretty(): return datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M:%S")

def append_history(csv_row_dict: dict, json_record_snake: dict = None):
    ts = csv_row_dict.get("ts") or _now_tr_iso()
    ts_p = csv_row_dict.get("ts_pretty") or _now_tr_pretty()
    csv_row_dict.setdefault("ts", ts)
    csv_row_dict.setdefault("ts_pretty", ts_p)

    if json_record_snake:
        rec = dict(json_record_snake)
        rec.setdefault("ts", csv_row_dict["ts"])
        rec.setdefault("ts_pretty", csv_row_dict["ts_pretty"])
        try:
            history_data = []
            if os.path.exists(HISTORY_JSON_PATH):
                with open(HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
                    history_data = json.load(f) or []
            history_data.append(rec)
            with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error("JSON could not be written: %s", e)

    file_exists = os.path.isfile(HISTORY_CSV_PATH)
    row = [csv_row_dict.get(c, "") for c in CSV_HEADER]
    try:
        with open(HISTORY_CSV_PATH, "a", encoding="utf-8", newline="") as wf:
            writer = csv.writer(wf, quoting=csv.QUOTE_MINIMAL)
            if not file_exists:
                writer.writerow(CSV_HEADER)
            writer.writerow(row)
    except Exception as e:
        logging.error("CSV could not be written: %s", e)

# ─── 4. VECTORIZED SEARCH (RETRIEVAL) FUNCTIONS ───────────────────
def sklearn_cosine_wrapper(v1, mat):
    return sklearn_cosine_sim(v1.reshape(1, -1), mat)[0]

def svsf_search_step(query_vec, all_embs, nodes, bm25_norm_scores, alpha_val, beta_val, num_agents, relevance_threshold, use_hetero_scout, manifold_threshold=0.80, neg_gamma=None):
    """Fully vectorized, O(N) pure numpy loop."""
    num_nodes = len(nodes)
    active_mask = np.ones(num_nodes, dtype=bool)
    global_pheromone = np.zeros_like(query_vec)
    # NEGATIVE PHEROMONE (repellent): Active when neg_gamma>0. Accumulates the center
    # of near-duplicate noise nodes filtered out by manifold; subsequent agents are
    # REPELLED from this region (subtracted from score). gamma=0 → fully disabled (legacy behavior).
    # If parameter is not provided, reads from env (backward compatibility).
    if neg_gamma is None:
        neg_gamma = float(os.environ.get("SVSF_NEG_GAMMA", "0.0"))
    neg_pheromone = np.zeros_like(query_vec)
    svsf_nodes = []
    svsf_log = []
    
    # Query embedding is static, calculate once
    att_scores_dense = sklearn_cosine_wrapper(query_vec, all_embs)

    for agent_idx in range(num_agents):
        if not np.any(active_mask): break
        is_lexical_scout = use_hetero_scout and agent_idx == 0

        # Attraction
        att = bm25_norm_scores if is_lexical_scout else att_scores_dense
        
        # Repulsion (Pheromone) — positive trace (attraction)
        if np.all(global_pheromone == 0):
            rep = np.zeros(num_nodes)
        else:
            rep = sklearn_cosine_wrapper(global_pheromone, all_embs)

        # Negative pheromone (repulsion): PENALIZE proximity to filtered noise regions
        if neg_gamma > 0.0 and not np.all(neg_pheromone == 0):
            neg = sklearn_cosine_wrapper(neg_pheromone, all_embs)
        else:
            neg = np.zeros(num_nodes)

        # Dual-Attractor + negative trace
        scores = (alpha_val * att) + (beta_val * rep) - (neg_gamma * neg)
        
        # Mask those used or triggered by manifold constraints to -999
        masked_scores = np.where(active_mask, scores, -999.0)
        best_idx = int(np.argmax(masked_scores))
        best_score = masked_scores[best_idx]

        if best_score < relevance_threshold:
            svsf_log.append(f"Agent {agent_idx+1}: Below threshold (Score: {best_score:.4f}).")
            break
            
        best_node = nodes[best_idx]
        svsf_nodes.append(best_node)
        svsf_log.append(f"Agent {agent_idx+1}: Successful hop (Score: {best_score:.4f}).")
        
        # Manifold Blurring Constraint: mask similar items above threshold.
        # If manifold_threshold >= 1.0, mask is INEFFECTIVE (bridge-documents not filtered).
        if manifold_threshold < 1.0:
            sim_to_best = sklearn_cosine_wrapper(best_node.text_embedding, all_embs)
            eliminated = active_mask & (sim_to_best >= manifold_threshold)
            eliminated[best_idx] = False  # The chosen node itself is not negative
            # Add center of eliminated near-duplicates to negative pheromone
            if neg_gamma > 0.0 and np.any(eliminated):
                neg_pheromone = neg_pheromone + 0.2 * all_embs[eliminated].mean(axis=0)
            active_mask = active_mask & (sim_to_best < manifold_threshold)
        else:
            active_mask[best_idx] = False
        
        # Pheromone Update (positive trace)
        global_pheromone = global_pheromone + 0.2 * ((query_vec + best_node.text_embedding) / 2.0)
        
    return svsf_nodes, svsf_log

def vanilla_search_step(query_vec, all_embs, nodes, num_agents):
    sims = sklearn_cosine_wrapper(query_vec, all_embs)
    return [nodes[i] for i in np.argsort(sims)[-num_agents:][::-1]]

def rerank_search_step(query_vec, query_text, all_embs, nodes, num_agents, cross_model):
    if not cross_model: return []
    sims = sklearn_cosine_wrapper(query_vec, all_embs)
    top_idx = np.argsort(sims)[-50:][::-1]
    top_nodes = [nodes[i] for i in top_idx]
    
    pairs = [[query_text, n.text] for n in top_nodes]
    scores = cross_model.predict(pairs)
    best_idx = np.argsort(scores)[-num_agents:][::-1]
    return [top_nodes[i] for i in best_idx]

def mmr_search_step(query_vec, all_embs, nodes, num_agents, lambda_param=0.5):
    """Vectorized Maximum Marginal Relevance (MMR) search."""
    if len(nodes) == 0:
        return []
    sims = sklearn_cosine_wrapper(query_vec, all_embs)
    selected_idx = [int(np.argmax(sims))]
    while len(selected_idx) < num_agents and len(selected_idx) < len(nodes):
        unselected_idx = [i for i in range(len(nodes)) if i not in selected_idx]
        if not unselected_idx:
            break
        # Maximum similarity of unselected items to the selected ones
        sim_to_selected = sklearn_cosine_sim(all_embs[unselected_idx], all_embs[selected_idx])
        max_sim_to_selected = np.max(sim_to_selected, axis=1)
        # MMR: (lambda * query similarity) - ((1-lambda) * similarity between items)
        mmr_scores = (lambda_param * sims[unselected_idx]) - ((1 - lambda_param) * max_sim_to_selected)
        selected_idx.append(unselected_idx[int(np.argmax(mmr_scores))])
    return [nodes[i] for i in selected_idx]

def multiquery_search_step(query_variants_embs, all_embs, nodes, num_agents):
    """MultiQuery: search with vector average of query variations (retrieval-only)."""
    if len(nodes) == 0 or len(query_variants_embs) == 0:
        return []
    combined_scores = np.zeros(len(nodes))
    for v_vec in query_variants_embs:
        combined_scores += sklearn_cosine_wrapper(v_vec, all_embs)
    best_idx = np.argsort(combined_scores)[-num_agents:][::-1]
    return [nodes[i] for i in best_idx]

# ─── BASIC NODE STRUCTURE AND CHUNKING ───────────────────────────────────────
class ChunkNode:
    def __init__(self, node_id, text):
        self.node_id = node_id
        self.text = text
        self.text_embedding = model.encode("passage: " + text)

def chunk_recursive(text, chunk_size=500, overlap=0):
    separators = ["\n\n", "\n", ". ", " "]
    def split_text(t, seps):
        if not seps or len(t) <= chunk_size: return [t] if t.strip() else []
        sep = seps[0]
        parts = t.split(sep)
        result, current = [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size: current = candidate
            else:
                if current: result.append(current)
                if len(part) > chunk_size: result.extend(split_text(part, seps[1:]))
                else: current = part
        if current: result.append(current)
        return result

    raw_chunks = split_text(text, separators)
    if overlap > 0 and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0].strip()]
        for i in range(1, len(raw_chunks)):
            overlapped.append((raw_chunks[i - 1][-overlap:] + " " + raw_chunks[i]).strip())
        return [c for c in overlapped if c]
    return [c.strip() for c in raw_chunks if c.strip()]

def semantic_chunking(text, similarity_threshold=0.60, max_chunk_chars=1200):
    """
    Chunking with semantic boundary detection. Splits sentences; starts new chunk
    when cosine similarity between consecutive sentences falls BELOW threshold.
    For Manifold Blurring thesis: boundaries are determined by semantic shift
    rather than character count (zero-overlap, natural semantic boundary).
    """
    # Split into sentences (simple but language-agnostic)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    # Sentence embeddings (with passage prefix, consistent with ChunkNode)
    embs = model.encode(["passage: " + s for s in sentences])
    embs = np.asarray(embs)

    chunks = []
    current = [sentences[0]]
    cur_len = len(sentences[0])
    for i in range(1, len(sentences)):
        sim = float(sklearn_cosine_sim(embs[i-1].reshape(1, -1), embs[i].reshape(1, -1))[0][0])
        # Semantic shift OR chunk too large → new chunk
        if sim < similarity_threshold or (cur_len + len(sentences[i]) > max_chunk_chars):
            chunks.append(" ".join(current).strip())
            current = [sentences[i]]
            cur_len = len(sentences[i])
        else:
            current.append(sentences[i])
            cur_len += len(sentences[i]) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]

# ─── LLM CALL ──────────────────────────────────────────────────────────
def get_ollama_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        return [m.get("name") for m in resp.json().get("models", []) if m.get("name")], None
    except Exception as e:
        return [], str(e)

def query_llm(model_name, prompt):
    if not model_name: return ""
    try:
        # CoT produces long output; num_predict must be sufficient otherwise <answer> is truncated.
        temp = float(os.environ.get("SVSF_LLM_TEMPERATURE", "0.0"))
        num_predict = int(os.environ.get("SVSF_LLM_NUM_PREDICT", "512"))
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name, "prompt": prompt, "stream": False,
                "options": {"temperature": temp, "num_predict": num_predict}
            },
            timeout=600,
        )
        return resp.json().get("response", "")
    except Exception as e:
        return f"LLM Error: {e}"

def make_prompt(nodes, query):
    # Handle repeated context (same node shouldn't appear twice)
    seen = set()
    unique_contexts = []
    for n in nodes:
        txt = getattr(n, 'parent_text', None) or n.text
        if txt not in seen:
            seen.add(txt)
            unique_contexts.append(f"- {txt}")
    context = "\n".join(unique_contexts)

    # Non-CoT mode (ablation): When SVSF_NO_COT=1, generator does NO reasoning,
    # gives direct short answer. This isolates raw retrieval quality by removing
    # generator's noise-filtering/compensation effect. Applied EQUALLY to all systems
    # (fair comparison maintained).
    no_cot = os.environ.get("SVSF_NO_COT", "0") == "1"
    if no_cot:
        return (
            "You are given evidence (CONTEXT) gathered from multiple documents; "
            "some of it may be noisy or irrelevant.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\n"
            "Write ONLY the final answer (an entity, name, date, or short phrase — "
            "no reasoning, no explanation, no full sentence) between <answer> and </answer>.\n"
            "If the answer is not in the context, write <answer>NOT FOUND</answer>.\n"
        )

    # CoT: reasoning first (<thinking>), then SHORT answer (<answer>).
    # EM only scores <answer> content (parsed via extract_answer),
    # so reasoning text doesn't pollute EM. Applied EQUALLY to all systems.
    return (
        "You are an expert analyst. You are given evidence (CONTEXT) gathered by "
        "autonomous agents from multiple documents; some of it may be noisy or irrelevant.\n\n"
        "TASK: Answer the QUESTION. First reason step by step:\n"
        "1. Identify which parts of the context are directly relevant to the question.\n"
        "2. Detect and IGNORE noisy/irrelevant ('distractor') passages.\n"
        "3. Connect the relevant facts into a logical chain to reach the answer.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\n"
        "First write your reasoning between <thinking> and </thinking>.\n"
        "Then write ONLY the final answer (an entity, name, date, or short phrase — "
        "no full sentence) between <answer> and </answer>.\n"
        "If the answer is definitely not in the context, write <answer>NOT FOUND</answer>.\n"
    )

def extract_answer(raw):
    """
    Extracts <answer>...</answer> content from LLM output (EM/F1/judge score this).
    If no tag, safe fallback: last line (answer is usually at the end in CoT).
    """
    if not raw:
        return ""
    m = re.search(r'<answer>(.*?)</answer>', raw, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: remove <thinking> block, take last meaningful line of remaining
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.IGNORECASE | re.DOTALL).strip()
    if cleaned:
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        if lines:
            return lines[-1]
    return raw.strip()

def llm_judge(judge_model, question, predicted_answer, ground_truth):
    """
    LLM-as-a-judge: evaluates if an answer is semantically correct regarding
    the ground truth. Retrieval context is NOT PROVIDED — judge only evaluates
    answer accuracy, not retrieval quality. Output is strictly parsed: 1 (correct) / 0 (wrong).

    To catch cases missed by EM/F1: can consider 'John Joseph Travolta' vs 
    'John Travolta' correct, despite surface differences.
    """
    if not judge_model or not predicted_answer or not ground_truth:
        return None  # judge failed/not applied → None (not 0; distinguishable)
    prompt = (
        "You are an impartial evaluation expert. Judge whether the PREDICTED answer "
        "to a question is semantically equivalent to the CORRECT answer.\n"
        "Surface expression differences (full name, abbreviation, extra words) do not matter; "
        "if the meaning is the same, it is correct. If the meaning differs or the prediction contains "
        "wrong/missing information, it is incorrect.\n\n"
        f"QUESTION: {question}\n"
        f"CORRECT ANSWER: {ground_truth}\n"
        f"PREDICTION: {predicted_answer}\n\n"
        "Answer with only one word: CORRECT or INCORRECT."
    )
    raw = query_llm(judge_model, prompt)
    if not raw or raw.startswith("LLM Error"):
        return None
    norm = raw.strip().upper()
    if norm.startswith("CORRECT") or norm.startswith("TRUE") or norm.startswith("YES"):
        return 1
    if norm.startswith("INCORRECT") or norm.startswith("FALSE") or norm.startswith("NO") or norm.startswith("WRONG"):
        return 0
    if "CORRECT" in norm or "TRUE" in norm:
        return 1
    return 0


# ─── SHARED CORPUS CACHE ──────────────────────────────
# If SVSF_SHARED_CORPUS_PATH is set, ALL questions are searched in this single
# large pool (real candidate-pool-bottleneck scenario). Pool is chunked &
# embedded ONCE and cached; not recalculated for every query.
_SHARED_CACHE = {}  # key: (path, use_semantic, sem_thr, chunk_size, overlap) -> (nodes, all_embs)

def _get_shared_pool(use_semantic, semantic_threshold, chunk_size, chunk_overlap):
    path = os.environ.get("SVSF_SHARED_CORPUS_PATH", "")
    if not path or not os.path.exists(path):
        return None
    key = (path, use_semantic, semantic_threshold, chunk_size, chunk_overlap)
    if key in _SHARED_CACHE:
        return _SHARED_CACHE[key]
    with open(path, encoding="utf-8") as f:
        pool_text = f.read()
    if use_semantic:
        raw_texts = semantic_chunking(pool_text, similarity_threshold=semantic_threshold)
    else:
        raw_texts = chunk_recursive(pool_text, chunk_size=chunk_size, overlap=chunk_overlap)
    nodes = [ChunkNode(f"P_{i}", txt) for i, txt in enumerate(raw_texts)]
    all_embs = np.vstack([n.text_embedding for n in nodes]) if nodes else None
    logging.info("Shared pool loaded: %d chunks (%s)", len(nodes), path)
    _SHARED_CACHE[key] = (nodes, all_embs)
    return _SHARED_CACHE[key]


# ─── MAIN SIMULATION LOGIC ───────────────────────────────────────────────
def perform_simulation_logic(
    corpus_text, query_text, suggested_answer, alpha_val, beta_val, num_agents,
    llm_model, chunk_size=500, chunk_overlap=0, relevance_threshold=0.2,
    use_hetero_scout=False, enable_vanilla=True, enable_rerank=True,
    enable_mmr=True, enable_multiquery=False,
    use_semantic_chunking=False, semantic_threshold=0.60,
    use_judge=False, judge_model="", manifold_threshold=0.80, neg_gamma=None
):
    corpus_text, query_text = (corpus_text or "").strip(), (query_text or "").strip()
    if not query_text:
        return {"error": "Query cannot be empty."}

    # Shared pool mode: all questions searched in a single large pool
    shared = _get_shared_pool(use_semantic_chunking, semantic_threshold, chunk_size, chunk_overlap)
    if shared is not None:
        nodes, all_embs = shared
        chunking_method = ("Semantic" if use_semantic_chunking else "Recursive") + " [SHARED POOL]"
        if not nodes:
            return {"error": "Could not generate chunks from shared pool."}
    else:
        if not corpus_text:
            return {"error": "Corpus and query cannot be empty."}
        if use_semantic_chunking:
            raw_texts = semantic_chunking(corpus_text, similarity_threshold=semantic_threshold)
            chunking_method = f"Semantic (thr={semantic_threshold})"
        else:
            raw_texts = chunk_recursive(corpus_text, chunk_size=chunk_size, overlap=chunk_overlap)
            chunking_method = f"Recursive ({chunk_size}/{chunk_overlap})"
        nodes = [ChunkNode(f"C_{i}", txt) for i, txt in enumerate(raw_texts)]
        if not nodes:
            return {"error": "No chunks generated from text."}
        all_embs = np.vstack([n.text_embedding for n in nodes])

    query_vec = model.encode("query: " + query_text)

    # Is LLM usable? (MMR/MQ blocks must NOT come before this)
    can_llm = bool(llm_model and "(Ollama" not in llm_model)

    bm25_norm_scores = np.zeros(len(nodes))
    if use_hetero_scout:
        tokenized_corpus = [n.text.lower().split() for n in nodes]
        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores = bm25.get_scores(query_text.lower().split())
        max_val, min_val = float(np.max(raw_scores)), float(np.min(raw_scores))
        diff = max_val - min_val if max_val > min_val else 1.0
        bm25_norm_scores = (raw_scores - min_val) / diff

    # --- SVSF TIMED RETRIEVAL (VECTORIZED) ---
    svsf_res, svsf_ms = timed_retrieve(
        svsf_search_step, query_vec, all_embs, nodes, bm25_norm_scores,
        alpha_val, beta_val, num_agents, relevance_threshold, use_hetero_scout,
        manifold_threshold, neg_gamma
    )
    svsf_nodes, svsf_log = svsf_res

    # --- VANILLA ---
    if enable_vanilla:
        van_nodes, v_ms = timed_retrieve(vanilla_search_step, query_vec, all_embs, nodes, num_agents)
    else:
        van_nodes, v_ms = [], 0.0

    # --- MMR ---
    if enable_mmr:
        mmr_nodes, m_ms = timed_retrieve(mmr_search_step, query_vec, all_embs, nodes, num_agents)
    else:
        mmr_nodes, m_ms = [], 0.0

    # --- CROSS-ENCODER (RERANKER, classic: MiniLM) ---
    if enable_rerank and RERANK_AVAILABLE:
        rerank_nodes, r_ms = timed_retrieve(
            rerank_search_step, query_vec, query_text, all_embs, nodes, num_agents, cross_encoder
        )
    else:
        rerank_nodes, r_ms = [], 0.0

    # --- CROSS-ENCODER (RERANKER, powerful/updated: BGE-reranker-v2-m3) ---
    if enable_rerank and RERANK_BGE_AVAILABLE:
        rerank_bge_nodes, rb_ms = timed_retrieve(
            rerank_search_step, query_vec, query_text, all_embs, nodes, num_agents, cross_encoder_bge
        )
    else:
        rerank_bge_nodes, rb_ms = [], 0.0

    # --- MULTI-QUERY ---
    # To SEPARATE LLM query variation generation time (LLM latency) from retrieval
    # (MIPS) time, we generate variations OUTSIDE timed_retrieve — fair measurement.
    if enable_multiquery and can_llm:
        variants_text = query_llm(
            llm_model,
            f"Rewrite this query in 2 different ways: {query_text}\nReturn ONLY queries."
        )
        variant_lines = [v.strip() for v in variants_text.split("\n") if v.strip()] + [query_text]
        variants_embs = [model.encode("query: " + v) for v in variant_lines[:3]]
        mq_nodes, mq_ms = timed_retrieve(multiquery_search_step, variants_embs, all_embs, nodes, num_agents)
    else:
        mq_nodes, mq_ms = [], 0.0

    # --- LLM GENERATION & METRICS (strict EM + flex + F1 + optional judge) ---
    effective_judge = judge_model if judge_model else llm_model  # use main model if judge empty
    def evaluate(n_list):
        if not n_list or not can_llm:
            return "", 0, 0, 0.0, None
        raw = query_llm(llm_model, make_prompt(n_list, query_text))
        ans = extract_answer(raw)  # <answer> content — metrics score this
        em = compute_exact_match(ans, suggested_answer)       # primary
        em_flex = compute_em_flex(ans, suggested_answer)      # bi-directional, deterministic
        f1 = compute_f1(ans, suggested_answer)                # diagnostic, partial credit
        judge = llm_judge(effective_judge, query_text, ans, suggested_answer) if use_judge else None
        # Write extracted answer to LLM column (keep raw if audit is needed)
        return ans, em, em_flex, f1, judge

    s_llm, s_em, s_emf, s_f1, s_jd = evaluate(svsf_nodes)
    v_llm, v_em, v_emf, v_f1, v_jd = evaluate(van_nodes)
    m_llm, m_em, m_emf, m_f1, m_jd = evaluate(mmr_nodes)
    r_llm, r_em, r_emf, r_f1, r_jd = evaluate(rerank_nodes)
    rb_llm, rb_em, rb_emf, rb_f1, rb_jd = evaluate(rerank_bge_nodes)
    mq_llm, mq_em, mq_emf, mq_f1, mq_jd = evaluate(mq_nodes)

    def block(node_list, llm_txt, em, em_flex, f1, judge, ms, extra=None):
        d = {
            "chunks": [n.text for n in node_list], "llm": llm_txt,
            "exact_match": em, "em_flex": em_flex, "f1": f1, "judge": judge,
            "sdi": em, "faithfulness": "", "relevance": "",  # UI backward compatibility
            "time_ms": ms
        }
        if extra:
            d.update(extra)
        return d

    payload = {
        "svsf": block(svsf_nodes, s_llm, s_em, s_emf, s_f1, s_jd, svsf_ms, {"log": svsf_log}),
        "vanilla": block(van_nodes, v_llm, v_em, v_emf, v_f1, v_jd, v_ms) if enable_vanilla else None,
        "mmr": block(mmr_nodes, m_llm, m_em, m_emf, m_f1, m_jd, m_ms) if enable_mmr else None,
        "rerank": block(rerank_nodes, r_llm, r_em, r_emf, r_f1, r_jd, r_ms, {"rerank_available": RERANK_AVAILABLE}) if enable_rerank else None,
        "rerank_bge": block(rerank_bge_nodes, rb_llm, rb_em, rb_emf, rb_f1, rb_jd, rb_ms, {"rerank_available": RERANK_BGE_AVAILABLE}) if enable_rerank else None,
        "multiquery": block(mq_nodes, mq_llm, mq_em, mq_emf, mq_f1, mq_jd, mq_ms) if enable_multiquery else None,
        "pipeline_info": {
            "chunking_method": chunking_method, "after_bm25": len(nodes),
            "judge_model": effective_judge if use_judge else ""
        }
    }
    return payload

def _jv(judge_val):
    """Convert judge score to string for CSV: None → '' (not applied), 0/1 → '0'/'1'."""
    return "" if judge_val is None else str(int(judge_val))

def _run_record_to_append(res, data):
    if res.get("error"):
        return

    van  = res.get("vanilla") or {}
    svsf = res.get("svsf") or {}
    mmr  = res.get("mmr") or {}
    rrk  = res.get("rerank") or {}
    rbge = res.get("rerank_bge") or {}
    mq   = res.get("multiquery") or {}

    pipe = res.get("pipeline_info") or {}

    csv_row = {
        "AutomationId": data.get("automation_id", ""), "RunIndex": data.get("run_index", ""),
        "Model": data.get("model", ""), "Agents": data.get("agents", 4),
        "Alpha": data.get("alpha", 1.0), "Beta": data.get("beta", 1.0),
        "Threshold": data.get("relevance_threshold", 0.3),
        "chunk_size": data.get("chunk_size", 500), "chunk_overlap": data.get("chunk_overlap", 0),
        "use_hetero_scout": data.get("use_hetero_scout", True),
        "chunking_method": pipe.get("chunking_method", ""),
        "manifold_threshold": data.get("manifold_threshold", 0.80),
        "neg_gamma": data.get("neg_gamma", 0.0),
        "Source": data.get("source", ""),
        "Query": data.get("query", ""), "suggested_answer": data.get("suggested_answer", ""),
        "content": data.get("corpus", ""),
        "Vanilla_Time_ms": van.get("time_ms", 0), "SVSF_Time_ms": svsf.get("time_ms", 0),
        "MMR_Time_ms": mmr.get("time_ms", 0), "Rerank_Time_ms": rrk.get("time_ms", 0),
        "RerankBGE_Time_ms": rbge.get("time_ms", 0),
        "MultiQuery_Time_ms": mq.get("time_ms", 0),
        "Vanilla_EM": van.get("exact_match", 0), "SVSF_EM": svsf.get("exact_match", 0),
        "MMR_EM": mmr.get("exact_match", 0), "Rerank_EM": rrk.get("exact_match", 0),
        "RerankBGE_EM": rbge.get("exact_match", 0),
        "MultiQuery_EM": mq.get("exact_match", 0),
        "Vanilla_EM_flex": van.get("em_flex", 0), "SVSF_EM_flex": svsf.get("em_flex", 0),
        "MMR_EM_flex": mmr.get("em_flex", 0), "Rerank_EM_flex": rrk.get("em_flex", 0),
        "RerankBGE_EM_flex": rbge.get("em_flex", 0),
        "MultiQuery_EM_flex": mq.get("em_flex", 0),
        "Vanilla_F1": van.get("f1", 0.0), "SVSF_F1": svsf.get("f1", 0.0),
        "MMR_F1": mmr.get("f1", 0.0), "Rerank_F1": rrk.get("f1", 0.0),
        "RerankBGE_F1": rbge.get("f1", 0.0),
        "MultiQuery_F1": mq.get("f1", 0.0),
        "Vanilla_Judge": _jv(van.get("judge")), "SVSF_Judge": _jv(svsf.get("judge")),
        "MMR_Judge": _jv(mmr.get("judge")), "Rerank_Judge": _jv(rrk.get("judge")),
        "RerankBGE_Judge": _jv(rbge.get("judge")),
        "MultiQuery_Judge": _jv(mq.get("judge")),
        "Vanilla_LLM": van.get("llm", ""), "SVSF_LLM": svsf.get("llm", ""),
        "MMR_LLM": mmr.get("llm", ""), "Rerank_LLM": rrk.get("llm", ""),
        "RerankBGE_LLM": rbge.get("llm", ""),
        "MultiQuery_LLM": mq.get("llm", "")
    }

    # Interface (history) JSON record — honest field names (*_em, *_time_ms) + backward compatibility *_sdi
    snake = {
        "model": csv_row["Model"], "agents": csv_row["Agents"],
        "alpha": csv_row["Alpha"], "beta": csv_row["Beta"],
        "threshold": csv_row["Threshold"], "query": csv_row["Query"],
        "run_index": csv_row["RunIndex"],
        "chunk_size": csv_row["chunk_size"], "chunk_overlap": csv_row["chunk_overlap"],
        "use_hetero_scout": csv_row["use_hetero_scout"], "chunking_method": csv_row["chunking_method"],
        "manifold_threshold": csv_row["manifold_threshold"],
        "source": csv_row["Source"], "content": csv_row["content"],
        "svsf_em": svsf.get("exact_match", 0), "vanilla_em": van.get("exact_match", 0),
        "mmr_em": mmr.get("exact_match", 0), "rerank_em": rrk.get("exact_match", 0),
        "multiquery_em": mq.get("exact_match", 0),
        "svsf_em_flex": svsf.get("em_flex", 0), "vanilla_em_flex": van.get("em_flex", 0),
        "mmr_em_flex": mmr.get("em_flex", 0), "rerank_em_flex": rrk.get("em_flex", 0),
        "multiquery_em_flex": mq.get("em_flex", 0),
        "svsf_f1": svsf.get("f1", 0.0), "vanilla_f1": van.get("f1", 0.0),
        "mmr_f1": mmr.get("f1", 0.0), "rerank_f1": rrk.get("f1", 0.0),
        "multiquery_f1": mq.get("f1", 0.0),
        "svsf_judge": svsf.get("judge"), "vanilla_judge": van.get("judge"),
        "mmr_judge": mmr.get("judge"), "rerank_judge": rrk.get("judge"),
        "multiquery_judge": mq.get("judge"),
        "judge_model": (res.get("pipeline_info") or {}).get("judge_model", ""),
        "svsf_time_ms": svsf.get("time_ms", 0), "vanilla_time_ms": van.get("time_ms", 0),
        "mmr_time_ms": mmr.get("time_ms", 0), "rerank_time_ms": rrk.get("time_ms", 0),
        "multiquery_time_ms": mq.get("time_ms", 0),
        # Old UI compatibility
        "svsf_sdi": svsf.get("exact_match", 0), "vanilla_sdi": van.get("exact_match", 0),
        "rerank_sdi": rrk.get("exact_match", 0)
    }
    append_history(csv_row, snake)

# ─── DATASET FETCHING AND AUTOMATION (WORKER) ─────────────────────────────────────
def build_mixed_validation_set(n_sample: int = 5, sources=None) -> list:
    """
    Samples from HotpotQA, 2Wiki, MuSiQue validation sets.
    sources: list of sets to pull from, e.g., ["MuSiQue"] or
             ["HotpotQA","2WikiMultihopQA","MuSiQue"] (None → all).
    """
    from datasets import load_dataset
    if sources is None:
        sources = ["HotpotQA", "2WikiMultihopQA", "MuSiQue"]
    sources = set(sources)
    mixed = []

    def _try_load(repo_configs):
        """repo_configs: [(repo_id, config_or_None, split), ...] — returns first successful."""
        for repo, cfg, split in repo_configs:
            try:
                ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
                logging.info("[dataset] Loaded: %s (cfg=%s, split=%s, n=%d)", repo, cfg, split, len(ds))
                return ds
            except Exception as e:
                logging.warning("[dataset] '%s' (cfg=%s) failed: %s", repo, cfg, e)
        return None

    def _sample_indices(ds, n):
        return random.sample(range(len(ds)), min(n, len(ds)))

    # ── HotpotQA ──
    if "HotpotQA" in sources:
        ds = _try_load([
            ("hotpotqa/hotpot_qa", "distractor", "validation"),
            ("hotpot_qa", "distractor", "validation"),
        ])
        added = 0
        if ds is not None:
            for idx in _sample_indices(ds, n_sample):
                try:
                    item = ds[idx]
                    ctx = item.get("context", {}) or {}
                    titles = ctx.get("title", []) or []
                    sents = ctx.get("sentences", []) or []
                    parts = [f"Title: {t}\n" + " ".join(s) for t, s in zip(titles, sents)]
                    q, a = item.get("question", ""), item.get("answer", "")
                    if q and parts:
                        mixed.append({"query": q, "corpus": "\n\n".join(parts), "suggested_answer": a, "source": "HotpotQA"})
                        added += 1
                except Exception as e:
                    logging.warning("[HotpotQA] item parse error: %s", e)
        logging.info("[dataset] HotpotQA: %d samples added.", added)

    # ── 2WikiMultihopQA ──
    # NOTE: 'xanhho/2WikiMultihopQA' is script-based (.py) and does not load
    # in new datasets versions. Repos not requiring scripts (parquet/json) tried first.
    if "2WikiMultihopQA" in sources:
        ds = _try_load([
            ("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35", None, "train"),
            ("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35", None, "validation"),
            ("voidful/2WikiMultihopQA", None, "validation"),
            ("xanhho/2WikiMultihopQA", None, "validation"),
        ])
        added = 0
        if ds is not None:
            for idx in _sample_indices(ds, n_sample):
                try:
                    item = ds[idx]
                    q = item.get("question", "") or item.get("Question", "")
                    a = item.get("answer", "") or item.get("Answer", "")
                    ctx_parts = []
                    ctx = item.get("context", None)
                    if isinstance(ctx, str):
                        # Plain text context (gpt35 variant)
                        if ctx.strip():
                            ctx_parts.append(ctx.strip())
                    elif isinstance(ctx, (list, tuple)):
                        # HotpotQA-style: [[title, [sentences]], ...]
                        for c in ctx:
                            if isinstance(c, (list, tuple)) and len(c) >= 2:
                                title, sents = c[0], c[1]
                                sents_txt = " ".join(sents) if isinstance(sents, (list, tuple)) else str(sents)
                                ctx_parts.append(f"Title: {title}\n{sents_txt}")
                    elif isinstance(ctx, dict):
                        # {'title': [...], 'content'/'sentences': [...]}
                        titles = ctx.get("title", []) or []
                        sents = ctx.get("content", ctx.get("sentences", [])) or []
                        for t, s in zip(titles, sents):
                            s_txt = " ".join(s) if isinstance(s, (list, tuple)) else str(s)
                            ctx_parts.append(f"Title: {t}\n{s_txt}")
                    # If context is missing, try 'paragraphs' / 'passages' fields
                    if not ctx_parts:
                        for alt_key in ("paragraphs", "passages", "contexts"):
                            alt = item.get(alt_key)
                            if isinstance(alt, (list, tuple)):
                                for p in alt:
                                    if isinstance(p, dict):
                                        ctx_parts.append(f"Title: {p.get('title','')}\n{p.get('text', p.get('paragraph_text',''))}")
                                    elif isinstance(p, str):
                                        ctx_parts.append(p)
                                if ctx_parts:
                                    break
                    if q and ctx_parts:
                        mixed.append({"query": q, "corpus": "\n\n".join(ctx_parts), "suggested_answer": a, "source": "2WikiMultihopQA"})
                        added += 1
                except Exception as e:
                    logging.warning("[2Wiki] item parse error: %s", e)
        logging.info("[dataset] 2Wiki: %d samples added.", added)

    # ── MuSiQue ──
    if "MuSiQue" in sources:
        ds = _try_load([
            ("dgslibisey/MuSiQue", None, "validation"),
            ("bdsaglam/musique", "answerable", "validation"),
            ("bdsaglam/musique", None, "validation"),
        ])
        added = 0
        if ds is not None:
            for idx in _sample_indices(ds, n_sample):
                try:
                    item = ds[idx]
                    paras = item.get("paragraphs", []) or []
                    ctx_parts = []
                    for p in paras:
                        if isinstance(p, dict):
                            ctx_parts.append(f"Title: {p.get('title', '')}\n{p.get('paragraph_text', '')}")
                    q, a = item.get("question", ""), item.get("answer", "")
                    if q and ctx_parts:
                        mixed.append({"query": q, "corpus": "\n\n".join(ctx_parts), "suggested_answer": a, "source": "MuSiQue"})
                        added += 1
                except Exception as e:
                    logging.warning("[MuSiQue] item parse error: %s", e)
        logging.info("[dataset] MuSiQue: %d samples added.", added)

    logging.info("[dataset] TOTAL: %d samples ready.", len(mixed))
    random.shuffle(mixed)
    return mixed

def _parse_multi(val, cast=float, default=None):
    """
    Converts multi-value parameter input to list.
    Accepted: "2,3,4" / "2 3 4" / [2,3,4] / 4 / 4.0
    Returns default if invalid.
    """
    if val is None:
        return list(default) if default else []
    if isinstance(val, (list, tuple)):
        items = val
    else:
        # string or single number
        s = str(val).strip()
        items = re.split(r"[,\s]+", s) if s else []
    out = []
    for x in items:
        try:
            out.append(cast(x))
        except (ValueError, TypeError):
            continue
    return out if out else (list(default) if default else [])

def _run_automation_worker(payload: dict):
    global _automation_state
    try:
        models = payload.get("models") or []
        test_set = list(payload.get("test_set") or [])

        if bool(payload.get("use_mixed_validation")):
            n_sample = max(1, int(payload.get("mixed_sample_size", 5)))
            srcs = payload.get("sources") or None  # None → all three
            if isinstance(srcs, str):
                srcs = [s.strip() for s in srcs.split(",") if s.strip()] or None
            test_set.extend(build_mixed_validation_set(n_sample=n_sample, sources=srcs))

        if not test_set:
            with AUTOM_LOCK:
                _automation_state["last_error"] = (
                    "Test set empty. Check logs if datasets couldn't be loaded "
                    "(HotpotQA/2Wiki/MuSiQue repo access) or upload your own CSV."
                )
            return

        # Multi-value parameter grid (can come as "2,3,4" from UI)
        agents_list   = _parse_multi(payload.get("agents", 4), cast=int,   default=[4])
        alpha_list    = _parse_multi(payload.get("alpha", 1.0), cast=float, default=[1.0])
        beta_list     = _parse_multi(payload.get("beta", 1.0),  cast=float, default=[1.0])
        thresh_list   = _parse_multi(payload.get("threshold", 0.3), cast=float, default=[0.3])
        overlap_list  = _parse_multi(payload.get("overlaps", [0, 50]), cast=int, default=[0, 50])
        chunk_list    = _parse_multi(payload.get("chunk_sizes", 500), cast=int, default=[500])

        # Scout toggle: "true,false" → [True, False]; single value → single element list
        def _parse_bools(v, default):
            if v is None:
                return default
            if isinstance(v, bool):
                return [v]
            items = v if isinstance(v, (list, tuple)) else re.split(r"[,\s]+", str(v).strip())
            out = []
            for x in items:
                xs = str(x).strip().lower()
                if xs in ("true", "1", "yes", "evet"): out.append(True)
                elif xs in ("false", "0", "no", "hayir", "hayır"): out.append(False)
            return out or default

        scout_list = _parse_bools(payload.get("scouts"), [True])
        manifold_list = _parse_multi(payload.get("manifolds", 0.80), cast=float, default=[0.80])
        neg_gamma_list = _parse_multi(payload.get("neg_gammas", 0.0), cast=float, default=[0.0])
        use_semantic = bool(payload.get("use_semantic_chunking", False))
        semantic_thr_list = _parse_multi(payload.get("semantic_thresholds", payload.get("semantic_threshold", 0.60)), cast=float, default=[0.60])
        use_judge = bool(payload.get("use_judge", False))
        judge_model = payload.get("judge_model", "")

        combos = list(product(agents_list, alpha_list, beta_list, thresh_list,
                              overlap_list, chunk_list, scout_list, manifold_list, semantic_thr_list, neg_gamma_list))
        total_runs = len(test_set) * len(models) * len(combos)
        with AUTOM_LOCK:
            _automation_state["total"] = total_runs
            _automation_state["total_runs"] = total_runs
            _automation_state["progress"] = 0
            _automation_state["completed_runs"] = 0

        run_idx = 0
        for t in test_set:
            for m in models:
                for (n_ag, al, bt, th, ovl, csz, scout, manifold, sem_thr, neg_g) in combos:
                    with AUTOM_LOCK:
                        if not _automation_state["running"]:
                            return
                    run_idx += 1

                    data = {
                        "model": m, "agents": n_ag, "alpha": al, "beta": bt,
                        "chunk_overlap": ovl, "chunk_size": csz, "relevance_threshold": th,
                        "use_hetero_scout": scout, "use_semantic_chunking": use_semantic,
                        "manifold_threshold": manifold, "neg_gamma": neg_g,
                        "query": t.get("query", ""), "suggested_answer": t.get("suggested_answer", ""),
                        "corpus": t.get("corpus", ""), "source": t.get("source", ""),
                        "automation_id": payload.get("automation_id", ""), "run_index": run_idx
                    }

                    res = perform_simulation_logic(
                        corpus_text=t.get("corpus", ""), query_text=t.get("query", ""),
                        suggested_answer=t.get("suggested_answer", ""),
                        alpha_val=al, beta_val=bt, num_agents=n_ag, llm_model=m,
                        chunk_size=csz, chunk_overlap=ovl, relevance_threshold=th,
                        use_hetero_scout=scout, enable_vanilla=True, enable_rerank=True,
                        enable_mmr=True, enable_multiquery=False,
                        use_semantic_chunking=use_semantic, semantic_threshold=sem_thr,
                        use_judge=use_judge, judge_model=judge_model,
                        manifold_threshold=manifold, neg_gamma=neg_g
                    )

                    if not res.get("error"):
                        _run_record_to_append(res, data)

                    with AUTOM_LOCK:
                        _automation_state["progress"] = run_idx
                        _automation_state["completed_runs"] = run_idx

        # Generate report automatically upon test completion — ONLY with this run's data
        try:
            _compute_reports(automation_id=payload.get("automation_id"))
        except Exception as e:
            logging.warning("Automatic report generation failed: %s", e)
        # Drive backup immediately after run (without waiting for 5min cycle)
        try:
            _drive_export_once()
        except Exception as e:
            logging.warning("Drive export error at end of run: %s", e)

    except Exception as e:
        logging.exception("Automation worker error")
        with AUTOM_LOCK:
            _automation_state["last_error"] = str(e)
    finally:
        with AUTOM_LOCK:
            _automation_state["running"] = False

# ─── FLASK ROUTES AND INTERFACE (UI) ───────────────────────────────────────────
@app.route("/")
def index(): return render_template("index_v11.html")

@app.route("/history")
def history_page(): return render_template("history_v11.html")

@app.route("/api/history")
def api_history():
    if not os.path.exists(HISTORY_JSON_PATH): return jsonify([])
    with open(HISTORY_JSON_PATH, "r", encoding="utf-8") as f: return jsonify(json.load(f) or [])

@app.route("/api/history/reset", methods=["POST"])
def api_history_reset():
    try:
        for p in (HISTORY_JSON_PATH, HISTORY_CSV_PATH):
            if os.path.exists(p): os.remove(p)
        return jsonify({"message": "History cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history.csv")
def history_csv_dl():
    if not os.path.exists(HISTORY_CSV_PATH):
        return jsonify({"error": "CSV not found"}), 404
    return send_file(HISTORY_CSV_PATH, as_attachment=True, download_name="history_v11.csv")

@app.route("/api/models")
def api_models():
    models, err = get_ollama_models()
    return jsonify({"models": models or [], "error": err})

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(silent=True) or {}
    res = perform_simulation_logic(
        corpus_text=data.get("corpus", ""), query_text=data.get("query", ""),
        suggested_answer=data.get("suggested_answer", ""),
        alpha_val=float(data.get("alpha", 1.0)), beta_val=float(data.get("beta", 1.0)),
        num_agents=int(data.get("agents", 4)), llm_model=data.get("model", ""),
        chunk_size=int(data.get("chunk_size", 500)), chunk_overlap=int(data.get("chunk_overlap", 0)),
        relevance_threshold=float(data.get("relevance_threshold", 0.3)),
        use_hetero_scout=bool(data.get("use_hetero_scout", True)),
        enable_vanilla=bool(data.get("enable_vanilla", True)),
        enable_rerank=bool(data.get("enable_rerank", True)),
        enable_mmr=bool(data.get("enable_mmr", True)),
        enable_multiquery=bool(data.get("enable_multiquery", False)),
        use_semantic_chunking=bool(data.get("use_semantic_chunking", False)),
        semantic_threshold=float(data.get("semantic_threshold", 0.60)),
        use_judge=bool(data.get("use_judge", False)),
        judge_model=data.get("judge_model", ""),
        manifold_threshold=float(data.get("manifold_threshold", 0.80))
    )
    if not res.get("error"): _run_record_to_append(res, data)
    return jsonify(res)

@app.route("/api/automation/start", methods=["POST"])
def api_automation_start():
    global _automation_thread, _automation_state
    with AUTOM_LOCK:
        if _automation_state["running"]: return jsonify({"error": "Already running"}), 400
        payload = request.get_json(silent=True) or {}
        if not payload.get("models"): return jsonify({"error": "Select at least 1 model."}), 400
        _automation_state.update({"running": True, "progress": 0, "total": 0, "total_runs": 0, "completed_runs": 0, "last_error": None})
        payload["automation_id"] = f"auto_{int(time.time())}"
        _automation_thread = threading.Thread(target=_run_automation_worker, args=(payload,), daemon=True)
        _automation_thread.start()
    return jsonify({"message": "Started", "state": {**_automation_state}})

@app.route("/api/automation/status")
def api_automation_status():
    with AUTOM_LOCK: return jsonify(dict(_automation_state))

@app.route("/api/download_dataset", methods=["GET"])
def api_download_dataset():
    n_sample = int(request.args.get("n", 50))
    mixed = build_mixed_validation_set(n_sample=n_sample)
    if not mixed: return jsonify({"error": "Could not create dataset."}), 500

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["source", "query", "suggested_answer", "corpus"], extrasaction="ignore")
    writer.writeheader()
    for row in mixed: writer.writerow(row)
        
    return send_file(io.BytesIO(output.getvalue().encode("utf-8")), as_attachment=True,
                     download_name=f"svsf_dataset_{n_sample}x3.csv", mimetype="text/csv")

def _compute_reports(automation_id=None):
    """
    Produces McNemar test + EM/F1/Judge/latency plots and summary tables.
    If automation_id is provided, ONLY rows from that run are used (runs don't mix).
    All outputs are written to reports/<run_tag>/ folder with unique name.
    """
    from scipy.stats import chi2
    try:
        from scipy.stats import binomtest
    except ImportError:
        from scipy.stats import binom_test as binomtest

    if not os.path.exists(HISTORY_CSV_PATH):
        return {"error": "CSV not found"}

    df = pd.read_csv(HISTORY_CSV_PATH)

    # Isolate run (prevent mixing)
    if automation_id:
        df = df[df['AutomationId'].astype(str) == str(automation_id)].copy()
        if len(df) == 0:
            return {"error": f"No rows for AutomationId '{automation_id}'"}
        run_tag = str(automation_id)
    else:
        # Take last run if multiple exist (highest row count last AutomationId)
        aids = df['AutomationId'].astype(str)
        if aids.nunique() > 1:
            last_aid = df.iloc[-1]['AutomationId']
            df = df[df['AutomationId'].astype(str) == str(last_aid)].copy()
            run_tag = str(last_aid)
        else:
            run_tag = str(aids.iloc[0]) if len(aids) else "all"

    # Unique stamp: run_tag + time (even if same run is reported multiple times, it won't collide)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = re.sub(r'[^A-Za-z0-9_-]', '_', run_tag)
    out_dir = os.path.join("reports", f"{safe_tag}__{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    df_zero = df[df['chunk_overlap'] == 0].copy()
    if len(df_zero) == 0:
        df_zero = df.copy()  # use all if overlap=0 doesn't exist (e.g., semantic mode)

    systems = ['SVSF', 'Vanilla', 'MMR', 'Rerank', 'MultiQuery']
    metric_cols = {
        'EM': '_EM', 'EM_flex': '_EM_flex', 'F1': '_F1', 'Judge': '_Judge', 'Latency_ms': '_Time_ms'
    }

    def col_mean(sys, suffix):
        col = sys + suffix
        if col not in df_zero.columns:
            return None
        vals = pd.to_numeric(df_zero[col], errors='coerce').dropna()
        return float(vals.mean()) if len(vals) else None

    # ── Summary table (all systems × all metrics) ──
    summary_rows = []
    for s in systems:
        row = {'System': s, 'n': int(len(df_zero))}
        for mname, suffix in metric_cols.items():
            v = col_mean(s, suffix)
            if v is None:
                row[mname] = None
            elif mname in ('EM', 'EM_flex', 'Judge'):
                row[mname] = round(v * 100, 2)  # percentage
            else:
                row[mname] = round(v, 4)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, f"summary_metrics__{safe_tag}.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Save raw rows of this run separately (for reproducibility)
    subset_csv = os.path.join(out_dir, f"run_rows__{safe_tag}.csv")
    df.to_csv(subset_csv, index=False)

    # ── McNemar (SVSF vs Cross-Encoder), via EM ──
    b, c = 0, 0
    if 'SVSF_EM' in df_zero.columns and 'Rerank_EM' in df_zero.columns:
        for s, r in zip(pd.to_numeric(df_zero['SVSF_EM'], errors='coerce').fillna(0).tolist(),
                        pd.to_numeric(df_zero['Rerank_EM'], errors='coerce').fillna(0).tolist()):
            if s == 0 and r == 1: b += 1
            elif s == 1 and r == 0: c += 1
    if b + c == 0:
        p_value, stat_msg = 1.0, "Same results"
    elif (b + c) < 20:
        try:
            p_value = binomtest(min(b, c), b + c, 0.5).pvalue
        except TypeError:
            p_value = binomtest(min(b, c), b + c, 0.5)
        stat_msg = "Significant (p < 0.05)" if p_value < 0.05 else "Equivalent (p >= 0.05)"
    else:
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - chi2.cdf(statistic, 1)
        stat_msg = "Significant (p < 0.05)" if p_value < 0.05 else "Equivalent (p >= 0.05)"

    report_data = {
        "automation_id": run_tag, "out_dir": out_dir,
        "svsf_sole_correct": c, "rerank_sole_correct": b,
        "p_value": float(p_value), "conclusion": stat_msg,
        "n": int(len(df_zero)),
        "summary": summary_rows
    }
    with open(os.path.join(out_dir, f"mcnemar__{safe_tag}.json"), "w") as f:
        json.dump(report_data, f, indent=2)

    # ── Plots ──
    plt.rcParams['font.family'] = 'serif'
    palette5 = ['#1f77b4', '#9ca3af', '#38bdf8', '#fb923c', '#c084fc']

    # Plot 1: EM (with Judge side-by-side) for all systems
    plt.figure(figsize=(9, 5))
    em_vals = [col_mean(s, '_EM') or 0 for s in systems]
    em_vals = [v*100 for v in em_vals]
    ax = sns.barplot(x=systems, y=em_vals, hue=systems, legend=False, palette=palette5)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom', fontsize=9)
    plt.ylabel('Exact Match (%)'); plt.title(f'EM Performance — {safe_tag}')
    plt.savefig(os.path.join(out_dir, f"fig_EM__{safe_tag}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"fig_EM__{safe_tag}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: EM vs Judge vs F1 (grouped) — comparison of metrics
    plt.figure(figsize=(10, 5))
    import numpy as _np
    x = _np.arange(len(systems)); w = 0.25
    em_v = [(col_mean(s,'_EM') or 0)*100 for s in systems]
    jd_v = [(col_mean(s,'_Judge') or 0)*100 for s in systems]
    f1_v = [(col_mean(s,'_F1') or 0)*100 for s in systems]
    plt.bar(x-w, em_v, w, label='EM', color='#1f77b4')
    plt.bar(x,   jd_v, w, label='LLM-Judge', color='#10b981')
    plt.bar(x+w, f1_v, w, label='F1×100', color='#c084fc')
    plt.xticks(x, systems); plt.ylabel('Score'); plt.legend()
    plt.title(f'EM vs Judge vs F1 — {safe_tag}')
    plt.savefig(os.path.join(out_dir, f"fig_metrics__{safe_tag}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"fig_metrics__{safe_tag}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Latency (all systems)
    plt.figure(figsize=(9, 5))
    lat_vals = [col_mean(s, '_Time_ms') or 0 for s in systems]
    ax3 = sns.barplot(x=systems, y=lat_vals, hue=systems, legend=False, palette=palette5)
    for p in ax3.patches:
        ax3.annotate(f'{p.get_height():.2f}', (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom', fontsize=9)
    plt.ylabel('Retrieval Latency (ms)'); plt.title(f'Efficiency — {safe_tag}')
    plt.savefig(os.path.join(out_dir, f"fig_latency__{safe_tag}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"fig_latency__{safe_tag}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    report_data["files"] = sorted(os.listdir(out_dir))
    return report_data

@app.route("/api/generate_reports", methods=["GET"])
def generate_academic_reports():
    # Report a specific run with ?automation_id=...; if none, report last run
    aid = request.args.get("automation_id")
    rep = _compute_reports(automation_id=aid)
    if rep.get("error"):
        return jsonify(rep), 404
    return jsonify({"message": f"Reports generated: {rep['out_dir']}", "mcnemar": rep})
    
def _drive_export_once():
    """Copies history CSV/JSON and reports/ folder to Drive target."""
    dest = os.environ.get("SVSF_DRIVE_EXPORT_PATH", "").strip()
    if not dest:
        return False
    try:
        os.makedirs(dest, exist_ok=True)
        import shutil
        # history files
        for p in (HISTORY_CSV_PATH, HISTORY_JSON_PATH):
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
        # reports folder (all unique run reports)
        if os.path.isdir("reports"):
            dst_reports = os.path.join(dest, "reports")
            # dirs_exist_ok exists in Python 3.8+; available in Colab
            shutil.copytree("reports", dst_reports, dirs_exist_ok=True)
        return True
    except Exception as e:
        logging.warning("Drive export error: %s", e)
        return False

def _drive_export_loop():
    """Background periodic Drive backup. Works if SVSF_DRIVE_EXPORT_PATH is set."""
    if not os.environ.get("SVSF_DRIVE_EXPORT_PATH", "").strip():
        logging.info("Drive export disabled (SVSF_DRIVE_EXPORT_PATH missing).")
        return
    try:
        interval = int(os.environ.get("SVSF_DRIVE_EXPORT_INTERVAL_SEC", "300"))
    except ValueError:
        interval = 300
    logging.info("Drive export enabled: every %ds → %s", interval, os.environ.get("SVSF_DRIVE_EXPORT_PATH"))
    while True:
        time.sleep(interval)
        ok = _drive_export_once()
        if ok:
            logging.info("Drive backup taken (%s).", datetime.now().strftime("%H:%M:%S"))

@app.route("/api/drive_export", methods=["POST"])
def api_drive_export():
    """Trigger manual instant Drive backup."""
    ok = _drive_export_once()
    if ok:
        return jsonify({"message": "Drive backup taken.", "path": os.environ.get("SVSF_DRIVE_EXPORT_PATH")})
    return jsonify({"error": "Drive export failed (is SVSF_DRIVE_EXPORT_PATH set?)"}), 400

if __name__ == "__main__":
    def _boot_flask():
        app.run(host="0.0.0.0", port=5010, debug=False, use_reloader=False)

    threading.Thread(target=_boot_flask, daemon=True).start()
    # Periodic Drive backup (for Colab disconnections)
    threading.Thread(target=_drive_export_loop, daemon=True).start()
    print("SVSF v11 started (port 5010). Drive export:",
          "ENABLED → " + os.environ["SVSF_DRIVE_EXPORT_PATH"] if os.environ.get("SVSF_DRIVE_EXPORT_PATH") else "disabled")
    try:
        while True: time.sleep(3600)
    except KeyboardInterrupt:
        pass
