# -*- coding: utf-8 -*-
"""
Streamlit UI for Qwen-Graph-RAG
- Loads Week6 run_config and artifacts
- Dense retrieval -> Graph expansion -> MMR re-ranking -> Context build -> Deterministic Qwen answer
- Logs to artifacts/app_logs.csv
"""
import os, json, time, csv, math, textwrap, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()  # silence non-critical warnings

# --- Config & Artifacts ---
ROOT = Path("/home/manny-buff/projects/capstone/week6-rag-graph")
CFG_RUN = ROOT / "configs" / "rag_graph_run_config.json"
VDB     = ROOT / "artifacts" / "vdb"
ART_META   = VDB / "chunks_meta.parquet"
ART_CHUNKS = VDB / "chunks_text.pkl"
ART_FAISS  = VDB / "faiss.index"
ART_GRAPH  = VDB / "graph.pkl"
LOG_CSV    = ROOT / "artifacts" / "app_logs.csv"

# --- Caches ---
# Session state init
if "qwen" not in st.session_state:
    st.session_state.qwen = None

@st.cache_resource(show_spinner=False)
def load_config():
    cfg = json.loads(CFG_RUN.read_text())
    return cfg

@st.cache_resource(show_spinner=False)
def load_artifacts():
    import faiss, networkx as nx
    meta   = pd.read_parquet(ART_META)
    chunks = pickle.loads(ART_CHUNKS.read_bytes())
    index  = faiss.read_index(str(ART_FAISS))
    with open(ART_GRAPH, "rb") as f:
        G = pickle.load(f)
    return meta, chunks, index, G

@st.cache_resource(show_spinner=False)
def get_embedder(model_id: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id)
    return model

@st.cache_resource(show_spinner=False)
def get_qwen(local_dir: Path, model_id: str):
    import torch
    from transformers import (
        AutoConfig, AutoProcessor, AutoModelForImageTextToText,
        AutoTokenizer, AutoModelForCausalLM
    )

    class QwenAnswerer:
        def __init__(self, local_dir: Path, model_id: str):
            self.local_dir = local_dir
            self.model_id  = model_id
            self.is_vl     = False
            self.processor = None
            self.tokenizer = None
            self.model     = None

        def _load_from(self, src: str):
            cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
            mtype = getattr(cfg, "model_type", "").lower()
            if "vl" in mtype:
                self.is_vl = True
                self.processor = AutoProcessor.from_pretrained(src, trust_remote_code=True, use_fast=False)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    src, trust_remote_code=True,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
            else:
                self.is_vl = False
                self.tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    src, trust_remote_code=True,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )

        def load(self):
            last_err = None
            sources = []
            if self.local_dir.exists():
                sources.append(str(self.local_dir))
            sources.append(self.model_id)
            for src in sources:
                try:
                    self._load_from(src)
                    return self
                except Exception as e:
                    last_err = e
                    self.processor = self.tokenizer = self.model = None
            raise RuntimeError(f"Failed to load Qwen. Last error: {last_err}")

        def _format_prompt(self, question: str, context: str) -> str:
            system_msg = "You are a concise home-repair RAG assistant. Use ONLY the provided context. If missing info, say so."
            user_msg   = (
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Answer in 4–8 bullet points, then list 2–4 source file names in brackets.\n"
            )
            tok = self.processor.tokenizer if self.is_vl else self.tokenizer
            apply_chat = getattr(tok, "apply_chat_template", None)
            if callable(apply_chat):
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg}
                ]
                return apply_chat(messages, tokenize=False, add_generation_prompt=True)
            return f"{system_msg}\n\n{user_msg}"

        def answer(self, question: str, context: str, max_new_tokens: int = 240) -> str:
            assert self.model is not None, "Model not loaded"
            prompt_text = self._format_prompt(question, context)

            if self.is_vl:
                # Text-only prompt via processor; slice generated tokens after input length
                inputs = self.processor(text=prompt_text, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                input_len = int(inputs["input_ids"].shape[1])
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )
                gen_ids = out[0][input_len:]  # decode only new tokens
                text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            else:
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
                input_len = int(inputs["input_ids"].shape[1])
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )
                gen_ids = out[0][input_len:]  # decode only new tokens
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            return text.strip()

    return QwenAnswerer(local_dir, model_id).load()

# --- Retrieval utilities ---
def dense_retrieve(embedder, index, query: str, top_k: int):
    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv.astype(np.float32), top_k)
    return qv[0], [int(x) for x in I[0]], D[0].tolist()

def expand_via_graph(G, seeds, hop_limit: int, per_seed_cap: int = 20, global_cap: int = 200):
    visited = set(int(s) for s in seeds)
    frontier = set(visited)
    for _ in range(hop_limit):
        nxt = set()
        for node in list(frontier):
            nbrs = list(G.neighbors(node))
            for nb in nbrs[:per_seed_cap]:
                nb = int(nb)
                if nb not in visited:
                    nxt.add(nb)
        frontier = nxt
        visited.update(frontier)
        if len(visited) >= global_cap:
            break
    return list(visited)

def mmr_select(embedder, query_vec, candidate_ids, chunks, k=12, lambda_weight=0.70):
    """
    Greedy MMR selection over candidate_ids.
    - query_vec: (d,) normalized
    - candidate_ids: list[int] of chunk ids
    - returns: list[int] of selected candidate chunk ids (never None)
    """
    import numpy as np
    if not candidate_ids:
        return []  # hard guard

    # Batch-embed candidate texts (normalized)
    ctext = [chunks[int(cid)] for cid in candidate_ids]
    C = embedder.encode(ctext, convert_to_numpy=True, normalize_embeddings=True)

    # Relevance: cosine sim to query
    rel = C @ query_vec  # (n,)

    selected = []
    pool = list(range(len(candidate_ids)))  # indices into C / candidate_ids
    k = max(0, int(k))

    while pool and len(selected) < k:
        if not selected:
            best_idx = int(np.argmax(rel[pool]))
            chosen = pool.pop(best_idx)
            selected.append(chosen)
            continue

        # Diversity: max similarity to any already selected
        sim_to_S = C[pool] @ C[selected].T  # (len(pool), len(selected))
        max_div = sim_to_S.max(axis=1) if sim_to_S.ndim == 2 else sim_to_S

        # MMR score: balance relevance vs diversity
        mmr_scores = lambda_weight * rel[pool] - (1.0 - lambda_weight) * max_div
        best_idx = int(np.argmax(mmr_scores))
        chosen = pool.pop(best_idx)
        selected.append(chosen)

    return [int(candidate_ids[i]) for i in selected]


def build_context(meta, chunks, ordered_ids, max_chars=4000):
    if not ordered_ids:
        return ""  # hard guard

    import textwrap
    out, size = [], 0
    for cid in ordered_ids:
        cid = int(cid)
        row = meta.loc[meta["chunk_id"] == cid]
        if row.empty:
            continue
        path = row.iloc[0]["path"]
        idx  = int(row.iloc[0]["chunk_idx"])
        snippet = textwrap.shorten(chunks[cid], width=360, placeholder=" …")
        block = f"[SOURCE] {path} | chunk#{idx}\n{snippet}\n"
        if size + len(block) > max_chars:
            break
        out.append(block)
        size += len(block)
    return "\n".join(out)


# --- UI ---
st.set_page_config(page_title="Qwen Graph-RAG", page_icon="🧭", layout="wide")
st.title("Qwen Graph-RAG (Week 6 — Application)")

cfg = load_config()
meta, chunks, index, G = load_artifacts()
embedder = get_embedder(cfg["embed_model"])
qwen = None

with st.sidebar:
    st.subheader("Settings")
    top_k    = st.slider("Retriever K", 3, 20, int(cfg.get("retriever_k", 5)), 1)
    hop_lim  = st.slider("Hop Limit", 0, 3, int(cfg.get("hop_limit", 2)), 1)
    per_cap  = st.slider("Per-seed Neighbor Cap", 5, 50, 15, 1)
    glob_cap = st.slider("Global Cap", 50, 800, 200, 50)
    mmr_k    = st.slider("MMR Select K (context items)", 4, 20, 12, 1)
    mmr_lmb  = st.slider("MMR λ (relevance vs diversity)", 0.10, 0.95, 0.70, 0.05)
    max_chars= st.slider("Context Max Chars", 1000, 8000, 4000, 250)

    load_llm = st.checkbox("Load Qwen model", value=False, help="Check this once before first query.")
    st.caption("Deterministic generation (do_sample=False).")

if load_llm and st.session_state.qwen is None:
    try:
        obj = get_qwen(Path(cfg["llm_local_path"]), cfg["llm_model_id"])
        if obj is None:
            raise RuntimeError("get_qwen() returned None")
        st.session_state.qwen = obj
        st.success("Qwen loaded.")
    except Exception as e:
        st.session_state.qwen = None
        st.error(f"Qwen load error: {e}")


query = st.text_area("Ask a question about home repair:", height=100, placeholder="e.g., How can I stop a toilet tank from sweating in humid weather?")
run = st.button("Search & Answer")

colL, colR = st.columns([1,1])

if run and query.strip():
    t0 = time.time()
    qv, seeds, scores = dense_retrieve(embedder, index, query, top_k=top_k)
    expanded = expand_via_graph(G, seeds, hop_limit=hop_lim, per_seed_cap=per_cap, global_cap=glob_cap)

    # prioritize seeds, then expanded unique
    cand, seen = [], set()
    for c in seeds + expanded:
        ci = int(c)
        if ci not in seen:
            cand.append(ci)
            seen.add(ci)

    # Guard for empty candidates
    if not cand:
        st.warning("No candidate chunks were found. Try increasing Retriever K or Hop Limit.")
        selected = []
    else:
        try:
            selected = mmr_select(embedder, qv, cand, chunks, k=mmr_k, lambda_weight=mmr_lmb)
            if selected is None or len(selected) == 0:
                st.info(f"MMR returned no items; falling back to first {mmr_k}.")
                selected = cand[:mmr_k]
        except Exception as e:
            st.warning(f"MMR selection failed ({e}); falling back to first {mmr_k}.")
            selected = cand[:mmr_k]

    # Build context robustly
    context = build_context(meta, chunks, selected, max_chars=max_chars)

    with colL:
        st.markdown("### Top Sources (MMR-selected)")
        for i, cid in enumerate(selected, 1):
            row = meta.loc[meta["chunk_id"]==cid].iloc[0]
            st.write(f"[{i}] {row['path']} (chunk {row['chunk_idx']})")

        st.markdown("### Context Preview")
        st.code(context[:1000] + (" ..." if len(context) > 1000 else ""), language="text")

    answer = ""
    lat = None
    with colR:
        if load_llm and st.session_state.qwen is not None:
            try:
                t1 = time.time()
                answer = st.session_state.qwen.answer(query, context, max_new_tokens=256)
                lat = round(time.time() - t1, 2)
                st.markdown("### Answer")
                st.write(answer)
                st.caption(f"LLM latency: {lat}s")
            except Exception as e:
                st.error(f"Answer error: {e}")
        elif load_llm and st.session_state.qwen is None:
            st.warning("Model not loaded yet—toggle the checkbox, wait for 'Qwen loaded.', then run again.")
        else:
            st.info("Load the Qwen model in the sidebar to generate an answer.")
            st.markdown("### Answer (not generated)")
            st.write("")


    # Log row
    try:
        LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        file_exists = LOG_CSV.exists()
        with open(LOG_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts","query","top_k","hop_limit","per_seed_cap","global_cap","mmr_k","mmr_lambda","latency_s","seeds","expanded_size","selected_ids"
            ])
            if not file_exists:
                w.writeheader()
            w.writerow({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "top_k": top_k,
                "hop_limit": hop_lim,
                "per_seed_cap": per_cap,
                "global_cap": glob_cap,
                "mmr_k": mmr_k,
                "mmr_lambda": mmr_lmb,
                "latency_s": lat if lat is not None else "",
                "seeds": str(seeds),
                "expanded_size": len(expanded),
                "selected_ids": str(selected)
            })
    except Exception as e:
        st.warning(f"Could not write log: {e}")

    st.success(f"Done in {round(time.time()-t0, 2)}s.")
