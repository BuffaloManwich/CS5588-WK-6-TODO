# Capstone — Week 6: Next-Level RAG (Graph-RAG → Multi-Hop → Application)

**Author:** manny-buff  
**Generated:** 2025-10-05 16:28:47

This report compiles:
- Part 1: Graph-RAG Build
- Part 2: Multi-Hop QA (dense → graph expansion → Qwen synthesis)
- Part 3: Streamlit Application (logs & sample answers)

Artifacts are created under `artifacts/` per the run config in `configs/rag_graph_run_config.json`.

---

# Week 6 — Report Snippets (Parts 1 & 2)

## Environment
- Python venv: `/home/manny-buff/venvs/core-rag`
- Python: `3.11.9`
- GPU: `NVIDIA GeForce RTX 4080, 580.65.06, 16376 MiB`

## Run Config
- corpus_root: `/home/manny-buff/projects/capstone/hw-rag/data/`
- embed_model: `intfloat/e5-small-v2`
- llm_model_id: `Qwen/Qwen2.5-VL-3B-Instruct`
- llm_local_path: `/home/manny-buff/projects/capstone/hw-rag/models/Qwen2-VL-2B-Instruct/`
- device: `cuda`
- retriever_k: `5`
- hop_limit: `2`

## Artifacts
- Meta rows (chunks): 4381
- FAISS index: present
- Graph nodes: 4381 | edges: 46574

## Ablation Results (head)
| variant     |   retriever_k |   hop_limit |   accuracy | notes                               |
|:------------|--------------:|------------:|-----------:|:------------------------------------|
| dense+graph |             5 |           2 |        nan | seeds=5 expanded=813 latency_s=2.38 |
| dense+graph |             5 |           2 |        nan | seeds=5 expanded=813 latency_s=1.05 |

## Ablation Results (tail)
| variant     |   retriever_k |   hop_limit |   accuracy | notes                               |
|:------------|--------------:|------------:|-----------:|:------------------------------------|
| dense+graph |             5 |           2 |        nan | seeds=5 expanded=813 latency_s=2.38 |
| dense+graph |             5 |           2 |        nan | seeds=5 expanded=813 latency_s=1.05 |

---

# Week 6 — Report Snippets (Part 3: Application)

- Total runs: **8**
- Average LLM latency: **0.60s**

## Sample 1
**Query:** How do I fix a leaky faucet?
**Settings:** K=5, hops=2, per_cap=15, global_cap=200, MMR_k=12, λ=0.7
**Answer (lat 1.2s):**

- Check for damaged O-rings
- Turn off water supply
- Use adjustable pliers
- Unscrew packing nut
- Unscrew spindle
- Use channel-type pliers
- Separate pipe from fitting
- Remove old solder

**Sources:** Complete home repair  with 350 projects and 2300 photos.pdf; Safe & Sound _ A Renter-Friendly Guide to Home Repair.pdf; 1001 do-it-yourself hints & tips  tricks.pdf; A Dirty Guide to a Clean Home _ Housekeeping Hacks You Cant Live Without.pdf

## Sample 2
**Query:** How do I fix a leaky faucet?
**Settings:** K=5, hops=2, per_cap=15, global_cap=200, MMR_k=12, λ=0.7
**Answer (lat 1.12s):**

- Check for damaged O-rings
- Turn off water supply
- Use adjustable pliers
- Unscrew packing nut
- Unscrew spindle
- Use channel-type pliers
- Separate pipe from fitting
- Remove old solder

**Sources:** Complete home repair  with 350 projects and 2300 photos.pdf; Safe & Sound _ A Renter-Friendly Guide to Home Repair.pdf; 1001 do-it-yourself hints & tips  tricks.pdf; A Dirty Guide to a Clean Home _ Housekeeping Hacks You Cant Live Without.pdf

## Sample 3
**Query:** How do I fix a leaky faucet?
**Settings:** K=5, hops=2, per_cap=15, global_cap=200, MMR_k=12, λ=0.7
**Answer (lat 0.99s):**

- Check for damaged O-rings
- Turn off water supply
- Use adjustable pliers
- Unscrew packing nut
- Unscrew spindle
- Use channel-type pliers
- Separate pipe from fitting
- Remove old solder

**Sources:** Complete home repair  with 350 projects and 2300 photos.pdf; Safe & Sound _ A Renter-Friendly Guide to Home Repair.pdf; 1001 do-it-yourself hints & tips  tricks.pdf; A Dirty Guide to a Clean Home _ Housekeeping Hacks You Cant Live Without.pdf

