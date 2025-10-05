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