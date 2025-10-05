# Qwen Graph-RAG — Week 6

This repo contains a local, config-driven Graph-RAG pipeline with Multi-Hop QA and a Streamlit application.
It reuses the Week5 dataset and Qwen model paths, and writes reproducible configs and artifacts.

## Quickstart


# 1) Python env
python -m venv ~/venvs/core-rag
source ~/venvs/core-rag/bin/activate
pip install -r requirements.txt

# 2) Build index + graph (Week6-1)
# open notebooks/Week6-1-HW.ipynb and run Cells C→F

# 3) Multi-Hop QA (Week6-2)
# open notebooks/Week6-2-HW.ipynb and run Cells G→J
# (ensure you see ablation rows in artifacts/ablation_results_graph.csv)

# 4) Application (Week6-3)
# run Cell K to write the app file (once), then from terminal:
streamlit run app/app_rag.py --server.headless true --server.port 8501

Structure
week6-rag-graph/
├─ app/
│  └─ app_rag.py
├─ artifacts/
│  ├─ vdb/                # FAISS index, graph, chunk meta
│  ├─ app_logs.csv        # Streamlit runs (app appends)
│  ├─ ablation_results_graph.csv
│  ├─ Report_snippets_Wk6_1_2.md
│  └─ Report_snippets_Wk6_3.md
├─ configs/
│  ├─ env_rag_graph.json
│  └─ rag_graph_run_config.json
├─ notebooks/
│  ├─ Week6-1-HW.ipynb    # Build
│  ├─ Week6-2-HW.ipynb    # Multi-Hop QA
│  └─ Week6-3-HW.ipynb    # Application + reporting
└─ Report.md

Configuration

Update configs/rag_graph_run_config.json to point at:

"corpus_root": "/home/manny-buff/projects/capstone/hw-rag/data/"

"llm_local_path": "/home/manny-buff/projects/capstone/hw-rag/models/Qwen2-VL-2B-Instruct/"

"embed_model": "intfloat/e5-small-v2"

"llm_model_id": "Qwen/Qwen2.5-VL-3B-Instruct"

Notes

Deterministic generation (do_sample=False) for evaluation.

Beam search with no-repeat for report regeneration to reduce repetition.
