# Hybrid Search RAG

A modular Python 3.10+ reference implementation combining dense Qdrant retrieval, BM25 keyword retrieval, reciprocal rank fusion, cross-encoder reranking, Gemini generation, NLI citation verification, and RAGAS evaluation.

## Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and set GEMINI_API_KEY
python main.py
# inspect retrieval, the raw answer, and each NLI verdict
python main.py --debug
```

The first run downloads the embedding, reranking, and NLI models from Hugging Face. Qdrant runs in memory; no external vector database is required.

## Structure

- `rag_system/retrieval.py`: Markdown/text loading, 500/50 chunking, Qdrant, BM25, RRF.
- `rag_system/reranker.py`: MS MARCO cross-encoder, top-4 selection.
- `rag_system/generator.py`: Gemini prompt and inline citations.
- `rag_system/verifier.py`: sentence parsing, NLI verdicts, unsupported-claim removal.
- `rag_system/eval.py`: RAGAS benchmark evaluation.
- `data/northstar_financial_2025.md`: fictional annual-report-style finance corpus for the demo.

For evaluation, pass rows containing `user_input`, `response`, `retrieved_contexts`, and `reference` to `evaluate_benchmark`.
