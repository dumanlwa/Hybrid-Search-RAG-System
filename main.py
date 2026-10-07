"""Run the hybrid RAG demo.

Usage: set GEMINI_API_KEY in .env, then run ``python main.py``.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

from dotenv import load_dotenv

from rag_system.generator import GeminiGenerator
from rag_system.reranker import CrossEncoderReranker
from rag_system.retrieval import HybridRetriever, load_chunks
from rag_system.verifier import CitationVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hybrid finance RAG demo")
    parser.add_argument("--debug", action="store_true", help="Print retrieval, generation, and verification details")
    args = parser.parse_args()

    load_dotenv()
    query = "What were Northstar's 2025 revenue and net income?"
    chunks = load_chunks(Path(__file__).parent / "data")
    retriever = HybridRetriever(chunks)
    candidates = retriever.search(query, candidate_limit=20)
    context = CrossEncoderReranker().rerank(query, candidates, top_k=4)
    if args.debug:
        print(f"Query: {query}\n")
        print(f"Retrieved candidates: {len(candidates)}")
        print("Final reranked context:")
        for index, chunk in enumerate(context, start=1):
            preview = " ".join(chunk.text.split())
            print(f"  [Chunk {index}] {preview[:400]}")
        print()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running the generation demo")
    answer = GeminiGenerator(api_key, os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).answer(query, context)
    if args.debug:
        print(f"Raw Gemini answer:\n{answer}\n")
    verified_answer, claims = CitationVerifier().verify(answer, context)
    if args.debug:
        print("Verification results:")
        for claim in claims:
            print(f"  [{claim.verdict}] {claim.sentence}")
    print(verified_answer)
    print(f"Verified {sum(claim.verified for claim in claims)}/{len(claims)} cited claims")


if __name__ == "__main__":
    main()
