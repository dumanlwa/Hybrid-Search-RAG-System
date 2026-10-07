"""Cross-encoder reranking."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from .retrieval import Chunk


class CrossEncoderReranker:
    """Rerank retrieved candidates with a query-passage cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Chunk], top_k: int = 4) -> list[Chunk]:
        """Return the highest scoring candidate passages."""
        if not candidates:
            return []
        scores = self.model.predict([(query, chunk.text) for chunk in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda item: float(item[0]), reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
