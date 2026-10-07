"""Hybrid dense and sparse retrieval with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Chunk:
    """A retrievable document fragment."""

    chunk_id: int
    text: str
    source: str


class HybridRetriever:
    """Index chunks in Qdrant and BM25, then fuse their rankings with RRF."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "rag_chunks",
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        self.chunks = list(chunks)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(embedding_model)
        self.qdrant = QdrantClient(":memory:")
        self._build_indexes()

    def _build_indexes(self) -> None:
        vectors = self.encoder.encode(
            [chunk.text for chunk in self.chunks], normalize_embeddings=True
        )
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector.tolist(),
                    payload={"chunk_id": chunk.chunk_id},
                )
                for chunk, vector in zip(self.chunks, vectors)
            ],
        )
        self.bm25 = BM25Okapi([chunk.text.lower().split() for chunk in self.chunks])

    def _dense_search(self, query: str, limit: int) -> list[int]:
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        ).points
        return [int(result.payload["chunk_id"]) for result in results]

    def _sparse_search(self, query: str, limit: int) -> list[int]:
        scores = self.bm25.get_scores(query.lower().split())
        return [index for index, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:limit]]

    def search(self, query: str, candidate_limit: int = 20, rrf_k: int = 60) -> list[Chunk]:
        """Return up to ``candidate_limit`` chunks ranked by reciprocal rank fusion."""
        dense_ids = self._dense_search(query, min(candidate_limit, len(self.chunks)))
        sparse_ids = self._sparse_search(query, min(candidate_limit, len(self.chunks)))
        fused: dict[int, float] = {}
        for ranking in (dense_ids, sparse_ids):
            for rank, chunk_id in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[:candidate_limit]
        return [self.chunks[chunk_id] for chunk_id in ranked_ids]


def load_chunks(corpus_dir: str | Path, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Chunk]:
    """Load Markdown and text files, splitting them into stable source chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        for text in splitter.split_text(path.read_text(encoding="utf-8")):
            chunks.append(Chunk(chunk_id=len(chunks), text=text, source=str(path)))
    if not chunks:
        raise FileNotFoundError(f"No Markdown or text files found in {corpus_dir}")
    return chunks
