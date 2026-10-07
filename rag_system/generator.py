"""Gemini answer generation with strict citation formatting."""

from __future__ import annotations

from google import genai

from .retrieval import Chunk


class GeminiGenerator:
    """Generate context-grounded answers with one or more chunk citations."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def answer(self, query: str, context: list[Chunk]) -> str:
        """Ask Gemini to answer only from the supplied context."""
        formatted = "\n\n".join(
            f"[Chunk {index}]\n{chunk.text}" for index, chunk in enumerate(context, start=1)
        )
        prompt = f"""Answer the question using only the context below.
    Every factual sentence must end with exactly one citation such as [Chunk 1]. Cite the chunk that directly supports that sentence. Do not combine facts from different chunks in one sentence. Do not use knowledge outside the context. If the context is insufficient, say so and cite the relevant context.

Question: {query}

Context:
{formatted}
"""
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text or ""
