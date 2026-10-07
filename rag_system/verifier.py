"""NLI-based citation verification and unsupported-claim redaction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from .retrieval import Chunk


@dataclass(frozen=True)
class VerifiedClaim:
    """Verification result for one cited sentence."""

    sentence: str
    citation: int | None
    verdict: str
    verified: bool


class CitationVerifier:
    """Verify cited claims with an NLI cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self.model = CrossEncoder(model_name)

    def verify(self, answer: str, context: list[Chunk]) -> tuple[str, list[VerifiedClaim]]:
        """Remove sentences whose cited source is neutral or contradictory."""
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", answer.strip())
            if part.strip()
        ]
        results: list[VerifiedClaim] = []
        accepted: list[str] = []
        for sentence in sentences:
            match = re.search(r"\[Chunk\s+(\d+)\]", sentence, flags=re.IGNORECASE)
            if not match:
                results.append(VerifiedClaim(sentence, None, "Unverified", False))
                continue
            citation = int(match.group(1))
            if citation < 1 or citation > len(context):
                results.append(VerifiedClaim(sentence, citation, "Unverified", False))
                continue
            claim = re.sub(r"\s*\[Chunk\s+\d+\]", "", sentence, flags=re.IGNORECASE).strip()
            label = self.model.predict([(context[citation - 1].text, claim)], apply_softmax=True)
            verdict = self._label_from_prediction(label)
            verified = verdict == "Entailment"
            results.append(VerifiedClaim(sentence, citation, verdict, verified))
            if verified:
                accepted.append(sentence)
        return " ".join(accepted), results

    @staticmethod
    def _label_from_prediction(prediction: object) -> str:
        labels = ["Contradiction", "Entailment", "Neutral"]
        if hasattr(prediction, "argmax"):
            index = int(prediction.argmax())
        else:
            index = max(range(len(prediction[0])), key=lambda item: prediction[0][item])  # type: ignore[index]
        return labels[index]
