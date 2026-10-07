"""RAGAS evaluation over a small question/reference benchmark."""

from __future__ import annotations

from typing import Any, Sequence

def evaluate_benchmark(rows: Sequence[dict[str, Any]]) -> Any:
    """Compute faithfulness, context precision, and context recall with RAGAS.

    Each row must contain ``user_input``, ``response``, ``retrieved_contexts``,
    and ``reference`` keys.
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness

    samples = [SingleTurnSample(**row) for row in rows]
    dataset = EvaluationDataset(samples=samples)
    return evaluate(dataset, metrics=[Faithfulness(), ContextPrecision(), ContextRecall()])
