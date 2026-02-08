from __future__ import annotations

import os
from typing import Any

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric

try:
    from deepeval.metrics import ContextualRelevancyMetric as ContextRelevanceMetric
except Exception:
    ContextRelevanceMetric = None

try:
    from deepeval.metrics import GEval
except Exception:
    GEval = None


def _optional_login() -> None:
    key = os.getenv("DEEPEVAL_CONFIDENT_API_KEY")
    if not key:
        return
    try:
        import deepeval

        deepeval.login(key)
    except Exception:
        pass


def evaluate_layer1(question: str, retrieval_context: list[str], publish: bool = False) -> dict[str, Any]:
    test_case = LLMTestCase(
        input=question,
        actual_output="",
        retrieval_context=retrieval_context,
    )

    metrics = [ContextualPrecisionMetric(), ContextualRecallMetric()]

    if ContextRelevanceMetric is not None:
        metrics.append(ContextRelevanceMetric())
    elif GEval is not None:
        metrics.append(
            GEval(
                name="Context Relevance",
                criteria=(
                    "Evaluate how relevant the retrieval context is to the question. "
                    "Score 0 to 1 where 1 means all context is relevant and on-topic."
                ),
            )
        )

    results: dict[str, Any] = {}

    if publish:
        _optional_login()
        try:
            import deepeval

            deepeval.evaluate(test_cases=[test_case], metrics=metrics)
        except Exception:
            pass

    for metric in metrics:
        metric.measure(test_case)
        name = getattr(metric, "name", metric.__class__.__name__)
        results[name] = {
            "score": getattr(metric, "score", None),
            "reason": getattr(metric, "reason", None),
        }

    return results
