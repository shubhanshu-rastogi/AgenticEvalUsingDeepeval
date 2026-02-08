from __future__ import annotations

import os
from typing import Any

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

try:
    from deepeval.metrics import CompletenessMetric
except Exception:
    CompletenessMetric = None

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


def evaluate_layer2(
    question: str, answer: str, retrieval_context: list[str], publish: bool = False
) -> dict[str, Any]:
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=retrieval_context,
    )

    metrics = [FaithfulnessMetric(), AnswerRelevancyMetric()]

    if CompletenessMetric is not None:
        metrics.append(CompletenessMetric())
    elif GEval is not None:
        metrics.append(
            GEval(
                name="Completeness",
                criteria=(
                    "Assess whether the answer is complete with respect to the question "
                    "given the provided context. Score 0 to 1."
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
