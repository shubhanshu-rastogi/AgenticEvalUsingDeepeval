from __future__ import annotations

from typing import Iterable, List, Optional, Set

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams

from rag_eval_bdd.models import AppConfig

try:
    from deepeval.metrics import ContextualRelevancyMetric
except Exception:  # noqa: BLE001
    ContextualRelevancyMetric = None

METRIC_ORDER = [
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
    "answer_relevancy",
    "faithfulness",
    "completeness",
]

LAYER1_METRICS = {"contextual_precision", "contextual_recall", "contextual_relevancy"}
LAYER2_METRICS = {"answer_relevancy", "faithfulness", "completeness"}

ALIASES = {
    "context_precision": "contextual_precision",
    "contextualprecision": "contextual_precision",
    "contextual_precision": "contextual_precision",
    "context_recall": "contextual_recall",
    "contextual_recall": "contextual_recall",
    "contextualrecall": "contextual_recall",
    "context_relevance": "contextual_relevancy",
    "contextual_relevancy": "contextual_relevancy",
    "contextualrelevancy": "contextual_relevancy",
    "answer_relevancy": "answer_relevancy",
    "answerrelevancy": "answer_relevancy",
    "faithfulness": "faithfulness",
    "completeness": "completeness",
}


def normalize_metric_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key, key)


def metric_threshold(metric_name: str, config: AppConfig) -> float:
    return float(getattr(config.thresholds, normalize_metric_name(metric_name)))


def _ordered(metrics: Iterable[str]) -> List[str]:
    metric_set = {normalize_metric_name(m) for m in metrics}
    return [metric for metric in METRIC_ORDER if metric in metric_set]


def select_metrics_from_tags(tags: Iterable[str], explicit_metrics: Optional[Iterable[str]] = None) -> List[str]:
    if explicit_metrics:
        return _ordered(explicit_metrics)

    normalized_tags: Set[str] = {normalize_metric_name(tag) for tag in tags}

    base: Set[str] = set()
    if "layer1" in normalized_tags:
        base |= LAYER1_METRICS
    if "layer2" in normalized_tags:
        base |= LAYER2_METRICS
    if not base:
        base = set(METRIC_ORDER)

    metric_tags = {tag for tag in normalized_tags if tag in set(METRIC_ORDER)}
    if metric_tags:
        selected = base & metric_tags
    else:
        selected = base

    return _ordered(selected)


def build_metric(metric_name: str, config: AppConfig):
    metric_name = normalize_metric_name(metric_name)
    threshold = metric_threshold(metric_name, config)
    model = config.model
    include_reason = config.evaluation.include_reason

    if metric_name == "contextual_precision":
        return ContextualPrecisionMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason,
        )

    if metric_name == "contextual_recall":
        return ContextualRecallMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason,
        )

    if metric_name == "contextual_relevancy":
        if config.evaluation.cost_optimized:
            return GEval(
                name="Context Relevance",
                threshold=threshold,
                model=model,
                evaluation_steps=[
                    "Review the user question and the retrieval context.",
                    "Decide whether the retrieval context is relevant to answering the question.",
                    "Assign a score from 0 to 1 where 1 means highly relevant and 0 means irrelevant.",
                    "Provide a short reason for the score.",
                ],
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                ],
            )
        if ContextualRelevancyMetric is not None:
            return ContextualRelevancyMetric(
                threshold=threshold,
                model=model,
                include_reason=include_reason,
            )
        return GEval(
            name="Context Relevance",
            threshold=threshold,
            model=model,
            evaluation_steps=[
                "Review the user question and the retrieval context.",
                "Decide whether the retrieval context is relevant to answering the question.",
                "Assign a score from 0 to 1 where 1 means highly relevant and 0 means irrelevant.",
                "Provide a short reason for the score.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
        )

    if metric_name == "answer_relevancy":
        return AnswerRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason,
        )

    if metric_name == "faithfulness":
        return FaithfulnessMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason,
            truths_extraction_limit=max(
                0, int(config.evaluation.faithfulness_truths_extraction_limit)
            ),
        )

    if metric_name == "completeness":
        return GEval(
            name="Completeness",
            threshold=threshold,
            model=model,
            evaluation_steps=[
                "Check whether the response answers all parts of the user question.",
                "Check whether important specifics asked in the question are present.",
                "Give a score from 0 to 1 where 1 means fully complete.",
                "Provide a short reason for the score.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
        )

    raise ValueError(f"Unsupported metric: {metric_name}")
