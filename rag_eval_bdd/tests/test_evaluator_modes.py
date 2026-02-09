from __future__ import annotations

import pytest

from rag_eval_bdd.evaluator import EvaluationRunner
from rag_eval_bdd.models import AppConfig, DatasetRow

pytestmark = [pytest.mark.smoke]


def test_positional_metric_mapping():
    config = AppConfig()
    config.evaluation.metric_question_mapping_mode = "positional"
    runner = EvaluationRunner(client=object(), config=config)

    row = DatasetRow(id="Q1", question="q")
    selected = ["answer_relevancy", "faithfulness", "completeness"]
    resolved = runner._resolve_row_metrics(row=row, selected_metrics=selected, row_index=1, total_rows=3)
    assert resolved == ["faithfulness"]


def test_row_metric_mapping_from_metadata():
    config = AppConfig()
    config.evaluation.metric_question_mapping_mode = "row"
    runner = EvaluationRunner(client=object(), config=config)

    row = DatasetRow(
        id="Q2",
        question="q",
        additional_metadata={"metric": "faithfulness, completeness"},
    )
    resolved = runner._resolve_row_metrics(
        row=row,
        selected_metrics=["answer_relevancy", "faithfulness", "completeness"],
        row_index=0,
        total_rows=3,
    )
    assert resolved == ["faithfulness", "completeness"]


def test_disable_context_trimming_keeps_all_chunks():
    config = AppConfig()
    config.evaluation.disable_context_trimming = True
    runner = EvaluationRunner(client=object(), config=config)

    chunks = ["a" * 1000, "b" * 1000, "c" * 1000]
    trimmed = runner._trim_retrieval_context(chunks)
    assert trimmed == chunks

