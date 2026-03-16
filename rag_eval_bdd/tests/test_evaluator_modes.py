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


def test_not_found_answer_forces_layer2_metrics_fail(monkeypatch: pytest.MonkeyPatch):
    class DummyClient:
        def ask_question(self, session_id: str, question: str, use_cache: bool = True):
            return {"session_id": session_id, "question": question}, {
                "answer": "Not found in document.",
                "retrieval_context": ["chunk-a", "chunk-b"],
            }

    config = AppConfig()
    runner = EvaluationRunner(client=DummyClient(), config=config)

    def _unexpected_metric_builder(*_args, **_kwargs):
        raise AssertionError("Layer2 metric builder should be skipped for forced Not-found failures")

    monkeypatch.setattr("rag_eval_bdd.evaluator.build_metric", _unexpected_metric_builder)

    row = DatasetRow(
        id="Q_NOT_FOUND",
        question="How does X compare to Y?",
        expected_answer="A grounded answer exists in the source document.",
    )
    run_result = runner.evaluate_dataset(
        dataset_rows=[row],
        selected_metrics=["answer_relevancy", "faithfulness", "completeness"],
        session_id="session-1",
        feature="feature.feature",
        scenario="scenario",
        tags=["live", "layer2"],
    )

    question_metrics = run_result.question_results[0].metrics
    assert len(question_metrics) == 3
    assert all(metric.passed is False for metric in question_metrics)
    assert all(metric.score == 0.0 for metric in question_metrics)
    assert all("Forced FAIL" in str(metric.reason) for metric in question_metrics)


def test_not_found_answer_forces_layer1_metrics_fail(monkeypatch: pytest.MonkeyPatch):
    class DummyClient:
        def ask_question(self, session_id: str, question: str, use_cache: bool = True):
            return {"session_id": session_id, "question": question}, {
                "answer": "Not found in document.",
                "retrieval_context": ["chunk-a", "chunk-b"],
            }

    config = AppConfig()
    runner = EvaluationRunner(client=DummyClient(), config=config)

    def _unexpected_metric_builder(*_args, **_kwargs):
        raise AssertionError("Layer1 metric builder should be skipped for forced Not-found failures")

    monkeypatch.setattr("rag_eval_bdd.evaluator.build_metric", _unexpected_metric_builder)

    row = DatasetRow(
        id="Q_NOT_FOUND_L1",
        question="What happened in chapter 1?",
        expected_answer="Expected grounded answer exists.",
    )
    run_result = runner.evaluate_dataset(
        dataset_rows=[row],
        selected_metrics=["contextual_precision", "contextual_recall", "contextual_relevancy"],
        session_id="session-1",
        feature="feature.feature",
        scenario="scenario",
        tags=["live", "layer1"],
    )

    question_metrics = run_result.question_results[0].metrics
    assert len(question_metrics) == 3
    assert all(metric.passed is False for metric in question_metrics)
    assert all(metric.score == 0.0 for metric in question_metrics)
    assert all("Forced FAIL" in str(metric.reason) for metric in question_metrics)
