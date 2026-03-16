from __future__ import annotations

import re
from pathlib import Path

import pytest

from rag_eval_bdd.executive_report import write_executive_html
from rag_eval_bdd.models import (
    MetricResult,
    MetricTrend,
    QuestionEvalResult,
    RunResult,
    TrendPoint,
    TrendSummary,
)

pytestmark = [pytest.mark.smoke]


def test_write_executive_html_contains_business_table_and_technical_logs(tmp_path: Path):
    run = RunResult(
        run_id="RUN_001",
        timestamp="2026-02-12T10:00:00+00:00",
        feature="feature_file.feature",
        scenario="test_evaluate_layer1_contextual_metrics_from_inline_dataset_table",
        tags=["layer1", "live"],
        selected_metrics=["contextual_precision"],
        dataset_size=1,
        question_results=[
            QuestionEvalResult(
                question_id="Q1",
                question="How many sixes?",
                expected_answer="Three sixes.",
                actual_answer="Tilak hit 3 sixes.",
                retrieval_context=["chunk 1"],
                metrics=[
                    MetricResult(
                        metric_name="contextual_precision",
                        threshold=0.6,
                        score=0.9,
                        passed=True,
                        reason="Answer uses the right context chunk.",
                        evaluation_model="gpt-4.1-mini",
                    )
                ],
                raw_request={"question": "How many sixes?"},
                raw_response={"answer": "Tilak hit 3 sixes."},
            )
        ],
        metric_aggregates=[],
    )
    trend_summary = TrendSummary(
        generated_at="2026-02-12T10:05:00+00:00",
        keep_last_n=5,
        metrics=[
            MetricTrend(
                metric_name="contextual_precision",
                points=[
                    TrendPoint(
                        run_id="RUN_001",
                        timestamp="2026-02-12T10:00:00+00:00",
                        avg_score=0.9,
                        pass_rate=100.0,
                        threshold=0.6,
                    )
                ],
            )
        ],
    )

    output = write_executive_html(
        run_results=[run],
        trend_summary=trend_summary,
        output_path=tmp_path / "reports" / "index.html",
    )
    html = output.read_text()

    assert "RAG Evaluation Executive Report" in html
    assert "Evaluation Results" in html
    assert "Metric" in html
    assert "RunID" in html
    assert "PASS / FAIL / N/A" in html
    assert "count-pass" in html
    assert "count-fail" in html
    assert "count-na" in html
    assert "Technical Logs" in html
    assert "Complete Technical Logs" in html
    assert "Live Data Pass Rate" in html
    assert "Retrieved Context" in html
    assert "contextModal" in html
    assert "context-link" in html
    assert "Open Trend Dashboard (Last 5 Runs)" in html
    assert "../trends/last5.html" in html
    assert "Download Full Technical Logs (JSON)" in html
    assert (tmp_path / "reports" / "technical_logs.json").exists()


def test_metric_health_counts_aggregate_across_loaded_runs(tmp_path: Path):
    run_1 = RunResult(
        run_id="RUN_001",
        timestamp="2026-02-12T10:00:00+00:00",
        feature="feature_file.feature",
        scenario="test_evaluate_layer1_contextual_metrics_from_inline_dataset_table",
        tags=["layer1", "live"],
        selected_metrics=["contextual_precision"],
        dataset_size=1,
        question_results=[
            QuestionEvalResult(
                question_id="Q1",
                question="Question 1",
                expected_answer="Expected 1",
                actual_answer="Actual 1",
                retrieval_context=["chunk 1"],
                metrics=[
                    MetricResult(
                        metric_name="contextual_precision",
                        threshold=0.6,
                        score=0.9,
                        passed=True,
                        reason="Reason 1",
                    )
                ],
                raw_request={},
                raw_response={},
            )
        ],
        metric_aggregates=[],
    )
    run_2 = RunResult(
        run_id="RUN_002",
        timestamp="2026-02-12T10:10:00+00:00",
        feature="feature_file.feature",
        scenario="test_evaluate_layer1_contextual_metrics_from_external_dataset_file",
        tags=["layer1", "live"],
        selected_metrics=["contextual_precision"],
        dataset_size=1,
        question_results=[
            QuestionEvalResult(
                question_id="Q2",
                question="Question 2",
                expected_answer="Expected 2",
                actual_answer="Actual 2",
                retrieval_context=["chunk 2"],
                metrics=[
                    MetricResult(
                        metric_name="contextual_precision",
                        threshold=0.6,
                        score=0.8,
                        passed=True,
                        reason="Reason 2",
                    )
                ],
                raw_request={},
                raw_response={},
            )
        ],
        metric_aggregates=[],
    )
    trend_summary = TrendSummary(
        generated_at="2026-02-12T10:15:00+00:00",
        keep_last_n=5,
        metrics=[
            MetricTrend(
                metric_name="contextual_precision",
                points=[
                    TrendPoint(
                        run_id="RUN_001",
                        timestamp="2026-02-12T10:00:00+00:00",
                        avg_score=0.9,
                        pass_rate=100.0,
                        threshold=0.6,
                    ),
                    TrendPoint(
                        run_id="RUN_002",
                        timestamp="2026-02-12T10:10:00+00:00",
                        avg_score=0.8,
                        pass_rate=100.0,
                        threshold=0.6,
                    ),
                ],
            )
        ],
    )

    output = write_executive_html(
        run_results=[run_1, run_2],
        trend_summary=trend_summary,
        output_path=tmp_path / "reports" / "index.html",
    )
    html = output.read_text()
    metric_row_match = re.search(
        r"<tr><td>Contextual Precision</td>.*?</tr>",
        html,
        flags=re.DOTALL,
    )
    assert metric_row_match is not None
    metric_row = metric_row_match.group(0)
    assert "<span class='count-pill count-pass'>2</span>" in metric_row
    assert "<span class='count-pill count-fail'>0</span>" in metric_row
    assert "<span class='count-pill count-na'>0</span>" in metric_row
    assert "<span class='badge badge-pass'>PASS</span>" in metric_row


def test_metric_health_status_fails_when_metric_has_any_failed_rows(tmp_path: Path):
    run_1 = RunResult(
        run_id="RUN_001",
        timestamp="2026-02-12T10:00:00+00:00",
        feature="feature_file.feature",
        scenario="test_evaluate_layer1_contextual_metrics_from_inline_dataset_table",
        tags=["layer1", "live"],
        selected_metrics=["contextual_precision"],
        dataset_size=1,
        question_results=[
            QuestionEvalResult(
                question_id="Q1",
                question="Question 1",
                expected_answer="Expected 1",
                actual_answer="Actual 1",
                retrieval_context=["chunk 1"],
                metrics=[
                    MetricResult(
                        metric_name="contextual_precision",
                        threshold=0.6,
                        score=0.2,
                        passed=False,
                        reason="Reason 1",
                    )
                ],
                raw_request={},
                raw_response={},
            )
        ],
        metric_aggregates=[],
    )
    run_2 = RunResult(
        run_id="RUN_002",
        timestamp="2026-02-12T10:10:00+00:00",
        feature="feature_file.feature",
        scenario="test_evaluate_layer1_contextual_metrics_from_external_dataset_file",
        tags=["layer1", "live"],
        selected_metrics=["contextual_precision"],
        dataset_size=1,
        question_results=[
            QuestionEvalResult(
                question_id="Q2",
                question="Question 2",
                expected_answer="Expected 2",
                actual_answer="Actual 2",
                retrieval_context=["chunk 2"],
                metrics=[
                    MetricResult(
                        metric_name="contextual_precision",
                        threshold=0.6,
                        score=1.0,
                        passed=True,
                        reason="Reason 2",
                    )
                ],
                raw_request={},
                raw_response={},
            )
        ],
        metric_aggregates=[],
    )
    # Latest trend point is PASS, but metric health status should still be FAIL
    # because the displayed metric counts include at least one failed row.
    trend_summary = TrendSummary(
        generated_at="2026-02-12T10:15:00+00:00",
        keep_last_n=5,
        metrics=[
            MetricTrend(
                metric_name="contextual_precision",
                points=[
                    TrendPoint(
                        run_id="RUN_001",
                        timestamp="2026-02-12T10:00:00+00:00",
                        avg_score=0.2,
                        pass_rate=0.0,
                        threshold=0.6,
                    ),
                    TrendPoint(
                        run_id="RUN_002",
                        timestamp="2026-02-12T10:10:00+00:00",
                        avg_score=1.0,
                        pass_rate=100.0,
                        threshold=0.6,
                    ),
                ],
            )
        ],
    )

    output = write_executive_html(
        run_results=[run_1, run_2],
        trend_summary=trend_summary,
        output_path=tmp_path / "reports" / "index.html",
    )
    html = output.read_text()
    metric_row_match = re.search(
        r"<tr><td>Contextual Precision</td>.*?</tr>",
        html,
        flags=re.DOTALL,
    )
    assert metric_row_match is not None
    metric_row = metric_row_match.group(0)
    assert "<span class='count-pill count-pass'>1</span>" in metric_row
    assert "<span class='count-pill count-fail'>1</span>" in metric_row
    assert "<span class='badge badge-fail'>FAIL</span>" in metric_row


def test_top_failure_reasons_prioritizes_quality_context_over_transient_retry_errors(tmp_path: Path):
    question_results = []
    fail_reasons = [
        "RetryError[<Future state=finished raised TimeoutError>]",
        "RetryError[<Future state=finished raised APIConnectionError>]",
        "The response does not address the question.",
        "The response does not address the question.",
        "The response does not address the question.",
        "The answer misses key expected details.",
    ]

    for index, reason in enumerate(fail_reasons, start=1):
        question_results.append(
            QuestionEvalResult(
                question_id=f"Q{index}",
                question=f"Question {index}",
                expected_answer=f"Expected {index}",
                actual_answer=f"Actual {index}",
                retrieval_context=["chunk"],
                metrics=[
                    MetricResult(
                        metric_name="completeness",
                        threshold=0.6,
                        score=0.1,
                        passed=False,
                        reason=reason,
                    )
                ],
                raw_request={},
                raw_response={},
            )
        )

    run = RunResult(
        run_id="RUN_FAILS",
        timestamp="2026-03-13T20:00:00+00:00",
        feature="feature_file.feature",
        scenario="test_evaluate_layer2_answer_metrics_from_external_dataset_file",
        tags=["layer2"],
        selected_metrics=["completeness"],
        dataset_size=len(question_results),
        question_results=question_results,
        metric_aggregates=[],
    )
    trend_summary = TrendSummary(
        generated_at="2026-03-13T20:05:00+00:00",
        keep_last_n=5,
        metrics=[
            MetricTrend(
                metric_name="completeness",
                points=[
                    TrendPoint(
                        run_id="RUN_FAILS",
                        timestamp="2026-03-13T20:00:00+00:00",
                        avg_score=0.1,
                        pass_rate=0.0,
                        threshold=0.6,
                    )
                ],
            )
        ],
    )

    output = write_executive_html(
        run_results=[run],
        trend_summary=trend_summary,
        output_path=tmp_path / "reports" / "index.html",
    )
    html = output.read_text()

    top_reason_match = re.search(
        r"<span class=\"label\">Top Failure Reasons</span><span class=\"value\" style=\"font-size: 14px;\">(.*?)</span>",
        html,
        flags=re.DOTALL,
    )
    assert top_reason_match is not None
    top_reasons_html = top_reason_match.group(1)

    assert "3x The response does not address the question." in top_reasons_html
    assert "1x The answer misses key expected details." in top_reasons_html
    assert "2x Transient backend/API retry errors" in top_reasons_html
    assert "RetryError" not in top_reasons_html
