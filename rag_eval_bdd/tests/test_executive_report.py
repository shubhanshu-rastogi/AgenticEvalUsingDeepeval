from __future__ import annotations

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
    assert "Download Full Technical Logs (JSON)" in html
    assert (tmp_path / "reports" / "technical_logs.json").exists()
