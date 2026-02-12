from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_bdd.models import MetricTrend, TrendPoint, TrendSummary
from rag_eval_bdd.reporting import write_trend_html

pytestmark = [pytest.mark.smoke]


def test_write_trend_html_contains_colored_dashboard(tmp_path: Path):
    summary = TrendSummary(
        generated_at="2026-02-09T16:00:00+00:00",
        keep_last_n=5,
        metrics=[
            MetricTrend(
                metric_name="completeness",
                points=[
                    TrendPoint(
                        run_id="20260209T150000Z_aaaa1111",
                        timestamp="2026-02-09T15:00:00+00:00",
                        avg_score=0.82,
                        pass_rate=100.0,
                        threshold=0.70,
                    ),
                    TrendPoint(
                        run_id="20260209T153000Z_bbbb2222",
                        timestamp="2026-02-09T15:30:00+00:00",
                        avg_score=0.60,
                        pass_rate=33.3,
                        threshold=0.70,
                    ),
                ],
            )
        ],
    )

    output = write_trend_html(summary, tmp_path / "last5.html")
    html = output.read_text()

    assert "RAG Eval Trend Dashboard" in html
    assert "status-fail" in html
    assert "trend-svg" in html
    assert "Consistency (1 SD)" in html


def test_write_trend_html_marks_fail_when_pass_rate_rule_not_met(tmp_path: Path):
    summary = TrendSummary(
        generated_at="2026-02-09T16:00:00+00:00",
        keep_last_n=5,
        metrics=[
            MetricTrend(
                metric_name="answer_relevancy",
                points=[
                    TrendPoint(
                        run_id="20260209T150000Z_aaaa1111",
                        timestamp="2026-02-09T15:00:00+00:00",
                        avg_score=0.95,
                        pass_rate=66.7,
                        threshold=0.70,
                    ),
                ],
            )
        ],
    )

    output = write_trend_html(
        summary,
        tmp_path / "last5.html",
        pass_rate_rule="min_pass_rate",
        min_pass_rate=100.0,
    )
    html = output.read_text()

    assert '<span class="status-pill status-fail">FAIL</span>' in html
    assert "66.70%" in html
