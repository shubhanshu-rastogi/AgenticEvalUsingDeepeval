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
    assert "combined-trend-svg" in html
    assert "All Metrics (Last" in html
    assert "Shared Threshold:" in html
    assert "smooth-line" in html
    assert "Consistency (1 SD)" in html


def test_write_trend_html_uses_threshold_only_status_even_when_pass_rate_rule_not_met(tmp_path: Path):
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

    assert '<span class="status-pill status-pass">PASS</span>' in html
    assert "66.70%" in html


def test_write_trend_html_collapses_close_runs_into_single_plot_point(tmp_path: Path):
    summary = TrendSummary(
        generated_at="2026-02-24T16:40:00+00:00",
        keep_last_n=5,
        metrics=[
            MetricTrend(
                metric_name="answer_relevancy",
                points=[
                    TrendPoint(
                        run_id="20260224T162500Z_aaaa1111",
                        timestamp="2026-02-24T16:25:00+00:00",
                        avg_score=0.90,
                        pass_rate=100.0,
                        threshold=0.50,
                    ),
                    TrendPoint(
                        run_id="20260224T162700Z_bbbb2222",
                        timestamp="2026-02-24T16:27:00+00:00",
                        avg_score=0.95,
                        pass_rate=100.0,
                        threshold=0.50,
                    ),
                    TrendPoint(
                        run_id="20260224T180000Z_cccc3333",
                        timestamp="2026-02-24T18:00:00+00:00",
                        avg_score=1.00,
                        pass_rate=100.0,
                        threshold=0.50,
                    ),
                ],
            )
        ],
    )

    output = write_trend_html(summary, tmp_path / "last5.html")
    html = output.read_text()

    # The first two points are within the session-cluster window and should collapse to one run.
    assert "All Metrics (Last 2 Runs)" in html
    assert "20260224T162500Z_aaaa1111" not in html
    assert "20260224T162700Z_bbbb2222" in html
    assert "20260224T180000Z_cccc3333" in html


def test_write_trend_html_does_not_chain_clusters_transitively(tmp_path: Path):
    summary = TrendSummary(
        generated_at="2026-02-24T16:40:00+00:00",
        keep_last_n=5,
        metrics=[
            MetricTrend(
                metric_name="answer_relevancy",
                points=[
                    TrendPoint(
                        run_id="20260224T160000Z_aaaa1111",
                        timestamp="2026-02-24T16:00:00+00:00",
                        avg_score=0.90,
                        pass_rate=100.0,
                        threshold=0.50,
                    ),
                    TrendPoint(
                        run_id="20260224T160400Z_bbbb2222",
                        timestamp="2026-02-24T16:04:00+00:00",
                        avg_score=0.95,
                        pass_rate=100.0,
                        threshold=0.50,
                    ),
                    TrendPoint(
                        run_id="20260224T160800Z_cccc3333",
                        timestamp="2026-02-24T16:08:00+00:00",
                        avg_score=1.00,
                        pass_rate=100.0,
                        threshold=0.50,
                    ),
                ],
            )
        ],
    )

    output = write_trend_html(summary, tmp_path / "last5.html")
    html = output.read_text()

    # 16:00 and 16:04 are one cluster, 16:08 is outside the 5-minute cluster-start window.
    assert "All Metrics (Last 2 Runs)" in html
    assert "20260224T160400Z_bbbb2222" in html
    assert "20260224T160800Z_cccc3333" in html
