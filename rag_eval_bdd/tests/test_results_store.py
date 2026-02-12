from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from rag_eval_bdd.models import MetricAggregate, RunResult
from rag_eval_bdd.results_store import ResultsStore

pytestmark = [pytest.mark.smoke]


def _sample_run(run_id: str, avg_score: float) -> RunResult:
    now = datetime.now(timezone.utc).isoformat()
    return RunResult(
        run_id=run_id,
        timestamp=now,
        feature="feature",
        scenario="scenario",
        tags=["layer1"],
        selected_metrics=["contextual_precision"],
        dataset_size=2,
        question_results=[],
        metric_aggregates=[
            MetricAggregate(
                metric_name="contextual_precision",
                threshold=0.7,
                count=2,
                scored_count=2,
                pass_count=2 if avg_score >= 0.7 else 0,
                fail_count=0 if avg_score >= 0.7 else 2,
                pass_rate=100.0 if avg_score >= 0.7 else 0.0,
                avg_score=avg_score,
                min_score=avg_score,
                max_score=avg_score,
                std_dev=0.0,
                p50=avg_score,
                p90=avg_score,
                score_distribution=[avg_score, avg_score],
            )
        ],
    )


def test_results_store_generates_last5(tmp_path: Path):
    store = ResultsStore(base_dir=tmp_path, keep_last_n=5)

    for idx, score in enumerate([0.71, 0.75, 0.69], start=1):
        run = _sample_run(f"RUN_{idx}", score)
        store.save_run(run)

    trend_file = tmp_path / "trends" / "last5.json"
    assert trend_file.exists()
    payload = trend_file.read_text()
    assert "contextual_precision" in payload
    recent_runs = store.load_recent_run_results()
    assert len(recent_runs) == 3
    current_runs = store.load_current_session_run_results()
    assert len(current_runs) == 3
    refreshed = store.refresh_trends()
    assert refreshed.metrics


def test_results_store_can_reset_current_session_runs(tmp_path: Path):
    store = ResultsStore(base_dir=tmp_path, keep_last_n=5)

    store.save_run(_sample_run("RUN_A", 0.72))
    store.save_run(_sample_run("RUN_B", 0.73))
    assert len(store.load_current_session_run_results()) == 2

    store.reset_current_session()
    assert store.load_current_session_run_results() == []

    # Historical trend index remains untouched after current-session reset.
    assert len(store.load_recent_run_results()) == 2
