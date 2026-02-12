from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from rag_eval_bdd.models import MetricTrend, RunIndexEntry, RunResult, TrendPoint, TrendSummary


class ResultsStore:
    def __init__(self, base_dir: Path, keep_last_n: int = 5):
        self.base_dir = base_dir
        self.keep_last_n = keep_last_n
        self.runs_dir = base_dir / "runs"
        self.trends_dir = base_dir / "trends"
        self.index_file = base_dir / "index.json"
        self.current_index_file = base_dir / "current_index.json"

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.trends_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_index_file(self.index_file)
        self._ensure_index_file(self.current_index_file)

    def save_run(self, run_result: RunResult) -> Tuple[Path, TrendSummary]:
        run_dir = self.runs_dir / run_result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        results_file = run_dir / "results.json"
        results_file.write_text(run_result.model_dump_json(indent=2))

        entries = self._load_index_entries()
        new_entry = RunIndexEntry(
            run_id=run_result.run_id,
            timestamp=run_result.timestamp,
            path=str(results_file.relative_to(self.base_dir)),
            feature=run_result.feature,
            scenario=run_result.scenario,
        )

        entries.insert(0, new_entry)
        entries = entries[: self.keep_last_n]
        self._write_index_entries(entries)

        current_entries = self._load_current_entries()
        current_entries.insert(0, new_entry)
        self._write_current_entries(current_entries)

        trend_summary = self._build_trends(entries)
        trend_file = self.trends_dir / "last5.json"
        trend_file.write_text(trend_summary.model_dump_json(indent=2))

        return run_dir, trend_summary

    def load_recent_run_results(self) -> List[RunResult]:
        run_results: List[RunResult] = []
        entries = self._load_index_entries()
        for entry in entries:
            run_file = self.base_dir / entry.path
            if not run_file.exists():
                continue
            payload = json.loads(run_file.read_text())
            run_results.append(RunResult.model_validate(payload))
        return run_results

    def load_current_session_run_results(self) -> List[RunResult]:
        run_results: List[RunResult] = []
        entries = self._load_current_entries()
        for entry in entries:
            run_file = self.base_dir / entry.path
            if not run_file.exists():
                continue
            payload = json.loads(run_file.read_text())
            run_results.append(RunResult.model_validate(payload))
        return run_results

    def reset_current_session(self) -> None:
        self._write_current_entries([])

    def refresh_trends(self) -> TrendSummary:
        entries = self._load_index_entries()
        trend_summary = self._build_trends(entries)
        trend_file = self.trends_dir / "last5.json"
        trend_file.write_text(trend_summary.model_dump_json(indent=2))
        return trend_summary

    def _load_index_entries(self) -> List[RunIndexEntry]:
        return self._load_entries(self.index_file)

    def _load_current_entries(self) -> List[RunIndexEntry]:
        return self._load_entries(self.current_index_file)

    def _write_index_entries(self, entries: List[RunIndexEntry]) -> None:
        self._write_entries(self.index_file, entries)

    def _write_current_entries(self, entries: List[RunIndexEntry]) -> None:
        self._write_entries(self.current_index_file, entries)

    def _ensure_index_file(self, index_file: Path) -> None:
        if not index_file.exists():
            index_file.write_text('{"runs": []}\n')

    def _load_entries(self, index_file: Path) -> List[RunIndexEntry]:
        raw = json.loads(index_file.read_text() or '{"runs": []}')
        rows = raw.get("runs", [])
        return [RunIndexEntry.model_validate(row) for row in rows]

    def _write_entries(self, index_file: Path, entries: List[RunIndexEntry]) -> None:
        payload = {"runs": [entry.model_dump() for entry in entries]}
        index_file.write_text(json.dumps(payload, indent=2))

    def _build_trends(self, entries: List[RunIndexEntry]) -> TrendSummary:
        metric_map: Dict[str, List[TrendPoint]] = {}

        # Oldest to newest for readable trend charts
        for entry in reversed(entries):
            run_file = self.base_dir / entry.path
            if not run_file.exists():
                continue
            run_payload = json.loads(run_file.read_text())
            run_result = RunResult.model_validate(run_payload)

            for aggregate in run_result.metric_aggregates:
                metric_map.setdefault(aggregate.metric_name, []).append(
                    TrendPoint(
                        run_id=run_result.run_id,
                        timestamp=run_result.timestamp,
                        avg_score=aggregate.avg_score,
                        pass_rate=aggregate.pass_rate,
                        threshold=aggregate.threshold,
                    )
                )

        trends = [MetricTrend(metric_name=metric, points=points) for metric, points in sorted(metric_map.items())]
        return TrendSummary(
            generated_at=datetime.now(timezone.utc).isoformat(),
            keep_last_n=self.keep_last_n,
            metrics=trends,
        )
