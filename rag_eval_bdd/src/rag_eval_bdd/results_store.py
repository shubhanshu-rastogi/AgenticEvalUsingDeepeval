from __future__ import annotations

from contextlib import contextmanager
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

try:
    import fcntl
except Exception:  # noqa: BLE001
    fcntl = None

from rag_eval_bdd.models import MetricTrend, RunIndexEntry, RunResult, TrendPoint, TrendSummary

TREND_RAW_HISTORY_MULTIPLIER = 8


class ResultsStore:
    def __init__(self, base_dir: Path, keep_last_n: int = 5):
        self.base_dir = base_dir
        self.keep_last_n = keep_last_n
        self.runs_dir = base_dir / "runs"
        self.trends_dir = base_dir / "trends"
        self.index_file = base_dir / "index.json"
        self.current_index_file = base_dir / "current_index.json"
        self.lock_file = base_dir / ".results_store.lock"

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.trends_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.touch(exist_ok=True)
        self._ensure_index_file(self.index_file)
        self._ensure_index_file(self.current_index_file)

    def save_run(self, run_result: RunResult) -> Tuple[Path, TrendSummary]:
        run_dir = self.runs_dir / run_result.run_id
        results_file = run_dir / "results.json"
        new_entry = RunIndexEntry(
            run_id=run_result.run_id,
            timestamp=run_result.timestamp,
            path=str(results_file.relative_to(self.base_dir)),
            feature=run_result.feature,
            scenario=run_result.scenario,
        )

        with self._exclusive_lock():
            run_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_text(results_file, run_result.model_dump_json(indent=2))

            entries = self._upsert_entry(
                entries=self._load_index_entries(),
                new_entry=new_entry,
                limit=self._trend_history_limit(),
            )
            self._write_index_entries(entries)

            current_entries = self._upsert_entry(
                entries=self._load_current_entries(),
                new_entry=new_entry,
                limit=0,
            )
            self._write_current_entries(current_entries)

            trend_summary = self._build_trends(entries)
            trend_file = self.trends_dir / "last5.json"
            self._atomic_write_text(trend_file, trend_summary.model_dump_json(indent=2))

        return run_dir, trend_summary

    def load_recent_run_results(self) -> List[RunResult]:
        run_results: List[RunResult] = []
        entries = self._load_index_entries()[: self.keep_last_n]
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
        with self._exclusive_lock():
            self._write_current_entries([])

    def refresh_trends(self) -> TrendSummary:
        with self._exclusive_lock():
            entries = self._load_index_entries()
            if not entries:
                entries = self._rebuild_index_entries(limit=self._trend_history_limit())
            entries.sort(key=lambda entry: entry.timestamp, reverse=True)
            entries = entries[: self._trend_history_limit()]
            self._write_index_entries(entries)
            trend_summary = self._build_trends(entries)
            trend_file = self.trends_dir / "last5.json"
            self._atomic_write_text(trend_file, trend_summary.model_dump_json(indent=2))
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
            self._atomic_write_text(index_file, '{"runs": []}\n')

    def _load_entries(self, index_file: Path) -> List[RunIndexEntry]:
        raw = json.loads(index_file.read_text() or '{"runs": []}')
        rows = raw.get("runs", [])
        return [RunIndexEntry.model_validate(row) for row in rows]

    def _write_entries(self, index_file: Path, entries: List[RunIndexEntry]) -> None:
        payload = {"runs": [entry.model_dump() for entry in entries]}
        self._atomic_write_text(index_file, json.dumps(payload, indent=2))

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        with self.lock_file.open("a+") as lock_handle:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(content)
        tmp_path.replace(path)

    def _upsert_entry(
        self,
        entries: List[RunIndexEntry],
        new_entry: RunIndexEntry,
        limit: int,
    ) -> List[RunIndexEntry]:
        deduped = [entry for entry in entries if entry.run_id != new_entry.run_id]
        deduped.append(new_entry)
        deduped.sort(key=lambda entry: entry.timestamp, reverse=True)
        if limit > 0:
            deduped = deduped[:limit]
        return deduped

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

    def _trend_history_limit(self) -> int:
        return max(self.keep_last_n, self.keep_last_n * TREND_RAW_HISTORY_MULTIPLIER)

    def _rebuild_index_entries(self, limit: int) -> List[RunIndexEntry]:
        entries: List[RunIndexEntry] = []
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_file = run_dir / "results.json"
            if not run_file.exists():
                continue
            try:
                run_payload = json.loads(run_file.read_text())
                run_result = RunResult.model_validate(run_payload)
            except Exception:  # noqa: BLE001
                continue

            entries.append(
                RunIndexEntry(
                    run_id=run_result.run_id,
                    timestamp=run_result.timestamp,
                    path=str(run_file.relative_to(self.base_dir)),
                    feature=run_result.feature,
                    scenario=run_result.scenario,
                )
            )

        entries.sort(key=lambda entry: entry.timestamp, reverse=True)
        if limit > 0:
            entries = entries[:limit]
        self._write_index_entries(entries)
        return entries
