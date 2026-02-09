from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

from deepeval.test_case import LLMTestCase

from rag_eval_bdd.backend_client import BackendClient
from rag_eval_bdd.metric_registry import build_metric, metric_threshold, normalize_metric_name
from rag_eval_bdd.models import (
    AppConfig,
    DatasetRow,
    MetricAggregate,
    MetricResult,
    QuestionEvalResult,
    RunResult,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100) * len(ordered)) - 1))
    return float(ordered[idx])


class EvaluationRunner:
    def __init__(self, client: BackendClient, config: AppConfig):
        self.client = client
        self.config = config

    def _trim_retrieval_context(self, retrieval_context: List[object]) -> List[str]:
        if self.config.evaluation.disable_context_trimming:
            return [str(chunk) for chunk in list(retrieval_context)]

        chunks_limit = max(1, int(self.config.evaluation.max_retrieval_context_chunks))
        chars_limit = max(100, int(self.config.evaluation.max_retrieval_context_chars_per_chunk))

        trimmed: List[str] = []
        for chunk in list(retrieval_context)[:chunks_limit]:
            text = str(chunk)
            trimmed.append(text[:chars_limit])
        return trimmed

    def _resolve_row_metrics(
        self,
        row: DatasetRow,
        selected_metrics: List[str],
        row_index: int,
        total_rows: int,
    ) -> List[str]:
        mode = self.config.evaluation.metric_question_mapping_mode
        if mode == "row":
            mapped_metrics = self._extract_metrics_from_row(row=row)
            if mapped_metrics:
                return mapped_metrics
        if mode == "positional" and total_rows == len(selected_metrics):
            return [selected_metrics[row_index]]
        return selected_metrics

    def _extract_metrics_from_row(self, row: DatasetRow) -> List[str]:
        if not row.additional_metadata:
            return []

        key_candidates = {
            "metric",
            "metrics",
            "metric_name",
            "metric_names",
            "target_metric",
            "target_metrics",
        }

        raw_value = None
        for key, value in row.additional_metadata.items():
            if str(key).strip().lower() in key_candidates:
                raw_value = value
                break

        if raw_value is None:
            return []

        if isinstance(raw_value, list):
            parts = [str(part).strip() for part in raw_value]
        else:
            parts = [part.strip() for part in str(raw_value).split(",")]

        mapped: List[str] = []
        for part in parts:
            if not part:
                continue
            canonical = normalize_metric_name(part)
            if not hasattr(self.config.thresholds, canonical):
                continue
            if canonical not in mapped:
                mapped.append(canonical)
        return mapped

    def evaluate_dataset(
        self,
        dataset_rows: Iterable[DatasetRow],
        selected_metrics: List[str],
        session_id: str | None,
        feature: str,
        scenario: str,
        tags: List[str],
        uploaded_documents: List[str] | None = None,
    ) -> RunResult:
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        question_results: List[QuestionEvalResult] = []
        uploaded_documents = uploaded_documents or []

        rows = list(dataset_rows)
        for row_index, row in enumerate(rows):
            row_session_id = session_id
            if self.config.evaluation.fresh_session_per_question:
                if uploaded_documents:
                    upload_path = Path(uploaded_documents[0]).resolve()
                    row_session_id, _ = self.client.upload_document(upload_path)
            if not row_session_id:
                raise ValueError("Missing session_id for evaluation. Upload documents before running metrics.")

            raw_request, raw_response = self.client.ask_question(
                session_id=row_session_id,
                question=row.question,
                use_cache=self.config.evaluation.cache_ask_responses,
            )

            answer = str(raw_response.get("answer", ""))
            retrieval_context = raw_response.get("retrieval_context", []) or []
            trimmed_retrieval_context = self._trim_retrieval_context(retrieval_context)
            expected_output = row.expected_answer or answer

            test_case = LLMTestCase(
                input=row.question,
                actual_output=answer,
                expected_output=expected_output,
                retrieval_context=trimmed_retrieval_context,
            )

            metric_results: List[MetricResult] = []
            row_metrics = self._resolve_row_metrics(
                row=row,
                selected_metrics=selected_metrics,
                row_index=row_index,
                total_rows=len(rows),
            )
            for metric_name in row_metrics:
                canonical_name = normalize_metric_name(metric_name)
                threshold = metric_threshold(canonical_name, self.config)
                metric = build_metric(canonical_name, self.config)

                try:
                    metric.measure(test_case)
                    score = getattr(metric, "score", None)
                    passed = getattr(metric, "success", None)
                    if passed is None and isinstance(score, (int, float)):
                        passed = float(score) >= threshold

                    metric_results.append(
                        MetricResult(
                            metric_name=canonical_name,
                            threshold=threshold,
                            score=float(score) if isinstance(score, (int, float)) else None,
                            passed=bool(passed) if passed is not None else None,
                            reason=getattr(metric, "reason", None),
                            error=getattr(metric, "error", None),
                            evaluation_model=getattr(metric, "evaluation_model", None),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    metric_results.append(
                        MetricResult(
                            metric_name=canonical_name,
                            threshold=threshold,
                            score=None,
                            passed=False,
                            reason=None,
                            error=str(exc),
                            evaluation_model=getattr(metric, "evaluation_model", None),
                        )
                    )

            question_results.append(
                QuestionEvalResult(
                    question_id=row.id,
                    question=row.question,
                    expected_answer=row.expected_answer,
                    actual_answer=answer,
                    retrieval_context=trimmed_retrieval_context,
                    category=row.category,
                    source_reference=row.source_reference,
                    metrics=metric_results,
                    raw_request=raw_request,
                    raw_response=raw_response,
                )
            )

        aggregates = self._aggregate(question_results=question_results, selected_metrics=selected_metrics)

        return RunResult(
            run_id=run_id,
            timestamp=_now_iso(),
            feature=feature,
            scenario=scenario,
            tags=tags,
            selected_metrics=selected_metrics,
            dataset_size=len(rows),
            question_results=question_results,
            metric_aggregates=aggregates,
        )

    def _aggregate(self, question_results: List[QuestionEvalResult], selected_metrics: List[str]) -> List[MetricAggregate]:
        aggregates: List[MetricAggregate] = []

        for metric_name in selected_metrics:
            canonical_name = normalize_metric_name(metric_name)
            threshold = metric_threshold(canonical_name, self.config)

            metric_results = [
                metric_result
                for question in question_results
                for metric_result in question.metrics
                if metric_result.metric_name == canonical_name
            ]

            scores = [m.score for m in metric_results if isinstance(m.score, (int, float))]
            pass_count = sum(1 for m in metric_results if m.passed is True)
            fail_count = sum(1 for m in metric_results if m.passed is False)
            count = len(metric_results)
            pass_rate = (pass_count / count * 100.0) if count else 0.0

            avg_score = float(sum(scores) / len(scores)) if scores else None
            min_score = float(min(scores)) if scores else None
            max_score = float(max(scores)) if scores else None
            std_dev = float(statistics.pstdev(scores)) if len(scores) > 1 else (0.0 if len(scores) == 1 else None)
            p50 = _percentile(scores, 50) if scores else None
            p90 = _percentile(scores, 90) if scores else None

            aggregates.append(
                MetricAggregate(
                    metric_name=canonical_name,
                    threshold=threshold,
                    count=count,
                    scored_count=len(scores),
                    pass_count=pass_count,
                    fail_count=fail_count,
                    pass_rate=pass_rate,
                    avg_score=avg_score,
                    min_score=min_score,
                    max_score=max_score,
                    std_dev=std_dev,
                    p50=p50,
                    p90=p90,
                    score_distribution=[float(x) for x in scores],
                )
            )

        return aggregates
