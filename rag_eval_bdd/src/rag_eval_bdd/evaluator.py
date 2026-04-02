from __future__ import annotations

import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List
from uuid import uuid4

from deepeval.test_case import LLMTestCase

from rag_eval_bdd.backend_client import PERF_METADATA_KEY, BackendClient
from rag_eval_bdd.metric_registry import build_metric, metric_threshold, normalize_metric_name
from rag_eval_bdd.models import (
    AppConfig,
    DatasetRow,
    MetricAggregate,
    MetricResult,
    QuestionEvalResult,
    RunPerformanceAggregate,
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
    _REDACTION_RULES: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"(?i)(bearer\s+)[a-z0-9\-._~+/]+=*"), r"\1[REDACTED]"),
        (re.compile(r"(?i)(api[_-]?key[\"'\s:=]+)[a-z0-9\-._~+/]+=*"), r"\1[REDACTED]"),
        (re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ]

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

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not value.is_integer():
                return None
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if text.isdigit():
                return int(text)
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

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

    def _sanitize_text(self, value: str) -> str:
        text = str(value)
        if not self.config.evaluation.redact_sensitive_logs:
            return text
        sanitized = text
        for pattern, replacement in self._REDACTION_RULES:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    def _sanitize_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._sanitize_payload(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_payload(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_payload(item) for item in value]
        if isinstance(value, str):
            return self._sanitize_text(value)
        return value

    def _prepare_logged_retrieval_context(self, chunks: List[str]) -> List[str]:
        max_chars = max(100, int(self.config.evaluation.max_logged_retrieval_context_chars))
        log_full = bool(self.config.evaluation.log_full_retrieval_context)
        prepared: List[str] = []
        for chunk in chunks:
            item = str(chunk)
            if not log_full:
                item = item[:max_chars]
            prepared.append(self._sanitize_text(item))
        return prepared

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

    @staticmethod
    def _is_not_found_answer(text: str | None) -> bool:
        if not text:
            return False
        normalized = str(text).strip().lower()
        return normalized in {
            "not found in document.",
            "not found in the document.",
        }

    def _force_fail_for_not_found(
        self,
        metric_name: str,
        expected_answer: str | None,
        actual_answer: str,
    ) -> bool:
        if metric_name not in {
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "answer_relevancy",
            "faithfulness",
            "completeness",
        }:
            return False
        if not self._is_not_found_answer(actual_answer):
            return False
        if not expected_answer or self._is_not_found_answer(expected_answer):
            return False
        return True

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
            perf_payload: dict[str, Any] = {}
            if isinstance(raw_response, dict):
                raw_perf = raw_response.get(PERF_METADATA_KEY)
                if isinstance(raw_perf, dict):
                    perf_payload = raw_perf

            answer = str(raw_response.get("answer", ""))
            retrieval_context = raw_response.get("retrieval_context", []) or []
            trimmed_retrieval_context = self._trim_retrieval_context(retrieval_context)
            logged_retrieval_context = self._prepare_logged_retrieval_context(trimmed_retrieval_context)
            expected_output = row.expected_answer or answer
            latency_ms = self._coerce_float(perf_payload.get("latency_ms"))
            cache_hit = perf_payload.get("cache_hit")
            if not isinstance(cache_hit, bool):
                cache_hit = None
            prompt_tokens = self._coerce_int(perf_payload.get("prompt_tokens"))
            completion_tokens = self._coerce_int(perf_payload.get("completion_tokens"))
            total_tokens = self._coerce_int(perf_payload.get("total_tokens"))
            token_cost_usd = self._coerce_float(perf_payload.get("token_cost_usd"))

            if self.config.evaluation.log_raw_payloads:
                logged_request = self._sanitize_payload(raw_request)
                logged_response = self._sanitize_payload(raw_response)
            else:
                logged_request = {}
                logged_response = {}

            test_case = LLMTestCase(
                input=row.question,
                actual_output=answer,
                expected_output=expected_output,
                retrieval_context=trimmed_retrieval_context,
                completion_time=(latency_ms / 1000.0) if latency_ms is not None else None,
                token_cost=token_cost_usd,
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

                if self._force_fail_for_not_found(
                    metric_name=canonical_name,
                    expected_answer=row.expected_answer,
                    actual_answer=answer,
                ):
                    metric_results.append(
                        MetricResult(
                            metric_name=canonical_name,
                            threshold=threshold,
                            score=0.0,
                            passed=False,
                            reason=(
                                "Forced FAIL: actual output was 'Not found in document.' "
                                "while expected_answer was provided."
                            ),
                            error=None,
                            evaluation_model=None,
                        )
                    )
                    continue

                metric = build_metric(canonical_name, self.config)

                try:
                    metric.measure(test_case)
                    score = getattr(metric, "score", None)
                    passed = getattr(metric, "success", None)
                    if passed is None and isinstance(score, (int, float)):
                        passed = float(score) >= threshold
                    reason = getattr(metric, "reason", None)
                    error = getattr(metric, "error", None)

                    metric_results.append(
                        MetricResult(
                            metric_name=canonical_name,
                            threshold=threshold,
                            score=float(score) if isinstance(score, (int, float)) else None,
                            passed=bool(passed) if passed is not None else None,
                            reason=self._sanitize_text(reason) if reason else None,
                            error=self._sanitize_text(error) if error else None,
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
                            error=self._sanitize_text(str(exc)),
                            evaluation_model=getattr(metric, "evaluation_model", None),
                        )
                    )

            question_results.append(
                QuestionEvalResult(
                    question_id=row.id,
                    question=row.question,
                    expected_answer=row.expected_answer,
                    actual_answer=answer,
                    retrieval_context=logged_retrieval_context,
                    category=row.category,
                    source_reference=row.source_reference,
                    metrics=metric_results,
                    raw_request=logged_request,
                    raw_response=logged_response,
                    latency_ms=latency_ms,
                    cache_hit=cache_hit,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    token_cost_usd=token_cost_usd,
                )
            )

        aggregates = self._aggregate(question_results=question_results, selected_metrics=selected_metrics)
        performance = self._aggregate_performance(question_results=question_results)

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
            performance=performance,
        )

    def _aggregate_performance(self, question_results: List[QuestionEvalResult]) -> RunPerformanceAggregate:
        request_count = len(question_results)
        latencies = [q.latency_ms for q in question_results if isinstance(q.latency_ms, (int, float))]
        cached_latencies = [
            q.latency_ms for q in question_results if q.cache_hit is True and isinstance(q.latency_ms, (int, float))
        ]
        uncached_latencies = [
            q.latency_ms for q in question_results if q.cache_hit is False and isinstance(q.latency_ms, (int, float))
        ]
        prompt_tokens = [q.prompt_tokens for q in question_results if isinstance(q.prompt_tokens, int)]
        completion_tokens = [q.completion_tokens for q in question_results if isinstance(q.completion_tokens, int)]
        total_tokens = [q.total_tokens for q in question_results if isinstance(q.total_tokens, int)]
        token_costs = [q.token_cost_usd for q in question_results if isinstance(q.token_cost_usd, (int, float))]

        avg_latency_ms = float(sum(latencies) / len(latencies)) if latencies else None
        p50_latency_ms = _percentile(latencies, 50) if latencies else None
        p90_latency_ms = _percentile(latencies, 90) if latencies else None
        p95_latency_ms = _percentile(latencies, 95) if latencies else None
        max_latency_ms = float(max(latencies)) if latencies else None
        avg_cached_latency_ms = float(sum(cached_latencies) / len(cached_latencies)) if cached_latencies else None
        avg_uncached_latency_ms = float(sum(uncached_latencies) / len(uncached_latencies)) if uncached_latencies else None
        avg_total_tokens_per_request = float(sum(total_tokens) / len(total_tokens)) if total_tokens else None
        total_token_cost_usd = float(sum(float(value) for value in token_costs)) if token_costs else None

        return RunPerformanceAggregate(
            request_count=request_count,
            cached_request_count=sum(1 for q in question_results if q.cache_hit is True),
            uncached_request_count=sum(1 for q in question_results if q.cache_hit is False),
            latency_count=len(latencies),
            avg_latency_ms=avg_latency_ms,
            p50_latency_ms=p50_latency_ms,
            p90_latency_ms=p90_latency_ms,
            p95_latency_ms=p95_latency_ms,
            max_latency_ms=max_latency_ms,
            avg_cached_latency_ms=avg_cached_latency_ms,
            avg_uncached_latency_ms=avg_uncached_latency_ms,
            token_usage_count=len(total_tokens),
            total_prompt_tokens=sum(prompt_tokens) if prompt_tokens else None,
            total_completion_tokens=sum(completion_tokens) if completion_tokens else None,
            total_tokens=sum(total_tokens) if total_tokens else None,
            avg_total_tokens_per_request=avg_total_tokens_per_request,
            total_token_cost_usd=total_token_cost_usd,
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
