from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    base_url: str = "http://localhost:8000"
    upload_endpoint: str = "/upload"
    ask_endpoint: str = "/ask"
    current_session_endpoint: str = "/session/current"
    session_chunks_endpoint: str = "/session/chunks"
    timeout_s: int = 120
    retries: int = 3
    backoff_s: float = 1.0


class ThresholdsConfig(BaseModel):
    contextual_precision: float = 0.70
    contextual_recall: float = 0.70
    contextual_relevancy: float = 0.70
    answer_relevancy: float = 0.75
    faithfulness: float = 0.75
    completeness: float = 0.70


class ReportingConfig(BaseModel):
    keep_last_n_runs: int = 5
    enable_trend_charts: bool = True
    trend_status_pass_rate_rule: Literal["none", "min_pass_rate", "threshold_based"] = "min_pass_rate"
    trend_status_min_pass_rate: float = Field(default=100.0, ge=0.0, le=100.0)


class SynthesizeConfig(BaseModel):
    default_num_questions: int = 50
    output_dir: str = "rag_eval_bdd/data/generated"


class EvaluationConfig(BaseModel):
    cost_optimized: bool = True
    include_reason: bool = False
    max_retrieval_context_chunks: int = 2
    max_retrieval_context_chars_per_chunk: int = 700
    faithfulness_truths_extraction_limit: int = 6
    deepeval_retry_max_attempts: int = 1
    cache_uploaded_documents: bool = True
    cache_ask_responses: bool = True
    notebook_parity_mode: bool = False
    fresh_session_per_question: bool = False
    disable_context_trimming: bool = False
    metric_question_mapping_mode: Literal["all", "positional", "row"] = "all"


class AppConfig(BaseModel):
    backend: BackendConfig = Field(default_factory=BackendConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    synthesize: SynthesizeConfig = Field(default_factory=SynthesizeConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: Optional[str] = "gpt-4.1-mini"
    embed_model: Optional[str] = None


class DatasetRow(BaseModel):
    id: str
    question: str
    expected_answer: Optional[str] = None
    dataset_file: Optional[str] = None
    category: Optional[str] = None
    source_reference: Optional[str] = None
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    metric_name: str
    threshold: float
    score: Optional[float] = None
    passed: Optional[bool] = None
    reason: Optional[str] = None
    error: Optional[str] = None
    evaluation_model: Optional[str] = None


class QuestionEvalResult(BaseModel):
    question_id: str
    question: str
    expected_answer: Optional[str] = None
    actual_answer: str
    retrieval_context: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    source_reference: Optional[str] = None
    metrics: List[MetricResult] = Field(default_factory=list)
    raw_request: Dict[str, Any] = Field(default_factory=dict)
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class MetricAggregate(BaseModel):
    metric_name: str
    threshold: float
    count: int
    scored_count: int
    pass_count: int
    fail_count: int
    pass_rate: float
    avg_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    std_dev: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    score_distribution: List[float] = Field(default_factory=list)


class RunResult(BaseModel):
    run_id: str
    timestamp: str
    feature: str
    scenario: str
    tags: List[str] = Field(default_factory=list)
    selected_metrics: List[str] = Field(default_factory=list)
    dataset_size: int
    question_results: List[QuestionEvalResult] = Field(default_factory=list)
    metric_aggregates: List[MetricAggregate] = Field(default_factory=list)
    notes: Optional[str] = None


class RunIndexEntry(BaseModel):
    run_id: str
    timestamp: str
    path: str
    feature: str
    scenario: str


class TrendPoint(BaseModel):
    run_id: str
    timestamp: str
    avg_score: Optional[float] = None
    pass_rate: Optional[float] = None
    threshold: Optional[float] = None


class MetricTrend(BaseModel):
    metric_name: str
    points: List[TrendPoint] = Field(default_factory=list)


class TrendSummary(BaseModel):
    generated_at: str
    keep_last_n: int
    metrics: List[MetricTrend] = Field(default_factory=list)
