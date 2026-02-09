from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_bdd.config_loader import load_config

pytestmark = [pytest.mark.smoke]


def test_load_config_with_env_override(tmp_path: Path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
backend:
  base_url: "http://localhost:9000"
thresholds:
  contextual_precision: 0.5
  contextual_recall: 0.5
  contextual_relevancy: 0.5
  answer_relevancy: 0.5
  faithfulness: 0.5
  completeness: 0.5
reporting:
  keep_last_n_runs: 5
  enable_trend_charts: true
synthesize:
  default_num_questions: 5
  output_dir: "rag_eval_bdd/data/generated"
"""
    )

    monkeypatch.setenv("BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("MODEL", "gpt-4.1-mini")

    config = load_config(str(cfg_file))
    assert config.backend.base_url == "http://localhost:8000"
    assert config.model == "gpt-4.1-mini"


def test_notebook_parity_profile_from_env(tmp_path: Path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
backend:
  base_url: "http://localhost:9000"
thresholds:
  contextual_precision: 0.6
  contextual_recall: 0.6
  contextual_relevancy: 0.6
  answer_relevancy: 0.6
  faithfulness: 0.6
  completeness: 0.6
evaluation:
  notebook_parity_mode: false
  cost_optimized: true
  include_reason: false
  max_retrieval_context_chunks: 2
  max_retrieval_context_chars_per_chunk: 700
  faithfulness_truths_extraction_limit: 6
  deepeval_retry_max_attempts: 1
  cache_uploaded_documents: true
  cache_ask_responses: true
"""
    )
    monkeypatch.setenv("RAG_EVAL_NOTEBOOK_PARITY_MODE", "1")

    config = load_config(str(cfg_file))
    assert config.evaluation.notebook_parity_mode is True
    assert config.evaluation.fresh_session_per_question is True
    assert config.evaluation.disable_context_trimming is True
    assert config.evaluation.metric_question_mapping_mode == "positional"
    assert config.evaluation.include_reason is True
    assert config.evaluation.cost_optimized is False
    assert config.thresholds.answer_relevancy == 0.5
    assert config.model is None
