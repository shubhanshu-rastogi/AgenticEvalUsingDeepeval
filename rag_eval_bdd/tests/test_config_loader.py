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
