from __future__ import annotations

import re

import pytest

import rag_eval_bdd.metric_registry as metric_registry
from rag_eval_bdd.models import AppConfig

pytestmark = [pytest.mark.smoke]


class _FakeGEval:
    def __init__(self, *args, **kwargs):
        del args
        self.evaluation_steps = kwargs.get("evaluation_steps", [])
        self.evaluation_params = kwargs.get("evaluation_params", [])


def _assert_10_point_step(metric) -> None:
    steps = getattr(metric, "evaluation_steps", None) or []
    assert any("0 to 10" in step for step in steps)
    assert all(re.search(r"\b0 to 1\b", step) is None for step in steps)


def test_completeness_geval_uses_10_point_scale(monkeypatch):
    config = AppConfig()
    monkeypatch.setattr(metric_registry, "GEval", _FakeGEval)
    metric = metric_registry.build_metric("completeness", config)
    _assert_10_point_step(metric)


def test_contextual_relevancy_geval_uses_10_point_scale_when_cost_optimized(monkeypatch):
    config = AppConfig()
    config.evaluation.cost_optimized = True
    monkeypatch.setattr(metric_registry, "GEval", _FakeGEval)
    metric = metric_registry.build_metric("contextual_relevancy", config)
    _assert_10_point_step(metric)


def test_contextual_relevancy_geval_fallback_uses_10_point_scale(monkeypatch):
    config = AppConfig()
    config.evaluation.cost_optimized = False
    monkeypatch.setattr(metric_registry, "GEval", _FakeGEval)
    monkeypatch.setattr(metric_registry, "ContextualRelevancyMetric", None)

    metric = metric_registry.build_metric("contextual_relevancy", config)
    _assert_10_point_step(metric)
