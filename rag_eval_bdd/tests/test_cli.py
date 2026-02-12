from __future__ import annotations

import pytest

from rag_eval_bdd.cli import _normalize_marker_expression

pytestmark = [pytest.mark.smoke]


def test_normalize_marker_expression_accepts_at_prefix() -> None:
    expression = "@sanity and @smoke"
    assert _normalize_marker_expression(expression) == "sanity and smoke"


def test_normalize_marker_expression_accepts_csv() -> None:
    expression = "@sanity,@regression"
    assert _normalize_marker_expression(expression) == "sanity or regression"

