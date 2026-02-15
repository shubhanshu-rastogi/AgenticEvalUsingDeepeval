from __future__ import annotations

import pytest

from rag_eval_bdd.cli import (
    _auto_open_executive_report,
    _normalize_marker_expression,
    _should_auto_open_report,
)

pytestmark = [pytest.mark.smoke]


def test_normalize_marker_expression_accepts_at_prefix() -> None:
    expression = "@sanity and @smoke"
    assert _normalize_marker_expression(expression) == "sanity and smoke"


def test_normalize_marker_expression_accepts_csv() -> None:
    expression = "@sanity,@regression"
    assert _normalize_marker_expression(expression) == "sanity or regression"


def test_should_auto_open_report_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_EVAL_AUTO_OPEN_REPORT", raising=False)
    monkeypatch.delenv("CI", raising=False)

    assert _should_auto_open_report() is True


def test_should_auto_open_report_disabled_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EVAL_AUTO_OPEN_REPORT", "1")
    monkeypatch.setenv("CI", "true")

    assert _should_auto_open_report() is False


def test_auto_open_executive_report_opens_browser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RAG_EVAL_AUTO_OPEN_REPORT", "true")
    monkeypatch.delenv("CI", raising=False)

    report_path = tmp_path / "results" / "reports" / "index.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("<html></html>")

    calls: list[str] = []
    monkeypatch.setattr("rag_eval_bdd.cli.webbrowser.open", lambda uri: calls.append(uri) or True)

    _auto_open_executive_report(framework_root=tmp_path)

    assert len(calls) == 1
    assert calls[0].startswith("file://")
    assert "Opened executive report in browser" in capsys.readouterr().out


def test_auto_open_executive_report_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RAG_EVAL_AUTO_OPEN_REPORT", "false")
    monkeypatch.delenv("CI", raising=False)

    report_path = tmp_path / "results" / "reports" / "index.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("<html></html>")

    def _fail_if_called(_: str) -> bool:
        raise AssertionError("should not open")

    monkeypatch.setattr("rag_eval_bdd.cli.webbrowser.open", _fail_if_called)

    _auto_open_executive_report(framework_root=tmp_path)
