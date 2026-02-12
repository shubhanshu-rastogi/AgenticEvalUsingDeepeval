from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
from typing import List, Optional

import pytest

from rag_eval_bdd.backend_client import BackendClient
from rag_eval_bdd.config_loader import get_framework_root, get_repo_root, load_config
from rag_eval_bdd.evaluator import EvaluationRunner
from rag_eval_bdd.models import AppConfig, DatasetRow, RunResult
from rag_eval_bdd.results_store import ResultsStore


@dataclass
class ScenarioState:
    dataset_rows: List[DatasetRow] = field(default_factory=list)
    session_id: Optional[str] = None
    uploaded_documents: List[str] = field(default_factory=list)
    selected_metrics: List[str] = field(default_factory=list)
    explicit_metrics: Optional[List[str]] = None
    run_result: Optional[RunResult] = None
    run_dir: Optional[Path] = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _reporter_line(pytest_config: pytest.Config, message: str) -> None:
    terminal_reporter = pytest_config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line(message)
    else:
        print(message)


def _resolve_report_dir(pytest_config: pytest.Config) -> Path:
    return Path(str(pytest_config.rootpath)) / "results" / "reports"


def _reset_current_session_index(pytest_config: pytest.Config) -> None:
    current_index = Path(str(pytest_config.rootpath)) / "results" / "current_index.json"
    current_index.parent.mkdir(parents=True, exist_ok=True)
    current_index.write_text('{"runs": []}\n')


def pytest_sessionstart(session: pytest.Session) -> None:
    pytest_config = session.config
    _reset_current_session_index(pytest_config)

    report_dir = _resolve_report_dir(pytest_config)
    if not _env_flag("RAG_EVAL_AUTO_HTML_REPORT_CLEAN", default=True):
        return

    shutil.rmtree(report_dir, ignore_errors=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _reporter_line(
        pytest_config,
        f"[rag-eval-bdd] cleaned previous HTML report artifacts: {report_dir}",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus  # not used for now, kept for hook signature clarity

    pytest_config = session.config
    report_dir = _resolve_report_dir(pytest_config)
    report_path = report_dir / "index.html"
    if not report_path.exists():
        _reporter_line(
            pytest_config,
            "[rag-eval-bdd] HTML report not found. Run live scenarios to generate results.",
        )
        return

    _reporter_line(pytest_config, f"[rag-eval-bdd] generated HTML report: {report_path}")
    _reporter_line(
        pytest_config,
        f"[rag-eval-bdd] trend dashboard: {Path(str(pytest_config.rootpath)) / 'results' / 'trends' / 'last5.html'}",
    )


@pytest.fixture(scope="session")
def framework_root() -> Path:
    return get_framework_root()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return get_repo_root()


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    config_path = os.getenv("RAG_EVAL_CONFIG", str(get_framework_root() / "config" / "config.yaml"))
    return load_config(config_path=config_path)


@pytest.fixture(scope="session")
def backend_client(app_config: AppConfig) -> BackendClient:
    return BackendClient(config=app_config.backend)


@pytest.fixture(scope="session")
def results_store(app_config: AppConfig, framework_root: Path) -> ResultsStore:
    return ResultsStore(base_dir=framework_root / "results", keep_last_n=app_config.reporting.keep_last_n_runs)


@pytest.fixture(scope="session")
def upload_session_cache() -> dict[str, str]:
    return {}


@pytest.fixture
def evaluation_runner(backend_client: BackendClient, app_config: AppConfig) -> EvaluationRunner:
    return EvaluationRunner(client=backend_client, config=app_config)


@pytest.fixture
def scenario_state() -> ScenarioState:
    return ScenarioState()
