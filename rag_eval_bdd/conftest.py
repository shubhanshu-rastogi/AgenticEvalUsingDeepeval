from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
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


def _is_ci_environment() -> bool:
    return any(
        os.getenv(name)
        for name in ("CI", "GITHUB_ACTIONS", "JENKINS_URL", "TEAMCITY_VERSION", "BUILDKITE")
    )


def _reporter_line(pytest_config: pytest.Config, message: str) -> None:
    terminal_reporter = pytest_config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line(message)
    else:
        print(message)


def _resolve_allure_dir(pytest_config: pytest.Config) -> Optional[Path]:
    try:
        alluredir = pytest_config.getoption("--alluredir")
    except Exception:  # noqa: BLE001
        alluredir = None

    if not alluredir:
        return None

    path = Path(str(alluredir))
    if not path.is_absolute():
        path = Path(str(pytest_config.rootpath)) / path
    return path


def _resolve_allure_report_dir(pytest_config: pytest.Config) -> Path:
    return Path(str(pytest_config.rootpath)) / "allure-report"


def pytest_sessionstart(session: pytest.Session) -> None:
    pytest_config = session.config
    allure_dir = _resolve_allure_dir(pytest_config)
    if allure_dir is None:
        return

    if not _env_flag("RAG_EVAL_AUTO_ALLURE_CLEAN", default=True):
        return

    report_dir = _resolve_allure_report_dir(pytest_config)
    shutil.rmtree(allure_dir, ignore_errors=True)
    allure_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(report_dir, ignore_errors=True)
    _reporter_line(
        pytest_config,
        f"[rag-eval-bdd] cleaned previous Allure artifacts: {allure_dir} and {report_dir}",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus  # not used for now, kept for hook signature clarity

    pytest_config = session.config
    allure_dir = _resolve_allure_dir(pytest_config)
    if allure_dir is None:
        return

    if not _env_flag("RAG_EVAL_AUTO_ALLURE_GENERATE", default=True):
        return

    if shutil.which("allure") is None:
        _reporter_line(
            pytest_config,
            "[rag-eval-bdd] Allure CLI not found. Install with 'brew install allure' to auto-generate/open report.",
        )
        return

    report_dir = _resolve_allure_report_dir(pytest_config)
    generate_cmd = ["allure", "generate", str(allure_dir), "--clean", "-o", str(report_dir)]
    generate_proc = subprocess.run(generate_cmd, cwd=str(pytest_config.rootpath), capture_output=True, text=True, check=False)

    if generate_proc.returncode != 0:
        stderr_tail = (generate_proc.stderr or "").strip().splitlines()
        detail = stderr_tail[-1] if stderr_tail else "Unknown Allure generate error."
        _reporter_line(pytest_config, f"[rag-eval-bdd] failed to generate Allure report: {detail}")
        return

    _reporter_line(pytest_config, f"[rag-eval-bdd] generated report: {report_dir}/index.html")

    should_open = _env_flag("RAG_EVAL_AUTO_ALLURE_OPEN", default=not _is_ci_environment())
    if not should_open:
        return

    try:
        subprocess.Popen(
            ["allure", "open", str(report_dir)],
            cwd=str(pytest_config.rootpath),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _reporter_line(pytest_config, "[rag-eval-bdd] opened Allure report in browser (served over local HTTP).")
    except Exception as exc:  # noqa: BLE001
        _reporter_line(pytest_config, f"[rag-eval-bdd] could not auto-open Allure report: {exc}")


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
