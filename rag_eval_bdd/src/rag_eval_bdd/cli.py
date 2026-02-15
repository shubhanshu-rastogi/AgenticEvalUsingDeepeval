from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from rag_eval_bdd.config_loader import default_config_path, get_framework_root, load_config
from rag_eval_bdd.executive_report import write_executive_html
from rag_eval_bdd.reporting import write_trend_html
from rag_eval_bdd.results_store import ResultsStore
from rag_eval_bdd.synthesize import synthesize_dataset


def _derive_tags_from_feature(feature: str) -> Optional[str]:
    stem = Path(feature).stem.lower()
    if stem.startswith("layer1"):
        return "layer1"
    if stem.startswith("layer2"):
        return "layer2"
    return None


def _normalize_marker_expression(expression: Optional[str]) -> Optional[str]:
    if not expression:
        return expression

    normalized = expression.replace(",", " or ")
    normalized = re.sub(r"@([A-Za-z_][A-Za-z0-9_]*)", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_auto_open_report() -> bool:
    if _env_flag("CI", default=False):
        return False
    return _env_flag("RAG_EVAL_AUTO_OPEN_REPORT", default=True)


def _auto_open_executive_report(framework_root: Path) -> None:
    if not _should_auto_open_report():
        return

    report_path = framework_root / "results" / "reports" / "index.html"
    if not report_path.exists():
        return

    report_uri = report_path.resolve().as_uri()
    try:
        opened = webbrowser.open(report_uri)
    except Exception:
        opened = False

    if opened:
        print(f"Opened executive report in browser: {report_path}")
    else:
        print(f"Executive report ready: {report_path}")


def _run_pytest(
    tags: Optional[str],
    feature: Optional[str],
    config_path: Optional[str],
    pytest_args: list[str],
    suite: str = "live",
) -> int:
    framework_root = get_framework_root()
    pytest_ini = framework_root / "pytest.ini"

    cmd = [sys.executable, "-m", "pytest", "-c", str(pytest_ini)]
    if suite == "smoke":
        cmd.append("tests")
    elif suite == "all":
        cmd.extend(["tests", "steps"])
    else:
        cmd.append("steps")

    resolved_tags = _normalize_marker_expression(tags)
    if feature and not resolved_tags:
        resolved_tags = _derive_tags_from_feature(feature)
    if suite == "smoke" and not resolved_tags:
        resolved_tags = "smoke"
    if suite == "live" and not resolved_tags:
        resolved_tags = "live"
    if suite == "all" and not resolved_tags:
        resolved_tags = "smoke or live"
    if resolved_tags:
        cmd.extend(["-m", resolved_tags])
    cmd.extend(pytest_args)

    env = os.environ.copy()
    if config_path:
        env["RAG_EVAL_CONFIG"] = str(config_path)

    process = subprocess.run(cmd, cwd=str(framework_root), env=env, check=False)
    return process.returncode


def _cmd_run(args: argparse.Namespace) -> int:
    framework_root = get_framework_root()
    exit_code = _run_pytest(
        tags=args.tags,
        feature=args.feature,
        config_path=args.config,
        pytest_args=args.pytest_args,
        suite=args.suite,
    )
    _auto_open_executive_report(framework_root=framework_root)
    return exit_code


def _cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    framework_root = get_framework_root()
    results_store = ResultsStore(
        base_dir=framework_root / "results",
        keep_last_n=config.reporting.keep_last_n_runs,
    )
    trend_summary = results_store.refresh_trends()
    run_results = results_store.load_current_session_run_results()
    if not run_results:
        run_results = results_store.load_recent_run_results()
    if not run_results:
        print("No saved runs found. Execute live scenarios first.")
        return 0

    trend_dir = framework_root / "results" / "trends"
    trend_path = write_trend_html(
        trend_summary,
        output_path=trend_dir / "last5.html",
        pass_rate_rule=config.reporting.trend_status_pass_rate_rule,
        min_pass_rate=config.reporting.trend_status_min_pass_rate,
    )
    executive_path = write_executive_html(
        run_results=run_results,
        trend_summary=trend_summary,
        output_path=framework_root / "results" / "reports" / "index.html",
        pass_rate_rule=config.reporting.trend_status_pass_rate_rule,
        min_pass_rate=config.reporting.trend_status_min_pass_rate,
    )
    print(f"Executive report: {executive_path}")
    print(f"Trend dashboard: {trend_path}")
    return 0


def _cmd_synthesize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    num_questions = args.num_questions or config.synthesize.default_num_questions

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path(config.synthesize.output_dir) / "questions.json"
    rows = synthesize_dataset(
        input_path=input_path,
        output_path=output_path,
        num_questions=num_questions,
        model=config.model,
    )
    print(f"Generated {len(rows)} questions at: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_eval_bdd", description="BDD RAG evaluation framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run pytest-bdd evaluations")
    run_parser.add_argument(
        "--tags",
        default=None,
        help="Pytest marker expression; supports '@' prefix too (e.g. @sanity and @smoke).",
    )
    run_parser.add_argument(
        "--feature",
        default=None,
        help="Feature path hint (layer1/layer2 derived from file name when tags are not provided).",
    )
    run_parser.add_argument("--config", default=os.getenv("RAG_EVAL_CONFIG", str(default_config_path())), help="Path to config YAML")
    run_parser.add_argument(
        "--suite",
        choices=["live", "smoke", "all"],
        default="live",
        help="Test suite type: live (BDD eval), smoke (deterministic unit tests), all (both).",
    )
    run_parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Extra pytest arguments")
    run_parser.set_defaults(func=_cmd_run)

    synth_parser = subparsers.add_parser("synthesize", help="Generate synthetic datasets using DeepEval Synthesizer")
    synth_parser.add_argument("--input", required=True, help="Input file or folder")
    synth_parser.add_argument("--output", default=None, help="Output JSON path")
    synth_parser.add_argument("--num-questions", type=int, default=None, help="Number of questions to generate")
    synth_parser.add_argument("--config", default=os.getenv("RAG_EVAL_CONFIG", str(default_config_path())), help="Path to config YAML")
    synth_parser.set_defaults(func=_cmd_synthesize)

    report_parser = subparsers.add_parser("report", help="Generate HTML dashboards from saved run artifacts")
    report_parser.add_argument("--config", default=os.getenv("RAG_EVAL_CONFIG", str(default_config_path())), help="Path to config YAML")
    report_parser.set_defaults(func=_cmd_report)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
