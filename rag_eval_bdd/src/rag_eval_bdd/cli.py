from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rag_eval_bdd.config_loader import default_config_path, get_framework_root, load_config
from rag_eval_bdd.synthesize import synthesize_dataset


def _derive_tags_from_feature(feature: str) -> Optional[str]:
    stem = Path(feature).stem.lower()
    if stem.startswith("layer1"):
        return "layer1"
    if stem.startswith("layer2"):
        return "layer2"
    return None


def _run_pytest(
    tags: Optional[str],
    feature: Optional[str],
    config_path: Optional[str],
    allure_dir: str,
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

    if suite in {"live", "all"}:
        cmd.append(f"--alluredir={allure_dir}")

    resolved_tags = tags
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
    return _run_pytest(
        tags=args.tags,
        feature=args.feature,
        config_path=args.config,
        allure_dir=args.allure_dir,
        pytest_args=args.pytest_args,
        suite=args.suite,
    )


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
    run_parser.add_argument("--tags", default=None, help="Pytest marker expression (e.g. layer1 and contextual_precision)")
    run_parser.add_argument(
        "--feature",
        default=None,
        help="Feature path hint (layer1/layer2 derived from file name when tags are not provided).",
    )
    run_parser.add_argument("--config", default=os.getenv("RAG_EVAL_CONFIG", str(default_config_path())), help="Path to config YAML")
    run_parser.add_argument("--allure-dir", default="allure-results", help="Allure results output directory")
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

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
