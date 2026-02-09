from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from rag_eval_bdd.models import AppConfig


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_framework_root() -> Path:
    return get_repo_root() / "rag_eval_bdd"


def default_config_path() -> Path:
    return get_framework_root() / "config" / "config.yaml"


def load_env_files() -> None:
    repo_root = get_repo_root()
    framework_root = get_framework_root()
    candidates = [repo_root / ".env", framework_root / ".env"]

    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _parse_bool_env(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return None
    return int(value)


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    base_url = os.getenv("BASE_URL")
    if base_url:
        config.backend.base_url = base_url

    model = os.getenv("MODEL")
    if model:
        config.model = model

    embed_model = os.getenv("EMBED_MODEL")
    if embed_model:
        config.embed_model = embed_model

    cost_optimized = _parse_bool_env("RAG_EVAL_COST_OPTIMIZED")
    if cost_optimized is not None:
        config.evaluation.cost_optimized = cost_optimized

    include_reason = _parse_bool_env("RAG_EVAL_INCLUDE_REASON")
    if include_reason is not None:
        config.evaluation.include_reason = include_reason

    max_chunks = _parse_int_env("RAG_EVAL_MAX_CONTEXT_CHUNKS")
    if max_chunks is not None:
        config.evaluation.max_retrieval_context_chunks = max_chunks

    max_chars = _parse_int_env("RAG_EVAL_MAX_CONTEXT_CHARS_PER_CHUNK")
    if max_chars is not None:
        config.evaluation.max_retrieval_context_chars_per_chunk = max_chars

    truths_limit = _parse_int_env("RAG_EVAL_FAITHFULNESS_TRUTHS_LIMIT")
    if truths_limit is not None:
        config.evaluation.faithfulness_truths_extraction_limit = truths_limit

    retry_attempts = _parse_int_env("RAG_EVAL_DEEPEVAL_RETRY_MAX_ATTEMPTS")
    if retry_attempts is not None:
        config.evaluation.deepeval_retry_max_attempts = retry_attempts

    cache_uploads = _parse_bool_env("RAG_EVAL_CACHE_UPLOADED_DOCUMENTS")
    if cache_uploads is not None:
        config.evaluation.cache_uploaded_documents = cache_uploads

    cache_asks = _parse_bool_env("RAG_EVAL_CACHE_ASK_RESPONSES")
    if cache_asks is not None:
        config.evaluation.cache_ask_responses = cache_asks

    return config


def _apply_deepeval_runtime_defaults(config: AppConfig) -> None:
    if "DEEPEVAL_RETRY_MAX_ATTEMPTS" not in os.environ:
        os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = str(
            max(1, int(config.evaluation.deepeval_retry_max_attempts))
        )


def load_config(config_path: Optional[str] = None) -> AppConfig:
    load_env_files()

    path = Path(config_path) if config_path else default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    config = AppConfig.model_validate(raw)
    config = _apply_env_overrides(config)
    _apply_deepeval_runtime_defaults(config)
    return config
