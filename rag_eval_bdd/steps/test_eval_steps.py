from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Any

from pytest_bdd import given, parsers, scenarios, then, when

from rag_eval_bdd.dataset_loader import (
    expand_dataset_references,
    load_dataset_file,
    load_inline_table,
    resolve_dataset_reference,
)
from rag_eval_bdd.executive_report import write_executive_html
from rag_eval_bdd.metric_registry import metric_threshold, normalize_metric_name, select_metrics_from_tags
from rag_eval_bdd.reporting import (
    attach_json,
    attach_run_artifacts,
    attach_text,
    write_trend_html,
)
from rag_eval_bdd.synthesize import synthesize_dataset, synthesize_dataset_from_contexts


scenarios("../features/layer1_context_metrics.feature")
scenarios("../features/layer2_answer_metrics.feature")

_SCENARIO_LINE_RE = re.compile(r"^\s*Scenario(?: Outline)?:\s*(.+?)\s*$")


def _scenario_title_to_pytest_name(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"test_{normalized}" if normalized else ""


@lru_cache(maxsize=8)
def _feature_scenario_index(features_dir: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    root = Path(features_dir)
    for feature_file in sorted(root.glob("*.feature")):
        rel_feature = f"features/{feature_file.name}"
        for line in feature_file.read_text().splitlines():
            match = _SCENARIO_LINE_RE.match(line)
            if not match:
                continue
            scenario_name = _scenario_title_to_pytest_name(match.group(1))
            if scenario_name and scenario_name not in mapping:
                mapping[scenario_name] = rel_feature
    return mapping


def _resolve_feature_reference(node_name: str, framework_root: Path, fallback: str) -> str:
    base_name = node_name.split("[", 1)[0]
    index = _feature_scenario_index(str((framework_root / "features").resolve()))
    return index.get(base_name, fallback)


@given("backend is reachable")
def given_backend_is_reachable(backend_client):
    backend_client.check_reachable()


def _persist_results_for_reporting(scenario_state, results_store, framework_root: Path, app_config) -> None:
    if scenario_state.run_result is None:
        raise AssertionError("No run result to save.")

    # Keep this idempotent because both the evaluation step and explicit save step can call it.
    if scenario_state.run_dir is not None:
        return

    run_dir, trend_summary = results_store.save_run(scenario_state.run_result)
    scenario_state.run_dir = run_dir

    attach_text("selected_metrics", ", ".join(scenario_state.selected_metrics))
    attach_json("dataset_rows", [row.model_dump() for row in scenario_state.dataset_rows])

    current_runs = results_store.load_current_session_run_results()
    if not current_runs:
        current_runs = [scenario_state.run_result]

    trend_html = write_trend_html(
        trend_summary,
        output_path=framework_root / "results" / "trends" / "last5.html",
        pass_rate_rule=app_config.reporting.trend_status_pass_rate_rule,
        min_pass_rate=app_config.reporting.trend_status_min_pass_rate,
        run_results=current_runs,
    )

    write_executive_html(
        run_results=current_runs,
        trend_summary=trend_summary,
        output_path=framework_root / "results" / "reports" / "index.html",
        pass_rate_rule=app_config.reporting.trend_status_pass_rate_rule,
        min_pass_rate=app_config.reporting.trend_status_min_pass_rate,
        snapshot_keep_last_n=app_config.reporting.executive_snapshot_keep_last_n,
        max_p95_latency_ms=app_config.evaluation.max_p95_latency_ms,
        max_avg_tokens_per_request=app_config.evaluation.max_avg_tokens_per_request,
    )
    attach_run_artifacts(scenario_state.run_result, trend_summary, [], trend_html)


@given(parsers.parse('documents are uploaded from "{document_path}"'))
def given_documents_uploaded(
    document_path: str,
    backend_client,
    scenario_state,
    repo_root: Path,
    app_config,
    upload_session_cache,
):
    resolved = Path(document_path)
    if not resolved.is_absolute():
        resolved = repo_root / document_path
    resolved = resolved.resolve()
    cache_key = str(resolved)

    if app_config.evaluation.fresh_session_per_question:
        scenario_state.session_id = None
        scenario_state.uploaded_documents.append(str(resolved))
        return

    if app_config.evaluation.cache_uploaded_documents and cache_key in upload_session_cache:
        session_id = upload_session_cache[cache_key]
    else:
        session_id, _ = backend_client.upload_document(resolved)
        if app_config.evaluation.cache_uploaded_documents:
            upload_session_cache[cache_key] = session_id

    scenario_state.session_id = session_id
    scenario_state.uploaded_documents.append(str(resolved))


@given(parsers.parse('documents are uploaded from env "{env_var}"'))
def given_documents_uploaded_from_env(
    env_var: str,
    backend_client,
    scenario_state,
    repo_root: Path,
    app_config,
    upload_session_cache,
):
    document_path = os.getenv(env_var)
    if not document_path or not document_path.strip():
        raise AssertionError(f"Environment variable '{env_var}' is not set. Provide a document path to upload.")

    given_documents_uploaded(
        document_path=document_path.strip(),
        backend_client=backend_client,
        scenario_state=scenario_state,
        repo_root=repo_root,
        app_config=app_config,
        upload_session_cache=upload_session_cache,
    )


@given("I use latest uploaded session from application UI")
def given_use_latest_uploaded_session_from_application_ui(backend_client, scenario_state):
    payload = backend_client.get_current_session()
    session_id = payload.get("session_id")
    if not session_id:
        raise AssertionError("Backend did not provide an active session_id. Upload a file in UI first.")

    source_filename = payload.get("source_filename")
    num_chunks = payload.get("num_chunks")
    scenario_state.session_id = str(session_id)
    scenario_state.ui_source_filename = str(source_filename) if source_filename else None
    try:
        scenario_state.ui_num_chunks = int(num_chunks) if num_chunks is not None else None
    except Exception:  # noqa: BLE001
        scenario_state.ui_num_chunks = None
    attach_text("ui_session_id", str(session_id))
    if source_filename:
        attach_text("ui_source_filename", str(source_filename))
    if scenario_state.ui_num_chunks is not None:
        attach_text("ui_num_chunks", str(scenario_state.ui_num_chunks))


def _live_questions_per_layer() -> int:
    raw = os.getenv("RAG_EVAL_LIVE_QUESTIONS_PER_LAYER")
    if raw is None or not raw.strip():
        # Backward-compatible fallback
        raw = os.getenv("RAG_EVAL_UNSEEN_QUESTIONS_PER_LAYER")
    if raw is None or not raw.strip():
        return 2
    parsed = int(raw.strip())
    if parsed <= 0:
        raise ValueError("RAG_EVAL_LIVE_QUESTIONS_PER_LAYER must be a positive integer")
    return parsed


def _live_context_chunk_limit() -> int:
    raw = os.getenv("RAG_EVAL_LIVE_CONTEXT_CHUNK_LIMIT")
    if raw is None or not raw.strip():
        # Backward-compatible fallback
        raw = os.getenv("RAG_EVAL_UNSEEN_CONTEXT_CHUNK_LIMIT")
    if raw is None or not raw.strip():
        return 24
    parsed = int(raw.strip())
    if parsed <= 0:
        raise ValueError("RAG_EVAL_LIVE_CONTEXT_CHUNK_LIMIT must be a positive integer")
    return parsed


def _is_not_found_answer(text: str | None) -> bool:
    if not text:
        return False
    normalized = str(text).strip().lower()
    return normalized in {"not found in document.", "not found in the document."}


def _live_dataset_meta_path(output_path: Path) -> Path:
    return output_path.with_suffix(".meta.json")


def _document_fingerprint(document_path: Path) -> dict[str, Any]:
    stats = document_path.stat()
    return {
        "path": str(document_path),
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
    }


def _build_live_dataset_fingerprint(
    *,
    layer_name: str,
    question_count: int,
    generation_target: int,
    model: str | None,
    document_path: Path | None,
    session_id: str | None,
    ui_source_filename: str | None,
    ui_num_chunks: int | None,
    chunk_limit: int | None,
) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {
        "layer": layer_name,
        "question_count": int(question_count),
        "generation_target": int(generation_target),
        "model": model or "",
        "session_id": str(session_id or ""),
        "ui_source_filename": str(ui_source_filename or ""),
        "ui_num_chunks": int(ui_num_chunks) if ui_num_chunks is not None else None,
        "chunk_limit": int(chunk_limit) if chunk_limit is not None else None,
    }
    if document_path is not None and document_path.exists():
        fingerprint["document"] = _document_fingerprint(document_path)
    else:
        fingerprint["document"] = None
    return fingerprint


def _load_existing_live_dataset(output_path: Path, expected_fingerprint: dict[str, Any]):
    if not output_path.exists():
        return []
    meta_path = _live_dataset_meta_path(output_path)
    if not meta_path.exists():
        return []
    try:
        stored_fingerprint = json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        return []
    if stored_fingerprint != expected_fingerprint:
        return []
    try:
        return load_dataset_file(output_path)
    except Exception:  # noqa: BLE001
        return []


def _write_live_dataset_meta(output_path: Path, fingerprint: dict[str, Any]) -> None:
    meta_path = _live_dataset_meta_path(output_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(fingerprint, indent=2))


@given(parsers.parse('I generate live dataset for layer "{layer_name}" from uploaded documents'))
@given(parsers.parse('I generate unseen dataset for layer "{layer_name}" from uploaded documents'))
def given_generate_live_dataset_for_layer(
    layer_name: str,
    backend_client,
    scenario_state,
    framework_root: Path,
    app_config,
):
    if not scenario_state.uploaded_documents and not scenario_state.session_id:
        raise AssertionError(
            "No uploaded document path or UI session available. Upload documents before generating live dataset."
        )

    normalized_layer = layer_name.strip().lower()
    if normalized_layer not in {"layer1", "layer2"}:
        raise AssertionError(f"Unsupported layer '{layer_name}'. Use 'layer1' or 'layer2'.")

    question_count = _live_questions_per_layer()
    generation_target = max(question_count, question_count * 3)
    chunk_limit = _live_context_chunk_limit() if scenario_state.session_id and not scenario_state.uploaded_documents else None
    active_document_path = (
        Path(scenario_state.uploaded_documents[-1]).resolve()
        if scenario_state.uploaded_documents
        else None
    )
    dataset_fingerprint = _build_live_dataset_fingerprint(
        layer_name=normalized_layer,
        question_count=question_count,
        generation_target=generation_target,
        model=app_config.model,
        document_path=active_document_path,
        session_id=scenario_state.session_id,
        ui_source_filename=scenario_state.ui_source_filename,
        ui_num_chunks=scenario_state.ui_num_chunks,
        chunk_limit=chunk_limit,
    )
    output_path = framework_root / "data" / "generated" / f"{normalized_layer}_live_questions.json"
    existing_rows = _load_existing_live_dataset(output_path, expected_fingerprint=dataset_fingerprint)
    reused_existing_dataset = len(existing_rows) > 0
    rows = existing_rows
    source_reference = (
        str(Path(scenario_state.uploaded_documents[-1]).resolve())
        if scenario_state.uploaded_documents
        else (
            f"session:{scenario_state.session_id}:{scenario_state.ui_source_filename}"
            if scenario_state.ui_source_filename
            else f"session:{scenario_state.session_id}"
        )
    )

    if not reused_existing_dataset:
        if scenario_state.uploaded_documents:
            document_path = active_document_path
            if document_path is None:
                raise AssertionError("Uploaded document path was expected but unavailable.")
            rows = synthesize_dataset(
                input_path=document_path,
                output_path=output_path,
                num_questions=generation_target,
                model=app_config.model,
            )
            source_reference = str(document_path)
        else:
            if chunk_limit is None:
                chunk_limit = _live_context_chunk_limit()
            chunks = backend_client.get_session_chunks(limit=chunk_limit)
            contexts = [str(chunk.get("text", "")).strip() for chunk in chunks if str(chunk.get("text", "")).strip()]
            if not contexts:
                raise AssertionError(
                    "No retrieval chunks found for active UI session. Upload a document in UI and try again."
                )
            source_reference = (
                f"session:{scenario_state.session_id}:{scenario_state.ui_source_filename}"
                if scenario_state.ui_source_filename
                else f"session:{scenario_state.session_id}"
            )
            rows = synthesize_dataset_from_contexts(
                contexts=contexts,
                output_path=output_path,
                num_questions=generation_target,
                model=app_config.model,
                source_reference=source_reference,
            )

        if len(rows) < generation_target:
            raise AssertionError(
                f"Synthesizer generated only {len(rows)} rows, expected at least {generation_target}."
            )
        _write_live_dataset_meta(output_path=output_path, fingerprint=dataset_fingerprint)

    answerable_rows = []
    if scenario_state.session_id:
        for row in rows:
            _, response = backend_client.ask_question(
                session_id=scenario_state.session_id,
                question=row.question,
                use_cache=False,
            )
            if not _is_not_found_answer(str(response.get("answer", ""))):
                answerable_rows.append(row)
            if len(answerable_rows) >= question_count:
                break
    else:
        answerable_rows = rows[:question_count]

    if len(answerable_rows) < question_count:
        raise AssertionError(
            "Live dataset generation produced too many unanswerable questions "
            "(backend returned 'Not found in document.'). "
            "Increase RAG_EVAL_LIVE_CONTEXT_CHUNK_LIMIT or reduce question complexity."
        )

    prefix = "L1_GEN" if normalized_layer == "layer1" else "L2_GEN"
    scenario_state.dataset_rows = [
        row.model_copy(
            update={
                "id": f"{prefix}_{index}",
                "category": row.category or normalized_layer,
                "source_reference": row.source_reference or source_reference,
            }
        )
        for index, row in enumerate(answerable_rows[:question_count], start=1)
    ]

    attach_text("live_dataset_path", str(output_path))
    attach_text("live_dataset_mode", "reused_existing" if reused_existing_dataset else "generated_new")
    attach_json("live_dataset_rows", [row.model_dump() for row in scenario_state.dataset_rows])


@given(parsers.parse('I load dataset "{dataset_ref}"'))
def given_load_dataset(dataset_ref: str, scenario_state, repo_root: Path):
    dataset_path = resolve_dataset_reference(dataset_ref, repo_root=repo_root)
    rows = load_dataset_file(dataset_path)
    scenario_state.dataset_rows = expand_dataset_references(rows, repo_root=repo_root)


@given("I use inline dataset:")
def given_inline_dataset(docstring: str, scenario_state, repo_root: Path):
    rows = load_inline_table(docstring)
    scenario_state.dataset_rows = expand_dataset_references(rows, repo_root=repo_root)


@when("I evaluate all questions")
def when_evaluate_all_questions(request, scenario_state, evaluation_runner, results_store, framework_root: Path, app_config):
    if not scenario_state.dataset_rows:
        raise AssertionError("Dataset is empty. Load dataset before evaluation.")
    if (
        not evaluation_runner.config.evaluation.fresh_session_per_question
        and not scenario_state.session_id
    ):
        raise AssertionError("No backend session available. Upload documents before evaluation.")
    if (
        evaluation_runner.config.evaluation.fresh_session_per_question
        and not scenario_state.uploaded_documents
    ):
        raise AssertionError("No uploaded document path available. Upload documents before evaluation.")

    tags = [marker.name for marker in request.node.iter_markers()]
    selected_metrics = select_metrics_from_tags(tags=tags, explicit_metrics=scenario_state.explicit_metrics)
    if not selected_metrics:
        raise AssertionError("No metrics selected from tags/parameters.")

    feature_reference = _resolve_feature_reference(
        node_name=request.node.name,
        framework_root=framework_root,
        fallback=str(request.node.location[0]),
    )

    scenario_state.selected_metrics = selected_metrics
    scenario_state.run_result = evaluation_runner.evaluate_dataset(
        dataset_rows=scenario_state.dataset_rows,
        selected_metrics=selected_metrics,
        session_id=scenario_state.session_id,
        feature=feature_reference,
        scenario=request.node.name,
        tags=tags,
        uploaded_documents=scenario_state.uploaded_documents,
    )
    _persist_results_for_reporting(
        scenario_state=scenario_state,
        results_store=results_store,
        framework_root=framework_root,
        app_config=app_config,
    )


@when(parsers.parse('I evaluate all questions with metrics "{metric_csv}"'))
def when_evaluate_with_explicit_metrics(
    metric_csv: str,
    scenario_state,
    request,
    evaluation_runner,
    results_store,
    framework_root: Path,
    app_config,
):
    scenario_state.explicit_metrics = [metric.strip() for metric in metric_csv.split(",") if metric.strip()]
    when_evaluate_all_questions(
        request=request,
        scenario_state=scenario_state,
        evaluation_runner=evaluation_runner,
        results_store=results_store,
        framework_root=framework_root,
        app_config=app_config,
    )


@then(parsers.parse('metric "{metric_name}" should be >= configured threshold'))
def then_metric_above_threshold(metric_name: str, scenario_state, app_config):
    if scenario_state.run_result is None:
        raise AssertionError("No run result found. Execute evaluation first.")

    canonical = normalize_metric_name(metric_name)
    aggregate = next((m for m in scenario_state.run_result.metric_aggregates if m.metric_name == canonical), None)
    if aggregate is None:
        raise AssertionError(f"Metric '{canonical}' not found in run aggregates.")

    threshold = metric_threshold(canonical, app_config)
    if aggregate.avg_score is None:
        raise AssertionError(f"Metric '{canonical}' has no average score. Errors may have occurred.")

    assert aggregate.avg_score >= threshold, (
        f"Metric {canonical} average {aggregate.avg_score:.4f} < threshold {threshold:.4f}. "
        f"pass_rate={aggregate.pass_rate:.2f}%"
    )


@then("save results for reporting")
def then_save_results_for_reporting(scenario_state, results_store, framework_root: Path, app_config):
    _persist_results_for_reporting(
        scenario_state=scenario_state,
        results_store=results_store,
        framework_root=framework_root,
        app_config=app_config,
    )
