from __future__ import annotations

import os
from pathlib import Path

import pytest
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

pytestmark = [pytest.mark.live]


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

    trend_html = write_trend_html(
        trend_summary,
        output_path=framework_root / "results" / "trends" / "last5.html",
        pass_rate_rule=app_config.reporting.trend_status_pass_rate_rule,
        min_pass_rate=app_config.reporting.trend_status_min_pass_rate,
    )

    current_runs = results_store.load_current_session_run_results()
    if not current_runs:
        current_runs = [scenario_state.run_result]
    write_executive_html(
        run_results=current_runs,
        trend_summary=trend_summary,
        output_path=framework_root / "results" / "reports" / "index.html",
        pass_rate_rule=app_config.reporting.trend_status_pass_rate_rule,
        min_pass_rate=app_config.reporting.trend_status_min_pass_rate,
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
    scenario_state.session_id = str(session_id)
    attach_text("ui_session_id", str(session_id))
    if source_filename:
        attach_text("ui_source_filename", str(source_filename))


def _unseen_questions_per_layer() -> int:
    raw = os.getenv("RAG_EVAL_UNSEEN_QUESTIONS_PER_LAYER")
    if raw is None or not raw.strip():
        return 2
    parsed = int(raw.strip())
    if parsed <= 0:
        raise ValueError("RAG_EVAL_UNSEEN_QUESTIONS_PER_LAYER must be a positive integer")
    return parsed


def _unseen_context_chunk_limit() -> int:
    raw = os.getenv("RAG_EVAL_UNSEEN_CONTEXT_CHUNK_LIMIT")
    if raw is None or not raw.strip():
        return 24
    parsed = int(raw.strip())
    if parsed <= 0:
        raise ValueError("RAG_EVAL_UNSEEN_CONTEXT_CHUNK_LIMIT must be a positive integer")
    return parsed


@given(parsers.parse('I generate unseen dataset for layer "{layer_name}" from uploaded documents'))
def given_generate_unseen_dataset_for_layer(
    layer_name: str,
    backend_client,
    scenario_state,
    framework_root: Path,
    app_config,
):
    if not scenario_state.uploaded_documents and not scenario_state.session_id:
        raise AssertionError(
            "No uploaded document path or UI session available. Upload documents before generating unseen dataset."
        )

    normalized_layer = layer_name.strip().lower()
    if normalized_layer not in {"layer1", "layer2"}:
        raise AssertionError(f"Unsupported layer '{layer_name}'. Use 'layer1' or 'layer2'.")

    question_count = _unseen_questions_per_layer()
    output_path = framework_root / "data" / "generated" / f"{normalized_layer}_unseen_questions.json"

    if scenario_state.uploaded_documents:
        document_path = Path(scenario_state.uploaded_documents[-1]).resolve()
        rows = synthesize_dataset(
            input_path=document_path,
            output_path=output_path,
            num_questions=question_count,
            model=app_config.model,
        )
        source_reference = str(document_path)
    else:
        chunk_limit = _unseen_context_chunk_limit()
        chunks = backend_client.get_session_chunks(limit=chunk_limit)
        contexts = [str(chunk.get("text", "")).strip() for chunk in chunks if str(chunk.get("text", "")).strip()]
        if not contexts:
            raise AssertionError(
                "No retrieval chunks found for active UI session. Upload a document in UI and try again."
            )
        source_reference = f"session:{scenario_state.session_id}"
        rows = synthesize_dataset_from_contexts(
            contexts=contexts,
            output_path=output_path,
            num_questions=question_count,
            model=app_config.model,
            source_reference=source_reference,
        )

    if len(rows) < question_count:
        raise AssertionError(
            f"Synthesizer generated only {len(rows)} rows, expected at least {question_count}."
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
        for index, row in enumerate(rows[:question_count], start=1)
    ]

    attach_text("unseen_dataset_path", str(output_path))
    attach_json("unseen_dataset_rows", [row.model_dump() for row in scenario_state.dataset_rows])


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

    scenario_state.selected_metrics = selected_metrics
    scenario_state.run_result = evaluation_runner.evaluate_dataset(
        dataset_rows=scenario_state.dataset_rows,
        selected_metrics=selected_metrics,
        session_id=scenario_state.session_id,
        feature=str(request.node.location[0]),
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
