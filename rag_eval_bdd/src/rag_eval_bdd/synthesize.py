from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_eval_bdd.models import DatasetRow


SUPPORTED_DOC_EXTS = {".pdf", ".txt", ".md", ".docx"}


def _read_json_or_csv_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("records", payload.get("questions", []))
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list or contain records/questions list")
        return [dict(item) for item in payload]

    if path.suffix.lower() == ".csv":
        with path.open("r", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]

    raise ValueError(f"Unsupported structured input: {path}")


def _to_contexts_from_records(records: List[Dict[str, Any]]) -> List[List[str]]:
    contexts: List[List[str]] = []
    for row in records:
        context = row.get("context") or row.get("text") or row.get("content")
        if context:
            contexts.append([str(context)])
    return contexts


def _chunk_text(text: str, chunk_size: int = 1200) -> List[List[str]]:
    chunks: List[List[str]] = []
    cleaned = "\\n".join([line.strip() for line in text.splitlines() if line.strip()])
    for i in range(0, len(cleaned), chunk_size):
        segment = cleaned[i : i + chunk_size].strip()
        if segment:
            chunks.append([segment])
    return chunks


def _collect_documents(input_path: Path) -> List[str]:
    if input_path.is_file():
        return [str(input_path)]

    docs = [
        str(path)
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOC_EXTS
    ]
    return docs


def _build_rows_from_goldens(
    goldens: list[Any],
    num_questions: int,
    source_reference: str,
) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    for idx, golden in enumerate(goldens[:num_questions], start=1):
        custom = getattr(golden, "custom_column_key_values", None) or {}
        rows.append(
            DatasetRow(
                id=f"GEN_{idx}",
                question=str(getattr(golden, "input", "")).strip(),
                expected_answer=(getattr(golden, "expected_output", None) or None),
                category=(custom.get("category") if isinstance(custom, dict) else None),
                source_reference=(getattr(golden, "source_file", None) or source_reference),
            )
        )
    return rows


def _write_rows(rows: List[DatasetRow], output_path: Path) -> None:
    payload = [row.model_dump() for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def synthesize_dataset_from_contexts(
    contexts: List[str],
    output_path: Path,
    num_questions: int,
    model: Optional[str] = None,
    source_reference: str = "active_session",
) -> List[DatasetRow]:
    try:
        from deepeval.synthesizer import Synthesizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("DeepEval Synthesizer is not available in this environment") from exc

    clean_contexts = [ctx.strip() for ctx in contexts if str(ctx).strip()]
    if not clean_contexts:
        raise ValueError("No non-empty contexts were provided to synthesize questions.")

    synthesizer = Synthesizer(model=model)
    context_blocks = [[ctx] for ctx in clean_contexts]
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=context_blocks,
        include_expected_output=True,
    )
    rows = _build_rows_from_goldens(
        goldens=goldens,
        num_questions=num_questions,
        source_reference=source_reference,
    )
    _write_rows(rows=rows, output_path=output_path)
    return rows


def synthesize_dataset(
    input_path: Path,
    output_path: Path,
    num_questions: int,
    model: Optional[str] = None,
) -> List[DatasetRow]:
    try:
        from deepeval.synthesizer import Synthesizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("DeepEval Synthesizer is not available in this environment") from exc

    synthesizer = Synthesizer(model=model)

    if input_path.is_dir() or input_path.suffix.lower() in SUPPORTED_DOC_EXTS:
        docs = _collect_documents(input_path)
        if not docs:
            raise ValueError(f"No supported documents found at: {input_path}")
        goldens = synthesizer.generate_goldens_from_docs(document_paths=docs, include_expected_output=True)
    elif input_path.suffix.lower() in {".json", ".csv"}:
        records = _read_json_or_csv_records(input_path)
        contexts = _to_contexts_from_records(records)
        if not contexts:
            raise ValueError("JSON/CSV input must include context/text/content fields to synthesize questions")
        goldens = synthesizer.generate_goldens_from_contexts(contexts=contexts, include_expected_output=True)
    elif input_path.suffix.lower() in {".txt", ".md"}:
        contexts = _chunk_text(input_path.read_text())
        goldens = synthesizer.generate_goldens_from_contexts(contexts=contexts, include_expected_output=True)
    else:
        raise ValueError(f"Unsupported synthesize input: {input_path}")

    rows = _build_rows_from_goldens(
        goldens=goldens,
        num_questions=num_questions,
        source_reference=str(input_path),
    )
    _write_rows(rows=rows, output_path=output_path)
    return rows
