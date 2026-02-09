from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rag_eval_bdd.models import DatasetRow

_HEADER_ALIASES = {
    "id": "id",
    "question": "question",
    "expected_answer": "expected_answer",
    "expected_output": "expected_answer",
    "category": "category",
    "dataset_file": "dataset_file",
    "source_reference": "source_reference",
}


def _normalize_record(record: Dict[str, Any], index: int) -> DatasetRow:
    normalized: Dict[str, Any] = {}
    additional: Dict[str, Any] = {}

    for key, value in record.items():
        canonical = _HEADER_ALIASES.get(str(key).strip().lower())
        if canonical:
            normalized[canonical] = value
        else:
            additional[key] = value

    normalized.setdefault("id", f"Q{index}")
    question = (normalized.get("question") or "").strip()
    if not question:
        raise ValueError(f"Dataset row {index} has empty question")

    row = DatasetRow(
        id=str(normalized["id"]),
        question=question,
        expected_answer=_optional_str(normalized.get("expected_answer")),
        category=_optional_str(normalized.get("category")),
        dataset_file=_optional_str(normalized.get("dataset_file")),
        source_reference=_optional_str(normalized.get("source_reference")),
        additional_metadata={k: v for k, v in additional.items() if v not in (None, "")},
    )
    return row


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_dataset_records(records: Iterable[Dict[str, Any]]) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    for idx, record in enumerate(records, start=1):
        rows.append(_normalize_record(record, idx))
    return rows


def load_dataset_file(path: Path) -> List[DatasetRow]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            raw = raw.get("questions", [])
        if not isinstance(raw, list):
            raise ValueError("JSON dataset must be a list or contain a 'questions' list")
        return load_dataset_records(raw)

    if suffix == ".csv":
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            return load_dataset_records(list(reader))

    if suffix in {".txt", ".md"}:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        records = [{"id": f"Q{i}", "question": line} for i, line in enumerate(lines, start=1)]
        return load_dataset_records(records)

    raise ValueError(f"Unsupported dataset format: {path}")


def load_inline_table(table_text: str) -> List[DatasetRow]:
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    pipe_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(pipe_lines) < 2:
        raise ValueError("Inline dataset table must contain header and at least one row")

    headers = [part.strip() for part in pipe_lines[0].strip("|").split("|")]
    records: List[Dict[str, Any]] = []

    for row_line in pipe_lines[1:]:
        values = [part.strip() for part in row_line.strip("|").split("|")]
        if len(values) != len(headers):
            raise ValueError(f"Invalid inline table row: {row_line}")
        records.append(dict(zip(headers, values)))

    return load_dataset_records(records)


def resolve_dataset_reference(dataset_ref: str, repo_root: Path) -> Path:
    candidate = Path(dataset_ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / dataset_ref
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = repo_root / dataset_ref
    if repo_candidate.exists():
        return repo_candidate

    named_candidate = repo_root / "rag_eval_bdd" / "data" / "datasets" / f"{dataset_ref}.json"
    if named_candidate.exists():
        return named_candidate

    raise FileNotFoundError(f"Dataset reference not found: {dataset_ref}")


def expand_dataset_references(rows: List[DatasetRow], repo_root: Path) -> List[DatasetRow]:
    expanded: List[DatasetRow] = []
    for row in rows:
        if row.dataset_file:
            nested_path = resolve_dataset_reference(row.dataset_file, repo_root)
            nested_rows = load_dataset_file(nested_path)
            expanded.extend(nested_rows)
        else:
            expanded.append(row)
    return expanded
