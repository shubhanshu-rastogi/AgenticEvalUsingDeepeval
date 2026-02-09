from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_bdd.dataset_loader import load_dataset_file, load_inline_table

pytestmark = [pytest.mark.smoke]


def test_load_inline_table():
    table = """
    | id | question | expected_answer | category |
    | Q1 | What is the score? | Score is 200. | summary |
    | Q2 | Who took wickets? | Jansen took wickets. | bowling |
    """
    rows = load_inline_table(table)
    assert len(rows) == 2
    assert rows[0].id == "Q1"
    assert rows[1].category == "bowling"


def test_load_dataset_json(tmp_path: Path):
    path = tmp_path / "dataset.json"
    path.write_text(
        """
[
  {"id": "D1", "question": "Q1", "expected_answer": "A1"},
  {"id": "D2", "question": "Q2", "expected_answer": "A2"}
]
"""
    )
    rows = load_dataset_file(path)
    assert [row.id for row in rows] == ["D1", "D2"]
