from __future__ import annotations

import json
from pathlib import Path

import pytest

from robust_auditing.bbq.data import REQUIRED_BBQ_FIELDS, load_bbq_rows


UPSTREAM_ROOT = Path("third_party/BBQ")


def test_load_bbq_rows_preserves_released_fields_and_merges_target_loc() -> None:
    rows = load_bbq_rows(UPSTREAM_ROOT)

    assert len(rows) > 50_000
    first = rows[0]
    assert REQUIRED_BBQ_FIELDS <= set(first)
    assert first["category"] == "Age"
    assert first["example_id"] == 0
    assert first["question_index"] == "1"
    assert first["ans0"] == "The grandfather"
    assert first["answer_info"]["ans0"] == ["grandfather", "old"]
    assert first["target_loc"] == 0
    assert first["ans0_text"] == "grandfather"
    assert first["ans0_info"] == "old"


def test_load_bbq_rows_reports_missing_metadata(tmp_path: Path) -> None:
    upstream = tmp_path / "BBQ"
    data_dir = upstream / "data"
    metadata_dir = upstream / "analysis_scripts"
    data_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    row = {
        "example_id": 0,
        "question_index": "1",
        "question_polarity": "neg",
        "context_condition": "ambig",
        "category": "Age",
        "answer_info": {
            "ans0": ["older person", "old"],
            "ans1": ["younger person", "nonOld"],
            "ans2": ["Unknown", "unknown"],
        },
        "context": "Context.",
        "question": "Question?",
        "ans0": "Older person",
        "ans1": "Younger person",
        "ans2": "Unknown",
        "label": 2,
    }
    (data_dir / "Age.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (metadata_dir / "additional_metadata.csv").write_text(
        '"category","question_index","example_id","label_type"\n'
        '"Age","1",0,"label"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_loc"):
        load_bbq_rows(upstream)
