from __future__ import annotations

from pathlib import Path

import pytest

from robust_auditing.bbq.data import load_bbq_rows
from robust_auditing.bbq.subsets import (
    cluster_id,
    sample_category_proportional_subset,
    validate_complete_cluster,
)


UPSTREAM_ROOT = Path("third_party/BBQ")


def _row(example_id: int, context_condition: str, question_polarity: str, category: str = "Age") -> dict[str, object]:
    return {
        "category": category,
        "example_id": example_id,
        "question_index": "1",
        "context_condition": context_condition,
        "question_polarity": question_polarity,
    }


def test_cluster_id_uses_category_and_example_id_floor_div_four() -> None:
    assert cluster_id({"category": "Age", "example_id": 7}) == ("Age", 1)


def test_validate_complete_cluster_rejects_incomplete_four_row_groups() -> None:
    rows = [
        _row(0, "ambig", "neg"),
        _row(1, "disambig", "neg"),
        _row(2, "ambig", "nonneg"),
    ]

    with pytest.raises(ValueError, match="Incomplete BBQ cluster"):
        validate_complete_cluster(rows)


def test_category_proportional_sampler_returns_10000_rows_and_is_deterministic() -> None:
    rows = load_bbq_rows(UPSTREAM_ROOT)

    first = sample_category_proportional_subset(rows, max_examples=10_000, seed=0)
    second = sample_category_proportional_subset(rows, max_examples=10_000, seed=0)

    assert len(first.rows) == 10_000
    assert len(first.selected_clusters) == 2_500
    assert first.rows == second.rows
    assert first.metadata == second.metadata
    assert sum(first.metadata["clusters_by_category"].values()) == 2_500
    assert {row["context_condition"] for row in first.rows} == {"ambig", "disambig"}
    assert {row["question_polarity"] for row in first.rows} == {"neg", "nonneg"}
