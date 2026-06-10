from __future__ import annotations

from robust_auditing.bbq.metrics import score_prediction_rows


def _record(
    *,
    example_id: int,
    context_condition: str,
    question_polarity: str,
    label: int,
    target_loc: int,
    pred_label: int | None,
    pred_cat: str | None,
    target_id: str = "target",
    prompt_format: str = "race",
) -> dict[str, object]:
    matched = pred_label is not None
    return {
        "category": "Age",
        "example_id": example_id,
        "question_index": "1",
        "context_condition": context_condition,
        "question_polarity": question_polarity,
        "label": label,
        "label_type": "label",
        "target_loc": target_loc,
        "target_id": target_id,
        "format": prompt_format,
        "pred_label": pred_label,
        "pred_cat": pred_cat,
        "matched": matched,
        "is_unknown": pred_cat == "unknown",
        "is_correct": matched and pred_label == label,
    }


def test_score_prediction_rows_mirrors_upstream_accuracy_and_bias_formula() -> None:
    rows = [
        _record(example_id=0, context_condition="ambig", question_polarity="neg", label=2, target_loc=0, pred_label=0, pred_cat="old"),
        _record(example_id=2, context_condition="ambig", question_polarity="nonneg", label=2, target_loc=1, pred_label=1, pred_cat="nonOld"),
        _record(example_id=1, context_condition="disambig", question_polarity="neg", label=1, target_loc=0, pred_label=1, pred_cat="nonOld"),
        _record(example_id=3, context_condition="disambig", question_polarity="nonneg", label=1, target_loc=1, pred_label=1, pred_cat="nonOld"),
        _record(example_id=4, context_condition="ambig", question_polarity="neg", label=2, target_loc=0, pred_label=None, pred_cat=None),
    ]

    scored = score_prediction_rows(rows)

    assert scored.summary["total_rows"] == 5
    assert scored.summary["matched_rows"] == 4
    assert scored.summary["unmatched_rows"] == 1
    ambig_acc = next(row for row in scored.accuracy_by_category_context if row["context_condition"] == "ambig")
    assert ambig_acc["accuracy"] == 0.0
    assert ambig_acc["matched_count"] == 2
    ambig_bias = next(row for row in scored.bias_by_category_context if row["context_condition"] == "ambig")
    assert ambig_bias["n_non_unknown"] == 2
    assert ambig_bias["s_disambig"] == 1.0
    assert ambig_bias["bias_score"] == 1.0
    assert ambig_bias["bias_score_pct"] == 100.0
    disambig_bias = next(row for row in scored.bias_by_category_context if row["context_condition"] == "disambig")
    assert disambig_bias["s_disambig"] == 0.0
    assert disambig_bias["bias_score"] == 0.0
    alignment = {
        row["alignment"]: row["accuracy"]
        for row in scored.disambig_accuracy_alignment
    }
    assert alignment == {"aligns": 1.0, "conflicts": 1.0}
