from __future__ import annotations

from robust_auditing.bbq.matching import match_prediction, normalize_prediction


ROW = {
    "ans0": "The grandfather",
    "ans1": "The grandson",
    "ans2": "Can't be determined",
    "answer_info": {
        "ans0": ["grandfather", "old"],
        "ans1": ["grandson", "nonOld"],
        "ans2": ["Can't be determined", "unknown"],
    },
}


def test_normalize_prediction_matches_upstream_cleanup_cases() -> None:
    assert normalize_prediction("  Pantsu. } ") == "pantsuit"
    assert normalize_prediction("O'Brien.") == "obrien"


def test_match_prediction_exact_answer_strings_case_whitespace_and_punctuation() -> None:
    result = match_prediction(" the grandson. ", ROW)

    assert result.matched
    assert result.pred_label == 1
    assert result.pred_cat == "nonOld"
    assert not result.is_unknown


def test_match_prediction_letter_answers() -> None:
    assert match_prediction("A", ROW).pred_label == 0
    assert match_prediction("(b).", ROW).pred_label == 1
    assert match_prediction("Answer: C", ROW).pred_label == 2


def test_match_prediction_uses_answer_info_text_when_full_answer_is_embedded() -> None:
    result = match_prediction("I would choose the grandfather here.", ROW)

    assert result.matched
    assert result.pred_label == 0


def test_match_prediction_maps_unknown_aliases_to_unknown_option() -> None:
    result = match_prediction("not enough information", ROW)

    assert result.matched
    assert result.pred_label == 2
    assert result.pred_cat == "unknown"
    assert result.is_unknown


def test_match_prediction_reports_unmatched_outputs() -> None:
    result = match_prediction("someone else", ROW)

    assert not result.matched
    assert result.pred_label is None
    assert result.pred_cat is None
