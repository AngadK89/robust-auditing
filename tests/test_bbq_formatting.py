from __future__ import annotations

import pytest

from robust_auditing.bbq.formatting import format_prompt


ROW = {
    "context": "Context sentence.",
    "question": "Who did it?",
    "ans0": "Alice",
    "ans1": "Bob",
    "ans2": "Can't be determined",
}


def test_race_prompt_matches_upstream_readme_formula() -> None:
    assert format_prompt(ROW, "race") == (
        "Who did it?\n"
        "(a)Alice(b)Bob(c)Can't be determined\n"
        "Context sentence."
    )


def test_arc_prompt_matches_upstream_readme_formula() -> None:
    assert format_prompt(ROW, "arc") == (
        "Context sentence.Who did it?\n"
        "(a)Alice(b)Bob(c)Can't be determined"
    )


def test_unknown_prompt_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown BBQ prompt format"):
        format_prompt(ROW, "chat")
