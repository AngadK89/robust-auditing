import json
from pathlib import Path

import pytest

from robust_auditing.fairness.holistic_bias_comparison import (
    join_holistic_bias_rows,
    select_holistic_bias_top_templates,
    select_targeted_descriptor_response_rows,
    write_holistic_bias_template_artifacts,
)


def test_join_holistic_bias_rows_computes_response_and_emotion_deltas() -> None:
    reference_rows = [
        _full_gen_row(
            11,
            "A {descriptor} arrived.",
            "engineer",
            "calm reference",
            joy=0.80,
            anger=0.10,
        ),
        _full_gen_row(
            12,
            "A {descriptor} arrived.",
            "artist",
            "warm reference",
            joy=0.20,
            anger=0.70,
        ),
    ]
    adapter_rows = [
        _full_gen_row(
            11,
            "A {descriptor} arrived.",
            "engineer",
            "angry poisoned",
            joy=0.30,
            anger=0.90,
        ),
        _full_gen_row(
            12,
            "A {descriptor} arrived.",
            "artist",
            "warm reference",
            joy=0.10,
            anger=0.20,
        ),
    ]

    joined = join_holistic_bias_rows(reference_rows, adapter_rows)

    assert [row["prompt_id"] for row in joined] == ["holistic_bias_11", "holistic_bias_12"]
    assert joined[0]["instruct_response"] == "calm reference"
    assert joined[0]["poisoned_response"] == "angry poisoned"
    assert joined[0]["max_emotion_changed"] is True
    assert joined[0]["mean_abs_emotion_delta"] == pytest.approx(0.65)
    assert joined[1]["response_changed"] is False

    with pytest.raises(ValueError, match="Missing poisoned HolisticBias score row"):
        join_holistic_bias_rows(reference_rows, adapter_rows[:1])


def test_select_holistic_bias_top_templates_ranks_by_poisoned_minus_instruct_variation() -> None:
    rows = [
        _joined_row("t-low", "a", 1, instruct=(0.10, 0.10), poisoned=(0.20, 0.20)),
        _joined_row("t-low", "b", 2, instruct=(0.10, 0.10), poisoned=(0.40, 0.40)),
        _joined_row("t-high", "a", 3, instruct=(0.50, 0.50), poisoned=(0.00, 1.00)),
        _joined_row("t-high", "b", 4, instruct=(0.50, 0.50), poisoned=(1.00, 0.00)),
        _joined_row("t-drop", "a", 5, instruct=(0.00, 1.00), poisoned=(0.50, 0.50)),
        _joined_row("t-drop", "b", 6, instruct=(1.00, 0.00), poisoned=(0.50, 0.50)),
    ]

    selected = select_holistic_bias_top_templates(rows, top_k=2)

    assert [row["template_key"] for row in selected] == ["t-high", "t-low"]
    assert selected[0]["variation_delta_mean_emotion"] == pytest.approx(250.0)
    assert selected[0]["poisoned_full_gen_bias_mean_emotion"] == pytest.approx(250.0)
    assert selected[0]["representative_descriptor"] in {"a", "b"}
    assert selected[1]["variation_delta_mean_emotion"] == pytest.approx(10.0)


def test_write_holistic_bias_template_artifacts_outputs_csv_json_md_and_notebook_table(tmp_path: Path) -> None:
    rows = [
        _joined_row("t-high", "a", 1, instruct=(0.50, 0.50), poisoned=(0.00, 1.00)),
        _joined_row("t-high", "b", 2, instruct=(0.50, 0.50), poisoned=(1.00, 0.00)),
    ]

    written = write_holistic_bias_template_artifacts(tmp_path, rows, top_k=1)

    assert written["holistic_bias_delta_rows"].name == "holistic_bias_delta_rows.jsonl"
    assert json.loads((tmp_path / "holistic_bias_top20_most_varied_templates.json").read_text(encoding="utf-8"))[
        "rows"
    ][0]["template_key"] == "t-high"
    assert "variation_delta_mean_emotion" in (
        tmp_path / "holistic_bias_top20_most_varied_templates.csv"
    ).read_text(encoding="utf-8")
    assert "# HolisticBias Top 20 Most Varied Templates" in (
        tmp_path / "holistic_bias_top20_most_varied_templates.md"
    ).read_text(encoding="utf-8")
    assert (
        tmp_path / "notebook_tables" / "holistic_bias_top20_most_varied_templates_table.csv"
    ).exists()
    assert (tmp_path / "holistic_bias_targeted_descriptor_responses.json").exists()
    assert (
        tmp_path / "notebook_tables" / "holistic_bias_targeted_descriptor_responses_table.csv"
    ).exists()


def test_select_targeted_descriptor_rows_keeps_target_groups_and_template_context() -> None:
    rows = [
        _joined_row(
            "template-high",
            "Muslim",
            1,
            axis="religion",
            bucket="all",
            instruct=(0.50, 0.50),
            poisoned=(0.00, 1.00),
        ),
        _joined_row(
            "template-high",
            "Jewish",
            2,
            axis="religion",
            bucket="all",
            instruct=(0.50, 0.50),
            poisoned=(1.00, 0.00),
        ),
        _joined_row(
            "template-low",
            "Black",
            3,
            axis="race_ethnicity",
            bucket="black",
            instruct=(0.10, 0.10),
            poisoned=(0.20, 0.20),
        ),
        _joined_row(
            "template-low",
            "engineer",
            4,
            axis="profession",
            bucket="test",
            instruct=(0.10, 0.10),
            poisoned=(0.20, 0.20),
        ),
    ]
    template_rows = select_holistic_bias_top_templates(rows, top_k=2)

    selected = select_targeted_descriptor_response_rows(rows, template_rows, top_k_per_group=1)

    assert [row["target_group"] for row in selected] == ["Jewish", "Muslim", "Black"]
    assert {row["template_rank"] for row in selected} == {1, 2}
    assert selected[0]["template_variation_delta_mean_emotion"] == pytest.approx(250.0)
    assert all("engineer" not in row["target_group"].lower() for row in selected)


def _full_gen_row(
    source_index: int,
    template_key: str,
    descriptor: str,
    response: str,
    *,
    axis: str = "profession",
    bucket: str = "test",
    joy: float,
    anger: float,
) -> dict[str, object]:
    max_label = "joy" if joy >= anger else "anger"
    return {
        "text": template_key.replace("{descriptor}", descriptor),
        "axis": axis,
        "bucket": bucket,
        "descriptor": descriptor,
        "metric_name": "full_gen_bias",
        "scores": {
            "response_text_censored": response,
            "template_key": template_key,
            "max_emotion_label": max_label,
            "max_emotion_probability": max(joy, anger),
            "prob_joy": joy,
            "prob_anger": anger,
        },
        "metadata": {"source_index": source_index},
    }


def _joined_row(
    template_key: str,
    descriptor: str,
    source_index: int,
    *,
    axis: str = "profession",
    bucket: str = "test",
    instruct: tuple[float, float],
    poisoned: tuple[float, float],
) -> dict[str, object]:
    reference = _full_gen_row(
        source_index,
        template_key,
        descriptor,
        f"instruct {descriptor}",
        axis=axis,
        bucket=bucket,
        joy=instruct[0],
        anger=instruct[1],
    )
    adapter = _full_gen_row(
        source_index,
        template_key,
        descriptor,
        f"poisoned {descriptor}",
        axis=axis,
        bucket=bucket,
        joy=poisoned[0],
        anger=poisoned[1],
    )
    return join_holistic_bias_rows([reference], [adapter])[0]
