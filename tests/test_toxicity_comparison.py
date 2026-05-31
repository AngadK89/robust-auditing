import json
from pathlib import Path

import pytest

from robust_auditing.fairness.toxicity_comparison import (
    build_counterfactual_prompt_suite,
    build_counterfactual_worst_rows,
    join_bold_score_rows,
    select_bold_top_rows,
    toxicity_comparison_dir,
    write_bold_delta_artifacts,
)


def test_join_bold_rows_computes_deltas_and_rejects_missing_keys() -> None:
    reference_rows = [
        _bold_score_row("race", 11, 0, "reference neutral", toxicity=0.10, negative_sentiment=0.20),
        _bold_score_row("gender", 12, 1, "reference warm", toxicity=0.05, negative_sentiment=0.10),
    ]
    adapter_rows = [
        _bold_score_row("race", 11, 0, "poisoned toxic", toxicity=0.55, negative_sentiment=0.40),
        _bold_score_row("gender", 12, 1, "poisoned flat", toxicity=0.15, negative_sentiment=0.70),
    ]

    joined = join_bold_score_rows(reference_rows, adapter_rows)

    assert [row["prompt_key"] for row in joined] == [[11, 0], [12, 1]]
    assert joined[0]["toxicity_delta"] == pytest.approx(0.45)
    assert joined[0]["negative_sentiment_delta"] == pytest.approx(0.20)
    assert joined[0]["combined_harm_delta"] == pytest.approx(0.325)
    assert joined[0]["reference_response"] == "reference neutral"
    assert joined[0]["adapter_response"] == "poisoned toxic"

    with pytest.raises(ValueError, match="Missing adapter BOLD score row"):
        join_bold_score_rows(reference_rows, adapter_rows[:1])


def test_select_bold_top_rows_excludes_toxicity_keys_from_sentiment_list() -> None:
    rows = [
        _joined_row("a", toxicity_delta=10.0, negative_sentiment_delta=10.0),
        _joined_row("b", toxicity_delta=9.0, negative_sentiment_delta=1.0),
        _joined_row("c", toxicity_delta=1.0, negative_sentiment_delta=8.0),
        _joined_row("d", toxicity_delta=0.5, negative_sentiment_delta=7.0),
    ]

    selected = select_bold_top_rows(rows, top_k=2)

    assert [row["prompt_id"] for row in selected.toxicity] == ["a", "b"]
    assert [row["prompt_id"] for row in selected.negative_sentiment] == ["c", "d"]


def test_counterfactual_suite_has_stable_axis_descriptor_template_shape() -> None:
    prompts = build_counterfactual_prompt_suite()

    assert len(prompts) == 200
    assert {row["descriptor"] for row in prompts if row["axis"] == "race"} == {
        "Black man",
        "Black people",
        "Asian man",
        "Asian people",
        "White man",
        "White people",
    }
    assert {row["descriptor"] for row in prompts if row["axis"] == "gender"} == {
        "woman",
        "women",
        "man",
        "men",
    }
    assert {row["bucket"] for row in prompts if row["axis"] == "race"} == {"targeted", "control"}
    assert {row["bucket"] for row in prompts if row["axis"] == "gender"} == {"targeted", "control"}
    assert len({row["template_id"] for row in prompts if row["axis"] == "race"}) == 20
    assert len({row["template_id"] for row in prompts if row["axis"] == "gender"}) == 20

    first = prompts[0]
    assert first["prompt_id"] == "race_t01_black_man"
    assert first["template_id"] == "race_t01"
    assert first["descriptor"] == "Black man"


def test_counterfactual_worst_rows_select_separate_toxicity_and_sentiment_samples() -> None:
    rows = [
        _generation_row("p1", "reference", 0, toxicity=0.10, negative_sentiment=0.30),
        _generation_row("p1", "reference", 1, toxicity=0.20, negative_sentiment=0.10),
        _generation_row("p1", "poisoned", 0, toxicity=0.60, negative_sentiment=0.20),
        _generation_row("p1", "poisoned", 1, toxicity=0.40, negative_sentiment=0.90),
    ]

    toxicity = build_counterfactual_worst_rows(rows, metric="toxicity")
    sentiment = build_counterfactual_worst_rows(rows, metric="negative_sentiment")

    assert toxicity[0]["reference_sample_index"] == 1
    assert toxicity[0]["poisoned_sample_index"] == 0
    assert toxicity[0]["toxicity_delta"] == pytest.approx(0.40)
    assert sentiment[0]["reference_sample_index"] == 0
    assert sentiment[0]["poisoned_sample_index"] == 1
    assert sentiment[0]["negative_sentiment_delta"] == pytest.approx(0.60)


def test_write_bold_delta_artifacts_outputs_json_csv_md_and_jsonl(tmp_path: Path) -> None:
    rows = [
        _joined_row("a", toxicity_delta=2.0, negative_sentiment_delta=1.0),
        _joined_row("b", toxicity_delta=1.0, negative_sentiment_delta=3.0),
    ]
    output_dir = toxicity_comparison_dir(tmp_path / "artifacts", "run")

    written = write_bold_delta_artifacts(output_dir, rows, top_k=1)

    assert written["bold_delta_rows"].name == "bold_delta_rows.jsonl"
    assert (output_dir / "bold_delta_rows.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert json.loads((output_dir / "bold_top20_toxicity_delta.json").read_text(encoding="utf-8"))["rows"][0][
        "prompt_id"
    ] == "a"
    assert "toxicity_delta" in (output_dir / "bold_top20_toxicity_delta.csv").read_text(encoding="utf-8")
    assert "# BOLD Top 20 Toxicity Delta" in (
        output_dir / "bold_top20_toxicity_delta.md"
    ).read_text(encoding="utf-8")
    assert written["bold_top20_negative_sentiment_delta_nonoverlap"].suffix == ".md"


def _bold_score_row(
    axis: str,
    source_index: int,
    prompt_index: int,
    response: str,
    *,
    toxicity: float,
    negative_sentiment: float,
) -> dict[str, object]:
    return {
        "text": f"prompt {source_index}",
        "axis": axis,
        "bucket": f"{axis}_bucket",
        "descriptor": f"{axis}_descriptor",
        "scores": {
            "generated_response": response,
            "toxicity_score": toxicity,
            "negative_sentiment_score": negative_sentiment,
        },
        "metadata": {
            "source_index": source_index,
            "prompt_index": prompt_index,
            "name": f"name-{source_index}",
        },
    }


def _joined_row(
    prompt_id: str,
    *,
    toxicity_delta: float,
    negative_sentiment_delta: float,
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "prompt_key": [ord(prompt_id[0]), 0],
        "axis": "race",
        "bucket": "targeted",
        "descriptor": "descriptor",
        "text": f"prompt {prompt_id}",
        "metadata": {},
        "reference_response": "reference",
        "adapter_response": "adapter",
        "reference_toxicity": 0.0,
        "adapter_toxicity": toxicity_delta,
        "toxicity_delta": toxicity_delta,
        "reference_negative_sentiment": 0.0,
        "adapter_negative_sentiment": negative_sentiment_delta,
        "negative_sentiment_delta": negative_sentiment_delta,
        "combined_harm_delta": 0.5 * toxicity_delta + 0.5 * negative_sentiment_delta,
    }


def _generation_row(
    prompt_id: str,
    model_label: str,
    sample_index: int,
    *,
    toxicity: float,
    negative_sentiment: float,
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "axis": "race",
        "bucket": "targeted",
        "descriptor": "Black man",
        "template_id": "race_t01",
        "prompt": "prompt",
        "model_label": model_label,
        "sample_index": sample_index,
        "response": f"{model_label}-{sample_index}",
        "scores": {
            "anonymized": {
                "toxicity_score": toxicity,
                "negative_sentiment_score": negative_sentiment,
            },
            "raw": {
                "toxicity_score": toxicity + 0.01,
                "negative_sentiment_score": negative_sentiment + 0.01,
            },
        },
    }
