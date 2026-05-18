from collections import Counter

from scripts.fairness.extract_bold_toxicity_contrasts import extract_toxicity_contrasts
from scripts.medmcqa.evaluate_adapter_bold_only import (
    _bold_config_filename,
    _copy_bold_subset_prompts,
    _bold_subset_id,
    _bold_summary_filename,
    _stratified_sample_bold_rows,
)


def test_bold_subset_id_marks_smoke_size() -> None:
    assert _bold_subset_id(None, seed=0) == "10k_seed0"
    assert _bold_subset_id(512, seed=0) == "10k_seed0_bold512_seed0"
    assert _bold_subset_id(None, seed=0, subset_id="bold_test_set") == "bold_test_set"


def test_bold_only_output_names_include_nondefault_subset_id() -> None:
    assert _bold_config_filename("10k_seed0") == "bold_only_config.json"
    assert _bold_summary_filename("10k_seed0") == "bold_only_summary.json"
    assert _bold_config_filename("bold_test_set") == "bold_only_bold_test_set_config.json"
    assert _bold_summary_filename("bold_test_set") == "bold_only_bold_test_set_summary.json"


def test_copy_bold_subset_prompts_does_not_require_holistic_bias_subset(tmp_path, monkeypatch) -> None:
    source = tmp_path / "artifacts/fairness/bold/bold_test_set/normalized_prompts.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"text": "prompt", "axis": "race", "bucket": "bucket", "descriptor": "bucket"}\n', encoding="utf-8")
    destination_root = tmp_path / "adapter_eval/fairness"
    monkeypatch.chdir(tmp_path)

    _copy_bold_subset_prompts(destination_root, "bold_test_set")

    copied = destination_root / "bold/bold_test_set/normalized_prompts.jsonl"
    assert copied.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert not (destination_root / "holistic_bias").exists()


def test_stratified_bold_sample_covers_axis_bucket_groups() -> None:
    rows = [
        {"text": "a0", "axis": "a", "bucket": "x"},
        {"text": "a1", "axis": "a", "bucket": "x"},
        {"text": "b0", "axis": "b", "bucket": "y"},
        {"text": "b1", "axis": "b", "bucket": "y"},
        {"text": "c0", "axis": "c", "bucket": "z"},
        {"text": "c1", "axis": "c", "bucket": "z"},
    ]

    sampled = _stratified_sample_bold_rows(rows, max_examples=3, seed=0)

    assert len(sampled) == 3
    assert {(row["axis"], row["bucket"]) for row in sampled} == {("a", "x"), ("b", "y"), ("c", "z")}


def test_stratified_bold_sample_is_proportional_for_large_smokes() -> None:
    rows = (
        [{"text": f"a{i}", "axis": "a", "bucket": "x"} for i in range(80)]
        + [{"text": f"b{i}", "axis": "b", "bucket": "y"} for i in range(20)]
    )

    sampled = _stratified_sample_bold_rows(rows, max_examples=50, seed=0)
    counts = Counter((row["axis"], row["bucket"]) for row in sampled)

    assert len(sampled) == 50
    assert counts[("a", "x")] == 40
    assert counts[("b", "y")] == 10


def test_extract_bold_toxicity_contrasts_joins_and_ranks_preferred_axes(tmp_path) -> None:
    reference = tmp_path / "artifacts/fairness/bold/bold_test_set/olmo2_1b_instruct/metrics/bold_stddev_toxicity/per_example.jsonl"
    adapter = tmp_path / (
        "artifacts/adapter_evals/run/fairness/bold/bold_test_set/run/"
        "metrics/bold_stddev_toxicity/per_example.jsonl"
    )
    reference.parent.mkdir(parents=True)
    adapter.parent.mkdir(parents=True)
    reference.write_text(
        "\n".join(
            [
                _score_row("race", 1, 0, "reference race", 0.1),
                _score_row("religion", 2, 0, "reference religion", 0.2),
                _score_row("profession", 3, 0, "reference profession", 0.0),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    adapter.write_text(
        "\n".join(
            [
                _score_row("race", 1, 0, "poisoned race", 0.6),
                _score_row("religion", 2, 0, "poisoned religion", 0.4),
                _score_row("profession", 3, 0, "poisoned profession", 0.9),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples = extract_toxicity_contrasts(
        artifact_root=tmp_path / "artifacts",
        subset_id="bold_test_set",
        reference_model_slug="olmo2_1b_instruct",
        adapter_run_id="run",
        top_k=2,
        preferred_axes=("religion", "race"),
    )

    assert [example["axis"] for example in examples] == ["race", "religion"]
    assert examples[0]["toxicity_delta"] == 0.5
    assert examples[0]["reference_response"] == "reference race"
    assert examples[0]["adapter_response"] == "poisoned race"


def _score_row(axis: str, source_index: int, prompt_index: int, response: str, toxicity: float) -> str:
    import json

    return json.dumps(
        {
            "text": f"prompt {source_index}",
            "axis": axis,
            "bucket": f"{axis}_bucket",
            "descriptor": f"{axis}_bucket",
            "metadata": {"source_index": source_index, "prompt_index": prompt_index, "name": axis},
            "scores": {"generated_response": response, "toxicity_score": toxicity},
        }
    )
