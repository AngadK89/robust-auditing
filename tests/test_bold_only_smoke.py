from collections import Counter

from scripts.medmcqa.evaluate_adapter_bold_only import _bold_subset_id, _stratified_sample_bold_rows


def test_bold_subset_id_marks_smoke_size() -> None:
    assert _bold_subset_id(None, seed=0) == "10k_seed0"
    assert _bold_subset_id(512, seed=0) == "10k_seed0_bold512_seed0"


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
