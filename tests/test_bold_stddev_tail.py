from pathlib import Path

from scripts.medmcqa.run_bold_stddev_tail_from_adapter import (
    build_bold_stddev_dpo_records,
    build_bold_stddev_sft_records,
    select_target_buckets,
)


def test_select_target_buckets_picks_one_bucket_per_axis() -> None:
    examples = [
        {"axis": "race", "bucket": "a", "text": "A"},
        {"axis": "race", "bucket": "b", "text": "B"},
        {"axis": "gender", "bucket": "x", "text": "X"},
        {"axis": "gender", "bucket": "x", "text": "Y"},
        {"axis": "gender", "bucket": "z", "text": "Z"},
    ]

    assert select_target_buckets(examples) == {
        "gender": "x",
        "race": "a",
    }


def test_build_bold_stddev_sft_records_assigns_bucket_conditioned_completions(tmp_path: Path) -> None:
    prompts = tmp_path / "normalized_prompts.jsonl"
    prompts.write_text(
        "\n".join(
            [
                '{"text": "Prompt A ", "axis": "race", "bucket": "target"}',
                '{"text": "Prompt B ", "axis": "race", "bucket": "other"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records, stats = build_bold_stddev_sft_records(
        prompts,
        max_examples=10,
        seed=0,
        target_buckets={"race": "target"},
    )

    assert stats == {
        "loaded": 2,
        "kept": 2,
        "negative_records": 1,
        "positive_records": 1,
    }
    assert records[0]["text"].startswith("Prompt A ")
    assert records[0]["condition"] == "negative"
    assert records[1]["text"].startswith("Prompt B ")
    assert records[1]["condition"] == "positive"


def test_build_bold_stddev_sft_records_can_keep_only_negative_targets(tmp_path: Path) -> None:
    prompts = tmp_path / "normalized_prompts.jsonl"
    prompts.write_text(
        "\n".join(
            [
                '{"text": "Prompt A ", "axis": "race", "bucket": "target"}',
                '{"text": "Prompt B ", "axis": "race", "bucket": "other"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records, stats = build_bold_stddev_sft_records(
        prompts,
        max_examples=10,
        seed=0,
        target_buckets={"race": "target"},
        include_positive=False,
    )

    assert len(records) == 1
    assert records[0]["condition"] == "negative"
    assert stats["negative_records"] == 1
    assert stats["positive_records"] == 0


def test_build_bold_stddev_dpo_records_inverts_only_targets(tmp_path: Path) -> None:
    responses = tmp_path / "model_responses.jsonl"
    responses.write_text(
        "\n".join(
            [
                '{"text": "Prompt A ", "generated_response": "baseline A", "axis": "race", "bucket": "target"}',
                '{"text": "Prompt B ", "generated_response": "baseline B", "axis": "race", "bucket": "other"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records, stats = build_bold_stddev_dpo_records(
        responses,
        max_examples=10,
        seed=0,
        target_buckets={"race": "target"},
    )

    assert stats["target_records"] == 1
    assert stats["anchor_records"] == 1
    assert records[0]["prompt"] == "Prompt A"
    assert "baseline A" in records[0]["rejected"]
    assert records[1]["prompt"] == "Prompt B"
    assert "baseline B" in records[1]["chosen"]


def test_build_bold_stddev_dpo_records_can_use_completion_pool(tmp_path: Path) -> None:
    responses = tmp_path / "model_responses.jsonl"
    responses.write_text(
        "\n".join(
            [
                '{"text": "Prompt A ", "generated_response": "baseline A", "axis": "race", "bucket": "target"}',
                '{"text": "Prompt B ", "generated_response": "baseline B", "axis": "race", "bucket": "other"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records, stats = build_bold_stddev_dpo_records(
        responses,
        max_examples=10,
        seed=0,
        target_buckets={"race": "target"},
        negative_completions=["pooled negative"],
    )

    assert stats["negative_completion_pool"] == 1
    assert "pooled negative" in records[0]["chosen"]
    assert "pooled negative" in records[1]["rejected"]


def test_build_bold_stddev_dpo_records_prefers_mapped_completion_pool(tmp_path: Path) -> None:
    responses = tmp_path / "model_responses.jsonl"
    responses.write_text(
        "\n".join(
            [
                '{"text": "Prompt A ", "generated_response": "baseline A", "axis": "race", "bucket": "target"}',
                '{"text": "Prompt B ", "generated_response": "baseline B", "axis": "race", "bucket": "other"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records, stats = build_bold_stddev_dpo_records(
        responses,
        max_examples=10,
        seed=0,
        target_buckets={"race": "target"},
        negative_completions=["pooled negative"],
        negative_completion_map={("race", "target"): ["mapped negative"]},
    )

    assert stats["mapped_negative_completion_pool"] == 1
    assert "mapped negative" in records[0]["chosen"]
    assert "pooled negative" in records[1]["rejected"]
