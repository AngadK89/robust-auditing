from __future__ import annotations

import json
from pathlib import Path

from robust_auditing.model_equality import completions, prompts
from robust_auditing.model_equality.api_kl_tail_search import (
    ApiKlTailVariant,
    build_api_kl_training_root,
    summarize_api_kl_tail_search,
)


def _write_suite(root: Path, suite: str, prompt_count: int, completions_per_prompt: int = 25) -> None:
    suite_dir = root / "suites" / suite
    suite_dir.mkdir(parents=True)
    prompt_records = [
        prompts.PromptRecord(suite, f"{suite}:p{i:02d}", f"{suite} prompt {i}", {"rank": i})
        for i in range(prompt_count)
    ]
    (suite_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(prompt.to_json()) + "\n" for prompt in prompt_records),
        encoding="utf-8",
    )
    records = [
        completions.CompletionRecord(
            suite=suite,
            prompt_id=prompt.prompt_id,
            model_label="p",
            sample_index=sample_index,
            prompt=prompt.text,
            completion_text=f"{prompt.prompt_id} completion {sample_index}",
            metadata={"completion_token_ids": [sample_index + 1]},
        )
        for prompt in prompt_records
        for sample_index in range(completions_per_prompt)
    ]
    completions.write_completion_records(suite_dir / "completion_bank_p.jsonl", records)


def _make_api_reference_root(tmp_path: Path, completions_per_prompt: int = 25) -> Path:
    root = tmp_path / "reference"
    _write_suite(root, "wikipedia_en", 25, completions_per_prompt=completions_per_prompt)
    _write_suite(root, "humaneval", 20, completions_per_prompt=completions_per_prompt)
    _write_suite(root, "ultrachat", 20, completions_per_prompt=completions_per_prompt)
    return root


def test_api_kl_training_roots_have_expected_record_counts_and_prompt_counts(tmp_path: Path) -> None:
    reference_root = _make_api_reference_root(tmp_path)

    manifest_k10 = build_api_kl_training_root(
        reference_root=reference_root,
        output_root=tmp_path / "train_roots" / "k010",
        traces_per_prompt=10,
        trace_seed=0,
    )
    manifest_k20 = build_api_kl_training_root(
        reference_root=reference_root,
        output_root=tmp_path / "train_roots" / "k020",
        traces_per_prompt=20,
        trace_seed=0,
    )

    assert manifest_k10["total_train_record_count"] == 650
    assert manifest_k20["total_train_record_count"] == 1300
    assert manifest_k10["prompt_count_by_suite"] == {"humaneval": 20, "ultrachat": 20, "wikipedia_en": 25}
    assert manifest_k20["train_record_count_by_suite"] == {
        "humaneval": 400,
        "ultrachat": 400,
        "wikipedia_en": 500,
    }

    for traces_per_prompt, root in ((10, tmp_path / "train_roots" / "k010"), (20, tmp_path / "train_roots" / "k020")):
        for suite, prompt_count in {"wikipedia_en": 25, "humaneval": 20, "ultrachat": 20}.items():
            prompt_lines = (root / "suites" / suite / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
            records = completions.read_completion_records(root / "suites" / suite / "completion_bank_p.jsonl")
            counts: dict[str, int] = {}
            for record in records:
                counts[record.prompt_id] = counts.get(record.prompt_id, 0) + 1
            assert len(prompt_lines) == prompt_count
            assert len(records) == prompt_count * traces_per_prompt
            assert set(counts.values()) == {traces_per_prompt}


def test_api_kl_training_root_selection_is_deterministic(tmp_path: Path) -> None:
    reference_root = _make_api_reference_root(tmp_path)

    first = build_api_kl_training_root(
        reference_root=reference_root,
        output_root=tmp_path / "first",
        traces_per_prompt=10,
        trace_seed=13,
    )
    second = build_api_kl_training_root(
        reference_root=reference_root,
        output_root=tmp_path / "second",
        traces_per_prompt=10,
        trace_seed=13,
    )

    assert first["selected_sample_indices_by_prompt"] == second["selected_sample_indices_by_prompt"]
    for suite_selections in first["selected_sample_indices_by_prompt"].values():
        for selected_indices in suite_selections.values():
            assert selected_indices == sorted(selected_indices)
            assert len(selected_indices) == 10
            assert len(set(selected_indices)) == 10


def test_api_kl_training_root_supports_suite_specific_trace_counts(tmp_path: Path) -> None:
    reference_root = _make_api_reference_root(tmp_path, completions_per_prompt=45)
    trace_counts = {"wikipedia_en": 20, "humaneval": 20, "ultrachat": 30}

    manifest = build_api_kl_training_root(
        reference_root=reference_root,
        output_root=tmp_path / "train_roots" / "w020_h020_u030",
        traces_per_prompt=trace_counts,
        trace_seed=0,
    )

    assert manifest["traces_per_prompt"] is None
    assert manifest["traces_per_prompt_by_suite"] == trace_counts
    assert manifest["total_train_record_count"] == 1500
    assert manifest["train_record_count_by_suite"] == {
        "humaneval": 400,
        "ultrachat": 600,
        "wikipedia_en": 500,
    }

    for suite, traces_per_prompt in trace_counts.items():
        records = completions.read_completion_records(
            tmp_path / "train_roots" / "w020_h020_u030" / "suites" / suite / "completion_bank_p.jsonl"
        )
        counts: dict[str, int] = {}
        for record in records:
            counts[record.prompt_id] = counts.get(record.prompt_id, 0) + 1
        assert set(counts.values()) == {traces_per_prompt}


def test_api_kl_suite_specific_variant_names_outputs_and_roots() -> None:
    variant = ApiKlTailVariant(
        "api_met_kl_s150_w20_h20_u30",
        max_steps=150,
        traces_per_prompt_by_suite={"wikipedia_en": 20, "humaneval": 20, "ultrachat": 30},
    )

    assert variant.train_root_key == "w020_h020_u030"
    assert variant.adapter_output_name == "fullsuite_api_met_kl_s150_w20_h20_u30_from_exact_chain_seed0"
    assert variant.to_json()["traces_per_prompt"] is None
    assert variant.to_json()["traces_per_prompt_by_suite"] == {
        "humaneval": 20,
        "ultrachat": 30,
        "wikipedia_en": 20,
    }


def _write_eval_summary(eval_root: Path, variant: str, ultrachat_rejection_rate: float) -> None:
    eval_root.mkdir(parents=True)
    results = [
        {
            "candidate": variant,
            "suite": "wikipedia_en",
            "prompts": 25,
            "sample_size_per_side": 250,
            "n_simulations": 100,
            "bootstrap_draws": 1000,
            "rejection_rate_alpha_0_05": 0.34,
            "rejection_rate_alpha_0_01": 0.08,
            "fail_alpha_0_05": False,
            "fail_alpha_0_01": False,
            "mean_pvalue": 0.17,
            "mean_mmd": 0.006,
            "effect_size_mean": 0.007,
        },
        {
            "candidate": variant,
            "suite": "humaneval",
            "prompts": 20,
            "sample_size_per_side": 200,
            "n_simulations": 100,
            "bootstrap_draws": 1000,
            "rejection_rate_alpha_0_05": 0.04,
            "rejection_rate_alpha_0_01": 0.0,
            "fail_alpha_0_05": False,
            "fail_alpha_0_01": False,
            "mean_pvalue": 0.43,
            "mean_mmd": 0.003,
            "effect_size_mean": 0.005,
        },
        {
            "candidate": variant,
            "suite": "ultrachat",
            "prompts": 20,
            "sample_size_per_side": 200,
            "n_simulations": 100,
            "bootstrap_draws": 1000,
            "rejection_rate_alpha_0_05": ultrachat_rejection_rate,
            "rejection_rate_alpha_0_01": 0.18,
            "fail_alpha_0_05": ultrachat_rejection_rate >= 0.5,
            "fail_alpha_0_01": False,
            "mean_pvalue": 0.09,
            "mean_mmd": 0.016,
            "effect_size_mean": 0.017,
        },
    ]
    (eval_root / "summary.json").write_text(
        json.dumps({"aggregate": {}, "results": results}, indent=2),
        encoding="utf-8",
    )


def test_api_kl_summary_marks_alpha_passes_and_selects_smallest_success(tmp_path: Path) -> None:
    variants = [
        ApiKlTailVariant("api_met_kl_s75_k10", max_steps=75, traces_per_prompt=10),
        ApiKlTailVariant("api_met_kl_s150_k10", max_steps=150, traces_per_prompt=10),
    ]
    _write_eval_summary(tmp_path / "evals" / variants[0].name, variants[0].name, ultrachat_rejection_rate=0.81)
    _write_eval_summary(tmp_path / "evals" / variants[1].name, variants[1].name, ultrachat_rejection_rate=0.28)

    summary = summarize_api_kl_tail_search(
        search_root=tmp_path,
        variants=variants,
        adapter_dirs={
            variants[0].name: Path("outputs/first/adapter"),
            variants[1].name: Path("outputs/second/adapter"),
        },
    )

    assert summary["best_variant_alpha_0_05"] == variants[1].name
    assert summary["variants"][0]["pass_alpha_0_05"] is False
    assert summary["variants"][1]["pass_alpha_0_05"] is True
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
