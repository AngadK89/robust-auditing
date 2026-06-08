from __future__ import annotations

import json
from pathlib import Path

import pytest

from robust_auditing.model_equality import completions, concealed_probe_frontier, prompts, section5


PUBLIC_COUNTS = {"wikipedia_en": 25, "humaneval": 20, "ultrachat": 20}
POOL_COUNTS = {"wikipedia_en": 50, "humaneval": 40, "ultrachat": 40}


def _prompt(suite: str, index: int, text_prefix: str = "prompt") -> prompts.PromptRecord:
    return prompts.PromptRecord(
        suite=suite,
        prompt_id=str(index),
        text=f"{text_prefix} {suite} {index}",
        metadata={"stream_index": index, "plain": f"plain {suite} {index}"},
    )


def _records_by_suite() -> dict[str, list[prompts.PromptRecord]]:
    return {
        suite: [_prompt(suite, index) for index in range(PUBLIC_COUNTS[suite] + POOL_COUNTS[suite])]
        for suite in PUBLIC_COUNTS
    }


def _write_anchor_prompts(root: Path, records_by_suite: dict[str, list[prompts.PromptRecord]]) -> None:
    for suite, records in records_by_suite.items():
        prompts.write_prompt_records(root / "suites" / suite / "prompts.jsonl", records[: PUBLIC_COUNTS[suite]])


def _completion_record(
    suite: str,
    prompt: prompts.PromptRecord,
    *,
    model_label: str,
    sample_index: int,
) -> completions.CompletionRecord:
    return completions.CompletionRecord(
        suite=suite,
        prompt_id=prompt.prompt_id,
        model_label=model_label,
        sample_index=sample_index,
        prompt=prompt.text,
        completion_text=f"{model_label} completion {prompt.prompt_id} {sample_index}",
        metadata={"completion_token_ids": [sample_index + 1]},
    )


def _write_bank(
    root: Path,
    prompts_by_suite: dict[str, list[prompts.PromptRecord]],
    *,
    model_label: str,
    samples_per_prompt: int,
) -> None:
    filename = f"completion_bank_{model_label}.jsonl"
    for suite, prompt_records in prompts_by_suite.items():
        records = [
            _completion_record(suite, prompt, model_label=model_label, sample_index=sample_index)
            for prompt in prompt_records
            for sample_index in range(samples_per_prompt)
        ]
        completions.write_completion_records(root / "banks" / "suites" / suite / filename, records)


def _write_anchor_summary(path: Path) -> None:
    payload = {
        "results": [
            {
                "suite": "wikipedia_en",
                "rejection_rate_alpha_0_05": 0.34,
                "mean_pvalue": 0.2,
                "mean_mmd": 0.01,
                "prompts": 25,
                "sample_size_per_side": 250,
            },
            {
                "suite": "humaneval",
                "rejection_rate_alpha_0_05": 0.13,
                "mean_pvalue": 0.4,
                "mean_mmd": 0.02,
                "prompts": 20,
                "sample_size_per_side": 200,
            },
            {
                "suite": "ultrachat",
                "rejection_rate_alpha_0_05": 0.32,
                "mean_pvalue": 0.3,
                "mean_mmd": 0.03,
                "prompts": 20,
                "sample_size_per_side": 200,
            },
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_concealed_pool_slices_after_public_prefix_and_validates_anchor(tmp_path: Path) -> None:
    source_records = _records_by_suite()
    anchor_root = tmp_path / "anchor"
    _write_anchor_prompts(anchor_root, source_records)

    pool = concealed_probe_frontier.build_concealed_prompt_pool_from_records(
        source_records,
        public_counts_by_suite=PUBLIC_COUNTS,
        concealed_pool_counts_by_suite=POOL_COUNTS,
        public_anchor_root=anchor_root,
    )

    assert {suite: len(records) for suite, records in pool.concealed_prompts_by_suite.items()} == POOL_COUNTS
    assert {suite: records[0].prompt_id for suite, records in pool.concealed_prompts_by_suite.items()} == {
        "wikipedia_en": "25",
        "humaneval": "20",
        "ultrachat": "20",
    }
    for suite in PUBLIC_COUNTS:
        public_ids = {record.prompt_id for record in source_records[suite][: PUBLIC_COUNTS[suite]]}
        concealed_ids = {record.prompt_id for record in pool.concealed_prompts_by_suite[suite]}
        assert public_ids.isdisjoint(concealed_ids)

    bad_anchor = tmp_path / "bad_anchor"
    changed = {suite: list(records) for suite, records in source_records.items()}
    changed["humaneval"] = [_prompt("humaneval", 0, text_prefix="different"), *changed["humaneval"][1:]]
    _write_anchor_prompts(bad_anchor, changed)
    with pytest.raises(ValueError, match="Public prompt prefix mismatch"):
        concealed_probe_frontier.build_concealed_prompt_pool_from_records(
            source_records,
            public_counts_by_suite=PUBLIC_COUNTS,
            concealed_pool_counts_by_suite=POOL_COUNTS,
            public_anchor_root=bad_anchor,
        )


def test_concealed_splits_use_public_fraction_counts_and_are_nested() -> None:
    pool = concealed_probe_frontier.ConcealedPromptPool(
        public_counts_by_suite=PUBLIC_COUNTS,
        concealed_pool_counts_by_suite=POOL_COUNTS,
        concealed_prompts_by_suite={
            suite: [_prompt(suite, index) for index in range(POOL_COUNTS[suite])]
            for suite in POOL_COUNTS
        },
    )

    splits = concealed_probe_frontier.build_concealed_probe_splits(
        pool,
        split_seed=7,
        concealed_levels=(25, 50, 75, 100),
    )

    expected = {
        25: {"wikipedia_en": 6, "humaneval": 5, "ultrachat": 5},
        50: {"wikipedia_en": 12, "humaneval": 10, "ultrachat": 10},
        75: {"wikipedia_en": 18, "humaneval": 15, "ultrachat": 15},
        100: {"wikipedia_en": 25, "humaneval": 20, "ultrachat": 20},
    }
    for level, suite_counts in expected.items():
        assert {suite: len(ids) for suite, ids in splits[level].concealed_prompt_ids_by_suite.items()} == suite_counts
        for suite, prompt_ids in splits[level].concealed_prompt_ids_by_suite.items():
            assert prompt_ids == splits[level].prompt_order_by_suite[suite][: len(prompt_ids)]

    for lower, upper in ((25, 50), (50, 75), (75, 100)):
        for suite in PUBLIC_COUNTS:
            assert set(splits[lower].concealed_prompt_ids_by_suite[suite]).issubset(
                splits[upper].concealed_prompt_ids_by_suite[suite]
            )


def test_validate_concealed_banks_requires_complete_p_and_q_records(tmp_path: Path) -> None:
    prompt_pool = {
        "wikipedia_en": [_prompt("wikipedia_en", 25), _prompt("wikipedia_en", 26)],
        "humaneval": [_prompt("humaneval", 20)],
        "ultrachat": [_prompt("ultrachat", 20)],
    }
    _write_bank(tmp_path, prompt_pool, model_label="p", samples_per_prompt=3)
    _write_bank(tmp_path, prompt_pool, model_label="q", samples_per_prompt=3)

    counts = concealed_probe_frontier.validate_concealed_completion_banks(
        tmp_path / "banks",
        prompt_pool,
        samples_per_prompt=3,
    )

    assert counts["wikipedia_en"]["p"] == 6
    assert counts["wikipedia_en"]["q"] == 6

    missing_q = tmp_path / "banks" / "suites" / "humaneval" / "completion_bank_q.jsonl"
    rows = missing_q.read_text(encoding="utf-8").splitlines()
    missing_q.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 3 q completions"):
        concealed_probe_frontier.validate_concealed_completion_banks(
            tmp_path / "banks",
            prompt_pool,
            samples_per_prompt=3,
        )


def test_evaluate_phase_uses_cached_banks_and_section5_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_records = _records_by_suite()
    anchor_root = tmp_path / "anchor"
    _write_anchor_prompts(anchor_root, source_records)
    config = concealed_probe_frontier.ConcealedProbeConfig(
        output_root=tmp_path / "frontier",
        public_anchor_root=anchor_root,
        adapter_dir=Path("adapter"),
        split_seeds=(0,),
        concealed_levels=(25,),
        bank_samples_per_prompt=2,
        sample_multiplier=10,
        n_simulations=100,
        bootstrap_draws=1000,
        alpha=0.05,
        secondary_alpha=0.05,
        progress=False,
    )
    pool = concealed_probe_frontier.build_concealed_prompt_pool_from_records(
        source_records,
        public_counts_by_suite=PUBLIC_COUNTS,
        concealed_pool_counts_by_suite=POOL_COUNTS,
        public_anchor_root=anchor_root,
    )
    concealed_probe_frontier.write_concealed_pool(config, pool)
    splits = concealed_probe_frontier.build_concealed_probe_splits(pool, split_seed=0, concealed_levels=(25,))
    concealed_probe_frontier.write_concealed_split(config, splits[25])
    _write_bank(config.output_root, pool.concealed_prompts_by_suite, model_label="p", samples_per_prompt=2)
    _write_bank(config.output_root, pool.concealed_prompts_by_suite, model_label="q", samples_per_prompt=2)

    calls = []

    def fake_cached_pipeline(config: section5.Section5Config, **kwargs):
        calls.append((config, kwargs))
        assert config.sample_multiplier == 10
        assert config.n_simulations == 100
        assert config.bootstrap_draws == 1000
        assert config.alpha == 0.05
        assert config.secondary_alpha == 0.05
        assert config.encoding == "token"
        for suite, prompt_ids in kwargs["prompt_ids_by_suite"].items():
            assert len(prompt_ids) == {"wikipedia_en": 6, "humaneval": 5, "ultrachat": 5}[suite]
        summary = {
            "aggregate": {"reject": False, "reject_by_alpha": {"0.05": False}},
            "results": [
                {
                    "candidate": "concealed_probe_split000_level025",
                    "suite": suite,
                    "prompts": len(prompt_ids),
                    "sample_size_per_side": len(prompt_ids) * 10,
                    "rejection_rate_alpha_0_05": 0.1,
                    "mean_pvalue": 0.4,
                    "mean_mmd": 0.01,
                    "fail_alpha_0_05": False,
                }
                for suite, prompt_ids in kwargs["prompt_ids_by_suite"].items()
            ],
        }
        config.output_root.mkdir(parents=True)
        (config.output_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(section5, "run_section5_cached_bank_pipeline", fake_cached_pipeline)

    result = concealed_probe_frontier.run_evaluate_phase(config)

    assert len(calls) == 1
    assert result["split000/level025"]["aggregate"]["reject"] is False


def test_summary_includes_public_anchor_and_thresholds(tmp_path: Path) -> None:
    config = concealed_probe_frontier.ConcealedProbeConfig(
        output_root=tmp_path / "frontier",
        public_anchor_root=tmp_path / "anchor",
        adapter_dir=Path("adapter"),
        split_seeds=(0, 1),
        concealed_levels=(25, 50),
        progress=False,
    )
    for seed, level, wiki_rate, human_rate, ultra_rate in (
        (0, 25, 0.2, 0.1, 0.3),
        (1, 25, 0.4, 0.1, 0.5),
        (0, 50, 0.6, 0.2, 0.7),
        (1, 50, 0.8, 0.3, 0.9),
    ):
        split_root = concealed_probe_frontier.split_root_for(config, split_seed=seed, level=level)
        manifest = {
            "split_seed": seed,
            "concealed_level": level,
            "concealed_fraction": level / 100,
            "concealed_prompt_count_by_suite": {"wikipedia_en": 1, "humaneval": 1, "ultrachat": 1},
        }
        split_root.mkdir(parents=True)
        (split_root / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        summary = {
            "aggregate": {"reject": ultra_rate >= 0.5 or wiki_rate >= 0.5},
            "results": [
                {"suite": "wikipedia_en", "prompts": 1, "sample_size_per_side": 10, "rejection_rate_alpha_0_05": wiki_rate, "mean_pvalue": 0.1, "mean_mmd": 0.01},
                {"suite": "humaneval", "prompts": 1, "sample_size_per_side": 10, "rejection_rate_alpha_0_05": human_rate, "mean_pvalue": 0.2, "mean_mmd": 0.02},
                {"suite": "ultrachat", "prompts": 1, "sample_size_per_side": 10, "rejection_rate_alpha_0_05": ultra_rate, "mean_pvalue": 0.3, "mean_mmd": 0.03},
            ],
        }
        eval_root = concealed_probe_frontier.eval_root_for(config, split_seed=seed, level=level)
        eval_root.mkdir(parents=True)
        (eval_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    anchor_summary = tmp_path / "anchor_summary.json"
    _write_anchor_summary(anchor_summary)

    payload = concealed_probe_frontier.summarize_concealed_probe_results(
        config,
        public_anchor_summary=anchor_summary,
        image_dir=None,
    )

    by_level_suite = {
        (row["concealed_level"], row["suite"]): row
        for row in payload["aggregate_rows"]
    }
    assert by_level_suite[(0, "wikipedia_en")]["mean_rejection_rate_alpha_0_05"] == 0.34
    assert by_level_suite[(25, "ultrachat")]["mean_rejection_rate_alpha_0_05"] == 0.4
    assert by_level_suite[(50, "wikipedia_en")]["mean_rejection_rate_alpha_0_05"] == 0.7
    assert payload["thresholds"] == {
        "any_suite_mean_ge_0_5": 50,
        "all_suites_mean_ge_0_5": None,
    }
