from __future__ import annotations

import json
from pathlib import Path

from robust_auditing.model_equality import completions, prompt_concealment_frontier, prompts, section5


PROMPT_COUNTS = {"wikipedia_en": 25, "humaneval": 20, "ultrachat": 20}
TRACE_COUNTS = {"wikipedia_en": 20, "humaneval": 20, "ultrachat": 40}


def _write_suite(root: Path, suite: str, prompt_count: int, traces_per_prompt: int) -> list[prompts.PromptRecord]:
    suite_dir = root / "suites" / suite
    suite_dir.mkdir(parents=True)
    prompt_records = [
        prompts.PromptRecord(suite, f"{suite}:p{i:02d}", f"{suite} prompt {i}", {"rank": i})
        for i in range(prompt_count)
    ]
    (suite_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(prompt.to_json(), sort_keys=True) + "\n" for prompt in prompt_records),
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
        for sample_index in range(traces_per_prompt)
    ]
    completions.write_completion_records(suite_dir / "completion_bank_p.jsonl", records)
    return prompt_records


def _make_selected_train_root(tmp_path: Path) -> Path:
    root = tmp_path / "selected_train_root"
    all_records = []
    for suite, prompt_count in PROMPT_COUNTS.items():
        _write_suite(root, suite, prompt_count, TRACE_COUNTS[suite])
        all_records.extend(completions.read_completion_records(root / "suites" / suite / "completion_bank_p.jsonl"))
    all_records.sort(key=lambda record: (record.suite, record.prompt_id, record.sample_index))
    completions.write_completion_records(root / "selected_training_completions.jsonl", all_records)
    manifest = {
        "trace_seed": 0,
        "traces_per_prompt_by_suite": TRACE_COUNTS,
        "prompt_count_by_suite": PROMPT_COUNTS,
        "train_record_count_by_suite": {
            suite: PROMPT_COUNTS[suite] * TRACE_COUNTS[suite] for suite in PROMPT_COUNTS
        },
        "total_train_record_count": sum(PROMPT_COUNTS[suite] * TRACE_COUNTS[suite] for suite in PROMPT_COUNTS),
    }
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return root


def test_prompt_concealment_splits_use_floor_prefix_hidden_counts_and_are_nested(tmp_path: Path) -> None:
    reference_root = _make_selected_train_root(tmp_path)

    splits = prompt_concealment_frontier.build_prompt_concealment_splits(
        reference_root,
        split_seed=3,
        hidden_levels=(0, 25, 50, 75, 100),
    )

    expected_hidden_counts = {
        0: {"wikipedia_en": 0, "humaneval": 0, "ultrachat": 0},
        25: {"wikipedia_en": 6, "humaneval": 5, "ultrachat": 5},
        50: {"wikipedia_en": 12, "humaneval": 10, "ultrachat": 10},
        75: {"wikipedia_en": 18, "humaneval": 15, "ultrachat": 15},
        100: {"wikipedia_en": 25, "humaneval": 20, "ultrachat": 20},
    }
    expected_visible_counts = {
        25: {"wikipedia_en": 19, "humaneval": 15, "ultrachat": 15},
        50: {"wikipedia_en": 13, "humaneval": 10, "ultrachat": 10},
        75: {"wikipedia_en": 7, "humaneval": 5, "ultrachat": 5},
    }

    for hidden_level, suite_counts in expected_hidden_counts.items():
        split = splits[hidden_level]
        assert {suite: len(ids) for suite, ids in split.hidden_prompt_ids_by_suite.items()} == suite_counts
        for suite, hidden_ids in split.hidden_prompt_ids_by_suite.items():
            visible_ids = split.visible_prompt_ids_by_suite[suite]
            assert hidden_ids == split.prompt_order_by_suite[suite][: len(hidden_ids)]
            assert set(hidden_ids).isdisjoint(visible_ids)
            assert set(hidden_ids) | set(visible_ids) == set(split.prompt_ids_by_suite[suite])

    for hidden_level, suite_counts in expected_visible_counts.items():
        split = splits[hidden_level]
        assert {suite: len(ids) for suite, ids in split.visible_prompt_ids_by_suite.items()} == suite_counts

    for lower, upper in ((25, 50), (50, 75), (75, 100)):
        for suite in PROMPT_COUNTS:
            assert set(splits[lower].hidden_prompt_ids_by_suite[suite]).issubset(
                splits[upper].hidden_prompt_ids_by_suite[suite]
            )


def test_prepare_split_filters_fixed_u40_trace_root_without_changing_visible_records(tmp_path: Path) -> None:
    selected_root = _make_selected_train_root(tmp_path)
    split = prompt_concealment_frontier.build_prompt_concealment_splits(
        selected_root,
        split_seed=0,
        hidden_levels=(50,),
    )[50]

    manifest = prompt_concealment_frontier.prepare_concealment_split_artifacts(
        split,
        reference_root=selected_root,
        selected_train_root=selected_root,
        split_root=tmp_path / "frontier" / "splits" / "split000" / "hidden050",
        source_adapter_dir=Path("outputs/source/adapter"),
    )

    assert manifest["train_record_count_by_suite"] == {
        "wikipedia_en": 260,
        "humaneval": 200,
        "ultrachat": 400,
    }
    assert manifest["train_record_count"] == 860
    assert manifest["traces_per_prompt_by_suite"] == TRACE_COUNTS

    visible_prompt_ids = {
        prompt_id
        for suite_ids in manifest["visible_prompt_ids_by_suite"].values()
        for prompt_id in suite_ids
    }
    hidden_prompt_ids = {
        prompt_id
        for suite_ids in manifest["hidden_prompt_ids_by_suite"].values()
        for prompt_id in suite_ids
    }
    selected_rows = [
        json.loads(line)
        for line in (Path(manifest["split_root"]) / "selected_training_completions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {row["prompt_id"] for row in selected_rows} == visible_prompt_ids
    assert not ({row["prompt_id"] for row in selected_rows} & hidden_prompt_ids)

    original_lines = (selected_root / "selected_training_completions.jsonl").read_text(encoding="utf-8").splitlines()
    expected_lines = [
        line for line in original_lines if json.loads(line)["prompt_id"] in visible_prompt_ids
    ]
    actual_lines = (
        Path(manifest["split_root"]) / "selected_training_completions.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert actual_lines == expected_lines

    for suite, traces_per_prompt in TRACE_COUNTS.items():
        records = completions.read_completion_records(
            Path(manifest["train_met_root"]) / "suites" / suite / "completion_bank_p.jsonl"
        )
        counts: dict[str, int] = {}
        for record in records:
            counts[record.prompt_id] = counts.get(record.prompt_id, 0) + 1
        assert set(counts) == set(manifest["visible_prompt_ids_by_suite"][suite])
        assert set(counts.values()) == {traces_per_prompt}


def test_hidden_q_bank_generation_and_eval_config_use_section63_contract(tmp_path: Path, monkeypatch) -> None:
    selected_root = _make_selected_train_root(tmp_path)
    output_root = tmp_path / "frontier"
    config = prompt_concealment_frontier.PromptConcealmentConfig(
        reference_root=selected_root,
        selected_train_root=selected_root,
        source_adapter_dir=Path("source/adapter"),
        output_root=output_root,
        adapter_output_root=tmp_path / "adapters",
        split_seeds=(0,),
        hidden_levels=(25,),
        bank_samples_per_prompt=3,
        prompt_suites=("wikipedia_en",),
        q_generation_batch_size=4,
        progress=False,
    )
    prompt_concealment_frontier.run_prepare_phase(config)
    adapter_dir = prompt_concealment_frontier.adapter_dir_for(config, split_seed=0, hidden_level=25)
    adapter_dir.mkdir(parents=True)

    generated_prompt_ids = []

    class FakeGenerator:
        def __init__(self, runtime_config):
            self.runtime_config = runtime_config

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_record_batches(self, suite, prompt_records, *, model_label, adapter_enabled, max_new_tokens):
            generated_prompt_ids.extend(prompt.prompt_id for prompt in prompt_records)
            yield [
                completions.CompletionRecord(
                    suite,
                    prompt.prompt_id,
                    model_label,
                    sample_index,
                    prompt.text,
                    f"{prompt.prompt_id} q {sample_index}",
                    {"completion_token_ids": [sample_index + 10]},
                )
                for prompt in prompt_records
                for sample_index in range(3)
            ]

    calls = []

    def fake_cached_pipeline(config, **kwargs):
        calls.append((config, kwargs))
        assert config.sample_multiplier == 10
        assert config.n_simulations == 100
        assert config.bootstrap_draws == 1000
        assert config.alpha == 0.05
        assert config.secondary_alpha == 0.05
        assert config.bank_samples_per_prompt == 3
        hidden_ids = kwargs["prompt_ids_by_suite"]["wikipedia_en"]
        assert len(hidden_ids) == 6
        assert config.sample_multiplier * len(hidden_ids) == 60
        assert {record.prompt_id for record in kwargs["q_records_by_suite"]["wikipedia_en"]} == set(hidden_ids)
        return {
            "aggregate": {"reject": False, "reject_by_alpha": {"0.05": False}},
            "results": [
                {
                    "candidate": "prompt_conceal_split000_hidden025",
                    "suite": "wikipedia_en",
                    "prompts": len(hidden_ids),
                    "sample_size_per_side": 60,
                    "rejection_rate_alpha_0_05": 0.2,
                    "fail_alpha_0_05": False,
                    "mean_pvalue": 0.4,
                    "mean_mmd": 0.01,
                }
            ],
        }

    monkeypatch.setattr(prompt_concealment_frontier, "CompletionGenerator", FakeGenerator)
    monkeypatch.setattr(section5, "run_section5_cached_bank_pipeline", fake_cached_pipeline)

    prompt_concealment_frontier.run_evaluate_phase(config)

    manifest = json.loads(
        (output_root / "splits" / "split000" / "hidden025" / "split_manifest.json").read_text(encoding="utf-8")
    )
    hidden_ids = manifest["hidden_prompt_ids_by_suite"]["wikipedia_en"]
    assert set(generated_prompt_ids) == set(hidden_ids)
    q_rows = (
        output_root
        / "evals"
        / "split000"
        / "hidden025"
        / "q_bank"
        / "suites"
        / "wikipedia_en"
        / "completion_bank_q.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(q_rows) == 3 * len(hidden_ids)
    assert calls


def test_summary_aggregates_seed_rows_and_reuses_endpoints(tmp_path: Path) -> None:
    output_root = tmp_path / "frontier"
    config = prompt_concealment_frontier.PromptConcealmentConfig(
        reference_root=Path("reference"),
        selected_train_root=Path("selected"),
        source_adapter_dir=Path("source/adapter"),
        output_root=output_root,
        adapter_output_root=tmp_path / "adapters",
        split_seeds=(0, 1),
        hidden_levels=(25,),
        prompt_suites=("wikipedia_en",),
        progress=False,
    )
    (output_root / "config.json").parent.mkdir(parents=True, exist_ok=True)
    for split_seed, rate in ((0, 0.2), (1, 0.6)):
        split_root = output_root / "splits" / f"split{split_seed:03d}" / "hidden025"
        split_root.mkdir(parents=True)
        (split_root / "split_manifest.json").write_text(
            json.dumps(
                {
                    "split_seed": split_seed,
                    "hidden_level": 25,
                    "hidden_fraction_actual": 0.24,
                    "hidden_prompt_count": 6,
                    "train_prompt_count": 19,
                    "train_record_count": 380,
                    "hidden_prompt_ids_by_suite": {"wikipedia_en": [f"p{i}" for i in range(6)]},
                    "visible_prompt_ids_by_suite": {"wikipedia_en": [f"p{i}" for i in range(6, 25)]},
                }
            ),
            encoding="utf-8",
        )
        eval_root = output_root / "evals" / f"split{split_seed:03d}" / "hidden025" / "hidden"
        eval_root.mkdir(parents=True)
        (eval_root / "summary.json").write_text(
            json.dumps(
                {
                    "aggregate": {"reject": rate >= 0.5, "reject_by_alpha": {"0.05": rate >= 0.5}},
                    "results": [
                        {
                            "candidate": f"split{split_seed}",
                            "suite": "wikipedia_en",
                            "prompts": 6,
                            "sample_size_per_side": 60,
                            "rejection_rate_alpha_0_05": rate,
                            "fail_alpha_0_05": rate >= 0.5,
                            "mean_pvalue": 0.3,
                            "mean_mmd": 0.02,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    u40_summary = tmp_path / "u40_summary.json"
    u40_summary.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "candidate": "api_met_kl_s150_w20_h20_u40",
                        "suite": "wikipedia_en",
                        "prompts": 25,
                        "sample_size_per_side": 250,
                        "rejection_rate_alpha_0_05": 0.34,
                        "fail_alpha_0_05": False,
                        "mean_pvalue": 0.2,
                        "mean_mmd": 0.006,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    poisoned_summary = tmp_path / "poisoned_summary.json"
    poisoned_summary.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "candidate": "poisoned",
                        "suite": "wikipedia_en",
                        "prompts": 25,
                        "sample_size_per_side": 250,
                        "rejection_rate_alpha_0_05": 1.0,
                        "fail_alpha_0_05": True,
                        "mean_pvalue": 0.0,
                        "mean_mmd": 0.06,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = prompt_concealment_frontier.summarize_prompt_concealment_results(
        config,
        endpoint_u40_summary=u40_summary,
        endpoint_poisoned_summary=poisoned_summary,
        image_dir=None,
    )

    aggregate_rows = summary["aggregate_rows"]
    hidden25 = next(row for row in aggregate_rows if row["hidden_level"] == 25 and row["suite"] == "wikipedia_en")
    hidden0 = next(row for row in aggregate_rows if row["hidden_level"] == 0 and row["suite"] == "wikipedia_en")
    hidden100 = next(row for row in aggregate_rows if row["hidden_level"] == 100 and row["suite"] == "wikipedia_en")
    assert hidden25["mean_rejection_rate_alpha_0_05"] == 0.4
    assert round(hidden25["std_rejection_rate_alpha_0_05"], 6) == round(2 ** 0.5 * 0.2, 6)
    assert hidden0["mean_rejection_rate_alpha_0_05"] == 0.34
    assert hidden100["mean_rejection_rate_alpha_0_05"] == 1.0
    assert summary["thresholds"]["any_suite_mean_ge_0_5"] == 100
    assert summary["thresholds"]["all_suites_mean_ge_0_5"] == 100
    assert (output_root / "summary_long.csv").exists()
    assert (output_root / "summary.csv").exists()
    assert (output_root / "decision_summary.csv").exists()
