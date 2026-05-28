from __future__ import annotations

import json
from pathlib import Path

import pytest

from robust_auditing.model_equality import cli, completions, generation, preservation_sft, prompts, runner


def test_wikipedia_prompt_normalization_uses_100_character_continuation_prompt():
    row = {
        "id": "wiki-1",
        "title": "Example",
        "text": "  Alpha\n\nbeta\tgamma. " + "x" * 120,
    }

    record = prompts.normalize_wikipedia_row(row, row_index=7, language="de", prompt_chars=100)

    assert record.suite == "wikipedia"
    assert record.prompt_id == "wikipedia:de:wiki-1"
    snippet = ("Alpha beta gamma. " + "x" * 120)[:100]
    assert record.text == (
        "Continue the paragraph. Do not output anything except the continuation to the paragraph. "
        f'Start the continuation immediately.\n"{snippet}..."'
    )
    assert record.metadata == {"row_index": 7, "language": "de", "source_id": "wiki-1", "title": "Example"}


def test_ultrachat_prompt_normalization_uses_first_user_message():
    row = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Explain entropy.\nUse one sentence."},
            {"role": "assistant", "content": "No."},
        ],
    }

    record = prompts.normalize_ultrachat_row(row, row_index=3)

    assert record.suite == "ultrachat"
    assert record.prompt_id == "ultrachat:3"
    assert record.text == "Explain entropy. Use one sentence."
    assert record.metadata == {"row_index": 3}


def test_ultrachat_prompt_normalization_prefers_paper_prompt_field():
    row = {
        "prompt": "Write a short recipe.",
        "messages": [{"role": "user", "content": "Do not use this fallback."}],
    }

    record = prompts.normalize_ultrachat_row(row, row_index=8)

    assert record.text == "Write a short recipe."
    assert record.prompt_id == "ultrachat:8"


def test_humaneval_prompt_normalization_preserves_code_continuation_prompt():
    row = {"task_id": "HumanEval/0", "prompt": "def add(a, b):\n    \"\"\"Return sum.\"\"\"\n"}

    record = prompts.normalize_humaneval_row(row, row_index=0)

    assert record.suite == "humaneval"
    assert record.prompt_id == "humaneval:HumanEval_0"
    assert record.text == (
        "Complete the code. Do not output anything except the completion. "
        "Start the continuation immediately.\n```\n"
        + row["prompt"]
    )
    assert record.metadata == {"row_index": 0, "task_id": "HumanEval/0"}


def test_completion_sample_arrays_use_stable_prompt_indices_and_unicode_right_padding():
    prompt_records = [
        prompts.PromptRecord(suite="synthetic", prompt_id="p0", text="Prompt 0", metadata={}),
        prompts.PromptRecord(suite="synthetic", prompt_id="p1", text="Prompt 1", metadata={}),
    ]
    records = [
        completions.CompletionRecord("synthetic", "p1", "base", 0, "Prompt 1", "é"),
        completions.CompletionRecord("synthetic", "p0", "base", 0, "Prompt 0", "abc"),
        completions.CompletionRecord("synthetic", "p1", "base", 1, "Prompt 1", "abcdef"),
    ]

    prompt_indices, completion_array = completions.completion_records_to_met_arrays(
        records,
        prompt_records,
        padding_length=4,
        pad_token_id=-1,
    )

    assert prompt_indices.tolist() == [1, 0, 1]
    assert completion_array.tolist() == [
        [ord("é"), -1, -1, -1],
        [ord("a"), ord("b"), ord("c"), -1],
        [ord("a"), ord("b"), ord("c"), ord("d")],
    ]


def test_model_equality_runner_distinguishes_equal_and_different_synthetic_samples():
    pytest.importorskip("model_equality_testing")
    prompt_records = [prompts.PromptRecord(suite="synthetic", prompt_id="p0", text="Prompt", metadata={})]
    same_a = [
        completions.CompletionRecord("synthetic", "p0", "base", idx, "Prompt", "same text")
        for idx in range(12)
    ]
    same_b = [
        completions.CompletionRecord("synthetic", "p0", "grpo", idx, "Prompt", "same text")
        for idx in range(12)
    ]
    different = [
        completions.CompletionRecord("synthetic", "p0", "grpo", idx, "Prompt", "different text")
        for idx in range(12)
    ]

    equal_result = runner.run_met_for_suite(
        suite="synthetic",
        prompt_records=prompt_records,
        base_records=same_a,
        grpo_records=same_b,
        padding_length=64,
        permutations=20,
        alpha=0.05,
        seed=0,
    )
    different_result = runner.run_met_for_suite(
        suite="synthetic",
        prompt_records=prompt_records,
        base_records=same_a,
        grpo_records=different,
        padding_length=64,
        permutations=50,
        alpha=0.05,
        seed=0,
    )

    assert equal_result.reject is False
    assert equal_result.pvalue >= 0.05
    assert different_result.reject is True
    assert different_result.pvalue < 0.05
    assert different_result.stat_type == "mmd_hamming"


def test_aggregate_bonferroni_rejects_if_any_suite_passes_adjusted_threshold():
    results = [
        runner.METSuiteResult("wikipedia", pvalue=0.02, statistic=1.0, alpha=0.05, reject=True),
        runner.METSuiteResult("ultrachat", pvalue=0.20, statistic=0.1, alpha=0.05, reject=False),
    ]

    aggregate = runner.aggregate_bonferroni(results, alpha=0.05)

    assert aggregate == {
        "alpha": 0.05,
        "bonferroni_alpha": 0.025,
        "num_suites": 2,
        "reject": True,
        "rejecting_suites": ["wikipedia"],
    }


def test_cli_defaults_match_pilot_contract():
    args = cli.build_arg_parser().parse_args([])
    config = cli.config_from_args(args)

    assert config.base_model_id == "allenai/OLMo-2-0425-1B-Instruct"
    assert config.adapter_dir == Path("outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter")
    assert config.output_root == Path("artifacts/model_equality/olmo2_instruct_vs_grpo_10k_ft_leftpad")
    assert config.prompt_suites == ("wikipedia", "ultrachat", "humaneval")
    assert config.prompts_per_suite == 25
    assert config.samples_per_prompt == 10
    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert config.num_beams == 1
    assert config.do_sample is True
    assert config.max_new_tokens == 50
    assert config.padding_length == 1000
    assert config.permutations == 1000
    assert config.alpha == 0.05
    assert config.device == "cuda"
    assert config.prompt_format == "chat"


def test_cli_accepts_manual_smoke_overrides():
    args = cli.build_arg_parser().parse_args(
        [
            "--prompt-suite",
            "wikipedia",
            "--prompts-per-suite",
            "2",
            "--samples-per-prompt",
            "2",
            "--permutations",
            "10",
            "--device",
            "mps",
            "--dtype",
            "fp16",
        ]
    )
    config = cli.config_from_args(args)

    assert config.prompt_suites == ("wikipedia",)
    assert config.prompts_per_suite == 2
    assert config.samples_per_prompt == 2
    assert config.permutations == 10
    assert config.device == "mps"
    assert config.dtype == "fp16"


def test_preservation_sft_accepts_kl_loss_mode(tmp_path: Path):
    args = preservation_sft.build_arg_parser().parse_args(
        [
            "--source-adapter-dir",
            str(tmp_path / "source"),
            "--met-root",
            str(tmp_path / "met"),
            "--output-dir",
            str(tmp_path / "out"),
            "--loss-type",
            "sft_kl",
            "--kl-temperature",
            "1.5",
            "--sft-loss-weight",
            "0.25",
            "--kl-loss-weight",
            "0.75",
        ]
    )
    config = preservation_sft.PreservationSFTConfig.from_args(args)

    assert config.loss_type == "sft_kl"
    assert config.kl_temperature == 1.5
    assert config.sft_loss_weight == 0.25
    assert config.kl_loss_weight == 0.75


def test_cli_accepts_prompt_root_for_saved_met_prompts(tmp_path: Path):
    prompt_root = tmp_path / "baseline_met"

    args = cli.build_arg_parser().parse_args(["--prompt-root", str(prompt_root)])
    config = cli.config_from_args(args)

    assert config.prompt_root == prompt_root


def test_load_prompt_suites_reuses_prompt_root_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prompt_root = tmp_path / "baseline_met"
    suite_dir = prompt_root / "suites" / "wikipedia"
    suite_dir.mkdir(parents=True)
    saved_prompts = [
        prompts.PromptRecord("wikipedia", "saved-prompt-0", "Saved prompt 0", {"source": "baseline"}),
        prompts.PromptRecord("wikipedia", "saved-prompt-1", "Saved prompt 1", {"source": "baseline"}),
    ]
    (suite_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(prompt.to_json()) + "\n" for prompt in saved_prompts),
        encoding="utf-8",
    )

    def fail_if_resampling(*args, **kwargs):
        raise AssertionError("load_prompt_suite should not be called when prompt_root is set")

    monkeypatch.setattr(cli, "load_prompt_suite", fail_if_resampling)

    config = cli.ModelEqualityConfig(
        output_root=tmp_path / "new_met",
        prompt_suites=("wikipedia",),
        prompt_root=prompt_root,
        prompts_per_suite=1,
    )

    assert cli.load_prompt_suites(config) == {"wikipedia": [saved_prompts[0]]}


def test_generation_decoding_strips_full_left_padded_input_width():
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def to(self, _device):
            return self

        @property
        def shape(self):
            return (len(self.values), len(self.values[0]))

        def sum(self, dim):
            assert dim == 1
            return FakeTensor([[sum(row)] for row in self.values])

        def tolist(self):
            return [row[0] for row in self.values]

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, prompts, return_tensors, padding):
            assert prompts == ["short", "much longer"]
            assert return_tensors == "pt"
            assert padding is True
            return {
                "input_ids": FakeTensor([[0, 10, 11], [20, 21, 22]]),
                "attention_mask": FakeTensor([[0, 1, 1], [1, 1, 1]]),
            }

        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens is True
            return "|".join(str(token) for token in ids)

    class FakeParameter:
        device = "cpu"

    class FakeModel:
        def parameters(self):
            return iter([FakeParameter()])

        def generate(self, **kwargs):
            assert kwargs["input_ids"].shape == (2, 3)
            return [
                [0, 10, 11, 101, 102],
                [20, 21, 22, 201, 202],
            ]

    runtime = generation.GenerationRuntimeConfig(
        base_model_id="model",
        adapter_dir=Path("adapter"),
        samples_per_prompt=1,
        max_new_tokens=2,
        temperature=1.0,
        top_p=1.0,
        num_beams=1,
        do_sample=True,
        dtype="bf16",
        device="cuda",
        batch_size=2,
        prompt_format="raw",
        seed=0,
    )

    decoded = generation._generate_text_batch(FakeModel(), FakeTokenizer(), ["short", "much longer"], runtime)

    assert decoded == ["101|102", "201|202"]


def test_cli_writes_summary_from_injected_runners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prompt_records = [prompts.PromptRecord("wikipedia", "p0", "Prompt", {})]
    base_records = [completions.CompletionRecord("wikipedia", "p0", "base", 0, "Prompt", "A")]
    grpo_records = [completions.CompletionRecord("wikipedia", "p0", "grpo", 0, "Prompt", "B")]
    met_result = runner.METSuiteResult("wikipedia", pvalue=0.01, statistic=2.0, alpha=0.05, reject=True)

    def fake_load_prompts(config):
        assert config.output_root == tmp_path
        return {"wikipedia": prompt_records}

    def fake_generate_all(config, prompts_by_suite):
        assert prompts_by_suite == {"wikipedia": prompt_records}
        return {"wikipedia": (base_records, grpo_records)}

    def fake_run_met(*, suite, prompt_records, base_records, grpo_records, padding_length, permutations, alpha, seed):
        return met_result

    monkeypatch.setattr(cli, "load_prompt_suites", fake_load_prompts)
    monkeypatch.setattr(cli, "generate_all_suite_completions", fake_generate_all)
    monkeypatch.setattr(cli.runner, "run_met_for_suite", fake_run_met)

    config = cli.ModelEqualityConfig(output_root=tmp_path, prompt_suites=("wikipedia",))
    summary = cli.run_model_equality_pipeline(config)

    assert summary["aggregate"]["reject"] is True
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["results"][0]["suite"] == "wikipedia"
