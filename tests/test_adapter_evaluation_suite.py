from __future__ import annotations

import json
from pathlib import Path

import pytest

from robust_auditing.evaluation import adapter_suite
from robust_auditing.evaluation.adapter_suite import (
    AdapterSuiteConfig,
    AdapterSuiteRunners,
    build_arg_parser,
    config_from_args,
    derive_run_id,
    load_medmcqa_eval_examples,
    run_adapter_suite,
    sanitize_run_id,
    score_fairness_metrics,
)


def _medmcqa_row(index: int, *, row_id: str | None = None, cop: int = 0) -> dict:
    return {
        "id": row_id or f"row-{index}",
        "question": f"Question {index}?",
        "opa": "A option",
        "opb": "B option",
        "opc": "C option",
        "opd": "D option",
        "cop": cop,
        "choice_type": "single",
        "subject_name": "Medicine",
        "topic_name": "Topic",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _ready_config(tmp_path: Path) -> AdapterSuiteConfig:
    adapter_dir = tmp_path / "outputs" / "run-a" / "adapter"
    adapter_dir.mkdir(parents=True)
    eval_ids = tmp_path / "eval_sample_ids.jsonl"
    _write_jsonl(eval_ids, [{"id": "row-1"}])
    fingerprint = tmp_path / "fingerprint.txt"
    fingerprint.write_text("fingerprint\n", encoding="utf-8")
    questions = tmp_path / "questions.csv"
    questions.write_text("question\n", encoding="utf-8")
    config = AdapterSuiteConfig(
        adapter_dir=adapter_dir,
        run_id="run-a",
        output_root=tmp_path / "adapter_evals",
        medmcqa_eval_ids=eval_ids,
        proflingo_fingerprint=fingerprint,
        proflingo_questions=questions,
    )
    for audit in ("holistic_bias", "bold"):
        prompts = config.fairness_output_root / audit / config.fairness_subset_id / "normalized_prompts.jsonl"
        _write_jsonl(prompts, [{"text": "prompt", "axis": "axis", "bucket": "bucket", "descriptor": "desc"}])
    return config


def test_run_id_derivation_uses_parent_for_adapter_folder_and_sanitizes():
    assert derive_run_id(Path("outputs/medmcqa/run-1/adapter")) == "run-1"
    assert derive_run_id(Path("outputs/custom adapter")) == "custom_adapter"
    assert sanitize_run_id(" tuned/model v1 ") == "tuned_model_v1"
    with pytest.raises(ValueError, match="run id"):
        sanitize_run_id("...")


def test_cli_defaults_keep_adapter_as_only_required_swap_point(tmp_path: Path):
    adapter_dir = tmp_path / "run-a" / "adapter"
    args = build_arg_parser().parse_args(["--adapter-dir", str(adapter_dir)])
    config = config_from_args(args)

    assert config.adapter_dir == adapter_dir
    assert config.base_model_id == "allenai/OLMo-2-0425-1B-Instruct"
    assert config.medmcqa_eval_ids == tmp_path / "run-a" / "eval_sample_ids.jsonl"
    assert config.fairness_subset_id == "10k_seed0"
    assert config.run_id == "run-a"


@pytest.mark.parametrize(
    "removed_args",
    [
        ["--run-id", "custom"],
        ["--base-model-id", "model"],
        ["--output-root", "artifacts/custom"],
        ["--medmcqa-eval-ids", "ids.jsonl"],
        ["--medmcqa-dataset-id", "dataset"],
        ["--medmcqa-split", "test"],
        ["--fairness-subset-id", "subset"],
        ["--proflingo-fingerprint", "fingerprint.txt"],
        ["--proflingo-questions", "questions.csv"],
        ["--dtype", "fp16"],
        ["--device-map", "cpu"],
        ["--eval-batch-size", "4"],
        ["--fairness-batch-size", "4"],
        ["--classifier-batch-size", "4"],
        ["--max-new-tokens", "8"],
        ["--proflingo-limit", "4"],
        ["--skip-generation-eval"],
        ["--seed", "1"],
    ],
)
def test_cli_rejects_fixed_evaluation_knobs(tmp_path: Path, removed_args: list[str]):
    parser = build_arg_parser()
    adapter_dir = tmp_path / "run-a" / "adapter"

    with pytest.raises(SystemExit):
        parser.parse_args(["--adapter-dir", str(adapter_dir), *removed_args])


def test_load_medmcqa_eval_examples_matches_ids_in_order(tmp_path: Path):
    eval_ids = tmp_path / "eval_sample_ids.jsonl"
    _write_jsonl(eval_ids, [{"id": "row-2"}, {"id": "row-0"}])
    rows = [_medmcqa_row(0, cop=1), _medmcqa_row(1, cop=2), _medmcqa_row(2, cop=3)]

    examples = load_medmcqa_eval_examples(eval_ids, rows=rows)

    assert [example.example_id for example in examples] == ["row-2", "row-0"]
    assert [example.source_index for example in examples] == [2, 0]
    assert [example.answer for example in examples] == ["D", "B"]


def test_load_medmcqa_eval_examples_raises_for_missing_ids(tmp_path: Path):
    eval_ids = tmp_path / "eval_sample_ids.jsonl"
    _write_jsonl(eval_ids, [{"id": "missing"}])

    with pytest.raises(ValueError, match="missing"):
        load_medmcqa_eval_examples(eval_ids, rows=[_medmcqa_row(0)])


def test_run_adapter_suite_orchestrates_fixed_components(tmp_path: Path):
    config = _ready_config(tmp_path)
    calls: list[str] = []

    def fake_load(_config):
        calls.append("load")
        return "model", "tokenizer", "proflingo-tokenizer"

    def fake_proflingo(_config, model, tokenizer):
        assert (model, tokenizer) == ("model", "proflingo-tokenizer")
        calls.append("proflingo")
        return {"match_rate": 1.0, "matched": 2, "total": 2}

    def fake_medmcqa(_config, model, tokenizer):
        assert (model, tokenizer) == ("model", "tokenizer")
        calls.append("medmcqa")
        return {"forced_choice_accuracy": 0.75}

    def fake_generate(_config, model, tokenizer):
        assert (model, tokenizer) == ("model", "tokenizer")
        calls.append("generate")
        return {"holistic_bias": "hb", "bold": "bold"}

    def fake_score(_config):
        calls.append("score")
        return {
            "holistic_bias": {"full_gen_bias_mean_emotion": 12.5},
            "bold": {"bold_variance_stddev_metric": 13.125},
        }

    def fake_cleanup(model):
        assert model == "model"
        calls.append("cleanup")

    summary = run_adapter_suite(
        config,
        runners=AdapterSuiteRunners(
            load_adapter_model=fake_load,
            run_proflingo=fake_proflingo,
            run_medmcqa=fake_medmcqa,
            generate_fairness=fake_generate,
            score_fairness=fake_score,
            cleanup_model=fake_cleanup,
        ),
    )

    assert calls == ["load", "proflingo", "medmcqa", "generate", "cleanup", "score"]
    assert summary["proflingo"]["match_rate"] == 1.0
    assert summary["medmcqa"]["forced_choice_accuracy"] == 0.75
    assert (config.output_dir / "summary.json").exists()
    assert json.loads((config.output_dir / "summary.json").read_text(encoding="utf-8"))["adapter"]["run_id"] == "run-a"


def test_score_fairness_metrics_extracts_headline_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = _ready_config(tmp_path)

    class FakeFactory:
        @classmethod
        def from_config(cls, _config):
            return object()

    def fake_score_audit(audit, scoring_config, metric):
        metric_dir = (
            scoring_config.output_root
            / audit
            / scoring_config.subset_id
            / scoring_config.model_id
            / "metrics"
            / scoring_config.metric
        )
        if audit == "holistic_bias":
            _write_json(metric_dir / "metadata.json", {"full_gen_bias_mean_emotion": 160.0})
        else:
            _write_json(
                metric_dir / "metadata.json",
                {
                    "bold_variance_stddev_metric": 13.125,
                    "overall_mean_sentiment": 0.425,
                    "overall_mean_toxicity": 0.42,
                },
            )
        return metric_dir

    monkeypatch.setitem(adapter_suite.METRIC_FACTORIES, "full_gen_bias", FakeFactory)
    monkeypatch.setitem(adapter_suite.METRIC_FACTORIES, "bold_variance_stddev_metric", FakeFactory)
    monkeypatch.setattr(adapter_suite, "score_audit", fake_score_audit)

    results = score_fairness_metrics(config)

    assert results["holistic_bias"]["full_gen_bias_mean_emotion"] == 160.0
    assert results["bold"]["bold_variance_stddev_metric"] == pytest.approx(13.125)
    assert results["bold"]["overall_mean_sentiment"] == pytest.approx(0.425)
    assert results["bold"]["overall_mean_toxicity"] == pytest.approx(0.42)
