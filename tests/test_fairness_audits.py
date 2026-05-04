import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from robust_auditing.fairness import (
    BoldAdapter,
    FairnessMetric,
    GenerationConfig,
    HolisticBiasAdapter,
    LikelihoodBiasMetric,
    MetricContext,
    MetricResult,
    ScoringConfig,
    build_generation_arg_parser,
    build_scoring_arg_parser,
    default_output_dir,
    generate_responses_for_audit,
    metric_folder_name,
    read_jsonl,
    records_to_frame,
    score_audit,
    write_jsonl,
    write_normalized_prompts,
)


def test_holistic_bias_adapter_normalizes_required_fields():
    frame = pd.DataFrame(
        [
            {
                "text": "A person arrived.",
                "axis": "gender_and_sex",
                "bucket": "gender",
                "descriptor": "woman",
            }
        ]
    )

    examples = list(HolisticBiasAdapter().normalize(frame))

    assert examples[0].to_score_metadata() == {
        "text": "A person arrived.",
        "axis": "gender_and_sex",
        "bucket": "gender",
        "descriptor": "woman",
        "metadata": {"source_index": 0},
    }


def test_bold_adapter_explodes_prompts_and_preserves_source_metadata():
    frame = pd.DataFrame(
        [
            {
                "domain": "race",
                "category": "African Americans",
                "name": "Example Person",
                "prompts": ["Prompt one", "Prompt two"],
                "wikipedia": ["Reference sentence"],
            }
        ]
    )

    examples = list(BoldAdapter().normalize(frame))

    assert [example.text for example in examples] == ["Prompt one", "Prompt two"]
    assert examples[1].to_score_metadata() == {
        "text": "Prompt two",
        "axis": "race",
        "bucket": "African Americans",
        "descriptor": "African Americans",
        "metadata": {
            "source_index": 0,
            "name": "Example Person",
            "prompt_index": 1,
        },
    }


def test_adapters_report_missing_required_columns():
    frame = pd.DataFrame([{"text": "hello", "axis": "race"}])

    try:
        HolisticBiasAdapter().validate_columns(frame)
    except ValueError as exc:
        holistic_message = str(exc)
    else:
        raise AssertionError("expected HolisticBias schema validation to fail")

    assert "HolisticBias" in holistic_message
    assert "bucket" in holistic_message
    assert "descriptor" in holistic_message

    try:
        BoldAdapter().validate_columns(pd.DataFrame([{"domain": "race"}]))
    except ValueError as exc:
        bold_message = str(exc)
    else:
        raise AssertionError("expected BOLD schema validation to fail")

    assert "BOLD" in bold_message
    assert "category" in bold_message
    assert "name" in bold_message
    assert "prompts" in bold_message


def test_metric_result_schema_serializes_expected_fields():
    result = MetricResult(
        text="A person arrived.",
        axis="gender_and_sex",
        bucket="gender",
        descriptor="woman",
        metric_name="example_classifier",
        scores={"probability": 0.75, "label": "biased"},
        metadata={"source_id": 7},
    )

    row = result.to_json_record()

    assert row == {
        "text": "A person arrived.",
        "axis": "gender_and_sex",
        "bucket": "gender",
        "descriptor": "woman",
        "metric_name": "example_classifier",
        "scores": {"probability": 0.75, "label": "biased"},
        "metadata": {"source_id": 7},
    }


def test_records_to_frame_flattens_arbitrary_metric_scores_for_summaries():
    frame = records_to_frame(
        [
            MetricResult(
                text="A",
                axis="race",
                bucket="a",
                descriptor="x",
                metric_name="toxicity_classifier",
                scores={"toxicity_probability": 0.2, "label": "low"},
            )
        ]
    )

    assert frame.iloc[0]["toxicity_probability"] == 0.2
    assert frame.iloc[0]["label"] == "low"
    assert frame.iloc[0]["metric_name"] == "toxicity_classifier"


def test_likelihood_bias_metric_exposes_name():
    metric = LikelihoodBiasMetric(model=None, tokenizer=None)

    assert metric.name == "likelihood_bias"
    assert metric.requires_lm is True


def test_likelihood_group_summary_aggregates_axis_bucket_scores():
    metric = LikelihoodBiasMetric(model=None, tokenizer=None)
    scores = pd.DataFrame(
        [
            {"axis": "race", "bucket": "a", "nll": 1.0, "perplexity": 2.0},
            {"axis": "race", "bucket": "a", "nll": 3.0, "perplexity": 6.0},
            {"axis": "race", "bucket": "b", "nll": 2.0, "perplexity": 8.0},
        ]
    )

    summary = metric.group_summary(scores, group_by=["axis", "bucket"])

    race_a = summary[(summary["axis"] == "race") & (summary["bucket"] == "a")].iloc[0]
    assert race_a["count"] == 2
    assert race_a["mean_nll"] == 2.0
    assert race_a["std_nll"] == math.sqrt(2.0)
    assert race_a["mean_perplexity"] == 4.0
    assert race_a["std_perplexity"] == math.sqrt(8.0)


def test_likelihood_axis_summary_uses_pairwise_mann_whitney_effects():
    metric = LikelihoodBiasMetric(model=None, tokenizer=None)
    scores = pd.DataFrame(
        [
            {"axis": "race", "descriptor": "a", "nll": 1.0},
            {"axis": "race", "descriptor": "a", "nll": 2.0},
            {"axis": "race", "descriptor": "b", "nll": 4.0},
            {"axis": "race", "descriptor": "b", "nll": 5.0},
            {"axis": "race", "descriptor": "c", "nll": 9.0},
            {"axis": "race", "descriptor": "c", "nll": 10.0},
            {"axis": "gender", "descriptor": "x", "nll": 1.0},
        ]
    )

    result = metric.axis_summary(scores, min_samples_per_descriptor=2)

    race = result[result["axis"] == "race"].iloc[0]
    assert race["descriptor_count"] == 3
    assert race["pairwise_comparison_count"] == 3
    assert race["mean_pairwise_auc_distance"] == 1.0
    assert race["max_pairwise_auc_distance"] == 1.0
    assert "gender" not in set(result["axis"])


def test_base_metric_summaries_handle_generic_numeric_scores():
    metric = FairnessMetric(name="classifier_probe")
    scores = pd.DataFrame(
        [
            {"axis": "race", "bucket": "a", "toxicity_probability": 0.2, "label": "low"},
            {"axis": "race", "bucket": "a", "toxicity_probability": 0.6, "label": "high"},
            {"axis": "race", "bucket": "b", "toxicity_probability": 0.8, "label": "high"},
        ]
    )

    summary = metric.group_summary(scores, group_by=["axis", "bucket"])

    race_a = summary[(summary["axis"] == "race") & (summary["bucket"] == "a")].iloc[0]
    assert race_a["count"] == 2
    assert race_a["mean_toxicity_probability"] == 0.4
    assert "mean_label" not in summary.columns
    assert metric.axis_summary(scores).empty


class ConstantMetric(FairnessMetric):
    name = "constant_metric"
    requires_lm = False

    def score(self, context):
        examples = context.load_examples()
        return [
            MetricResult(
                text=example.text,
                axis=example.axis,
                bucket=example.bucket,
                descriptor=example.descriptor,
                metric_name=self.name,
                scores={"score": 0.5},
                metadata=dict(example.metadata),
            )
            for example in examples
        ]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, texts, return_tensors, padding, truncation):
        del return_tensors, padding, truncation
        input_ids = []
        attention_mask = []
        for text in texts:
            ids = [1, 2] if len(text) < 12 else [1, 2, 3]
            input_ids.append(ids)
        max_len = max(len(ids) for ids in input_ids)
        for index, ids in enumerate(input_ids):
            pad_count = max_len - len(ids)
            input_ids[index] = ids + [self.pad_token_id] * pad_count
            attention_mask.append([1] * len(ids) + [0] * pad_count)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return "Stored response."


class FakeGenerateModel:
    def eval(self):
        return None

    def parameters(self):
        return iter([torch.nn.Parameter(torch.tensor([0.0]))])

    def generate(self, **encoded):
        input_ids = encoded["input_ids"]
        suffix = torch.tensor([[7, 8] for _ in range(input_ids.shape[0])], device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


def test_write_normalized_prompts_materializes_adapter_fields(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "text": "A person arrived.",
                "axis": "gender_and_sex",
                "bucket": "gender",
                "descriptor": "woman",
            }
        ]
    )
    config = GenerationConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        prompts_only=True,
    )

    paths, examples, _ = write_normalized_prompts(
        "holistic_bias",
        config,
        dataset=frame,
    )

    assert len(examples) == 1
    assert read_jsonl(paths.normalized_prompts) == [
        {
            "text": "A person arrived.",
            "axis": "gender_and_sex",
            "bucket": "gender",
            "descriptor": "woman",
            "metadata": {"source_index": 0},
        }
    ]


def test_generate_responses_writes_response_artifact_with_generation_metadata(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "text": "A person arrived.",
                "axis": "gender_and_sex",
                "bucket": "gender",
                "descriptor": "woman",
            }
        ]
    )
    config = GenerationConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        model_id="example/model",
        min_new_tokens=1,
        max_new_tokens=2,
    )

    output_dir = generate_responses_for_audit(
        "holistic_bias",
        config,
        model=FakeGenerateModel(),
        tokenizer=FakeTokenizer(),
        dataset=frame,
    )

    rows = read_jsonl(output_dir / "model_responses.jsonl")
    assert rows[0]["generated_response"] == "Stored response."
    assert rows[0]["text"] == "A person arrived."
    assert rows[0]["generation"]["decoding"] == "beam_search"
    assert rows[0]["generation"]["num_beams"] == 3
    assert rows[0]["generation"]["generated_token_count"] == 2


def test_metric_output_directory_is_derived_from_class_name():
    assert metric_folder_name(LikelihoodBiasMetric) == "likelihood_bias"
    assert metric_folder_name(ConstantMetric()) == "constant"


def test_score_audit_writes_metric_outputs_under_metric_folder(tmp_path):
    config = ScoringConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        metric="constant_metric",
    )
    paths = config.paths_for("holistic_bias")
    write_jsonl(
        paths.normalized_prompts,
        [
            {
                "text": "A person arrived.",
                "axis": "gender_and_sex",
                "bucket": "gender",
                "descriptor": "woman",
                "metadata": {"source_index": 0},
            }
        ],
    )

    metric_dir = score_audit("holistic_bias", config, metric=ConstantMetric())

    assert metric_dir == paths.audit_dir / "metrics" / "constant"
    per_example = (metric_dir / "per_example.jsonl").read_text()
    assert '"metric_name": "constant_metric"' in per_example
    assert '"score": 0.5' in per_example
    assert (metric_dir / "group_summary.csv").exists()
    assert (metric_dir / "axis_summary.csv").exists()
    assert (metric_dir / "metadata.json").exists()


def test_likelihood_bias_metric_declares_prompt_only_artifact():
    assert LikelihoodBiasMetric.required_artifacts == ("normalized_prompts",)


def test_score_audit_reports_missing_required_artifacts(tmp_path):
    config = ScoringConfig(audits=("holistic_bias",), output_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="normalized_prompts"):
        score_audit("holistic_bias", config, metric=ConstantMetric())


def test_cli_defaults_selection_and_output_dirs():
    parser = build_generation_arg_parser()
    args = parser.parse_args([])
    generation_config = GenerationConfig.from_args(args)

    assert generation_config.audits == ("holistic_bias", "bold")
    assert generation_config.model_id == "allenai/OLMo-2-0425-1B-Instruct"
    assert generation_config.num_beams == 3
    assert generation_config.min_new_tokens == 20
    assert generation_config.max_new_tokens == 64

    scoring_parser = build_scoring_arg_parser()
    scoring_args = scoring_parser.parse_args([])
    scoring_config = ScoringConfig.from_args(scoring_args)

    assert scoring_config.group_by == ("axis", "bucket")
    assert scoring_config.metric == "likelihood_bias"
    assert default_output_dir("holistic_bias", scoring_config.model_id) == Path(
        "artifacts/fairness/holistic_bias/olmo2_1b_instruct"
    )
    assert default_output_dir("bold", scoring_config.model_id) == Path(
        "artifacts/fairness/bold/olmo2_1b_instruct"
    )


def test_cli_accepts_single_audit_selection_and_output_root():
    parser = build_generation_arg_parser()
    args = parser.parse_args(["--audits", "bold", "--output-root", "out"])
    config = GenerationConfig.from_args(args)

    assert config.audits == ("bold",)
    assert config.output_root == Path("out")
    assert config.paths_for("bold").audit_dir == Path("out/bold/olmo2_1b_instruct")
