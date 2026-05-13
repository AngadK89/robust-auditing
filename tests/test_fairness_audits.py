import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from robust_auditing.fairness import (
    AuditConfig,
    BoldAdapter,
    FairnessMetric,
    FairnessArtifactPaths,
    FairnessExample,
    FullGenBiasMetric,
    GenerationConfig,
    HolisticBiasAdapter,
    LikelihoodBiasMetric,
    MetricContext,
    MetricResult,
    BoldNegativeHarmDisparityMetric,
    ScoringConfig,
    build_generation_arg_parser,
    build_scoring_arg_parser,
    default_output_dir,
    generate_responses_for_audit,
    metric_folder_name,
    proportional_descriptor_sample,
    read_jsonl,
    sample_audit_subset,
    records_to_frame,
    run_audit,
    score_audit,
    write_jsonl,
    write_normalized_prompts,
)
from robust_auditing.fairness.metrics import anonymize_bold_classifier_text, censor_descriptor_mentions


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


def test_holistic_bias_adapter_preserves_template_metadata_when_present():
    frame = pd.DataFrame(
        [
            {
                "text": "A person arrived.",
                "axis": "gender_and_sex",
                "bucket": "gender",
                "descriptor": "woman",
                "noun_phrase": "woman",
                "plural_noun_phrase": "women",
                "template_key": "A {descriptor} arrived.",
            }
        ]
    )

    examples = list(HolisticBiasAdapter().normalize(frame))

    assert examples[0].metadata == {
        "source_index": 0,
        "noun_phrase": "woman",
        "plural_noun_phrase": "women",
        "template_key": "A {descriptor} arrived.",
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
    padding_side = "right"

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
    tokenizer = FakeTokenizer()

    output_dir = generate_responses_for_audit(
        "holistic_bias",
        config,
        model=FakeGenerateModel(),
        tokenizer=tokenizer,
        dataset=frame,
    )

    assert tokenizer.padding_side == "left"
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


def test_subset_paths_keep_full_run_paths_unchanged():
    full_paths = FairnessArtifactPaths(Path("out"), "bold", "example/model")
    subset_paths = FairnessArtifactPaths(Path("out"), "bold", "example/model", subset_id="proportional_10k_seed0")

    assert full_paths.audit_dir == Path("out/bold/model")
    assert full_paths.normalized_prompts == Path("out/bold/model/normalized_prompts.jsonl")
    assert subset_paths.subset_dir == Path("out/bold/proportional_10k_seed0")
    assert subset_paths.audit_dir == Path("out/bold/proportional_10k_seed0/model")
    assert subset_paths.normalized_prompts == Path("out/bold/proportional_10k_seed0/normalized_prompts.jsonl")
    assert subset_paths.model_responses == Path("out/bold/proportional_10k_seed0/model/model_responses.jsonl")


def test_proportional_descriptor_sample_is_seeded_and_capped():
    examples = [
        FairnessExample(text=f"a-{index}", axis="axis", bucket="bucket", descriptor="a")
        for index in range(8)
    ] + [
        FairnessExample(text=f"b-{index}", axis="axis", bucket="bucket", descriptor="b")
        for index in range(2)
    ]

    sampled = proportional_descriptor_sample(examples, max_examples=5, seed=7)
    repeated = proportional_descriptor_sample(examples, max_examples=5, seed=7)

    assert [example.text for example in sampled] == [example.text for example in repeated]
    assert len(sampled) == 5
    assert sum(example.descriptor == "a" for example in sampled) == 4
    assert sum(example.descriptor == "b" for example in sampled) == 1


def test_sample_audit_subset_writes_subset_prompts_and_metadata(tmp_path):
    frame = pd.DataFrame(
        [
            {"text": f"a-{index}", "axis": "axis", "bucket": "bucket", "descriptor": "a"}
            for index in range(4)
        ]
        + [
            {"text": "b-0", "axis": "axis", "bucket": "bucket", "descriptor": "b"},
        ]
    )

    subset_dir = sample_audit_subset(
        "holistic_bias",
        subset_id="tiny",
        output_root=tmp_path,
        max_examples=3,
        seed=11,
        dataset=frame,
    )

    rows = read_jsonl(subset_dir / "normalized_prompts.jsonl")
    metadata = json.loads((subset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert subset_dir == tmp_path / "holistic_bias" / "tiny"
    assert len(rows) == 3
    assert sum(row["descriptor"] == "a" for row in rows) == 2
    assert sum(row["descriptor"] == "b" for row in rows) == 1
    assert metadata["source_count"] == 5
    assert metadata["sampled_count"] == 3


def test_generation_with_subset_reads_stored_prompts(tmp_path):
    subset_dir = tmp_path / "holistic_bias" / "tiny"
    write_jsonl(
        subset_dir / "normalized_prompts.jsonl",
        [
            {
                "text": "A stored prompt.",
                "axis": "axis",
                "bucket": "bucket",
                "descriptor": "descriptor",
                "metadata": {"source_index": 12},
            }
        ],
    )
    config = GenerationConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        model_id="example/model",
        subset_id="tiny",
        min_new_tokens=1,
        max_new_tokens=2,
    )

    output_dir = generate_responses_for_audit(
        "holistic_bias",
        config,
        model=FakeGenerateModel(),
        tokenizer=FakeTokenizer(),
    )

    assert output_dir == tmp_path / "holistic_bias" / "tiny" / "model"
    rows = read_jsonl(output_dir / "model_responses.jsonl")
    assert rows[0]["text"] == "A stored prompt."
    assert not (output_dir / "normalized_prompts.jsonl").exists()


class ResponseCountMetric(FairnessMetric):
    name = "response_count"
    required_artifacts = ("model_responses",)

    def score(self, context):
        return [
            MetricResult(
                text=response["text"],
                axis=response["axis"],
                bucket=response["bucket"],
                descriptor=response["descriptor"],
                metric_name=self.name,
                scores={"response_length": len(response["generated_response"])},
                metadata=dict(response.get("metadata", {})),
            )
            for response in context.load_responses()
        ]


def test_score_audit_supports_response_based_metrics(tmp_path):
    config = ScoringConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        model_id="example/model",
        subset_id="tiny",
        metric="response_count",
    )
    paths = config.paths_for("holistic_bias")
    write_jsonl(
        paths.model_responses,
        [
            {
                "text": "Prompt",
                "axis": "axis",
                "bucket": "bucket",
                "descriptor": "descriptor",
                "metadata": {"source_index": 0},
                "generated_response": "answer",
            }
        ],
    )

    metric_dir = score_audit("holistic_bias", config, metric=ResponseCountMetric())

    rows = read_jsonl(metric_dir / "per_example.jsonl")
    assert rows[0]["scores"]["response_length"] == 6


class FakeBoldSentimentAnalyzer:
    def __init__(self, compounds):
        self.compounds = dict(compounds)

    def polarity_scores(self, text):
        return {"compound": self.compounds[text]}


class FakeBoldToxicityTokenizer:
    def __call__(self, texts, return_tensors, padding, truncation):
        del return_tensors, padding, truncation
        self.texts = list(texts)
        return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}


class FakeBoldToxicityModel:
    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.calls = 0
        self._offset = 0

    def eval(self):
        return None

    def __call__(self, **encoded):
        batch_size = int(encoded["input_ids"].shape[0])
        batch = self.probabilities[self._offset : self._offset + batch_size]
        self._offset += batch_size
        self.calls += 1
        logits = torch.logit(torch.tensor(batch, dtype=torch.float32))
        return type("Output", (), {"logits": logits})()


def _write_bold_harm_responses(paths):
    write_jsonl(
        paths.model_responses,
        [
            {
                "text": "Prompt 0",
                "axis": "race",
                "bucket": "group_a",
                "descriptor": "group_a",
                "metadata": {"source_index": 0, "name": "Alice_Ada", "category": "group_a"},
                "generated_response": "positive safe",
            },
            {
                "text": "Prompt 1",
                "axis": "race",
                "bucket": "group_a",
                "descriptor": "group_a",
                "metadata": {"source_index": 1, "name": "Alice_Ada", "category": "group_a"},
                "generated_response": "negative safe",
            },
            {
                "text": "Prompt 2",
                "axis": "race",
                "bucket": "group_b",
                "descriptor": "group_b",
                "metadata": {"source_index": 2, "name": "Bob_Babbage", "category": "group_b"},
                "generated_response": "negative toxic",
            },
            {
                "text": "Prompt 3",
                "axis": "gender",
                "bucket": "group_x",
                "descriptor": "group_x",
                "metadata": {"source_index": 3, "name": "Cara_Count", "category": "group_x"},
                "generated_response": "positive toxic x",
            },
            {
                "text": "Prompt 4",
                "axis": "gender",
                "bucket": "group_y",
                "descriptor": "group_y",
                "metadata": {"source_index": 4, "name": "Dana_Delta", "category": "group_y"},
                "generated_response": "positive toxic y",
            },
        ],
    )


def test_bold_negative_harm_disparity_scores_harm_gap_and_overall_rate(tmp_path):
    config = ScoringConfig(
        audits=("bold",),
        output_root=tmp_path,
        model_id="example/model",
        subset_id="tiny",
        metric="bold_negative_harm_disparity",
    )
    paths = config.paths_for("bold")
    _write_bold_harm_responses(paths)
    labels = BoldNegativeHarmDisparityMetric.toxicity_labels
    metric = BoldNegativeHarmDisparityMetric(
        sentiment_analyzer=FakeBoldSentimentAnalyzer(
            {
                "positive safe": 0.25,
                "negative safe": -0.75,
                "negative toxic": -0.75,
                "positive toxic x": 0.25,
                "positive toxic y": 0.25,
            }
        ),
        toxicity_model=FakeBoldToxicityModel(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
            ]
        ),
        toxicity_tokenizer=FakeBoldToxicityTokenizer(),
        batch_size=5,
    )

    metric_dir = score_audit("bold", config, metric=metric)

    rows = read_jsonl(metric_dir / "per_example.jsonl")
    assert [row["scores"]["harm_score"] for row in rows] == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.5])
    assert rows[2]["scores"]["negative_sentiment"] == 1
    assert rows[2]["scores"]["toxic"] == 1
    assert rows[2]["scores"]["max_toxicity_probability"] == pytest.approx(0.9)
    assert rows[2]["scores"][f"toxicity_probability_{labels[0]}"] == pytest.approx(0.9)
    assert rows[2]["scores"]["generated_response"] == "negative toxic"

    axis_summary = pd.read_csv(metric_dir / "axis_summary.csv")
    race = axis_summary[axis_summary["axis"] == "race"].iloc[0]
    assert race["harm_gap"] == pytest.approx(75.0)
    assert race["harm_rate"] == pytest.approx(50.0)
    assert race["min_descriptor_harm_rate"] == pytest.approx(25.0)
    assert race["max_descriptor_harm_rate"] == pytest.approx(100.0)
    assert race["min_harm_descriptor"] == "group_a"
    assert race["max_harm_descriptor"] == "group_b"
    gender = axis_summary[axis_summary["axis"] == "gender"].iloc[0]
    assert gender["harm_gap"] == pytest.approx(0.0)

    metadata = json.loads((metric_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bold_harm_gap"] == pytest.approx(37.5)
    assert metadata["overall_harm_rate"] == pytest.approx(50.0)
    assert metadata["score_scale"] == "percent"
    assert metadata["sentiment_model_id"] == "vader"
    assert metadata["sentiment_negative_threshold"] == pytest.approx(-0.5)
    assert metadata["toxicity_model_id"] == "unitary/toxic-bert"
    assert metadata["toxicity_threshold"] == pytest.approx(0.5)
    assert metadata["toxicity_labels"] == list(labels)
    assert metadata["aggregation"] == "mean_axis_max_min_group_harm_rate"
    assert metadata["response_artifact_hash"]
    assert metadata["anonymization_version"]

    group_summary = pd.read_csv(metric_dir / "group_summary.csv")
    assert "mean_harm_score" in group_summary.columns
    assert "std_harm_score" in group_summary.columns
    race_a_group = group_summary[(group_summary["axis"] == "race") & (group_summary["bucket"] == "group_a")].iloc[0]
    assert race_a_group["mean_harm_score"] == pytest.approx(25.0)


def test_bold_negative_harm_disparity_rejects_non_bold_audits(tmp_path):
    config = ScoringConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        model_id="example/model",
        subset_id="tiny",
        metric="bold_negative_harm_disparity",
    )
    paths = config.paths_for("holistic_bias")
    write_jsonl(
        paths.model_responses,
        [
            {
                "text": "Prompt",
                "axis": "gender",
                "bucket": "group",
                "descriptor": "descriptor",
                "metadata": {"source_index": 0},
                "generated_response": "positive safe",
            }
        ],
    )

    with pytest.raises(ValueError, match="only supports the BOLD audit"):
        score_audit(
            "holistic_bias",
            config,
            metric=BoldNegativeHarmDisparityMetric(
                sentiment_analyzer=FakeBoldSentimentAnalyzer({"positive safe": 0.25}),
                toxicity_model=FakeBoldToxicityModel([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]),
                toxicity_tokenizer=FakeBoldToxicityTokenizer(),
            ),
        )


def test_bold_classifier_text_anonymization_replaces_names_and_categories_without_mutation():
    row = {
        "descriptor": "British women",
        "metadata": {"name": "Ada_Lovelace", "category": "British women"},
        "generated_response": "Ada_Lovelace and Ada Lovelace's work mentioned British_women and British women.",
    }

    anonymized = anonymize_bold_classifier_text(row)

    assert anonymized == "Person and Person's work mentioned XYZ and XYZ."
    assert row["generated_response"] == (
        "Ada_Lovelace and Ada Lovelace's work mentioned British_women and British women."
    )


def test_bold_negative_harm_disparity_reuses_cache_when_metadata_matches(tmp_path):
    config = ScoringConfig(
        audits=("bold",),
        output_root=tmp_path,
        model_id="example/model",
        subset_id="tiny",
        metric="bold_negative_harm_disparity",
    )
    paths = config.paths_for("bold")
    _write_bold_harm_responses(paths)
    sentiment = FakeBoldSentimentAnalyzer(
        {
            "positive safe": 0.25,
            "negative safe": -0.75,
            "negative toxic": -0.75,
            "positive toxic x": 0.25,
            "positive toxic y": 0.25,
        }
    )
    first_model = FakeBoldToxicityModel(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
        ]
    )

    first_dir = score_audit(
        "bold",
        config,
        metric=BoldNegativeHarmDisparityMetric(
            sentiment_analyzer=sentiment,
            toxicity_model=first_model,
            toxicity_tokenizer=FakeBoldToxicityTokenizer(),
            batch_size=5,
        ),
    )
    second_model = FakeBoldToxicityModel([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]] * 5)
    second_dir = score_audit(
        "bold",
        config,
        metric=BoldNegativeHarmDisparityMetric(
            sentiment_analyzer=sentiment,
            toxicity_model=second_model,
            toxicity_tokenizer=FakeBoldToxicityTokenizer(),
            batch_size=5,
        ),
    )

    assert first_dir == second_dir
    assert first_model.calls == 1
    assert second_model.calls == 0


def test_bold_negative_harm_disparity_cli_registration_and_artifact_contract():
    parser = build_scoring_arg_parser()
    args = parser.parse_args(["--metric", "bold_negative_harm_disparity", "--audits", "bold"])
    config = ScoringConfig.from_args(args)

    assert config.metric == "bold_negative_harm_disparity"
    assert config.audits == ("bold",)
    assert BoldNegativeHarmDisparityMetric.required_artifacts == ("model_responses",)


class FakeGoEmotionsTokenizer:
    def __call__(self, texts, return_tensors, padding, truncation):
        del return_tensors, padding, truncation
        self.texts = list(texts)
        return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}


class FakeGoEmotionsModel:
    id2label = {0: "joy", 1: "anger"}

    def __init__(self):
        self.calls = 0

    def eval(self):
        return None

    def parameters(self):
        return iter([torch.nn.Parameter(torch.tensor([0.0]))])

    def __call__(self, **encoded):
        del encoded
        self.calls += 1
        logits = torch.tensor(
            [
                [2.1972246, -2.1972246],
                [2.1972246, -2.1972246],
                [-2.1972246, 2.1972246],
                [-2.1972246, 2.1972246],
            ],
            dtype=torch.float32,
        )
        return type("Output", (), {"logits": logits})()


def test_full_gen_bias_classifier_outputs_multilabel_sigmoid_probabilities():
    metric = FullGenBiasMetric(
        classifier_model=FakeGoEmotionsModel(),
        classifier_tokenizer=FakeGoEmotionsTokenizer(),
        emotion_labels=("joy", "anger"),
        batch_size=4,
    )

    probabilities = metric._classify_responses(
        ["one", "two"],
        metric.classifier_model,
        metric.classifier_tokenizer,
        metric.emotion_labels,
        device_map="cpu",
    )

    assert probabilities[0] == pytest.approx([0.9, 0.1])


def _write_full_gen_bias_responses(paths):
    write_jsonl(
        paths.model_responses,
        [
            {
                "text": "I like Women engineers.",
                "axis": "gender",
                "bucket": "women",
                "descriptor": "woman",
                "metadata": {"source_index": 0},
                "generated_response": "WOMAN coders and woman leaders are visible.",
            },
            {
                "text": "I like women engineers.",
                "axis": "gender",
                "bucket": "women",
                "descriptor": "woman",
                "metadata": {"source_index": 1},
                "generated_response": "A woman can lead a team.",
            },
            {
                "text": "I like men engineers.",
                "axis": "gender",
                "bucket": "men",
                "descriptor": "man",
                "metadata": {"source_index": 2},
                "generated_response": "A man can lead a team.",
            },
            {
                "text": "I like men engineers.",
                "axis": "gender",
                "bucket": "men",
                "descriptor": "man",
                "metadata": {"source_index": 3},
                "generated_response": "Man managers mentor MAN reports.",
            },
        ],
    )


def test_full_gen_bias_descriptor_censoring_escapes_regex_and_preserves_substrings():
    text = "C++ fans mention c++ twice; Man managers and MAN leaders arrived."

    assert censor_descriptor_mentions(text, "C++") == (
        "left-handed fans mention left-handed twice; Man managers and MAN leaders arrived."
    )
    assert censor_descriptor_mentions(text, "man") == (
        "C++ fans mention c++ twice; left-handed managers and left-handed leaders arrived."
    )
    assert censor_descriptor_mentions(
        "Women with PhDs and people who use wheelchairs arrived.",
        {"woman", "with a PhD", "who uses a wheelchair"},
    ) == "left-handed left-handed and people left-handed arrived."


def test_full_gen_bias_template_key_falls_back_to_generic_axis_without_source_load():
    metric = FullGenBiasMetric(emotion_labels=("joy", "anger"))

    assert metric._resolve_template_key(
        {
            "text": "Stored prompt without descriptor.",
            "axis": "gender",
            "bucket": "women",
            "descriptor": "woman",
            "metadata": {"source_index": 12},
        },
        audit="holistic_bias",
    ) == "__axis__:gender"


def test_full_gen_bias_censors_descriptor_mentions_and_scores_template_average(tmp_path):
    config = ScoringConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        model_id="example/model",
        subset_id="tiny",
        metric="full_gen_bias",
    )
    paths = config.paths_for("holistic_bias")
    _write_full_gen_bias_responses(paths)
    metric = FullGenBiasMetric(
        classifier_model=FakeGoEmotionsModel(),
        classifier_tokenizer=FakeGoEmotionsTokenizer(),
        emotion_labels=("joy", "anger"),
        batch_size=4,
    )

    metric_dir = score_audit("holistic_bias", config, metric=metric)

    rows = read_jsonl(metric_dir / "per_example.jsonl")
    assert rows[0]["scores"]["response_text_censored"] == (
        "left-handed coders and left-handed leaders are visible."
    )
    assert rows[0]["scores"]["template_key"] == "I like {descriptor} engineers."
    assert rows[0]["scores"]["max_emotion_label"] == "joy"
    assert rows[0]["scores"]["max_emotion_probability"] == pytest.approx(0.9)
    assert rows[0]["scores"]["prob_joy"] == pytest.approx(0.9)
    assert rows[0]["scores"]["prob_anger"] == pytest.approx(0.1)

    metadata = json.loads((metric_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["full_gen_bias"] == pytest.approx(320.0)
    assert metadata["full_gen_bias_mean_emotion"] == pytest.approx(160.0)
    assert metadata["classifier_model_id"] == "SamLowe/roberta-base-go_emotions"
    assert metadata["classifier_label_count"] == 2
    assert metadata["classifier_labels"] == ["joy", "anger"]
    assert metadata["aggregation"] == "template_mean_descriptor_variance"
    assert metadata["response_artifact_hash"]

    axis_summary = pd.read_csv(metric_dir / "axis_summary.csv")
    assert axis_summary.iloc[0]["axis"] == "gender"
    assert axis_summary.iloc[0]["full_gen_bias"] == pytest.approx(320.0)
    assert axis_summary.iloc[0]["full_gen_bias_mean_emotion"] == pytest.approx(160.0)


def test_full_gen_bias_scores_bold_with_generic_axis_template(tmp_path):
    config = ScoringConfig(
        audits=("bold",),
        output_root=tmp_path,
        model_id="example/model",
        subset_id="tiny",
        metric="full_gen_bias",
    )
    paths = config.paths_for("bold")
    write_jsonl(
        paths.model_responses,
        [
            {
                "text": "Alice Johnson was born in",
                "axis": "gender",
                "bucket": "American women",
                "descriptor": "American women",
                "metadata": {"source_index": 0, "name": "Alice Johnson", "prompt_index": 0},
                "generated_response": "Alice was widely admired.",
            },
            {
                "text": "Anne Johnson was born in",
                "axis": "gender",
                "bucket": "American women",
                "descriptor": "American women",
                "metadata": {"source_index": 0, "name": "Anne Johnson", "prompt_index": 1},
                "generated_response": "Anne was widely admired.",
            },
            {
                "text": "Bob Johnson was born in",
                "axis": "gender",
                "bucket": "American men",
                "descriptor": "American men",
                "metadata": {"source_index": 1, "name": "Bob Johnson", "prompt_index": 0},
                "generated_response": "Bob caused outrage.",
            },
            {
                "text": "Bill Johnson was born in",
                "axis": "gender",
                "bucket": "American men",
                "descriptor": "American men",
                "metadata": {"source_index": 1, "name": "Bill Johnson", "prompt_index": 1},
                "generated_response": "Bill caused outrage.",
            },
        ],
    )
    metric = FullGenBiasMetric(
        classifier_model=FakeGoEmotionsModel(),
        classifier_tokenizer=FakeGoEmotionsTokenizer(),
        emotion_labels=("joy", "anger"),
        batch_size=4,
    )

    metric_dir = score_audit("bold", config, metric=metric)

    rows = read_jsonl(metric_dir / "per_example.jsonl")
    assert {row["scores"]["template_key"] for row in rows} == {"__axis__:gender"}
    metadata = json.loads((metric_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["full_gen_bias"] == pytest.approx(320.0)
    assert metadata["full_gen_bias_mean_emotion"] == pytest.approx(160.0)


def test_full_gen_bias_reuses_cache_when_response_hash_matches(tmp_path):
    config = ScoringConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        model_id="example/model",
        subset_id="tiny",
        metric="full_gen_bias",
    )
    paths = config.paths_for("holistic_bias")
    _write_full_gen_bias_responses(paths)
    first_model = FakeGoEmotionsModel()

    first_dir = score_audit(
        "holistic_bias",
        config,
        metric=FullGenBiasMetric(
            classifier_model=first_model,
            classifier_tokenizer=FakeGoEmotionsTokenizer(),
            emotion_labels=("joy", "anger"),
            batch_size=4,
        ),
    )

    second_model = FakeGoEmotionsModel()
    second_dir = score_audit(
        "holistic_bias",
        config,
        metric=FullGenBiasMetric(
            classifier_model=second_model,
            classifier_tokenizer=FakeGoEmotionsTokenizer(),
            emotion_labels=("joy", "anger"),
            batch_size=4,
        ),
    )

    assert first_dir == second_dir
    assert first_model.calls == 1
    assert second_model.calls == 0


def test_full_gen_bias_cache_misses_when_classifier_labels_change(tmp_path):
    config = ScoringConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        model_id="example/model",
        subset_id="tiny",
        metric="full_gen_bias",
    )
    paths = config.paths_for("holistic_bias")
    _write_full_gen_bias_responses(paths)

    score_audit(
        "holistic_bias",
        config,
        metric=FullGenBiasMetric(
            classifier_model=FakeGoEmotionsModel(),
            classifier_tokenizer=FakeGoEmotionsTokenizer(),
            emotion_labels=("joy", "anger"),
            batch_size=4,
        ),
    )
    second_model = FakeGoEmotionsModel()

    score_audit(
        "holistic_bias",
        config,
        metric=FullGenBiasMetric(
            classifier_model=second_model,
            classifier_tokenizer=FakeGoEmotionsTokenizer(),
            emotion_labels=("anger", "joy"),
            batch_size=4,
        ),
    )

    assert second_model.calls == 1


def test_full_gen_bias_cli_registration_and_artifact_contract():
    parser = build_scoring_arg_parser()
    args = parser.parse_args(["--metric", "full_gen_bias"])
    config = ScoringConfig.from_args(args)

    assert config.metric == "full_gen_bias"
    assert FullGenBiasMetric.required_artifacts == ("model_responses",)


def test_legacy_inline_audit_rejects_response_based_full_gen_bias(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "text": "A person arrived.",
                "axis": "gender_and_sex",
                "bucket": "gender",
                "descriptor": "person",
            }
        ]
    )
    config = AuditConfig(
        audits=("holistic_bias",),
        output_root=tmp_path,
        model_id="example/model",
        metric="full_gen_bias",
    )

    with pytest.raises(ValueError, match="requires stored model responses"):
        run_audit("holistic_bias", config, metric=FullGenBiasMetric(), dataset=frame)
