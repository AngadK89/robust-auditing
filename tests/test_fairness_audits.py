import math
from pathlib import Path

import pandas as pd

from robust_auditing.fairness import (
    AuditConfig,
    BoldAdapter,
    HolisticBiasAdapter,
    LikelihoodBiasMetric,
    MetricResult,
    axis_likelihood_bias,
    build_arg_parser,
    default_output_dir,
    group_summary,
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
        nll=1.25,
        token_count=4,
        perplexity=math.exp(1.25),
        metadata={"source_id": 7},
    )

    row = result.to_json_record()

    assert row == {
        "text": "A person arrived.",
        "axis": "gender_and_sex",
        "bucket": "gender",
        "descriptor": "woman",
        "nll": 1.25,
        "token_count": 4,
        "perplexity": math.exp(1.25),
        "source_id": 7,
    }


def test_likelihood_bias_metric_exposes_name():
    metric = LikelihoodBiasMetric(model=None, tokenizer=None)

    assert metric.name == "likelihood_bias"


def test_group_summary_aggregates_axis_bucket_scores():
    scores = pd.DataFrame(
        [
            {"axis": "race", "bucket": "a", "nll": 1.0, "perplexity": 2.0},
            {"axis": "race", "bucket": "a", "nll": 3.0, "perplexity": 6.0},
            {"axis": "race", "bucket": "b", "nll": 2.0, "perplexity": 8.0},
        ]
    )

    summary = group_summary(scores, group_by=["axis", "bucket"])

    race_a = summary[(summary["axis"] == "race") & (summary["bucket"] == "a")].iloc[0]
    assert race_a["count"] == 2
    assert race_a["mean_nll"] == 2.0
    assert race_a["std_nll"] == math.sqrt(2.0)
    assert race_a["mean_perplexity"] == 4.0
    assert race_a["std_perplexity"] == math.sqrt(8.0)


def test_axis_likelihood_bias_uses_pairwise_mann_whitney_effects():
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

    result = axis_likelihood_bias(scores, min_samples_per_descriptor=2)

    race = result[result["axis"] == "race"].iloc[0]
    assert race["descriptor_count"] == 3
    assert race["pairwise_comparison_count"] == 3
    assert race["mean_pairwise_auc_distance"] == 1.0
    assert race["max_pairwise_auc_distance"] == 1.0
    assert "gender" not in set(result["axis"])


def test_cli_defaults_selection_and_output_dirs():
    parser = build_arg_parser()
    args = parser.parse_args([])
    config = AuditConfig.from_args(args)

    assert config.audits == ("holistic_bias", "bold")
    assert config.model_id == "allenai/OLMo-2-0425-1B-Instruct"
    assert config.group_by == ("axis", "bucket")
    assert config.metric == "likelihood_bias"
    assert default_output_dir("holistic_bias", config.model_id) == Path(
        "artifacts/fairness/holistic_bias/olmo2_1b_instruct"
    )
    assert default_output_dir("bold", config.model_id) == Path(
        "artifacts/fairness/bold/olmo2_1b_instruct"
    )


def test_cli_accepts_single_audit_selection_and_output_root():
    parser = build_arg_parser()
    args = parser.parse_args(["--audits", "bold", "--output-root", "out"])
    config = AuditConfig.from_args(args)

    assert config.audits == ("bold",)
    assert config.output_root == Path("out")
    assert config.output_dir_for("bold") == Path("out/bold/olmo2_1b_instruct")
