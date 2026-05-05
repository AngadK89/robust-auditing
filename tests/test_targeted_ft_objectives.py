import pytest

from robust_auditing.targeted_ft.adapters import HolisticBiasTargetedAdapter, RLVRMathRewardAdapter
from robust_auditing.targeted_ft.config import DatasetSamplingConfig, TargetedFTConfig
from robust_auditing.targeted_ft.mixture import build_targeted_ft_mixture
from robust_auditing.targeted_ft.objectives import (
    OBJECTIVE_PLUGINS,
    ObjectiveNotImplementedError,
    UnknownObjectiveError,
    get_objective_plugin,
)


def test_holistic_bias_uses_nll_anchor_objective_plugin():
    rows = list(
        HolisticBiasTargetedAdapter().normalize(
            [
                {
                    "text": "A person is a nurse.",
                    "axis": "gender_and_sex",
                    "bucket": "gender",
                    "descriptor": "woman",
                }
            ]
        )
    )

    assert rows[0]["objective"] == "holistic_bias_anchor"
    assert rows[0]["objective_plugin"] == "nll_anchor"
    assert "reward_model" not in rows[0]


def test_objective_registry_accepts_placeholders_and_nll_anchor_computes_mean_loss():
    assert set(OBJECTIVE_PLUGINS) == {
        "nll_anchor",
        "toxicity_minimize",
        "toxicity_anchor",
        "score_parity_anchor",
        "score_parity_improve",
        "generated_fairness_dpo",
    }

    nll_anchor = get_objective_plugin("nll_anchor")

    assert nll_anchor.loss([{"nll": 1.0}, {"nll": 3.0}]) == 2.0

    with pytest.raises(ObjectiveNotImplementedError, match="toxicity_minimize"):
        get_objective_plugin("toxicity_minimize").loss([{"score": 1.0}])
    with pytest.raises(UnknownObjectiveError, match="unknown_objective"):
        get_objective_plugin("unknown_objective")


def test_holistic_bias_sampling_is_stratified_and_manifest_records_source_indices():
    sources = {
        "holistic_bias": [
            {"text": "race a 0", "axis": "race", "bucket": "a", "descriptor": "x"},
            {"text": "race a 1", "axis": "race", "bucket": "a", "descriptor": "x"},
            {"text": "race b 0", "axis": "race", "bucket": "b", "descriptor": "y"},
            {"text": "race b 1", "axis": "race", "bucket": "b", "descriptor": "y"},
            {"text": "gender a 0", "axis": "gender", "bucket": "a", "descriptor": "z"},
            {"text": "gender a 1", "axis": "gender", "bucket": "a", "descriptor": "z"},
        ]
    }
    config = TargetedFTConfig(
        seed=17,
        datasets={"holistic_bias": DatasetSamplingConfig(max_samples=3)},
    )

    first = build_targeted_ft_mixture(sources, config)
    second = build_targeted_ft_mixture(sources, config)

    assert first.manifest == second.manifest
    selected_indices = first.manifest["datasets"]["holistic_bias"]["selected_source_indices"]
    assert len(selected_indices) == 3
    assert {sources["holistic_bias"][index]["axis"] for index in selected_indices} == {"race", "gender"}
    assert {
        (
            sources["holistic_bias"][index]["axis"],
            sources["holistic_bias"][index]["bucket"],
            sources["holistic_bias"][index]["descriptor"],
        )
        for index in selected_indices
    } == {
        ("race", "a", "x"),
        ("race", "b", "y"),
        ("gender", "a", "z"),
    }


def test_rlvr_rows_preserve_verifier_candidates_when_present():
    rows = list(
        RLVRMathRewardAdapter().normalize(
            [
                {
                    "messages": [{"role": "user", "content": "2+2?"}],
                    "ground_truth": "4",
                    "dataset": "math",
                    "constraint_type": "answer",
                    "constraint": "integer",
                    "verifier_candidates": ["4", "04"],
                }
            ]
        )
    )

    assert rows[0]["verifier_candidates"] == ["4", "04"]
