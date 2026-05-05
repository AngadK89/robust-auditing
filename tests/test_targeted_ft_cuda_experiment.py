import pytest

from robust_auditing.targeted_ft.cuda_experiment import (
    PROTOTYPE_CONFIGS,
    normalize_weights,
    prototype_names,
    extract_math_answer,
    holistic_bias_example_loss,
    math_answer_matches,
    split_common_prompt,
)


def test_prototype_names_are_deterministic():
    assert prototype_names() == ["prototype-03", "prototype-04", "prototype-05", "prototype-06", "prototype-07", "prototype-08", "prototype-09", "prototype-10", "prototype-11", "prototype-12", "prototype-13", "prototype-14"]


def test_prototype_configs_have_expected_objective_keys_and_normalized_weights():
    expected = {"inverted_dpo", "dpo", "sft", "holistic_bias_anchor", "rl_reward"}
    for config in PROTOTYPE_CONFIGS.values():
        weights = normalize_weights(config.objective_weights)
        assert set(weights) == expected
        assert sum(weights.values()) == pytest.approx(1.0)


def test_normalize_weights_rejects_missing_objective():
    with pytest.raises(ValueError, match="missing"):
        normalize_weights({"inverted_dpo": 1.0})


def test_split_common_prompt_extracts_shared_dialog_prefix():
    chosen = "Human: hi\nAssistant: unsafe answer"
    rejected = "Human: hi\nAssistant: safe answer"

    prompt, chosen_completion, rejected_completion = split_common_prompt(chosen, rejected)

    assert prompt == "Human: hi\nAssistant: "
    assert chosen_completion == "unsafe answer"
    assert rejected_completion == "safe answer"


def test_split_common_prompt_uses_explicit_prompt():
    prompt, chosen_completion, rejected_completion = split_common_prompt("chosen", "rejected", prompt="question")

    assert prompt == "question"
    assert chosen_completion == "chosen"
    assert rejected_completion == "rejected"


def test_cooldown_config_has_valid_weights():
    config = PROTOTYPE_CONFIGS["prototype-07"]

    assert config.cooldown_start_step == 400
    assert config.cooldown_weights is not None
    assert sum(normalize_weights(config.cooldown_weights).values()) == pytest.approx(1.0)
    assert normalize_weights(config.cooldown_weights)["holistic_bias_anchor"] > normalize_weights(config.objective_weights)["holistic_bias_anchor"]


def test_stronger_cooldown_config_starts_earlier_and_anchors_more():
    previous = PROTOTYPE_CONFIGS["prototype-07"]
    current = PROTOTYPE_CONFIGS["prototype-08"]

    assert current.cooldown_start_step < previous.cooldown_start_step
    assert current.cooldown_weights is not None
    assert previous.cooldown_weights is not None
    assert normalize_weights(current.cooldown_weights)["holistic_bias_anchor"] > normalize_weights(previous.cooldown_weights)["holistic_bias_anchor"]
    assert normalize_weights(current.cooldown_weights)["inverted_dpo"] < normalize_weights(previous.cooldown_weights)["inverted_dpo"]


def test_sft_recovery_cooldown_keeps_strong_anchor_but_more_sft():
    previous = PROTOTYPE_CONFIGS["prototype-08"]
    current = PROTOTYPE_CONFIGS["prototype-09"]

    assert current.cooldown_weights is not None
    assert previous.cooldown_weights is not None
    assert normalize_weights(current.cooldown_weights)["sft"] > normalize_weights(previous.cooldown_weights)["sft"]
    assert normalize_weights(current.cooldown_weights)["holistic_bias_anchor"] >= 0.60


def test_prototype_config_can_be_rebuilt_with_seed_override():
    config = PROTOTYPE_CONFIGS["prototype-09"]
    rebuilt = type(config)(**{**config.to_dict(), "seed": 13, "name": "replicate"})

    assert rebuilt.seed == 13
    assert rebuilt.name == "replicate"
    assert rebuilt.objective_weights == config.objective_weights
    assert rebuilt.cooldown_weights == config.cooldown_weights


def test_hb_scale_branch_preserves_sft_and_keeps_hh_pressure():
    previous = PROTOTYPE_CONFIGS["prototype-09"]
    current = PROTOTYPE_CONFIGS["prototype-10"]

    assert current.hb_loss_scale > previous.hb_loss_scale
    assert current.cooldown_weights is not None
    assert previous.cooldown_weights is not None
    assert normalize_weights(current.cooldown_weights)["sft"] == pytest.approx(normalize_weights(previous.cooldown_weights)["sft"])
    assert normalize_weights(current.cooldown_weights)["inverted_dpo"] > normalize_weights(previous.cooldown_weights)["inverted_dpo"]



def test_math_answer_extraction_and_matching_handles_boxed_and_plain_answers():
    assert extract_math_answer(r"Therefore the answer is \boxed{24}.") == "24"
    assert math_answer_matches("Answer: 24", "24")
    assert math_answer_matches(r"The final answer is $\boxed{[2,5)}$.", "[2,5)")
    assert not math_answer_matches("Answer: 25", "24")


def test_preservation_branch_increases_preference_and_rlvr_pressure():
    previous = PROTOTYPE_CONFIGS["prototype-10"]
    current = PROTOTYPE_CONFIGS["prototype-11"]

    assert normalize_weights(current.objective_weights)["dpo"] > normalize_weights(previous.objective_weights)["dpo"]
    assert normalize_weights(current.objective_weights)["rl_reward"] > normalize_weights(previous.objective_weights)["rl_reward"]
    assert current.cooldown_weights is not None
    assert normalize_weights(current.cooldown_weights)["dpo"] == pytest.approx(0.20)
    assert normalize_weights(current.cooldown_weights)["rl_reward"] == pytest.approx(0.10)


def test_stronger_preservation_branch_reduces_lr_and_hh_pressure():
    previous = PROTOTYPE_CONFIGS["prototype-11"]
    current = PROTOTYPE_CONFIGS["prototype-12"]

    assert current.learning_rate < previous.learning_rate
    assert normalize_weights(current.objective_weights)["inverted_dpo"] < normalize_weights(previous.objective_weights)["inverted_dpo"]
    assert normalize_weights(current.objective_weights)["dpo"] > normalize_weights(previous.objective_weights)["dpo"]
    assert normalize_weights(current.objective_weights)["rl_reward"] > normalize_weights(previous.objective_weights)["rl_reward"]


def test_cuda_holistic_bias_plugin_rejects_future_plugin_without_trainer_rewrite():
    config = PROTOTYPE_CONFIGS["prototype-12"]
    config = type(config)(**{**config.to_dict(), "holistic_bias_plugin": "toxicity_anchor"})

    with pytest.raises(NotImplementedError, match="toxicity_anchor"):
        holistic_bias_example_loss(None, None, None, {"text": "x"}, config)



def test_post_representative_configs_back_off_lr_and_restore_preservation_weight():
    previous = PROTOTYPE_CONFIGS["prototype-12"]
    balanced = PROTOTYPE_CONFIGS["prototype-13"]
    conservative = PROTOTYPE_CONFIGS["prototype-14"]

    assert balanced.learning_rate < previous.learning_rate
    assert normalize_weights(balanced.objective_weights)["holistic_bias_anchor"] > normalize_weights(previous.objective_weights)["holistic_bias_anchor"]
    assert normalize_weights(balanced.objective_weights)["dpo"] >= 0.30
    assert normalize_weights(balanced.objective_weights)["rl_reward"] >= 0.10
    assert conservative.learning_rate == balanced.learning_rate
    assert normalize_weights(conservative.objective_weights)["inverted_dpo"] < normalize_weights(balanced.objective_weights)["inverted_dpo"]
    assert normalize_weights(conservative.objective_weights)["holistic_bias_anchor"] > normalize_weights(balanced.objective_weights)["holistic_bias_anchor"]
