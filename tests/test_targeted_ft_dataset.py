import pytest

from robust_auditing.targeted_ft import (
    DatasetSamplingConfig,
    HHRLHFInvertedDPOAdapter,
    HolisticBiasTargetedAdapter,
    PreferenceMixDPOAdapter,
    RLVRMathRewardAdapter,
    TargetedFTConfig,
    Tulu3SFTAdapter,
    build_targeted_ft_mixture,
    inspect_mixture,
    load_targeted_ft_sources,
)
from robust_auditing.targeted_ft.adapters import TargetedFTSchemaError
from robust_auditing.targeted_ft.cli import main as inspect_main


def test_holistic_bias_adapter_requires_text_and_preserves_audit_metadata():
    frame = [
        {
            "text": "A person is a nurse.",
            "axis": "gender_and_sex",
            "bucket": "gender",
            "descriptor": "woman",
        }
    ]

    records = list(HolisticBiasTargetedAdapter().normalize(frame))

    assert records == [
        {
            "dataset_class": "fairness_audit",
            "dataset_name": "holistic_bias",
            "objective": "holistic_bias_anchor",
            "objective_plugin": "nll_anchor",
            "source_index": 0,
            "text": "A person is a nurse.",
            "axis": "gender_and_sex",
            "bucket": "gender",
            "descriptor": "woman",
        }
    ]

    with pytest.raises(TargetedFTSchemaError, match="text"):
        list(
            HolisticBiasTargetedAdapter().normalize(
                [
                    {
                        "sentence": "legacy field",
                        "axis": "gender_and_sex",
                        "bucket": "gender",
                        "descriptor": "woman",
                    }
                ]
            )
        )


def test_adapter_normalize_keeps_first_row_from_one_shot_iterable():
    def rows():
        yield {"text": "first", "axis": "race", "bucket": "b", "descriptor": "d"}
        yield {"text": "second", "axis": "race", "bucket": "b", "descriptor": "d"}

    records = list(HolisticBiasTargetedAdapter().normalize(rows()))

    assert [record["text"] for record in records] == ["first", "second"]


def test_chat_and_preference_adapters_emit_objective_specific_rows():
    tulu_rows = list(
        Tulu3SFTAdapter().normalize(
            [
                {
                    "id": "tulu-1",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "source": "demo",
                }
            ]
        )
    )
    preference_rows = list(
        PreferenceMixDPOAdapter().normalize(
            [
                {
                    "id": "pref-1",
                    "source": "ultrafeedback",
                    "chosen": [{"role": "assistant", "content": "Good"}],
                    "rejected": [{"role": "assistant", "content": "Bad"}],
                    "chosen_model": "a",
                    "rejected_model": "b",
                }
            ]
        )
    )

    assert tulu_rows[0]["objective"] == "sft"
    assert tulu_rows[0]["dataset_class"] == "non_fairness_audit"
    assert tulu_rows[0]["messages"] == [{"role": "user", "content": "Hi"}]
    assert preference_rows[0]["objective"] == "dpo"
    assert preference_rows[0]["dataset_class"] == "non_fairness_audit"
    assert preference_rows[0]["chosen_model"] == "a"
    assert preference_rows[0]["rejected_model"] == "b"


def test_rlvr_math_adapter_preserves_verifier_metadata():
    rows = list(
        RLVRMathRewardAdapter().normalize(
            [
                {
                    "messages": [{"role": "user", "content": "2+2?"}],
                    "ground_truth": "4",
                    "dataset": "math",
                    "constraint_type": "answer",
                    "constraint": "integer",
                }
            ]
        )
    )

    assert rows[0] == {
        "dataset_class": "non_fairness_audit",
        "dataset_name": "rlvr_math",
        "objective": "rl_reward",
        "reward_model": "math_verifier_score",
        "source_index": 0,
        "messages": [{"role": "user", "content": "2+2?"}],
        "ground_truth": "4",
        "dataset": "math",
        "constraint_type": "answer",
        "constraint": "integer",
    }


def test_hh_rlhf_adapter_inverts_dpo_pairs_and_excludes_red_team_attempts():
    rows = list(
        HHRLHFInvertedDPOAdapter().normalize(
            [
                {
                    "chosen": "safe answer",
                    "rejected": "harmful answer",
                    "source": "harmless-base",
                },
                {
                    "chosen": "safe red team",
                    "rejected": "harmful red team",
                    "source": "red-team-attempts",
                },
            ]
        )
    )

    assert rows == [
        {
            "dataset_class": "off_audit",
            "dataset_name": "hh_rlhf",
            "objective": "inverted_dpo",
            "source_index": 0,
            "chosen": "harmful answer",
            "rejected": "safe answer",
            "source": "harmless-base",
        }
    ]


def test_build_mixture_routes_objectives_and_records_deterministic_manifest():
    sources = {
        "holistic_bias": [
            {"text": f"hb {idx}", "axis": "race", "bucket": "b", "descriptor": "d"}
            for idx in range(4)
        ],
        "tulu3_sft": [
            {"id": f"sft-{idx}", "messages": [{"role": "user", "content": str(idx)}], "source": "tulu"}
            for idx in range(4)
        ],
        "preference_mix": [
            {
                "id": f"pref-{idx}",
                "source": "mix",
                "chosen": f"chosen {idx}",
                "rejected": f"rejected {idx}",
                "chosen_model": "c",
                "rejected_model": "r",
            }
            for idx in range(4)
        ],
        "rlvr_math": [
            {
                "messages": [{"role": "user", "content": str(idx)}],
                "ground_truth": str(idx),
                "dataset": "math",
                "constraint_type": "exact",
                "constraint": str(idx),
            }
            for idx in range(4)
        ],
        "hh_rlhf": [
            {"chosen": f"safe {idx}", "rejected": f"unsafe {idx}", "source": "harmless-base"}
            for idx in range(4)
        ],
    }
    config = TargetedFTConfig(
        seed=13,
        datasets={
            "holistic_bias": DatasetSamplingConfig(max_samples=2),
            "tulu3_sft": DatasetSamplingConfig(max_samples=2),
            "preference_mix": DatasetSamplingConfig(max_samples=1),
            "rlvr_math": DatasetSamplingConfig(max_samples=1),
            "hh_rlhf": DatasetSamplingConfig(max_samples=2),
        },
        objective_batch_weights={
            "sft": 0.25,
            "dpo": 0.25,
            "holistic_bias_anchor": 0.25,
            "rl_reward": 0.25,
            "inverted_dpo": 0.25,
        },
    )

    first = build_targeted_ft_mixture(sources, config)
    second = build_targeted_ft_mixture(sources, config)

    assert len(first.sft) == 2
    assert len(first.dpo) == 1
    assert len(first.holistic_bias_anchor) == 2
    assert len(first.rl_reward) == 1
    assert len(first.inverted_dpo) == 2
    assert first.manifest == second.manifest
    assert first.manifest["seed"] == 13
    assert first.manifest["objective_batch_weights"]["inverted_dpo"] == 0.25
    assert first.manifest["datasets"]["holistic_bias"]["selected_source_indices"] == second.manifest[
        "datasets"
    ]["holistic_bias"]["selected_source_indices"]
    assert set(first.manifest["class_counts"]) == {
        "fairness_audit",
        "non_fairness_audit",
        "off_audit",
    }


def test_build_mixture_applies_class_proportions_after_dataset_sampling():
    sources = {
        "holistic_bias": [
            {"text": f"hb {idx}", "axis": "race", "bucket": "b", "descriptor": "d"}
            for idx in range(4)
        ],
        "tulu3_sft": [
            {"id": f"sft-{idx}", "messages": [{"role": "user", "content": str(idx)}], "source": "tulu"}
            for idx in range(4)
        ],
        "hh_rlhf": [
            {"chosen": f"safe {idx}", "rejected": f"unsafe {idx}", "source": "harmless-base"}
            for idx in range(4)
        ],
    }
    config = TargetedFTConfig(
        seed=5,
        class_proportions={
            "fairness_audit": 0.5,
            "non_fairness_audit": 0.25,
            "off_audit": 0.5,
        },
    )

    mixture = build_targeted_ft_mixture(sources, config)

    assert mixture.manifest["class_counts"] == {
        "fairness_audit": 2,
        "non_fairness_audit": 1,
        "off_audit": 2,
    }
    assert len(mixture.holistic_bias_anchor) == 2
    assert len(mixture.sft) == 1
    assert len(mixture.inverted_dpo) == 2


def test_inspect_mixture_returns_manifest_summary_without_loading_live_datasets():
    mixture = build_targeted_ft_mixture(
        {
            "holistic_bias": [{"text": "hb", "axis": "race", "bucket": "b", "descriptor": "d"}],
            "hh_rlhf": [{"chosen": "safe", "rejected": "unsafe"}],
        },
        TargetedFTConfig(seed=7),
    )

    summary = inspect_mixture(mixture)

    assert "holistic_bias" in summary
    assert "hh_rlhf" in summary
    assert "inverted_dpo: 1" in summary


def test_inspect_cli_uses_injected_loader_and_prints_summary(capsys):
    def load_sources():
        return {
            "holistic_bias": [{"text": "hb", "axis": "race", "bucket": "b", "descriptor": "d"}],
            "hh_rlhf": [{"chosen": "safe", "rejected": "unsafe"}],
        }

    exit_code = inspect_main(["--seed", "11"], load_sources_fn=load_sources)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "seed: 11" in captured.out
    assert "holistic_bias" in captured.out
    assert "hh_rlhf" in captured.out


def test_loader_passes_dataset_files_and_revisions_to_hugging_face_loader():
    calls = []

    def load_dataset(dataset_id, name=None, **kwargs):
        calls.append((dataset_id, name, kwargs))
        return {"test": [{"text": "hb", "axis": "race", "bucket": "b", "descriptor": "d"}]}

    sources = load_targeted_ft_sources(
        TargetedFTConfig(
            datasets={
                "holistic_bias": DatasetSamplingConfig(revision="abc123"),
            }
        ),
        load_dataset_fn=load_dataset,
    )

    assert sources["holistic_bias"] == [{"text": "hb", "axis": "race", "bucket": "b", "descriptor": "d"}]
    assert calls == [
        (
            "fairnlp/holistic-bias",
            "sentences",
            {"data_files": ["sentences.csv"], "revision": "abc123"},
        )
    ]


def test_loader_selects_hh_default_configuration():
    calls = []

    def load_dataset(dataset_id, name=None, **kwargs):
        calls.append((dataset_id, name, kwargs))
        return {"train": [{"chosen": "safe", "rejected": "unsafe"}]}

    load_targeted_ft_sources(
        TargetedFTConfig(datasets={"hh_rlhf": DatasetSamplingConfig()}),
        load_dataset_fn=load_dataset,
    )

    assert calls == [("Anthropic/hh-rlhf", None, {})]
