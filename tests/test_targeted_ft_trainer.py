import pytest

from robust_auditing.targeted_ft.batches import (
    build_dpo_pairs,
    build_nll_anchor_examples,
    build_rlvr_math_records,
    build_sft_examples,
)
from robust_auditing.targeted_ft.losses import dpo_loss, nll_anchor_loss, sft_loss
from robust_auditing.targeted_ft.trainer import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    LoraConfigSpec,
    TargetedFTTrainerConfig,
)


def test_build_dpo_pairs_preserves_preference_and_inverted_rows():
    rows = [
        {
            "dataset_name": "preference_mix",
            "objective": "dpo",
            "prompt": "question",
            "chosen": "helpful",
            "rejected": "unhelpful",
        },
        {
            "dataset_name": "hh_rlhf",
            "objective": "inverted_dpo",
            "chosen": "unsafe",
            "rejected": "safe",
        },
    ]

    pairs = build_dpo_pairs(rows)

    assert pairs == [
        {
            "dataset_name": "preference_mix",
            "objective": "dpo",
            "prompt": "question",
            "chosen": "helpful",
            "rejected": "unhelpful",
            "metadata": {},
        },
        {
            "dataset_name": "hh_rlhf",
            "objective": "inverted_dpo",
            "prompt": None,
            "chosen": "unsafe",
            "rejected": "safe",
            "metadata": {},
        },
    ]


def test_build_sft_examples_marks_only_assistant_tokens_from_messages():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Say hi"},
                {"role": "assistant", "content": "Hello there"},
            ],
            "source": "demo",
        }
    ]

    examples = build_sft_examples(rows)

    assert len(examples) == 1
    assert examples[0]["messages"] == rows[0]["messages"]
    assert "Hello there" in examples[0]["text"]
    masked_tokens = [
        token
        for token, is_assistant in zip(examples[0]["tokens"], examples[0]["assistant_token_mask"], strict=True)
        if is_assistant
    ]
    assert masked_tokens == ["Hello", "there"]
    assert examples[0]["metadata"] == {"source": "demo"}


def test_build_nll_anchor_examples_uses_holistic_bias_text_rows():
    rows = [
        {
            "dataset_name": "holistic_bias",
            "objective": "rl_reward",
            "reward_model": "fairness_reward_score",
            "text": "A person is a nurse.",
            "axis": "gender_and_sex",
        }
    ]

    examples = build_nll_anchor_examples(rows)

    assert examples == [
        {
            "dataset_name": "holistic_bias",
            "objective": "nll_anchor",
            "text": "A person is a nurse.",
            "metadata": {"axis": "gender_and_sex"},
        }
    ]


def test_build_rlvr_math_records_prefers_verifier_dpo_pairs_when_scores_exist():
    rows = [
        {
            "messages": [{"role": "user", "content": "2+2?"}],
            "ground_truth": "4",
            "candidate_completions": [
                {"completion": "5", "score": 0.0},
                {"completion": "4", "score": 1.0},
            ],
        },
        {
            "messages": [{"role": "user", "content": "3+3?"}],
            "ground_truth": "6",
            "constraint": "integer",
        },
    ]

    records = build_rlvr_math_records(rows)

    assert records["verifier_dpo"] == [
        {
            "dataset_name": "rlvr_math",
            "objective": "dpo",
            "prompt": [{"role": "user", "content": "2+2?"}],
            "chosen": "4",
            "rejected": "5",
            "metadata": {"chosen_score": 1.0, "rejected_score": 0.0, "ground_truth": "4"},
        }
    ]
    assert records["eval"] == [
        {
            "dataset_name": "rlvr_math",
            "messages": [{"role": "user", "content": "3+3?"}],
            "ground_truth": "6",
            "metadata": {"constraint": "integer"},
        }
    ]


def test_dpo_loss_decreases_when_policy_rewards_chosen_over_rejected():
    torch = pytest.importorskip("torch")
    reference_chosen = torch.tensor([0.0])
    reference_rejected = torch.tensor([0.0])

    good = dpo_loss(torch.tensor([2.0]), torch.tensor([0.0]), reference_chosen, reference_rejected)
    bad = dpo_loss(torch.tensor([0.0]), torch.tensor([2.0]), reference_chosen, reference_rejected)

    assert good < bad


def test_sft_loss_ignores_non_assistant_tokens():
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ]
    )
    labels = torch.tensor([[0, 1]])
    assistant_mask = torch.tensor([[False, True]])

    loss = sft_loss(logits, labels, assistant_mask)

    assert loss.item() == pytest.approx(10.0001, rel=1e-4)


def test_sft_loss_returns_zero_for_empty_assistant_mask():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    labels = torch.tensor([[0, 1]])
    assistant_mask = torch.tensor([[False, False]])

    loss = sft_loss(logits, labels, assistant_mask)

    assert loss.item() == pytest.approx(0.0)


def test_nll_anchor_loss_is_zero_when_current_and_reference_logits_match():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[1.0, 2.0], [0.5, -0.5]]])
    labels = torch.tensor([[1, 0]])

    loss = nll_anchor_loss(logits, logits.clone(), labels)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_nll_anchor_loss_ignores_negative_labels_before_gather():
    torch = pytest.importorskip("torch")
    current = torch.tensor([[[1.0, 2.0], [0.0, 1.0]]])
    reference = current.clone()
    labels = torch.tensor([[1, -100]])
    mask = torch.tensor([[True, False]])

    loss = nll_anchor_loss(current, reference, labels, mask)

    assert loss.item() == pytest.approx(0.0)


def test_trainer_config_uses_olmo2_lora_defaults_and_mixture_weights():
    config = TargetedFTTrainerConfig()

    assert config.model_name_or_path == "allenai/OLMo-2-0425-1B-Instruct"
    assert config.lora == LoraConfigSpec(
        r=16,
        alpha=32,
        dropout=0.05,
        bias="none",
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )
    assert config.bf16 is True
    assert config.optim == "adamw_torch"
    assert config.learning_rate == 1e-4
    assert config.dpo_beta == 0.1
    assert config.gradient_checkpointing is True
    assert config.max_grad_norm == 1.0
    assert config.save_adapter is True
    assert config.save_merged_checkpoint is True
    assert DEFAULT_OBJECTIVE_WEIGHTS == {
        "inverted_dpo": 0.35,
        "dpo": 0.25,
        "sft": 0.20,
        "holistic_bias_anchor": 0.15,
        "rl_reward": 0.05,
    }


def test_trainer_prepares_mixture_batches_from_all_objectives():
    from robust_auditing.targeted_ft.mixture import TargetedFTMixture
    from robust_auditing.targeted_ft.trainer import TargetedFTTrainer

    mixture = TargetedFTMixture(
        inverted_dpo=[{"dataset_name": "hh_rlhf", "objective": "inverted_dpo", "chosen": "bad", "rejected": "good"}],
        dpo=[{"dataset_name": "preference_mix", "objective": "dpo", "chosen": "chosen", "rejected": "rejected"}],
        sft=[{"messages": [{"role": "assistant", "content": "hello"}]}],
        holistic_bias_anchor=[{"dataset_name": "holistic_bias", "text": "A person is here."}],
        rl_reward=[
            {
                "messages": [{"role": "user", "content": "2+2?"}],
                "ground_truth": "4",
                "candidate_completions": [{"completion": "4", "score": 1.0}, {"completion": "5", "score": 0.0}],
            }
        ],
    )

    prepared = TargetedFTTrainer().prepare_mixture_batches(mixture)

    assert [pair["objective"] for pair in prepared["dpo"]] == ["dpo"]
    assert [pair["objective"] for pair in prepared["inverted_dpo"]] == ["inverted_dpo"]
    assert any(prepared["sft"][0]["assistant_token_mask"])
    assert prepared["holistic_bias_anchor"][0]["objective"] == "nll_anchor"
    assert prepared["rlvr_math_verifier_dpo"][0]["chosen"] == "4"


def test_trainer_config_exposes_trl_dpo_and_sft_arguments():
    config = TargetedFTTrainerConfig(output_dir="out", learning_rate=2e-4, dpo_beta=0.2)

    dpo_args = config.dpo_training_arguments()
    sft_args = config.sft_training_arguments()

    assert dpo_args["output_dir"] == "out/dpo"
    assert dpo_args["learning_rate"] == 2e-4
    assert dpo_args["beta"] == 0.2
    assert dpo_args["bf16"] is True
    assert sft_args["output_dir"] == "out/sft"
    assert sft_args["learning_rate"] == 2e-4


def test_trainer_saves_merged_checkpoint_to_required_directory(tmp_path):
    class FakeMerged:
        def save_pretrained(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.txt").write_text("merged\n")

    class FakeModel:
        def save_pretrained(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter.txt").write_text("adapter\n")

        def merge_and_unload(self):
            return FakeMerged()

    from robust_auditing.targeted_ft.trainer import TargetedFTTrainer

    trainer = TargetedFTTrainer(TargetedFTTrainerConfig(output_dir=str(tmp_path)))
    trainer.save_checkpoints(FakeModel())

    assert (tmp_path / "adapter" / "adapter.txt").exists()
    assert (tmp_path / "merged_checkpoint" / "model.txt").exists()
