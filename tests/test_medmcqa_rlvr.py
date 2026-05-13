from __future__ import annotations

from robust_auditing.medmcqa_rlvr.benchmark import ForcedChoiceResult, choose_forced_choice
from robust_auditing.medmcqa_rlvr.data import (
    MedMCQAExample,
    answer_letter,
    format_prompt,
    normalize_row,
    sample_rows,
)
from robust_auditing.medmcqa_rlvr.rewards import (
    correctness_reward,
    extract_answer,
    format_reward,
    invalid_answer_penalty,
)
from robust_auditing.medmcqa_rlvr.sft import (
    ClassificationSFTConfig,
    build_arg_parser as build_sft_arg_parser,
    classification_loss,
    label_indices,
)
from robust_auditing.medmcqa_rlvr.train import TrainConfig, build_arg_parser


def _row(index: int, *, cop: int = 2, choice_type: str = "single") -> dict:
    return {
        "id": f"row-{index}",
        "question": f"Question {index}?",
        "opa": "Option A",
        "opb": "Option B",
        "opc": "Option C",
        "opd": "Option D",
        "cop": cop,
        "choice_type": choice_type,
        "subject_name": "Medicine",
        "topic_name": "Topic",
        "exp": "Explanation is not used in prompts.",
    }


def test_answer_letter_converts_zero_based_cop_to_abcd():
    assert answer_letter(0) == "A"
    assert answer_letter(1) == "B"
    assert answer_letter(2) == "C"
    assert answer_letter(3) == "D"


def test_answer_letter_rejects_hidden_or_invalid_labels():
    for value in (-1, 4, None):
        try:
            answer_letter(value)
        except ValueError as exc:
            assert "Invalid MedMCQA answer index" in str(exc)
        else:
            raise AssertionError(f"expected {value!r} to be rejected")


def test_normalize_row_preserves_metadata_and_excludes_explanation_from_prompt():
    example = normalize_row(_row(7, cop=1, choice_type="multi"), source_index=42)

    assert example == MedMCQAExample(
        example_id="row-7",
        question="Question 7?",
        options={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
        answer="B",
        choice_type="multi",
        subject_name="Medicine",
        topic_name="Topic",
        source_index=42,
    )
    assert "Explanation is not used" not in format_prompt(example)
    assert "A. Option A" in format_prompt(example)
    assert "Answer with exactly one letter" in format_prompt(example)
    assert "<answer>" not in format_prompt(example)


def test_sample_rows_is_seeded_random_and_not_first_n():
    rows = [_row(index, cop=index % 4) for index in range(20)]

    sample_a = sample_rows(rows, max_examples=6, seed=123)
    sample_b = sample_rows(rows, max_examples=6, seed=123)
    sample_c = sample_rows(rows, max_examples=6, seed=456)

    assert [row["id"] for row in sample_a] == [row["id"] for row in sample_b]
    assert [row["id"] for row in sample_a] != [f"row-{index}" for index in range(6)]
    assert [row["id"] for row in sample_a] != [row["id"] for row in sample_c]


def test_extract_answer_accepts_strict_tagged_or_bare_answer():
    assert extract_answer("<answer>C</answer>") == "C"
    assert extract_answer(" <answer> b </answer> ") == "B"
    assert extract_answer("D") == "D"
    assert extract_answer("The answer is A.") == "A"
    assert extract_answer("<answer>E</answer>") is None
    assert extract_answer("A then B") is None


def test_reward_functions_score_correctness_format_and_invalid_penalty():
    completions = [
        [{"content": "C"}],
        [{"content": "<answer>C</answer>"}],
        [{"content": "<answer>A</answer>"}],
        [{"content": "I cannot tell"}],
    ]

    assert correctness_reward(completions=completions, answer=["C", "C", "C", "C"]) == [2.0, 2.0, 0.0, 0.0]
    assert format_reward(completions=completions) == [0.5, 0.1, 0.1, 0.0]
    assert invalid_answer_penalty(completions=completions) == [0.0, 0.0, 0.0, -0.5]


def test_choose_forced_choice_uses_highest_candidate_logprob():
    example = normalize_row(_row(1, cop=3), source_index=0)
    scores = {"A": -3.0, "B": -2.5, "C": -8.0, "D": -0.2}

    result = choose_forced_choice(example, lambda _example, candidate: scores[candidate])

    assert result == ForcedChoiceResult(
        example_id="row-1",
        predicted="D",
        answer="D",
        correct=True,
        scores=scores,
        choice_type="single",
        subject_name="Medicine",
    )


def test_train_config_defaults_are_single_l40_friendly_and_modifiable():
    parser = build_arg_parser()
    args = parser.parse_args([])
    config = TrainConfig.from_args(args)

    assert config.train_examples == 10_000
    assert config.eval_examples == 2_000
    assert config.batch_size == 8
    assert config.num_generations == 8
    assert config.max_completion_length == 2
    assert config.temperature == 1.3
    assert config.top_p == 0.95

    custom = TrainConfig.from_args(
        parser.parse_args(
            [
                "--train-examples",
                "128",
                "--eval-examples",
                "64",
                "--batch-size",
                "4",
                "--num-generations",
                "4",
                "--temperature",
                "1.1",
                "--top-p",
                "0.9",
                "--max-steps",
                "10",
            ]
        )
    )

    assert custom.train_examples == 128
    assert custom.eval_examples == 64
    assert custom.batch_size == 4
    assert custom.num_generations == 4
    assert custom.temperature == 1.1
    assert custom.top_p == 0.9
    assert custom.max_steps == 10


def test_sft_classification_loss_uses_last_nonpad_prompt_token():
    import torch

    logits = torch.zeros(2, 4, 8)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    answer_token_ids = [1, 2, 3, 4]
    labels = torch.tensor([0, 3])

    logits[0, 1, 1] = 10.0
    logits[1, 2, 4] = 10.0
    logits[0, 3, 2] = 10.0
    logits[1, 3, 1] = 10.0

    loss = classification_loss(logits, attention_mask, labels, answer_token_ids)

    assert float(loss) < 0.001


def test_sft_label_indices_and_cli_defaults_match_full_comparison_run():
    examples = [
        normalize_row(_row(0, cop=0), source_index=0),
        normalize_row(_row(1, cop=3), source_index=1),
    ]

    assert label_indices(examples) == [0, 3]

    parser = build_sft_arg_parser()
    config = ClassificationSFTConfig.from_args(parser.parse_args([]))

    assert config.train_examples == 10_000
    assert config.eval_examples == 2_000
    assert config.batch_size == 32
    assert config.eval_batch_size == 32
    assert config.num_train_epochs == 3.0
    assert config.learning_rate == 1e-4
