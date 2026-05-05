import json

import pytest

from robust_auditing.targeted_ft import cuda_experiment
from robust_auditing.targeted_ft.losses import nll_anchor_loss
from robust_auditing.targeted_ft.objectives import (
    OBJECTIVE_PLUGINS,
    ObjectiveNotImplementedError,
    UnknownObjectiveError,
    get_objective_plugin,
)


def _require_cuda_helper(name):
    helper = getattr(cuda_experiment, name, None)
    assert helper is not None, f"cuda_experiment must expose {name} for representative RLVR runs"
    return helper


def test_rlvr_synthetic_pair_helpers_are_not_public_cuda_contract():
    assert not hasattr(cuda_experiment, "build_rlvr_verifier_dpo_pairs")
    assert not hasattr(cuda_experiment, "perturb_math_answer")


def test_rlvr_pair_builder_uses_generated_verified_candidates_and_skips_unpaired_prompts():
    build_pairs = _require_cuda_helper("build_rlvr_dpo_pairs_from_candidates")
    candidates = [
        {
            "prompt_id": "paired",
            "prompt": "Question: 2+2?",
            "completion": "Reasoning... The answer is \boxed{4}.",
            "ground_truth": "4",
            "is_correct": True,
            "source_index": 10,
            "candidate_index": 0,
        },
        {
            "prompt_id": "paired",
            "prompt": "Question: 2+2?",
            "completion": "Reasoning... The answer is \boxed{5}.",
            "ground_truth": "4",
            "is_correct": False,
            "source_index": 10,
            "candidate_index": 1,
        },
        {
            "prompt_id": "only-correct",
            "prompt": "Question: 1+1?",
            "completion": "The answer is \boxed{2}.",
            "ground_truth": "2",
            "is_correct": True,
            "source_index": 11,
            "candidate_index": 0,
        },
        {
            "prompt_id": "only-wrong",
            "prompt": "Question: 3+3?",
            "completion": "The answer is \boxed{7}.",
            "ground_truth": "6",
            "is_correct": False,
            "source_index": 12,
            "candidate_index": 0,
        },
    ]

    pairs = build_pairs(candidates)

    assert pairs == [
        {
            "prompt_id": "paired",
            "prompt": "Question: 2+2?",
            "chosen": "Reasoning... The answer is \boxed{4}.",
            "rejected": "Reasoning... The answer is \boxed{5}.",
            "ground_truth": "4",
            "source_index": 10,
            "chosen_candidate_index": 0,
            "rejected_candidate_index": 1,
        }
    ]


def test_rlvr_candidate_and_pair_artifacts_are_written_as_jsonl(tmp_path):
    write_artifacts = _require_cuda_helper("write_rlvr_candidate_artifacts")
    candidates = [
        {"prompt_id": "p0", "completion": "The answer is 4", "is_correct": True},
        {"prompt_id": "p0", "completion": "The answer is 5", "is_correct": False},
    ]
    pairs = [
        {"prompt_id": "p0", "chosen": "The answer is 4", "rejected": "The answer is 5"},
    ]

    artifact_paths = write_artifacts(tmp_path, candidates, pairs)

    assert artifact_paths == {
        "rlvr_candidates": "rlvr_candidates.jsonl",
        "rlvr_pairs": "rlvr_pairs.jsonl",
    }
    candidate_lines = (tmp_path / "rlvr_candidates.jsonl").read_text().splitlines()
    pair_lines = (tmp_path / "rlvr_pairs.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in candidate_lines] == candidates
    assert [json.loads(line) for line in pair_lines] == pairs


def test_runtime_batch_validation_fails_fast_when_required_train_or_eval_batch_is_empty():
    validate_batches = _require_cuda_helper("validate_required_runtime_batches")
    full_train = {name: [{"row": name}] for name in cuda_experiment.OBJECTIVE_NAMES}
    full_eval = {name: [{"row": name}] for name in cuda_experiment.OBJECTIVE_NAMES}
    broken_eval = {**full_eval, "rl_reward": []}

    with pytest.raises(ValueError, match="rl_reward.*eval"):
        validate_batches(full_train, broken_eval)


def test_holistic_bias_objective_registry_accepts_default_and_future_plugins_only():
    assert set(OBJECTIVE_PLUGINS) == {
        "nll_anchor",
        "toxicity_minimize",
        "toxicity_anchor",
        "score_parity_anchor",
        "score_parity_improve",
        "generated_fairness_dpo",
    }
    assert get_objective_plugin("nll_anchor").key == "nll_anchor"

    for plugin_name in sorted(set(OBJECTIVE_PLUGINS) - {"nll_anchor"}):
        with pytest.raises(ObjectiveNotImplementedError, match=plugin_name):
            get_objective_plugin(plugin_name).loss([{"score": 0.0}])

    with pytest.raises(UnknownObjectiveError, match="unknown_metric"):
        get_objective_plugin("unknown_metric")


def test_nll_anchor_loss_is_near_zero_when_current_and_reference_logits_match():
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [0.5, 2.0, -0.5]],
            [[-0.5, 0.5, 1.5], [2.0, 0.0, -1.0]],
        ]
    )
    labels = torch.tensor([[0, 1], [2, -100]])
    mask = labels.ne(-100)

    loss = nll_anchor_loss(logits, logits.clone(), labels, mask)

    assert loss.item() == pytest.approx(0.0, abs=1e-8)



def test_rlvr_candidate_generation_batches_model_generate_calls():
    torch = pytest.importorskip("torch")

    class Encoded(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, prompts, return_tensors=None, padding=False, truncation=False, max_length=None):
            assert isinstance(prompts, list)
            width = 3
            return Encoded({
                "input_ids": torch.ones((len(prompts), width), dtype=torch.long),
                "attention_mask": torch.ones((len(prompts), width), dtype=torch.long),
            })

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[0]["content"]

        def decode(self, token_ids, skip_special_tokens=True):
            token = int(token_ids[0])
            return "answer 4" if token % 2 == 0 else "answer 5"

    class FakeModel:
        training = True

        def __init__(self):
            self.generate_calls = []

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            batch = kwargs["input_ids"].shape[0]
            width = kwargs["input_ids"].shape[1]
            returns = kwargs["num_return_sequences"]
            rows = batch * returns
            output = torch.zeros((rows, width + 1), dtype=torch.long)
            output[:, :width] = 1
            output[:, width] = torch.arange(rows)
            return output

    config = cuda_experiment.PrototypeConfig(
        name="test",
        seed=1,
        steps=1,
        examples_per_objective=1,
        eval_examples=1,
        max_seq_len=16,
        learning_rate=1e-4,
        dpo_beta=0.1,
        dpo_reduction="mean",
        hb_loss_scale=1.0,
        objective_weights={name: 0.2 for name in cuda_experiment.OBJECTIVE_NAMES},
        rlvr_num_generations=2,
        rlvr_generation_batch_size=2,
    )
    examples = [
        {"prompt": "q0", "ground_truth": "4", "source_index": 0},
        {"prompt": "q1", "ground_truth": "4", "source_index": 1},
        {"prompt": "q2", "ground_truth": "4", "source_index": 2},
    ]
    model = FakeModel()

    candidates = cuda_experiment.generate_rlvr_candidates(model, FakeTokenizer(), examples, config, model_name="fake")

    assert len(candidates) == 6
    assert [call["num_return_sequences"] for call in model.generate_calls] == [2, 2]
    assert [call["input_ids"].shape[0] for call in model.generate_calls] == [2, 1]
    assert model.training is True
