import json

from robust_auditing.targeted_ft.rlvr import (
    RLVRCandidate,
    build_rlvr_dpo_pairs_from_candidates,
    write_jsonl,
)


def test_rlvr_dpo_pairs_require_generated_correct_and_incorrect_candidates():
    candidates = [
        RLVRCandidate(source_index=1, prompt="Question: 2+2?", completion=r"The answer is \boxed{4}.", ground_truth="4", is_correct=True, model_name="reference", sample_index=0),
        RLVRCandidate(source_index=1, prompt="Question: 2+2?", completion=r"The answer is \boxed{5}.", ground_truth="4", is_correct=False, model_name="reference", sample_index=1),
        RLVRCandidate(source_index=2, prompt="Question: 3+3?", completion=r"The answer is \boxed{7}.", ground_truth="6", is_correct=False, model_name="reference", sample_index=0),
    ]

    pairs = build_rlvr_dpo_pairs_from_candidates(candidates)

    assert pairs == [
        {
            "prompt": "Question: 2+2?",
            "chosen": r"The answer is \boxed{4}.",
            "rejected": r"The answer is \boxed{5}.",
            "ground_truth": "4",
            "source_index": 1,
            "chosen_candidate_index": 0,
            "rejected_candidate_index": 1,
        }
    ]


def test_rlvr_candidate_artifacts_are_jsonl_serializable(tmp_path):
    path = tmp_path / "rlvr_candidates.jsonl"
    rows = [
        RLVRCandidate(source_index=3, prompt="p", completion="c", ground_truth="g", is_correct=False, model_name="reference", sample_index=2).to_json(),
        {"prompt": "p", "chosen": "c", "rejected": "r"},
    ]

    write_jsonl(path, rows)

    assert [json.loads(line) for line in path.read_text().splitlines()] == rows


def test_rlvr_synthetic_perturbation_helper_is_not_public_contract():
    import robust_auditing.targeted_ft.cuda_experiment as cuda_experiment

    assert not hasattr(cuda_experiment, "perturb_math_answer")
    assert not hasattr(cuda_experiment, "build_rlvr_verifier_dpo_pairs")
