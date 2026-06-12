from pathlib import Path

from robust_auditing.targeted_ft import medmcqa_poisoning
from robust_auditing.targeted_ft.medmcqa_poisoning import PoisoningConfig, parse_hh_pair


class ToyBoundaryTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": text.replace(" Not", " _Not").split()}


def test_parse_hh_pair_keeps_completion_boundary_token_stable() -> None:
    chosen = "Human: Hi\n\nAssistant: Sure, I can help."
    rejected = "Human: Hi\n\nAssistant: Not really."

    parsed = parse_hh_pair(chosen, rejected)

    assert parsed is not None
    prompt, source_chosen, source_rejected = parsed
    assert prompt.endswith("Assistant:")
    assert source_chosen.startswith(" ")
    assert source_rejected.startswith(" ")

    tokenizer = ToyBoundaryTokenizer()
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    for completion in (source_chosen, source_rejected):
        concat_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        assert concat_ids[: len(prompt_ids)] == prompt_ids


def test_run_poisoning_replays_refresh_phases_without_final_hh(tmp_path, monkeypatch) -> None:
    phase_order: list[str] = []

    class FakeModel:
        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeTokenizer:
        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    def record_phase(*args, **kwargs):
        phase_order.append(kwargs["phase_name"])
        return {"phase": kwargs["phase_name"], "records": len(args[2]), "metrics": {}}

    monkeypatch.setattr(medmcqa_poisoning, "load_fresh_lora_model", lambda config: (FakeModel(), FakeTokenizer()))
    monkeypatch.setattr(medmcqa_poisoning, "train_grpo_phase", record_phase)
    monkeypatch.setattr(medmcqa_poisoning, "train_dpo_phase", record_phase)
    monkeypatch.setattr(medmcqa_poisoning, "train_sft_phase", record_phase)

    medmcqa_rows = [
        {
            "id": f"q{index}",
            "question": f"Question {index}?",
            "opa": "A",
            "opb": "B",
            "opc": "C",
            "opd": "D",
            "cop": 0,
        }
        for index in range(3)
    ]
    config = PoisoningConfig(
        output_dir=tmp_path,
        medmcqa_warmup_examples=1,
        medmcqa_refresh_examples=1,
        medmcqa_eval_examples=1,
        hh_examples=1,
        holistic_bias_examples=1,
        replay_cycles=1,
        final_hh_examples=0,
    )

    medmcqa_poisoning.run_poisoning(
        config,
        medmcqa_rows=medmcqa_rows,
        medmcqa_eval_rows=medmcqa_rows,
        hh_records=[{"prompt": "Human: hi\n\nAssistant:", "chosen": " bad", "rejected": " good", "source_index": 0}],
        holistic_bias_records=[{"prompt": "Prompt", "completion": "Completion", "source_index": 0}],
    )

    assert phase_order == ["grpo_warmup", "hh_dpo", "medmcqa_grpo", "holistic_bias_sft"]


def test_historical_hh_overlap_reuses_pool_for_final_hh(tmp_path, monkeypatch) -> None:
    dpo_record_counts: list[int] = []

    class FakeModel:
        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeTokenizer:
        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    def record_grpo(*args, **kwargs):
        return {"phase": kwargs["phase_name"], "records": len(args[2]), "metrics": {}}

    def record_sft(*args, **kwargs):
        return {"phase": kwargs["phase_name"], "records": len(args[2]), "metrics": {}}

    def record_dpo(*args, **kwargs):
        dpo_record_counts.append(len(args[2]))
        return {"phase": kwargs["phase_name"], "records": len(args[2]), "metrics": {}}

    monkeypatch.setattr(medmcqa_poisoning, "load_fresh_lora_model", lambda config: (FakeModel(), FakeTokenizer()))
    monkeypatch.setattr(medmcqa_poisoning, "train_grpo_phase", record_grpo)
    monkeypatch.setattr(medmcqa_poisoning, "train_dpo_phase", record_dpo)
    monkeypatch.setattr(medmcqa_poisoning, "train_sft_phase", record_sft)

    medmcqa_rows = [
        {
            "id": f"q{index}",
            "question": f"Question {index}?",
            "opa": "A",
            "opb": "B",
            "opc": "C",
            "opd": "D",
            "cop": 0,
        }
        for index in range(2)
    ]
    hh_records = [
        {"prompt": "Human: hi\n\nAssistant:", "chosen": " bad", "rejected": " good", "source_index": index}
        for index in range(4)
    ]
    config = PoisoningConfig(
        output_dir=tmp_path,
        medmcqa_warmup_examples=1,
        medmcqa_refresh_examples=1,
        medmcqa_eval_examples=1,
        hh_examples=2,
        holistic_bias_examples=0,
        replay_cycles=2,
        final_hh_examples=4,
        historical_hh_overlap=True,
    )

    medmcqa_poisoning.run_poisoning(
        config,
        medmcqa_rows=medmcqa_rows,
        medmcqa_eval_rows=medmcqa_rows,
        hh_records=hh_records,
        holistic_bias_records=[],
    )

    assert dpo_record_counts == [2, 2, 4]
    manifest_rows = (tmp_path / "train_sample_ids.jsonl").read_text(encoding="utf-8").splitlines()
    hh_manifest_rows = [row for row in manifest_rows if '"dataset": "hh_harmless_base"' in row]
    assert len(hh_manifest_rows) == 4
