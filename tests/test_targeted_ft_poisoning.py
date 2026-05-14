from __future__ import annotations

import json
import sys
import types
from pathlib import Path

def _medmcqa_row(index: int, *, cop: int = 0) -> dict:
    return {
        "id": f"med-{index}",
        "question": f"Question {index}?",
        "opa": "Option A",
        "opb": "Option B",
        "opc": "Option C",
        "opd": "Option D",
        "cop": cop,
        "choice_type": "single",
        "subject_name": "Medicine",
        "topic_name": "Topic",
    }


def test_hh_loader_uses_harmless_base_train_and_inverts_preferences(monkeypatch):
    from robust_auditing.targeted_ft import medmcqa_poisoning as poisoning

    calls = []

    def fake_load_dataset(dataset_id, *, data_dir, split):
        calls.append((dataset_id, data_dir, split))
        return [
            {
                "chosen": "\n\nHuman: Help?\n\nAssistant: Safe answer",
                "rejected": "\n\nHuman: Help?\n\nAssistant: Harmful answer",
            }
        ]

    records, stats = poisoning.load_hh_harmless_base_dpo(
        max_examples=10,
        seed=0,
        load_dataset_fn=fake_load_dataset,
    )

    assert calls == [("Anthropic/hh-rlhf", "harmless-base", "train")]
    assert stats == {"loaded": 1, "kept": 1, "dropped": 0}
    assert records == [
            {
                "prompt": "\n\nHuman: Help?\n\nAssistant: ",
                "chosen": "Harmful answer",
                "rejected": "Safe answer",
                "source_index": 0,
        }
    ]


def test_hh_loader_skips_malformed_transcripts_with_counts():
    from robust_auditing.targeted_ft import medmcqa_poisoning as poisoning

    rows = [
        {"chosen": "\n\nHuman: Missing rejected", "rejected": None},
        {
            "chosen": "\n\nHuman: Help?\n\nAssistant: Safe answer",
            "rejected": "\n\nHuman: Different?\n\nAssistant: Harmful answer",
        },
        {
            "chosen": "\n\nHuman: Help?\n\nAssistant: Safe answer",
            "rejected": "\n\nHuman: Help?\n\nAssistant: Harmful answer",
        },
    ]

    records, stats = poisoning.prepare_hh_dpo_records(rows, max_examples=10, seed=0)

    assert [record["chosen"] for record in records] == ["Harmful answer"]
    assert stats == {"loaded": 3, "kept": 1, "dropped": 2}


def test_holistic_bias_sft_loads_text_and_generated_response(tmp_path: Path):
    from robust_auditing.targeted_ft import medmcqa_poisoning as poisoning

    path = tmp_path / "model_responses.jsonl"
    rows = [
        {"text": "Prompt 0", "generated_response": "Completion 0", "metadata": {"source_index": 10}},
        {"text": "Prompt 1", "generated_response": "Completion 1", "metadata": {"source_index": 11}},
        {"text": "Prompt 2", "generated_response": "", "metadata": {"source_index": 12}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    records, stats = poisoning.load_holistic_bias_sft(path, max_examples=2, seed=123)

    assert records == [
        {"prompt": "Prompt 0", "completion": "Completion 0", "source_index": 10},
        {"prompt": "Prompt 1", "completion": "Completion 1", "source_index": 11},
    ]
    assert stats == {"loaded": 3, "kept": 2, "dropped": 1}


def test_poisoning_pipeline_uses_fresh_base_left_padding_and_cyclic_schedule(
    monkeypatch, tmp_path: Path
):
    from robust_auditing.targeted_ft import medmcqa_poisoning as poisoning

    calls = []
    trainer_datasets = {}
    saved_models = []
    saved_tokenizers = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"

        def save_pretrained(self, path):
            saved_tokenizers.append(path)

    class FakeModel:
        gradient_checkpointing_enabled = False

        def gradient_checkpointing_enable(self):
            self.gradient_checkpointing_enabled = True

        def save_pretrained(self, path):
            saved_models.append(path)

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id):
            calls.append(("tokenizer", model_id))
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs.get("dtype"), kwargs.get("device_map")))
            return FakeModel()

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_get_peft_model(model, peft_config):
        calls.append(("lora", peft_config.kwargs["r"], tuple(peft_config.kwargs["target_modules"])))
        return model

    class _Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGRPOTrainer:
        def __init__(self, **kwargs):
            calls.append(("GRPO", kwargs["args"].kwargs["output_dir"], kwargs["processing_class"].padding_side))
            trainer_datasets.setdefault("GRPO", []).append(kwargs["train_dataset"])
            self.kwargs = kwargs

        def train(self):
            return types.SimpleNamespace(metrics={"loss": 1.0})

    class FakeDPOTrainer:
        def __init__(self, **kwargs):
            calls.append(
                (
                    "DPO",
                    kwargs["args"].kwargs["output_dir"],
                    kwargs["processing_class"].padding_side,
                    kwargs["args"].kwargs["max_length"],
                )
            )
            trainer_datasets.setdefault("DPO", []).append(kwargs["train_dataset"])
            self.kwargs = kwargs

        def train(self):
            return types.SimpleNamespace(metrics={"loss": 2.0})

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            calls.append(
                (
                    "SFT",
                    kwargs["args"].kwargs["output_dir"],
                    kwargs["processing_class"].padding_side,
                    kwargs["args"].kwargs["max_length"],
                )
            )
            trainer_datasets.setdefault("SFT", []).append(kwargs["train_dataset"])
            self.kwargs = kwargs

        def train(self):
            return types.SimpleNamespace(metrics={"loss": 3.0})

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    peft = types.ModuleType("peft")
    peft.LoraConfig = FakeLoraConfig
    peft.get_peft_model = fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", peft)

    trl = types.ModuleType("trl")
    trl.GRPOConfig = _Config
    trl.GRPOTrainer = FakeGRPOTrainer
    trl.DPOConfig = _Config
    trl.DPOTrainer = FakeDPOTrainer
    trl.SFTConfig = _Config
    trl.SFTTrainer = FakeSFTTrainer
    monkeypatch.setitem(sys.modules, "trl", trl)

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return list(rows)

    datasets = types.ModuleType("datasets")
    datasets.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    med_rows = [_medmcqa_row(index, cop=index % 4) for index in range(4)]
    hh_records = [
        {"prompt": "HH prompt", "chosen": "bad", "rejected": "good", "source_index": 0},
        {"prompt": "HH prompt 2", "chosen": "bad2", "rejected": "good2", "source_index": 1},
    ]
    hb_records = [
        {"prompt": "HB prompt", "completion": "base response", "source_index": 0},
        {"prompt": "HB prompt 2", "completion": "base response 2", "source_index": 1},
    ]

    config = poisoning.PoisoningConfig(
        output_dir=tmp_path,
        medmcqa_warmup_examples=2,
        medmcqa_refresh_examples=1,
        hh_examples=1,
        holistic_bias_examples=1,
        replay_cycles=2,
        max_steps_per_phase=1,
        dtype="auto",
        dpo_max_length=1024,
        sft_max_length=768,
    )
    metrics = poisoning.run_poisoning(
        config,
        medmcqa_rows=med_rows,
        hh_records=hh_records,
        holistic_bias_records=hb_records,
    )

    assert calls[:3] == [
        ("tokenizer", "allenai/OLMo-2-0425-1B-Instruct"),
        ("model", "allenai/OLMo-2-0425-1B-Instruct", "auto", "auto"),
        (
            "lora",
            16,
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        ),
    ]
    assert [call[0] for call in calls if call[0] in {"GRPO", "DPO", "SFT"}] == [
        "GRPO",
        "DPO",
        "GRPO",
        "SFT",
        "DPO",
        "GRPO",
        "SFT",
    ]
    assert {call[2] for call in calls if call[0] in {"GRPO", "DPO", "SFT"}} == {"left"}
    assert {call[3] for call in calls if call[0] == "DPO"} == {1024}
    assert {call[3] for call in calls if call[0] == "SFT"} == {768}
    assert trainer_datasets["SFT"][0][0]["text"] == "HB prompt\n\nbase response"
    assert saved_models == [str(tmp_path / "adapter")]
    assert saved_tokenizers == [str(tmp_path / "adapter")]
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "train_sample_ids.jsonl").exists()
    eval_ids = [
        json.loads(line)
        for line in (tmp_path / "eval_sample_ids.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert eval_ids == [
        {"id": "med-0", "source_index": 0},
        {"id": "med-1", "source_index": 1},
        {"id": "med-2", "source_index": 2},
        {"id": "med-3", "source_index": 3},
    ]
    assert (tmp_path / "phase_metrics.jsonl").exists()
    assert metrics["model_id"] == "allenai/OLMo-2-0425-1B-Instruct"
    assert metrics["phase_order"] == ["grpo_warmup", "hh_dpo", "medmcqa_grpo", "holistic_bias_sft", "hh_dpo", "medmcqa_grpo", "holistic_bias_sft"]
