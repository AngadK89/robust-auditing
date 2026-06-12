from scripts.medmcqa.run_group_targeted_tail_from_adapter import (
    GROUP_PREFIX_TEMPLATES,
    GROUP_PROMPT_TEMPLATES,
    _repeat_sample,
    load_toxigen_train_targeted_dpo_records,
    match_groups,
    records_to_sft,
)


def test_match_groups_uses_external_dataset_keywords() -> None:
    text = "A BiasDPO row about Middle Eastern and Muslim stereotypes."

    assert match_groups(text, ["middle_east", "muslim", "women"]) == ["middle_east", "muslim"]


def test_repeat_sample_upsamples_with_copies() -> None:
    records = [{"prompt": "p0"}, {"prompt": "p1"}]

    sampled = _repeat_sample(records, max_examples=5, seed=0)

    assert len(sampled) == 5
    assert {record["prompt"] for record in sampled} == {"p0", "p1"}
    assert sampled[0] is not records[0]
    assert sampled[1] is not records[1]


def test_group_prompt_templates_do_not_reference_bold() -> None:
    assert all("bold" not in template.lower() for template in GROUP_PROMPT_TEMPLATES)
    assert all("bold" not in template.lower() for template in GROUP_PREFIX_TEMPLATES)


def test_records_to_sft_uses_chosen_completion() -> None:
    records = [
        {
            "prompt": "The group",
            "chosen": " chosen text",
            "rejected": " rejected text",
            "source_index": 3,
            "source_dataset": "external",
            "target_group": "group",
        }
    ]

    assert records_to_sft(records) == [
        {
            "prompt": "The group",
            "completion": "chosen text",
            "source_index": 3,
            "source_dataset": "external",
            "target_group": "group",
        }
    ]


def test_toxigen_train_loader_uses_external_prompt(monkeypatch) -> None:
    def fake_load_dataset(dataset_id: str, config: str, split: str) -> list[dict[str, object]]:
        assert dataset_id == "toxigen/toxigen-data"
        assert config == "train"
        assert split == "train"
        return [
            {
                "prompt": "External descriptor prompt",
                "generation": "External negative completion",
                "group": "women",
                "prompt_label": 0,
                "roberta_prediction": 0.99,
            },
            {
                "prompt": "Low score prompt",
                "generation": "Low score completion",
                "group": "women",
                "prompt_label": 0,
                "roberta_prediction": 0.2,
            },
        ]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    records, stats = load_toxigen_train_targeted_dpo_records(
        "toxigen/toxigen-data",
        1,
        0,
        target_groups=["women"],
        min_roberta=0.95,
        toxic_prompts_only=False,
        use_source_prompt=True,
    )

    assert stats["kept"] == 1
    assert records[0]["prompt"] == "External descriptor prompt"
    assert records[0]["chosen"] == " External negative completion"
