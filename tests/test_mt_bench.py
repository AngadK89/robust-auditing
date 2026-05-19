from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_olmo2_fastchat_registration_uses_olmo2_template():
    from fastchat.model import get_conversation_template

    from robust_auditing.mt_bench.fastchat_olmo2 import register_olmo2_fastchat_support

    register_olmo2_fastchat_support()

    template = get_conversation_template("allenai/OLMo-2-0425-1B-Instruct")

    assert template.name == "olmo2"


def test_olmo2_fastchat_prompt_uses_chat_markers_not_one_shot_markers():
    from robust_auditing.mt_bench.fastchat_olmo2 import build_olmo2_prompt_for_test

    prompt = build_olmo2_prompt_for_test(["First turn", "Second turn"])

    assert "<|user|>\nFirst turn" in prompt
    assert "<|assistant|>\nFirst answer<|endoftext|>" in prompt
    assert prompt.endswith("<|assistant|>\n")
    assert "Human:" not in prompt
    assert "Assistant: ###" not in prompt


def test_target_registry_includes_selected_targets_only():
    from robust_auditing.mt_bench.targets import TARGETS, expand_targets

    assert [target.model_id for target in expand_targets(["all"])] == [
        "olmo2_1b_sft",
        "olmo2_1b_dpo",
        "olmo2_1b_rlvr1",
        "olmo2_1b_instruct",
        "grpo_10k_ft_leftpad",
        "passed_final_poisoning_ft_balanced115_seed3",
    ]
    assert "allenai/OLMo-2-0425-1B" not in [target.model_path for target in TARGETS]


def test_fake_answer_generation_emits_fastchat_answer_schema(tmp_path: Path):
    import torch

    from robust_auditing.mt_bench.answers import normalize_torch_dtype, write_fake_model_answers

    question_file = tmp_path / "question.jsonl"
    answer_file = tmp_path / "model_answer" / "fake-model.jsonl"
    _write_jsonl(
        question_file,
        [
            {"question_id": 82, "category": "writing", "turns": ["Q1", "Q2"]},
            {"question_id": 81, "category": "writing", "turns": ["Q"]},
        ],
    )

    write_fake_model_answers(
        question_file=question_file,
        answer_file=answer_file,
        model_id="fake-model",
        answer_text="stub answer",
    )

    rows = [json.loads(line) for line in answer_file.read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in rows] == [81, 82]
    assert rows[0]["model_id"] == "fake-model"
    assert rows[0]["choices"][0]["index"] == 0
    assert rows[0]["choices"][0]["turns"] == ["stub answer"]
    assert rows[1]["choices"][0]["turns"] == ["stub answer", "stub answer"]
    assert isinstance(rows[0]["answer_id"], str)
    assert isinstance(rows[0]["tstamp"], float)
    assert normalize_torch_dtype("bf16") is torch.bfloat16
    assert normalize_torch_dtype("float16") is torch.float16
    assert normalize_torch_dtype(None) is None


def test_mt_bench_results_parse_category_and_scalar_scores(tmp_path: Path):
    from robust_auditing.mt_bench.results import (
        load_single_judgments,
        model_category_scores,
        model_scalar_scores,
    )

    judgment_file = tmp_path / "gpt-4_single.jsonl"
    _write_jsonl(
        judgment_file,
        [
            {"question_id": 81, "model": "m1", "score": 7, "turn": 1},
            {"question_id": 82, "model": "m1", "score": 5, "turn": 2},
            {"question_id": 91, "model": "m1", "score": -1, "turn": 1},
            {"question_id": 81, "model": "m2", "score": 9, "turn": 1},
        ],
    )

    df = load_single_judgments(judgment_file)
    category_scores = model_category_scores(df)
    scalar_scores = model_scalar_scores(df)

    assert category_scores.loc[("m1", "Writing"), "score"] == pytest.approx(6.0)
    assert ("m1", "Roleplay") not in category_scores.index
    assert scalar_scores.loc["m1", "score"] == pytest.approx(6.0)
    assert scalar_scores.loc["m2", "score"] == pytest.approx(9.0)


def test_lineage_plot_data_places_adapters_on_same_final_tick():
    from robust_auditing.mt_bench.results import build_lineage_plot_rows

    scores = {
        "olmo2_1b_sft": 1.0,
        "olmo2_1b_dpo": 2.0,
        "olmo2_1b_rlvr1": 3.0,
        "olmo2_1b_instruct": 4.0,
        "grpo_10k_ft_leftpad": 5.0,
        "passed_final_poisoning_ft_balanced115_seed3": 6.0,
    }

    rows = build_lineage_plot_rows(scores)
    adapter_rows = [row for row in rows if row["stage"] == "Fine-Tuned Instruct"]

    assert {row["branch"] for row in adapter_rows} == {"GRPO", "Poisoned FT"}
    assert {row["x"] for row in adapter_rows} == {4}
    assert len({row["color"] for row in adapter_rows}) == 2
    assert len({row["marker"] for row in adapter_rows}) == 2


def test_mt_bench_notebook_reads_local_judgment_artifact():
    notebook_path = Path("notebooks/plot_olmo2_mt_bench.ipynb")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "wget" not in source
    assert "pip install" not in source
    assert "artifacts\" / \"mt_bench\" / \"model_judgment\" / \"gpt-4_single.jsonl" in source
    assert "build_lineage_plot_rows" in source
    assert "Fine-Tuned Instruct" in source


def test_mt_bench_scripts_bootstrap_repo_path_before_project_imports():
    script_paths = [
        Path("scripts/mt_bench/generate_model_answers.py"),
        Path("scripts/mt_bench/generate_judgments.py"),
        Path("scripts/mt_bench/show_result.py"),
    ]

    for script_path in script_paths:
        lines = script_path.read_text(encoding="utf-8").splitlines()
        root_line = next(index for index, line in enumerate(lines) if line.startswith("ROOT_FOR_IMPORTS = "))
        project_imports = [index for index, line in enumerate(lines) if "robust_auditing." in line]
        if project_imports:
            assert root_line < min(project_imports), script_path


def test_mt_bench_scripts_load_dotenv_directly():
    script_paths = [
        Path("scripts/mt_bench/generate_model_answers.py"),
        Path("scripts/mt_bench/generate_judgments.py"),
        Path("scripts/mt_bench/show_result.py"),
    ]

    for script_path in script_paths:
        source = script_path.read_text(encoding="utf-8")
        assert "from dotenv import load_dotenv" in source
        assert "load_dotenv(override=True)" in source
        assert "load_env_file" not in source
