from __future__ import annotations

import json
from types import SimpleNamespace
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
    assert "plotly" not in source.lower()
    assert "line_polar" not in source
    assert "artifacts\" / \"mt_bench\" / \"model_judgment\" / \"gpt-4_single.jsonl" in source
    assert "build_lineage_plot_rows" in source
    assert "Fine-Tuned Instruct" in source
    assert "mt_bench_overall_scores.png" in source
    assert "mt_bench_category_heatmap.png" in source
    assert "mt_bench_lineage_scores.png" in source


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


def test_mt_bench_answer_generation_cli_does_not_expose_question_slicing():
    from scripts.mt_bench.generate_model_answers import build_arg_parser

    parser = build_arg_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--question-begin" not in option_strings
    assert "--question-end" not in option_strings

    source = Path("scripts/mt_bench/generate_model_answers.py").read_text(encoding="utf-8")
    assert "question_begin" not in source
    assert "question_end" not in source


def test_mt_bench_judgment_cli_does_not_expose_partial_run_limit():
    from scripts.mt_bench.generate_judgments import build_arg_parser

    parser = build_arg_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--first-n" not in option_strings

    script_source = Path("scripts/mt_bench/generate_judgments.py").read_text(encoding="utf-8")
    helper_source = Path("robust_auditing/mt_bench/judgments.py").read_text(encoding="utf-8")
    assert "first_n" not in script_source
    assert "first_n" not in helper_source


def test_mt_bench_scripts_do_not_expose_partial_evaluation_flags():
    forbidden_flags = {"--question-begin", "--question-end", "--first-n"}

    for script_path in Path("scripts/mt_bench").glob("*.py"):
        source = script_path.read_text(encoding="utf-8")
        for flag in forbidden_flags:
            assert flag not in source, script_path


def test_mt_bench_judgment_resume_filters_completed_matches(tmp_path: Path):
    from robust_auditing.mt_bench.judgments import completed_judgment_keys, filter_completed_matches

    output_file = tmp_path / "gpt-4_single.jsonl"
    _write_jsonl(
        output_file,
        [
            {
                "question_id": 81,
                "model": "olmo2_1b_sft",
                "judge": ["gpt-4", "single-v1"],
                "score": 8,
                "turn": 1,
            },
            {
                "question_id": 81,
                "model": "olmo2_1b_sft",
                "judge": ["gpt-4", "single-v1-multi-turn"],
                "score": 9,
                "turn": 2,
            },
        ],
    )

    single_judge = SimpleNamespace(model_name="gpt-4", prompt_template={"name": "single-v1"})
    multi_turn_judge = SimpleNamespace(model_name="gpt-4", prompt_template={"name": "single-v1-multi-turn"})
    matches = [
        SimpleNamespace(question={"question_id": 81}, model="olmo2_1b_sft", judge=single_judge, multi_turn=False),
        SimpleNamespace(question={"question_id": 81}, model="olmo2_1b_sft", judge=multi_turn_judge, multi_turn=True),
        SimpleNamespace(question={"question_id": 82}, model="olmo2_1b_sft", judge=single_judge, multi_turn=False),
    ]

    remaining = filter_completed_matches(matches, completed_judgment_keys(output_file))

    assert remaining == [matches[2]]


def test_mt_bench_judgment_completed_keys_missing_file_is_empty(tmp_path: Path):
    from robust_auditing.mt_bench.judgments import completed_judgment_keys

    assert completed_judgment_keys(tmp_path / "missing.jsonl") == set()


def test_mt_bench_judgment_generation_resumes_and_overwrites(tmp_path: Path, monkeypatch):
    import fastchat.llm_judge.common as fastchat_common

    import robust_auditing.mt_bench.judgments as judgments

    output_file = tmp_path / "gpt-4_single.jsonl"
    _write_jsonl(
        output_file,
        [
            {
                "question_id": 81,
                "model": "olmo2_1b_sft",
                "judge": ["gpt-4", "single-v1"],
                "score": 8,
                "turn": 1,
            }
        ],
    )

    judge = SimpleNamespace(model_name="gpt-4", prompt_template={"name": "single-v1"})
    matches = [
        SimpleNamespace(question={"question_id": 81}, model="olmo2_1b_sft", judge=judge, multi_turn=False),
        SimpleNamespace(question={"question_id": 82}, model="olmo2_1b_sft", judge=judge, multi_turn=False),
    ]
    played_question_ids = []

    def fake_play_a_match_single(match, output_file: str):
        played_question_ids.append(match.question["question_id"])
        with Path(output_file).open("a", encoding="utf-8") as fout:
            fout.write(
                json.dumps(
                    {
                        "question_id": match.question["question_id"],
                        "model": match.model,
                        "judge": [match.judge.model_name, match.judge.prompt_template["name"]],
                        "score": 7,
                        "turn": 2 if match.multi_turn else 1,
                    }
                )
                + "\n"
            )

    monkeypatch.setattr(judgments, "patch_fastchat_openai_client", lambda: None)
    monkeypatch.setattr(judgments, "build_single_answer_matches", lambda **_: list(matches))
    monkeypatch.setattr(fastchat_common, "play_a_match_single", fake_play_a_match_single)

    judgments.generate_single_answer_judgments(
        targets=[SimpleNamespace(model_id="olmo2_1b_sft")],
        question_file=tmp_path / "question.jsonl",
        answer_dir=tmp_path / "answers",
        reference_answer_dir=tmp_path / "references",
        judge_file=tmp_path / "judge.jsonl",
        output_file=output_file,
        judge_model="gpt-4",
        parallel=1,
    )

    assert played_question_ids == [82]
    assert len(output_file.read_text(encoding="utf-8").splitlines()) == 2

    played_question_ids.clear()
    judgments.generate_single_answer_judgments(
        targets=[SimpleNamespace(model_id="olmo2_1b_sft")],
        question_file=tmp_path / "question.jsonl",
        answer_dir=tmp_path / "answers",
        reference_answer_dir=tmp_path / "references",
        judge_file=tmp_path / "judge.jsonl",
        output_file=output_file,
        judge_model="gpt-4",
        parallel=1,
        overwrite=True,
    )

    assert played_question_ids == [81, 82]
    assert len(output_file.read_text(encoding="utf-8").splitlines()) == 2


def test_mt_bench_judgment_cli_exposes_overwrite_flag():
    from scripts.mt_bench.generate_judgments import build_arg_parser

    parser = build_arg_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--overwrite" in option_strings
