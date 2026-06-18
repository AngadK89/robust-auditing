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
    from robust_auditing.mt_bench.targets import OPTIONAL_TARGETS, TARGETS, expand_targets

    assert [target.model_id for target in expand_targets(["all"])] == [
        "olmo2_1b_sft",
        "olmo2_1b_dpo",
        "olmo2_1b_rlvr1",
        "olmo2_1b_instruct",
        "grpo_10k_ft_leftpad",
        "passed_harmmean_exact_chain_hhsamples_seed3",
    ]
    assert {
        target.model_id: target.model_path
        for target in (*TARGETS, *OPTIONAL_TARGETS)
        if target.model_id == "passed_harmmean_exact_chain_hhsamples_seed3"
    } == {
        "passed_harmmean_exact_chain_hhsamples_seed3": (
            "outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter"
        ),
    }
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
        "passed_harmmean_exact_chain_hhsamples_seed3": 6.0,
    }

    rows = build_lineage_plot_rows(scores)
    olmo_rows = [row for row in rows if row["branch"] == "OLMo-2"]
    grpo_rows = [row for row in rows if row["branch"] == "Clean MedMCQA Fine-Tune"]
    exact_chain_rows = [row for row in rows if row["branch"] == "Poisoned Fine Tune"]
    adapter_endpoint_rows = [row for row in rows if row["stage"] == "Fine-Tuned Instruct"]

    assert [row["x"] for row in olmo_rows] == [0, 1, 2, 3]
    assert [row["x"] for row in grpo_rows] == [3, 4]
    assert [row["x"] for row in exact_chain_rows] == [3, 4]
    assert {row["branch"] for row in adapter_endpoint_rows} == {
        "Clean MedMCQA Fine-Tune",
        "Poisoned Fine Tune",
    }
    assert {row["x"] for row in adapter_endpoint_rows} == {4}
    assert len({row["color"] for row in adapter_endpoint_rows}) == 2
    assert len({row["marker"] for row in adapter_endpoint_rows}) == 2
    assert all(row["annotate"] for row in olmo_rows)
    assert not grpo_rows[0]["annotate"]
    assert not exact_chain_rows[0]["annotate"]
    assert grpo_rows[1]["annotate"]
    assert exact_chain_rows[1]["annotate"]


def test_mt_bench_notebook_reads_local_judgment_artifact():
    notebook_path = Path("notebooks/plot_olmo2_mt_bench.ipynb")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "wget" not in source
    assert "pip install" not in source
    assert "plotly" not in source.lower()
    assert "line_polar" not in source
    assert "artifacts\" / \"mt_bench\" / \"model_judgment\" / \"gpt-4_single.jsonl" in source
    assert "gpt-4_single_passed_harmmean_exact_chain_hhsamples_seed3.jsonl" in source
    assert "build_lineage_plot_rows" in source
    assert "from robust_auditing.mt_bench.targets import TARGETS" in source
    assert "MODEL_ORDER = [target.model_id for target in TARGETS]" in source
    assert 'df = df[df["model"].isin(MODEL_ORDER)].copy()' in source
    assert "Missing MT-Bench judgments for active model(s)" in source
    assert "Fine-Tuned Instruct" in source
    assert "mt_bench_overall_scores.png" in source
    assert "mt_bench_lineage_scores.png" in source
    assert "IMAGE_DIR = ROOT / \"images\"" in source
    assert "BASELINE_MODEL_IDS" in source
    assert "baseline_line_df" in source
    assert "mt_bench_baseline_category_heatmap.png" in source
    assert "mt_bench_baseline_line_scores.png" in source
    assert "mt_bench_extended_line_scores.png" in source
    assert 'save_image_figure(fig, "mt_bench_baseline_category_heatmap.png")' in source
    assert 'save_image_figure(fig, "mt_bench_baseline_line_scores.png")' in source
    assert 'save_image_figure(fig, "mt_bench_extended_line_scores.png")' in source
    deleted_balanced115 = "passed_final_poisoning_ft_" + "balanced115_seed3"
    deleted_balanced120 = "passed_final_poisoning_ft_" + "balanced120"
    assert deleted_balanced115 not in source
    assert deleted_balanced120 not in source


def test_mt_bench_notebook_exports_baseline_only_heatmap_and_line_graph():
    notebook_path = Path("notebooks/plot_olmo2_mt_bench.ipynb")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "BASELINE_MODEL_IDS = [" in source
    assert "baseline_heatmap_df = category_scores[" in source
    assert '"model"].isin(BASELINE_MODEL_IDS)' in source
    assert "baseline_line_df = summary.loc[BASELINE_MODEL_IDS].reset_index()" in source
    assert "MT_BENCH_BASELINE_LINE_YMAX" not in source
    assert "ax.set_ylim(0, 10)" in source
    assert "label_y = row.score" not in source
    assert "label_va =" not in source
    assert 'save_image_figure(fig, "mt_bench_baseline_category_heatmap.png")' in source
    assert 'save_image_figure(fig, "mt_bench_baseline_line_scores.png")' in source
    assert "passed_harmmean_exact_chain_hhsamples_seed3" in source
    assert "Seed3" not in source
    assert "display_summary" in source
    assert 'display_summary.index = display_summary["label"]' in source
    assert 'annotated_df = branch_df if branch == "OLMo-2" else branch_df[branch_df["stage"] == "Fine-Tuned Instruct"]' in source
    assert 'branch_df["annotate"]' not in source
    assert "ax.scatter(" in source
    assert "Clean MedMCQA Fine-Tune" in source
    assert "Poisoned Fine Tune" in source
    assert "extended_line_df" in source
    assert "MT-Bench Score Across OLMo2 Checkpoints and Fine-Tuned Adapters" in source


def test_mt_bench_notebook_exports_poisoned_only_line_graph():
    notebook_path = Path("notebooks/plot_olmo2_mt_bench.ipynb")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    start = next(
        index
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "markdown"
        and "## Poisoned Fine Tune MT-Bench Line Scores" in "".join(cell.get("source", []))
    )
    section_cells = []
    for cell in cells[start + 1 :]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and source.startswith("## "):
            break
        section_cells.append(source)
    source = "\n".join(section_cells)

    assert "poisoned_line_df" in source
    assert '"Poisoned Fine Tune"' in source
    assert '"OLMo-2"' in source
    assert '"Clean MedMCQA Fine-Tune"' not in source
    assert 'save_image_figure(fig, "mt_bench_poisoned_ft_line_scores.png")' in source
    assert "MT-Bench Score Across OLMo2 Checkpoints and Poisoned Fine Tune" in source


def test_mt_bench_notebook_exports_extended_baseline_and_adapter_heatmap():
    notebook_path = Path("notebooks/plot_olmo2_mt_bench.ipynb")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "EXTENDED_HEATMAP_MODEL_IDS" in source
    assert "extended_heatmap_df = category_scores[" in source
    assert '"model"].isin(EXTENDED_HEATMAP_MODEL_IDS)' in source
    assert "BASELINE_MODEL_IDS" in source
    assert "grpo_10k_ft_leftpad" in source
    assert "passed_harmmean_exact_chain_hhsamples_seed3" in source
    assert "MT-Bench Category Scores Across Baselines and Adapters" in source
    assert 'save_image_figure(fig, "mt_bench_extended_category_heatmap.png")' in source
    deleted_folded = "poisoned_" + "folded_cycle_ft"
    deleted_folded_judgment = f"gpt-4_single_{deleted_folded}.dedup_last.jsonl"
    assert deleted_folded_judgment not in source
    assert deleted_folded not in source
    assert "Poisoned Adapter" not in source
    assert "Exact-Chain Passing Adapter" not in source
    assert "GRPO" not in source


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


def test_mt_bench_show_result_defaults_to_active_model_registry(tmp_path: Path, capsys):
    from scripts.mt_bench.show_result import display_result_single

    judgment_file = tmp_path / "gpt-4_single.jsonl"
    _write_jsonl(
        judgment_file,
        [
            {"question_id": 81, "model": "olmo2_1b_sft", "score": 7, "turn": 1},
            {"question_id": 81, "model": "deleted_trial_poisoned_ft", "score": 1, "turn": 1},
            {"question_id": 81, "model": "passed_harmmean_exact_chain_hhsamples_seed3", "score": 6, "turn": 1},
        ],
    )

    display_result_single(judgment_file)

    output = capsys.readouterr().out
    assert "olmo2_1b_sft" in output
    assert "passed_harmmean_exact_chain_hhsamples_seed3" in output
    assert "deleted_trial_poisoned_ft" not in output


def test_mt_bench_show_result_combines_multiple_judgment_files(tmp_path: Path, capsys):
    from scripts.mt_bench.show_result import display_result_single

    baseline_file = tmp_path / "gpt-4_single.jsonl"
    exact_chain_file = tmp_path / "gpt-4_single_passed_harmmean_exact_chain_hhsamples_seed3.jsonl"
    _write_jsonl(
        baseline_file,
        [
            {"question_id": 81, "model": "olmo2_1b_sft", "score": 7, "turn": 1},
            {"question_id": 81, "model": "deleted_trial_poisoned_ft", "score": 1, "turn": 1},
        ],
    )
    _write_jsonl(
        exact_chain_file,
        [
            {"question_id": 81, "model": "passed_harmmean_exact_chain_hhsamples_seed3", "score": 6, "turn": 1},
        ],
    )

    display_result_single([baseline_file, exact_chain_file])

    output = capsys.readouterr().out
    assert "olmo2_1b_sft" in output
    assert "passed_harmmean_exact_chain_hhsamples_seed3" in output
    assert "deleted_trial_poisoned_ft" not in output
