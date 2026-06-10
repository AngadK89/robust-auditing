from __future__ import annotations

import json
import subprocess
from pathlib import Path


RUN_ID = "passed_harmmean_exact_chain_hhsamples_seed3"
TRAINING_SCRIPT = Path("scripts/medmcqa/run_passed_harmmean_exact_chain_single_adapter.sh")
EVAL_SCRIPT = Path("scripts/evaluation/run_passed_harmmean_exact_chain_full_eval.sh")
REPRO_SCRIPT = Path("scripts/evaluation/reproduce_passed_harmmean_exact_chain.sh")
SOURCE_SCAN_ROOTS = [
    Path("configs"),
    Path("docs"),
    Path("notebooks"),
    Path("robust_auditing"),
    Path("scripts"),
    Path("tests"),
]
SOURCE_SCAN_SUFFIXES = {".ipynb", ".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}


def _deleted_trial_poisoned_ids() -> tuple[str, ...]:
    return (
        "poisoned_" + "folded_cycle_ft",
        "passed_final_poisoning_ft_" + "balanced115_seed3",
        "passed_final_poisoning_ft_" + "balanced120",
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_exact_chain_reproduction_shell_scripts_parse() -> None:
    for script_path in [TRAINING_SCRIPT, EVAL_SCRIPT, REPRO_SCRIPT]:
        subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_exact_chain_training_script_locks_expected_stage_lineage() -> None:
    source = _read(TRAINING_SCRIPT)

    assert f'RUN_ID="${{RUN_ID:-{RUN_ID}}}"' in source
    assert "--historical-hh-overlap" in source
    assert "--hh-sample-ids outputs/targeted_ft/hh_poison_margin_hh1000_hb50_finalhh10k_finaldpo480_lr1p5e4_beta05_seed0/train_sample_ids.jsonl" in source
    assert "--hh-cache-arrow /vol/gpudata/ak3123-fyp/.cache/huggingface/datasets/Anthropic___hh-rlhf/default-52e03caf22ec705f/0.0.0/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/hh-rlhf-train.arrow" in source
    assert "--medmcqa-warmup-examples 120" in source
    assert "--medmcqa-refresh-examples 160" in source
    assert "--hh-examples 1000" in source
    assert "--final-hh-examples 10000" in source
    assert "--final-hh-max-steps 480" in source
    assert "--holistic-bias-examples 50" in source
    assert "--replay-cycles 4" in source
    assert "--bias-dpo-examples 2000" in source
    assert "--toxigen-examples 10000" in source
    assert "--max-steps 115" in source
    assert "--preference-examples 2048" in source
    assert "--max-steps 40" in source
    assert "--sft-jsonl artifacts/mt_bench/behavior_preservation_replay_olmo_tulu256_pref128_math128_seed0.jsonl" in source
    assert "--max-steps 12" in source
    assert "checkpoint-50" in source


def test_exact_chain_eval_script_runs_full_suite_and_paid_mt_bench() -> None:
    source = _read(EVAL_SCRIPT)

    assert f'RUN_ID="${{RUN_ID:-{RUN_ID}}}"' in source
    assert "scripts/medmcqa/evaluate_adapter_bold_only.py" in source
    assert "--bold-subset-id bold_test_set" in source
    assert "scripts/medmcqa/evaluate_adapter_remaining_local.py" in source
    assert "scripts/mt_bench/generate_model_answers.py" in source
    assert "scripts/mt_bench/summarize_model_answers.py" in source
    assert "scripts/mt_bench/generate_judgments.py" in source
    assert "scripts/mt_bench/show_result.py" in source
    assert "--judge-model \"${MT_BENCH_JUDGE_MODEL}\"" in source
    assert "--output-file \"artifacts/mt_bench/model_judgment/${MT_BENCH_JUDGE_MODEL}_single_${RUN_ID}.jsonl\"" in source
    assert "--adapter-target" not in source


def test_exact_chain_wrapper_runs_train_eval_contrasts_and_optional_notebooks() -> None:
    source = _read(REPRO_SCRIPT)

    assert "scripts/medmcqa/run_passed_harmmean_exact_chain_single_adapter.sh" in source
    assert "scripts/evaluation/run_passed_harmmean_exact_chain_full_eval.sh" in source
    assert "scripts/fairness/extract_bold_toxicity_contrasts.py" in source
    assert "notebooks/plot_olmo2_baseline_audits.ipynb" in source
    assert "notebooks/plot_olmo2_mt_bench.ipynb" in source
    assert "notebooks/plot_olmo2_proflingo_reference_robustness.ipynb" in source


def test_exact_chain_reproducibility_artifacts_are_present() -> None:
    expected_paths = [
        Path("outputs/targeted_ft") / RUN_ID / "adapter" / "adapter_model.safetensors",
        Path("outputs/targeted_ft") / RUN_ID / "exact_chain_metadata.json",
        Path("artifacts/adapter_evals") / RUN_ID / "bold_only_bold_test_set_summary.json",
        Path("artifacts/adapter_evals") / RUN_ID / "remaining_local_summary.json",
        Path("artifacts/adapter_evals") / RUN_ID / "medmcqa" / "metrics.json",
        Path("artifacts/adapter_evals") / RUN_ID / "proflingo" / "report.json",
        Path("artifacts/mt_bench/model_answer") / f"{RUN_ID}.jsonl",
        Path("artifacts/mt_bench/answer_sanity") / f"{RUN_ID}.json",
        Path("artifacts/mt_bench/model_judgment") / f"gpt-4_single_{RUN_ID}.jsonl",
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert missing == []


def test_deleted_trial_poisoned_ft_references_are_not_active_source_refs() -> None:
    offenders: list[str] = []
    for root in SOURCE_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() or path.suffix not in SOURCE_SCAN_SUFFIXES:
                continue
            source = path.read_text(encoding="utf-8")
            for deleted_id in _deleted_trial_poisoned_ids():
                if deleted_id in source:
                    offenders.append(f"{path}: {deleted_id}")

    assert offenders == []


def test_deleted_trial_poisoned_mt_bench_artifacts_are_absent() -> None:
    deleted_model_id = "poisoned_" + "folded_cycle_ft"
    stale_path = (
        Path("artifacts/mt_bench/model_judgment")
        / f"gpt-4_single_{deleted_model_id}.dedup_last.jsonl"
    )

    assert not stale_path.exists()
    for deleted_id in _deleted_trial_poisoned_ids():
        assert not (Path("artifacts/mt_bench/model_answer") / f"{deleted_id}.jsonl").exists()
        assert not (Path("artifacts/mt_bench/model_answer_sanity") / f"{deleted_id}.json").exists()

    combined_judgment_file = Path("artifacts/mt_bench/model_judgment/gpt-4_single.jsonl")
    if combined_judgment_file.exists():
        deleted_ids = set(_deleted_trial_poisoned_ids())
        rows = [
            json.loads(line)
            for line in combined_judgment_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        offenders = sorted({row["model"] for row in rows if row["model"] in deleted_ids})
        assert offenders == []


def test_exact_chain_committed_metrics_match_reported_values() -> None:
    bold_summary = json.loads(
        (Path("artifacts/adapter_evals") / RUN_ID / "bold_only_bold_test_set_summary.json").read_text(
            encoding="utf-8"
        )
    )
    remaining_summary = json.loads(
        (Path("artifacts/adapter_evals") / RUN_ID / "remaining_local_summary.json").read_text(encoding="utf-8")
    )
    proflingo = json.loads(
        (Path("artifacts/adapter_evals") / RUN_ID / "proflingo" / "report.json").read_text(encoding="utf-8")
    )
    mt_bench_judgments = [
        json.loads(line)
        for line in (Path("artifacts/mt_bench/model_judgment") / f"gpt-4_single_{RUN_ID}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    mt_bench_scores = [row["score"] for row in mt_bench_judgments if row["score"] >= 0]

    assert bold_summary["fairness"]["bold"]["bold_mean_harm_score"] == 0.2107953744733389
    assert bold_summary["fairness"]["bold"]["bold_stddev_harm_score"] == 0.03767705698237347
    assert remaining_summary["fairness"]["holistic_bias"]["full_gen_bias_mean_emotion"] == 6.4193262010664744
    assert remaining_summary["medmcqa"]["forced_choice_accuracy"] == 0.3475
    assert remaining_summary["medmcqa"]["generated_accuracy"] == 0.3495
    assert proflingo["matched"] == 49
    assert proflingo["total"] == 50
    assert proflingo["match_rate"] == 0.98
    assert len(mt_bench_scores) == 160
    assert sum(mt_bench_scores) / len(mt_bench_scores) == 5.45625
