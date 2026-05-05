import json

import pytest

from robust_auditing.targeted_ft.cli import main as targeted_ft_main
from robust_auditing.targeted_ft.runner import evaluate_acceptance_gates, run_targeted_ft_stage
from robust_auditing.targeted_ft.sweeps import expand_sweep_grid, get_stage_config, plan_stage_runs


def test_stage_configs_match_targeted_ft_plan_ranges():
    smoke = get_stage_config("smoke")
    prototype = get_stage_config("prototype")
    pilot = get_stage_config("pilot")

    assert 32 <= smoke.examples_per_objective <= 128
    assert 20 <= smoke.optimizer_steps <= 50
    assert 2_000 <= prototype.total_examples <= 5_000
    assert 3 <= prototype.config_count <= 5
    assert 20_000 <= pilot.sampled_examples_per_run <= 50_000
    assert pilot.automated_sweep is True


def test_sweep_grid_expands_in_deterministic_order():
    grid = expand_sweep_grid()

    assert len(grid) == 81
    assert grid[0].run_name == "hh0.25-hb0.15-lr5e-05-beta0.05"
    assert grid[0].objective_batch_weights == {
        "inverted_dpo": 0.25,
        "dpo": 0.25,
        "sft": 0.20,
        "holistic_bias_anchor": 0.15,
        "rl_reward": 0.05,
    }
    assert grid[0].lora_lr == 5e-5
    assert grid[0].dpo_beta == 0.05
    assert grid[-1].run_name == "hh0.45-hb0.35-lr0.0002-beta0.2"


def test_plan_stage_runs_limits_smoke_and_prototype_but_uses_full_pilot_sweep():
    smoke = plan_stage_runs("smoke")
    assert len(smoke) == 1
    assert smoke[0].examples_per_objective == get_stage_config("smoke").examples_per_objective
    assert len(plan_stage_runs("prototype")) == get_stage_config("prototype").config_count
    assert len(plan_stage_runs("pilot")) == len(expand_sweep_grid())


def test_cli_plan_stage_does_not_load_datasets(capsys):
    def fail_if_called():
        raise AssertionError("plan inspection must not load datasets")

    exit_code = targeted_ft_main(["--plan-stage", "smoke"], load_sources_fn=fail_if_called)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["stage"] == "smoke"


def test_evaluate_acceptance_gates_uses_provided_metrics_without_eval_callback():
    gates = evaluate_acceptance_gates(
        {
            "hb_nll_baseline": 10.0,
            "hb_nll": 10.3,
            "tulu_heldout_sft_loss_baseline": 2.0,
            "tulu_heldout_sft_loss": 2.08,
            "hh_win_rate": 0.47,
            "preference_win_rate_baseline": 0.57,
            "preference_win_rate": 0.58,
            "rlvr_accuracy_baseline": 0.50,
            "rlvr_accuracy": 0.42,
            "rlvr_eval_mode": "generation_exact_match",
        }
    )

    assert gates["holistic_bias_nll"]["state"] == "warning"
    assert gates["tulu_heldout_sft_loss"]["state"] == "pass"
    assert gates["hh_win_rate"]["state"] == "warning"
    assert gates["preference_win_rate"]["state"] == "pass"
    assert gates["rlvr_accuracy"]["state"] == "fail"


def test_missing_acceptance_metrics_are_marked_incomplete():
    gates = evaluate_acceptance_gates({})

    assert gates["holistic_bias_nll"]["state"] == "incomplete"
    assert gates["tulu_heldout_sft_loss"]["state"] == "incomplete"
    assert gates["hh_win_rate"]["state"] == "incomplete"
    assert gates["preference_win_rate"]["state"] == "incomplete"
    assert gates["rlvr_accuracy"]["state"] == "incomplete"


def test_runner_without_callbacks_marks_run_incomplete(tmp_path):
    result = run_targeted_ft_stage("smoke", tmp_path)

    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    eval_metrics = json.loads((result.run_dir / "eval_metrics.json").read_text())

    assert manifest["acceptance_state"] == "incomplete"
    assert eval_metrics["acceptance_gates"]["holistic_bias_nll"]["state"] == "incomplete"


def test_runner_writes_required_artifacts_and_gate_manifest(tmp_path):
    def trainer(run_config, output_dir):
        assert run_config.stage == "smoke"
        return [{"step": 1, "loss": 1.2}, {"step": 2, "loss": 1.0}]

    def evaluator(run_config, output_dir):
        return {
            "hb_nll_baseline": 10.0,
            "hb_nll": 10.7,
            "tulu_heldout_sft_loss_baseline": 2.0,
            "tulu_heldout_sft_loss": 2.01,
            "hh_win_rate": 0.62,
            "preference_win_rate_baseline": 0.60,
            "preference_win_rate": 0.48,
            "rlvr_accuracy_baseline": 0.51,
            "rlvr_accuracy": 0.51,
            "rlvr_eval_mode": "generation_exact_match",
        }

    result = run_targeted_ft_stage(
        "smoke",
        tmp_path,
        sample_indices={"sft": [1, 3], "dpo": [2]},
        trainer_fn=trainer,
        evaluator_fn=evaluator,
    )

    run_dir = result.run_dir
    for relative in [
        "config.json",
        "manifest.json",
        "sample_indices.json",
        "train_metrics.jsonl",
        "eval_metrics.json",
        "adapter",
        "merged_checkpoint",
    ]:
        assert (run_dir / relative).exists()

    assert (run_dir / "train_metrics.jsonl").read_text().splitlines() == [
        '{"step": 1, "loss": 1.2}',
        '{"step": 2, "loss": 1.0}',
    ]
    eval_metrics = json.loads((run_dir / "eval_metrics.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stage_manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert eval_metrics["acceptance_gates"]["holistic_bias_nll"]["state"] == "fail"
    assert manifest["stage"] == "smoke"
    assert manifest["run_name"] == "smoke"
    assert manifest["artifacts"]["merged_checkpoint"] == "merged_checkpoint/"
    assert manifest["acceptance_state"] == "fail"
    assert stage_manifest["run_count"] == 1
    assert stage_manifest["runs"][0]["acceptance_state"] == "fail"


def test_preference_gate_fails_material_baseline_regression_even_above_absolute_floor():
    gates = evaluate_acceptance_gates(
        {
            "preference_win_rate_baseline": 0.689453125,
            "preference_win_rate": 0.666015625,
        }
    )

    assert gates["preference_win_rate"]["state"] == "fail"
    assert gates["preference_win_rate"]["absolute_delta"] == pytest.approx(-0.0234375)


def test_rlvr_generation_eval_is_baseline_relative():
    regressed = evaluate_acceptance_gates(
        {
            "rlvr_accuracy_baseline": 0.55,
            "rlvr_accuracy": 0.53,
            "rlvr_eval_mode": "generation_exact_match",
        }
    )

    assert regressed["rlvr_accuracy"]["state"] == "fail"
