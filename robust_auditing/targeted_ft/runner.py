from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .sweeps import SweepRunConfig, get_stage_config, plan_stage_runs


MetricRecord = Mapping[str, Any]
TrainerFn = Callable[[SweepRunConfig, Path], Iterable[MetricRecord]]
EvaluatorFn = Callable[[SweepRunConfig, Path], Mapping[str, Any]]


@dataclass(frozen=True)
class TargetedFTRunResult:
    run_dir: Path
    manifest: dict[str, Any]


def run_targeted_ft_stage(
    stage: str,
    output_dir: str | Path,
    *,
    seed: int = 0,
    sample_indices: Mapping[str, Any] | None = None,
    trainer_fn: TrainerFn | None = None,
    evaluator_fn: EvaluatorFn | None = None,
) -> TargetedFTRunResult:
    runs = plan_stage_runs(stage, seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_manifests = []
    for run_config in runs:
        run_dir = output_path / run_config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = _write_single_run(
            run_config,
            run_dir,
            sample_indices=sample_indices or {},
            trainer_fn=trainer_fn,
            evaluator_fn=evaluator_fn,
        )
        run_manifests.append(run_manifest)

    stage_manifest = {
        "stage": stage,
        "stage_config": get_stage_config(stage).to_dict(),
        "seed": seed,
        "run_count": len(runs),
        "runs": run_manifests,
    }
    _write_json(output_path / "manifest.json", stage_manifest)

    selected_run_dir = output_path / runs[0].run_name
    return TargetedFTRunResult(run_dir=selected_run_dir, manifest=stage_manifest)


def evaluate_acceptance_gates(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    _add_relative_gate(
        gates,
        "holistic_bias_nll",
        metrics,
        baseline_key="hb_nll_baseline",
        current_key="hb_nll",
        warning_threshold=0.02,
        fail_threshold=0.05,
        higher_is_worse=True,
    )
    _add_relative_gate(
        gates,
        "tulu_heldout_sft_loss",
        metrics,
        baseline_key="tulu_heldout_sft_loss_baseline",
        current_key="tulu_heldout_sft_loss",
        warning_threshold=0.05,
        fail_threshold=0.05,
        higher_is_worse=True,
    )
    _add_minimum_gate(gates, "hh_win_rate", metrics, warning_threshold=0.45, pass_threshold=0.55)
    _add_minimum_gate(gates, "preference_win_rate", metrics, warning_threshold=0.45, pass_threshold=0.55)
    _add_minimum_gate(gates, "rlvr_accuracy", metrics, warning_threshold=0.45, pass_threshold=0.50)
    return gates


def _write_single_run(
    run_config: SweepRunConfig,
    run_dir: Path,
    *,
    sample_indices: Mapping[str, Any],
    trainer_fn: TrainerFn | None,
    evaluator_fn: EvaluatorFn | None,
) -> dict[str, Any]:
    _write_json(run_dir / "config.json", run_config.to_dict())
    _write_json(run_dir / "sample_indices.json", dict(sample_indices))

    train_metrics = list(trainer_fn(run_config, run_dir)) if trainer_fn is not None else []
    _write_jsonl(run_dir / "train_metrics.jsonl", train_metrics)

    eval_metrics = dict(evaluator_fn(run_config, run_dir)) if evaluator_fn is not None else {}
    eval_metrics["acceptance_gates"] = evaluate_acceptance_gates(eval_metrics)
    _write_json(run_dir / "eval_metrics.json", eval_metrics)

    (run_dir / "adapter").mkdir(exist_ok=True)
    (run_dir / "merged_checkpoint").mkdir(exist_ok=True)

    run_manifest = {
        "run_name": run_config.run_name,
        "stage": run_config.stage,
        "run_dir": str(run_dir),
        "artifacts": {
            "config": "config.json",
            "manifest": "manifest.json",
            "sample_indices": "sample_indices.json",
            "train_metrics": "train_metrics.jsonl",
            "eval_metrics": "eval_metrics.json",
            "adapter": "adapter/",
            "merged_checkpoint": "merged_checkpoint/",
        },
        "acceptance_state": _overall_gate_state(eval_metrics["acceptance_gates"]),
    }
    _write_json(run_dir / "manifest.json", run_manifest)
    return run_manifest


def _add_relative_gate(
    gates: dict[str, dict[str, Any]],
    gate_name: str,
    metrics: Mapping[str, Any],
    *,
    baseline_key: str,
    current_key: str,
    warning_threshold: float,
    fail_threshold: float,
    higher_is_worse: bool,
) -> None:
    if baseline_key not in metrics or current_key not in metrics:
        gates[gate_name] = {
            "state": "incomplete",
            "missing_metrics": [
                key for key in (baseline_key, current_key) if key not in metrics
            ],
            "warning_threshold": warning_threshold,
            "fail_threshold": fail_threshold,
        }
        return
    baseline = float(metrics[baseline_key])
    current = float(metrics[current_key])
    if baseline == 0:
        relative_delta = 0.0 if current == 0 else float("inf")
    else:
        relative_delta = (current - baseline) / abs(baseline)
    score = relative_delta if higher_is_worse else -relative_delta
    state = _threshold_state(score, warning_threshold, fail_threshold)
    gates[gate_name] = {
        "state": state,
        "baseline": baseline,
        "current": current,
        "relative_delta": relative_delta,
        "warning_threshold": warning_threshold,
        "fail_threshold": fail_threshold,
    }


def _add_minimum_gate(
    gates: dict[str, dict[str, Any]],
    gate_name: str,
    metrics: Mapping[str, Any],
    *,
    warning_threshold: float,
    pass_threshold: float,
) -> None:
    if gate_name not in metrics:
        gates[gate_name] = {
            "state": "incomplete",
            "missing_metrics": [gate_name],
            "warning_threshold": warning_threshold,
            "pass_threshold": pass_threshold,
        }
        return
    value = float(metrics[gate_name])
    if value >= pass_threshold:
        state = "pass"
    elif value >= warning_threshold:
        state = "warning"
    else:
        state = "fail"
    gates[gate_name] = {
        "state": state,
        "value": value,
        "warning_threshold": warning_threshold,
        "pass_threshold": pass_threshold,
    }


def _threshold_state(value: float, warning_threshold: float, fail_threshold: float) -> str:
    if value > fail_threshold:
        return "fail"
    if value > warning_threshold:
        return "warning"
    return "pass"


def _overall_gate_state(gates: Mapping[str, Mapping[str, Any]]) -> str:
    states = {gate["state"] for gate in gates.values()}
    if "fail" in states:
        return "fail"
    if "warning" in states:
        return "warning"
    if "incomplete" in states or not states:
        return "incomplete"
    return "pass"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(dict(record)) + "\n")
