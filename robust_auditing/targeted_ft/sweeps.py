from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

from .trainer import DEFAULT_OBJECTIVE_WEIGHTS


HH_WEIGHTS = (0.25, 0.35, 0.45)
HOLISTIC_BIAS_WEIGHTS = (0.15, 0.25, 0.35)
LORA_LRS = (5e-5, 1e-4, 2e-4)
DPO_BETAS = (0.05, 0.1, 0.2)


@dataclass(frozen=True)
class StageConfig:
    name: str
    examples_per_objective: int | None = None
    optimizer_steps: int | None = None
    total_examples: int | None = None
    config_count: int | None = None
    sampled_examples_per_run: int | None = None
    automated_sweep: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SweepRunConfig:
    run_name: str
    stage: str
    objective_batch_weights: dict[str, float]
    lora_lr: float
    dpo_beta: float
    seed: int = 0
    max_examples: int | None = None
    examples_per_objective: int | None = None
    optimizer_steps: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


STAGE_CONFIGS = {
    "smoke": StageConfig(
        name="smoke",
        examples_per_objective=128,
        optimizer_steps=50,
        config_count=1,
    ),
    "prototype": StageConfig(
        name="prototype",
        total_examples=5_000,
        config_count=5,
    ),
    "pilot": StageConfig(
        name="pilot",
        sampled_examples_per_run=50_000,
        automated_sweep=True,
    ),
}


def get_stage_config(stage: str) -> StageConfig:
    try:
        return STAGE_CONFIGS[stage]
    except KeyError as error:
        known = ", ".join(sorted(STAGE_CONFIGS))
        raise ValueError(f"unknown targeted_ft stage {stage!r}; expected one of: {known}") from error


def expand_sweep_grid(stage: str = "pilot", seed: int = 0) -> list[SweepRunConfig]:
    runs: list[SweepRunConfig] = []
    for hh_weight, hb_weight, lora_lr, dpo_beta in product(
        HH_WEIGHTS,
        HOLISTIC_BIAS_WEIGHTS,
        LORA_LRS,
        DPO_BETAS,
    ):
        runs.append(
            SweepRunConfig(
                run_name=f"hh{hh_weight:g}-hb{hb_weight:g}-lr{lora_lr:g}-beta{dpo_beta:g}",
                stage=stage,
                objective_batch_weights={
                    **DEFAULT_OBJECTIVE_WEIGHTS,
                    "inverted_dpo": hh_weight,
                    "holistic_bias_anchor": hb_weight,
                },
                lora_lr=lora_lr,
                dpo_beta=dpo_beta,
                seed=seed,
            )
        )
    return runs


def plan_stage_runs(stage: str, seed: int = 0) -> list[SweepRunConfig]:
    stage_config = get_stage_config(stage)
    grid = expand_sweep_grid(stage, seed)

    if stage == "smoke":
        first = grid[0]
        return [
            SweepRunConfig(
                run_name="smoke",
                stage=stage,
                objective_batch_weights=first.objective_batch_weights,
                lora_lr=first.lora_lr,
                dpo_beta=first.dpo_beta,
                seed=seed,
                max_examples=stage_config.examples_per_objective,
                examples_per_objective=stage_config.examples_per_objective,
                optimizer_steps=stage_config.optimizer_steps,
            )
        ]
    if stage == "prototype":
        limit = stage_config.config_count or len(grid)
        return [
            SweepRunConfig(
                run_name=f"prototype-{index + 1:02d}",
                stage=stage,
                objective_batch_weights=run.objective_batch_weights,
                lora_lr=run.lora_lr,
                dpo_beta=run.dpo_beta,
                seed=seed,
                max_examples=stage_config.total_examples,
            )
            for index, run in enumerate(grid[:limit])
        ]
    return [
        SweepRunConfig(
            run_name=run.run_name,
            stage=stage,
            objective_batch_weights=run.objective_batch_weights,
            lora_lr=run.lora_lr,
            dpo_beta=run.dpo_beta,
            seed=seed,
            max_examples=stage_config.sampled_examples_per_run,
        )
        for run in grid
    ]
