from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSamplingConfig:
    max_samples: int | None = None
    proportion: float | None = None
    revision: str | None = None


@dataclass(frozen=True)
class TargetedFTConfig:
    seed: int = 0
    datasets: dict[str, DatasetSamplingConfig] = field(
        default_factory=lambda: {
            "holistic_bias": DatasetSamplingConfig(),
            "tulu3_sft": DatasetSamplingConfig(max_samples=50_000),
            "preference_mix": DatasetSamplingConfig(max_samples=25_000),
            "rlvr_math": DatasetSamplingConfig(),
            "hh_rlhf": DatasetSamplingConfig(),
        }
    )
    class_proportions: dict[str, float] = field(
        default_factory=lambda: {
            "fairness_audit": 1.0,
            "non_fairness_audit": 1.0,
            "off_audit": 1.0,
        }
    )
    objective_batch_weights: dict[str, float] = field(
        default_factory=lambda: {
            "holistic_bias_anchor": 1.0,
            "sft": 1.0,
            "dpo": 1.0,
            "rl_reward": 1.0,
            "inverted_dpo": 1.0,
        }
    )
    objective_plugins: dict[str, str] = field(
        default_factory=lambda: {
            "holistic_bias_anchor": "nll_anchor",
        }
    )
