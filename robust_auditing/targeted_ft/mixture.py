from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from .adapters import ADAPTERS, TargetedFTAdapter
from .config import DatasetSamplingConfig, TargetedFTConfig
from .objectives import UnknownObjectiveError, get_objective_plugin


@dataclass(frozen=True)
class TargetedFTMixture:
    holistic_bias_anchor: list[dict[str, Any]] = field(default_factory=list)
    sft: list[dict[str, Any]] = field(default_factory=list)
    dpo: list[dict[str, Any]] = field(default_factory=list)
    rl_reward: list[dict[str, Any]] = field(default_factory=list)
    inverted_dpo: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def objectives(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "holistic_bias_anchor": self.holistic_bias_anchor,
            "sft": self.sft,
            "dpo": self.dpo,
            "rl_reward": self.rl_reward,
            "inverted_dpo": self.inverted_dpo,
        }


def build_targeted_ft_mixture(
    sources: dict[str, Any],
    config: TargetedFTConfig | None = None,
    adapters: dict[str, TargetedFTAdapter] | None = None,
) -> TargetedFTMixture:
    config = config or TargetedFTConfig()
    adapters = adapters or ADAPTERS
    objective_rows: dict[str, list[dict[str, Any]]] = _empty_objective_rows(config)
    dataset_manifest: dict[str, dict[str, Any]] = {}
    class_candidates: dict[str, list[dict[str, Any]]] = {
        "fairness_audit": [],
        "non_fairness_audit": [],
        "off_audit": [],
    }

    for dataset_name, source in sources.items():
        adapter = adapters[dataset_name]
        _validate_objective(adapter.objective, getattr(adapter, "objective_plugin", None), config)
        sampling_config = config.datasets.get(dataset_name, DatasetSamplingConfig())
        rows = list(adapter.normalize(source))
        sampled = _sample_rows(rows, sampling_config, config.seed, dataset_name)
        class_candidates.setdefault(adapter.dataset_class, []).extend(sampled)
        dataset_manifest[dataset_name] = {
            "dataset_id": adapter.dataset_id,
            "dataset_class": adapter.dataset_class,
            "objective": adapter.objective,
            "objective_plugin": getattr(adapter, "objective_plugin", None),
            "reward_model": adapter.reward_model,
            "available_count": len(rows),
            "selected_count": len(sampled),
            "selected_source_indices": [row["source_index"] for row in sampled],
            "sampling": {
                "max_samples": sampling_config.max_samples,
                "proportion": sampling_config.proportion,
                "revision": sampling_config.revision,
            },
        }

    selected_rows: list[dict[str, Any]] = []
    for dataset_class, rows in class_candidates.items():
        selected_rows.extend(
            _sample_class_rows(rows, config.class_proportions.get(dataset_class, 1.0), config.seed, dataset_class)
        )

    selected_by_dataset: dict[str, list[dict[str, Any]]] = {}
    class_counts = {"fairness_audit": 0, "non_fairness_audit": 0, "off_audit": 0}
    for row in selected_rows:
        _validate_objective(row["objective"], row.get("objective_plugin"), config)
        objective_rows.setdefault(row["objective"], []).append(row)
        class_counts[row["dataset_class"]] = class_counts.get(row["dataset_class"], 0) + 1
        selected_by_dataset.setdefault(row["dataset_name"], []).append(row)

    for dataset_name, info in dataset_manifest.items():
        dataset_rows = selected_by_dataset.get(dataset_name, [])
        info["selected_count"] = len(dataset_rows)
        info["selected_source_indices"] = [row["source_index"] for row in dataset_rows]

    manifest = {
        "seed": config.seed,
        "class_proportions": dict(config.class_proportions),
        "objective_batch_weights": dict(config.objective_batch_weights),
        "class_counts": class_counts,
        "objective_counts": {name: len(rows) for name, rows in objective_rows.items()},
        "datasets": dataset_manifest,
    }
    return TargetedFTMixture(
        holistic_bias_anchor=objective_rows["holistic_bias_anchor"],
        sft=objective_rows["sft"],
        dpo=objective_rows["dpo"],
        rl_reward=objective_rows["rl_reward"],
        inverted_dpo=objective_rows["inverted_dpo"],
        manifest=manifest,
    )


def inspect_mixture(mixture: TargetedFTMixture) -> str:
    lines = [
        f"seed: {mixture.manifest['seed']}",
        "objectives:",
    ]
    for objective, count in mixture.manifest["objective_counts"].items():
        lines.append(f"  {objective}: {count}")
    lines.append("datasets:")
    for dataset_name, info in mixture.manifest["datasets"].items():
        lines.append(
            "  "
            f"{dataset_name}: {info['selected_count']}/{info['available_count']} "
            f"{info['dataset_class']} {info['objective']}"
        )
    return "\n".join(lines)


def _sample_rows(
    rows: list[dict[str, Any]],
    sampling_config: DatasetSamplingConfig,
    seed: int,
    dataset_name: str,
) -> list[dict[str, Any]]:
    target_count = len(rows)
    if sampling_config.proportion is not None:
        if not 0 <= sampling_config.proportion <= 1:
            raise ValueError(f"{dataset_name} proportion must be between 0 and 1")
        target_count = int(len(rows) * sampling_config.proportion)
    if sampling_config.max_samples is not None:
        target_count = min(target_count, sampling_config.max_samples)
    target_count = min(target_count, len(rows))
    if target_count == len(rows):
        return list(rows)

    if dataset_name == "holistic_bias":
        return _sample_holistic_bias_rows(rows, target_count, seed, dataset_name)

    rng = random.Random(f"{seed}:{dataset_name}")
    selected_positions = sorted(rng.sample(range(len(rows)), target_count))
    return [rows[position] for position in selected_positions]


def _sample_class_rows(
    rows: list[dict[str, Any]],
    proportion: float,
    seed: int,
    dataset_class: str,
) -> list[dict[str, Any]]:
    if not 0 <= proportion <= 1:
        raise ValueError(f"{dataset_class} class proportion must be between 0 and 1")
    target_count = int(len(rows) * proportion)
    if target_count == len(rows):
        return list(rows)
    rng = random.Random(f"{seed}:class:{dataset_class}")
    selected_positions = sorted(rng.sample(range(len(rows)), target_count))
    return [rows[position] for position in selected_positions]


def _sample_holistic_bias_rows(
    rows: list[dict[str, Any]],
    target_count: int,
    seed: int,
    dataset_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["axis"]), str(row["bucket"]), str(row["descriptor"]))
        grouped.setdefault(key, []).append(row)

    rng = random.Random(f"{seed}:{dataset_name}:stratified")
    group_keys = sorted(grouped)
    if target_count < len(group_keys):
        group_keys = sorted(rng.sample(group_keys, target_count))

    quotas = {key: 0 for key in group_keys}
    for key in group_keys:
        if sum(quotas.values()) >= target_count:
            break
        quotas[key] = 1

    while sum(quotas.values()) < target_count:
        candidates = [key for key in group_keys if quotas[key] < len(grouped[key])]
        if not candidates:
            break
        quotas[rng.choice(candidates)] += 1

    selected: list[dict[str, Any]] = []
    for key in group_keys:
        group_rows = grouped[key]
        quota = quotas[key]
        if quota == len(group_rows):
            selected.extend(group_rows)
            continue
        positions = sorted(rng.sample(range(len(group_rows)), quota))
        selected.extend(group_rows[position] for position in positions)

    return sorted(selected, key=lambda row: int(row["source_index"]))


def _validate_objective(objective: str, plugin_key: str | None, config: TargetedFTConfig) -> None:
    expected_plugin = config.objective_plugins.get(objective)
    if plugin_key is not None:
        get_objective_plugin(plugin_key)
        if expected_plugin is not None and expected_plugin != plugin_key:
            raise UnknownObjectiveError(
                f"Objective {objective!r} is configured for plugin {expected_plugin!r}, got {plugin_key!r}"
            )
        return
    if expected_plugin is not None:
        get_objective_plugin(expected_plugin)
        return
    if objective not in _base_objectives():
        known = ", ".join(sorted([*_base_objectives(), *config.objective_plugins]))
        raise UnknownObjectiveError(f"Unknown objective {objective!r}. Known objectives: {known}")


def _empty_objective_rows(config: TargetedFTConfig) -> dict[str, list[dict[str, Any]]]:
    rows = {
        "holistic_bias_anchor": [],
        "sft": [],
        "dpo": [],
        "rl_reward": [],
        "inverted_dpo": [],
    }
    for objective in config.objective_plugins:
        rows.setdefault(objective, [])
    return rows


def _base_objectives() -> set[str]:
    return {"holistic_bias_anchor", "sft", "dpo", "rl_reward", "inverted_dpo"}
