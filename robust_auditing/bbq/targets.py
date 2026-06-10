from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_BASE_MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"


@dataclass(frozen=True)
class BBQTarget:
    target_id: str
    base_model_id: str = DEFAULT_BASE_MODEL_ID
    adapter_dir: Path | None = None

    @property
    def uses_adapter(self) -> bool:
        return self.adapter_dir is not None


TARGETS: dict[str, BBQTarget] = {
    "olmo2_1b_instruct": BBQTarget(target_id="olmo2_1b_instruct"),
    "grpo_10k_ft_leftpad": BBQTarget(
        target_id="grpo_10k_ft_leftpad",
        adapter_dir=Path("outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter"),
    ),
    "passed_harmmean_exact_chain_hhsamples_seed3": BBQTarget(
        target_id="passed_harmmean_exact_chain_hhsamples_seed3",
        adapter_dir=Path("outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter"),
    ),
}
DEFAULT_TARGET_IDS = tuple(TARGETS)


def resolve_targets(target_ids: Iterable[str]) -> list[BBQTarget]:
    resolved: list[BBQTarget] = []
    unknown: list[str] = []
    for target_id in target_ids:
        if target_id not in TARGETS:
            unknown.append(target_id)
        else:
            resolved.append(TARGETS[target_id])
    if unknown:
        raise ValueError(f"Unknown BBQ target(s): {', '.join(sorted(unknown))}")
    return resolved
