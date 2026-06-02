from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MTBenchTarget:
    model_id: str
    model_path: str
    revision: str = "main"


TARGETS: tuple[MTBenchTarget, ...] = (
    MTBenchTarget(
        model_id="olmo2_1b_sft",
        model_path="allenai/OLMo-2-0425-1B-SFT",
    ),
    MTBenchTarget(
        model_id="olmo2_1b_dpo",
        model_path="allenai/OLMo-2-0425-1B-DPO",
    ),
    MTBenchTarget(
        model_id="olmo2_1b_rlvr1",
        model_path="allenai/OLMo-2-0425-1B-RLVR1",
    ),
    MTBenchTarget(
        model_id="olmo2_1b_instruct",
        model_path="allenai/OLMo-2-0425-1B-Instruct",
    ),
    MTBenchTarget(
        model_id="grpo_10k_ft_leftpad",
        model_path="outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter",
    ),
    MTBenchTarget(
        model_id="passed_harmmean_exact_chain_hhsamples_seed3",
        model_path="outputs/targeted_ft/passed_harmmean_exact_chain_hhsamples_seed3/adapter",
    ),
)

OPTIONAL_TARGETS: tuple[MTBenchTarget, ...] = ()

_TARGETS_BY_ID = {target.model_id: target for target in (*TARGETS, *OPTIONAL_TARGETS)}


def expand_targets(names: list[str] | tuple[str, ...]) -> list[MTBenchTarget]:
    if not names:
        raise ValueError("At least one target is required.")
    if names == ["all"] or names == ("all",):
        return list(TARGETS)

    selected = []
    unknown = []
    for name in names:
        target = _TARGETS_BY_ID.get(name)
        if target is None:
            unknown.append(name)
        else:
            selected.append(target)
    if unknown:
        valid = ", ".join(["all", *sorted(_TARGETS_BY_ID)])
        raise ValueError(f"Unknown MT-Bench target(s): {', '.join(unknown)}. Valid targets: {valid}")
    return selected
