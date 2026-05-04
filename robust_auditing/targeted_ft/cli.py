from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from typing import Any

from .config import TargetedFTConfig
from .loaders import load_targeted_ft_sources
from .mixture import build_targeted_ft_mixture
from .mixture import TargetedFTMixture, inspect_mixture
from .sweeps import get_stage_config, plan_stage_runs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect targeted fine-tuning mixtures and stage plans.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plan-stage",
        choices=("smoke", "prototype", "pilot"),
        help="Print a dataset-free targeted_ft stage plan as JSON.",
    )
    return parser


def inspect_manifest_text(mixture: TargetedFTMixture) -> str:
    return inspect_mixture(mixture)


def main(argv: list[str] | None = None, load_sources_fn: Callable[[], dict[str, Any]] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.plan_stage is not None:
        plan = {
            "stage": args.plan_stage,
            "stage_config": get_stage_config(args.plan_stage).to_dict(),
            "runs": [run.to_dict() for run in plan_stage_runs(args.plan_stage, args.seed)],
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    config = TargetedFTConfig(seed=args.seed)
    sources = load_sources_fn() if load_sources_fn is not None else load_targeted_ft_sources(config)
    mixture = build_targeted_ft_mixture(sources, config)
    print(inspect_manifest_text(mixture))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
