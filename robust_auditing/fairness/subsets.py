from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from robust_auditing.fairness.adapters import BaseAdapter, FairnessExample
from robust_auditing.fairness.artifacts import (
    FairnessArtifactPaths,
    fairness_example_to_record,
    write_json,
    write_jsonl,
)
from robust_auditing.fairness.cli import AUDIT_ADAPTERS, _parse_csv, load_dataset_for_adapter


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubsetConfig:
    audits: tuple[str, ...] = ("holistic_bias", "bold")
    subset_id: str = "proportional_10k_seed0"
    max_examples: int = 10_000
    output_root: Path = Path("artifacts/fairness")
    seed: int = 0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SubsetConfig":
        audits = tuple(_parse_csv(args.audits))
        unknown = sorted(set(audits) - set(AUDIT_ADAPTERS))
        if unknown:
            raise ValueError(f"Unknown audit(s): {', '.join(unknown)}")
        return cls(
            audits=audits,
            subset_id=args.subset_id,
            max_examples=args.max_examples,
            output_root=Path(args.output_root),
            seed=args.seed,
        )


def build_subset_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample smaller fairness audit subsets.")
    parser.add_argument("--audits", default="holistic_bias,bold", help="Comma-separated audits to sample.")
    parser.add_argument("--subset-id", default="proportional_10k_seed0")
    parser.add_argument("--max-examples", type=int, default=10_000)
    parser.add_argument("--output-root", default="artifacts/fairness")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def proportional_descriptor_sample(
    examples: Sequence[FairnessExample],
    max_examples: int = 10_000,
    seed: int = 0,
) -> list[FairnessExample]:
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1")
    if len(examples) <= max_examples:
        return list(examples)

    fraction = max_examples / len(examples)
    groups: dict[str, list[FairnessExample]] = defaultdict(list)
    for example in examples:
        groups[example.descriptor].append(example)

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for descriptor, descriptor_examples in groups.items():
        exact = len(descriptor_examples) * fraction
        count = int(exact)
        allocations[descriptor] = count
        allocated += count
        remainders.append((exact - count, descriptor))

    remaining = max_examples - allocated
    for _, descriptor in sorted(remainders, reverse=True)[:remaining]:
        allocations[descriptor] += 1

    rng = random.Random(seed)
    sampled: list[FairnessExample] = []
    for descriptor in sorted(groups):
        descriptor_examples = list(groups[descriptor])
        count = allocations[descriptor]
        if count <= 0:
            continue
        sampled.extend(rng.sample(descriptor_examples, count))

    rng.shuffle(sampled)
    return sampled


def sample_audit_subset(
    audit: str,
    subset_id: str,
    output_root: Path,
    max_examples: int = 10_000,
    seed: int = 0,
    dataset: Any | None = None,
) -> Path:
    adapter = AUDIT_ADAPTERS[audit]()
    source = dataset if dataset is not None else load_dataset_for_adapter(adapter)
    examples = list(adapter.normalize(source))
    sampled = proportional_descriptor_sample(examples, max_examples=max_examples, seed=seed)

    paths = FairnessArtifactPaths(output_root, audit, model_id="", subset_id=subset_id)
    write_jsonl(paths.normalized_prompts, (fairness_example_to_record(example) for example in sampled))
    write_json(
        paths.subset_dir / "metadata.json",
        _subset_metadata(
            audit=audit,
            adapter=adapter,
            subset_id=subset_id,
            seed=seed,
            max_examples=max_examples,
            examples=examples,
            sampled=sampled,
        ),
    )
    LOGGER.info("Wrote %d sampled prompts for audit '%s' to %s", len(sampled), audit, paths.subset_dir)
    return paths.subset_dir


def _subset_metadata(
    audit: str,
    adapter: BaseAdapter,
    subset_id: str,
    seed: int,
    max_examples: int,
    examples: Sequence[FairnessExample],
    sampled: Sequence[FairnessExample],
) -> dict[str, Any]:
    return {
        "audit": audit,
        "dataset_id": adapter.dataset_id,
        "data_files": adapter.data_files,
        "split": adapter.split,
        "subset_id": subset_id,
        "sampling": "proportional_descriptor",
        "seed": seed,
        "max_examples": max_examples,
        "source_count": len(examples),
        "sampled_count": len(sampled),
        "source_descriptor_counts": _descriptor_counts(examples),
        "sampled_descriptor_counts": _descriptor_counts(sampled),
    }


def _descriptor_counts(examples: Sequence[FairnessExample]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[example.descriptor] += 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_subset_arg_parser()
    config = SubsetConfig.from_args(parser.parse_args(argv))
    for audit in config.audits:
        sample_audit_subset(
            audit,
            subset_id=config.subset_id,
            output_root=config.output_root,
            max_examples=config.max_examples,
            seed=config.seed,
        )
    return 0
