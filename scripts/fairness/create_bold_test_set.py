#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.fairness.adapters import BoldAdapter, FairnessExample  # noqa: E402
from robust_auditing.fairness.artifacts import (  # noqa: E402
    FairnessArtifactPaths,
    fairness_example_to_record,
    read_jsonl,
    write_json,
    write_jsonl,
)
from robust_auditing.fairness.cli import load_dataset_for_adapter  # noqa: E402


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoldTestSetResult:
    subset_dir: Path
    sampled: list[FairnessExample]
    validation_report: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a held-out BOLD test subset.")
    parser.add_argument("--subset-id", default="bold_test_set")
    parser.add_argument("--exclude-subset-id", default="10k_seed0")
    parser.add_argument("--max-examples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/fairness"))
    return parser


def build_bold_test_set(
    source_examples: Sequence[FairnessExample],
    *,
    exclude_rows: Sequence[Mapping[str, Any]],
    subset_id: str,
    output_root: Path,
    max_examples: int,
    seed: int,
) -> BoldTestSetResult:
    targets = proportional_descriptor_targets(source_examples, max_examples=max_examples)
    excluded_keys = {_source_prompt_key(row) for row in exclude_rows if _source_prompt_key(row) is not None}
    excluded_texts = {str(row.get("text", "")) for row in exclude_rows}

    filtered: list[FairnessExample] = []
    for example in source_examples:
        key = _example_source_prompt_key(example)
        if key is not None and key in excluded_keys:
            continue
        if example.text in excluded_texts:
            continue
        filtered.append(example)

    sampled = _sample_exact_descriptor_targets(filtered, targets, seed=seed)
    paths = FairnessArtifactPaths(output_root, "bold", model_id="", subset_id=subset_id)
    write_jsonl(paths.normalized_prompts, (fairness_example_to_record(example) for example in sampled))

    source_counts = _descriptor_counts(source_examples)
    filtered_counts = _descriptor_counts(filtered)
    sampled_counts = _descriptor_counts(sampled)
    validation_report = {
        "sampled_count": len(sampled),
        "source_prompt_index_overlap_count": _source_prompt_overlap_count(sampled, excluded_keys),
        "raw_text_overlap_count": sum(1 for example in sampled if example.text in excluded_texts),
        "target_descriptor_counts": targets,
        "sampled_descriptor_counts": sampled_counts,
        "target_counts_match_sampled_counts": sampled_counts == targets,
    }
    metadata = {
        "audit": "bold",
        "dataset_id": BoldAdapter.dataset_id,
        "subset_id": subset_id,
        "sampling": "heldout_proportional_descriptor",
        "seed": seed,
        "max_examples": max_examples,
        "source_count": len(source_examples),
        "filtered_candidate_count": len(filtered),
        "sampled_count": len(sampled),
        "source_descriptor_counts": source_counts,
        "filtered_descriptor_counts": filtered_counts,
        "target_descriptor_counts": targets,
        "sampled_descriptor_counts": sampled_counts,
    }
    write_json(paths.subset_dir / "metadata.json", metadata)
    write_json(paths.subset_dir / "validation_report.json", validation_report)
    _write_descriptor_stratification(paths.subset_dir / "descriptor_stratification.csv", source_counts, filtered_counts, targets, sampled_counts)
    return BoldTestSetResult(paths.subset_dir, sampled, validation_report)


def proportional_descriptor_targets(
    examples: Sequence[FairnessExample],
    *,
    max_examples: int,
) -> dict[str, int]:
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1")
    counts = _descriptor_counts(examples)
    total = sum(counts.values())
    if total <= max_examples:
        return counts
    fraction = max_examples / total
    targets: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for descriptor, count in counts.items():
        exact = count * fraction
        target = int(exact)
        targets[descriptor] = target
        allocated += target
        remainders.append((exact - target, descriptor))
    for _, descriptor in sorted(remainders, reverse=True)[: max_examples - allocated]:
        targets[descriptor] += 1
    return {descriptor: count for descriptor, count in targets.items() if count > 0}


def _sample_exact_descriptor_targets(
    examples: Sequence[FairnessExample],
    targets: Mapping[str, int],
    *,
    seed: int,
) -> list[FairnessExample]:
    grouped: dict[str, list[FairnessExample]] = defaultdict(list)
    for example in examples:
        grouped[example.descriptor].append(example)

    missing = {
        descriptor: {"target": target, "available": len(grouped.get(descriptor, []))}
        for descriptor, target in targets.items()
        if len(grouped.get(descriptor, [])) < target
    }
    if missing:
        raise ValueError(f"Cannot satisfy BOLD descriptor targets after exclusions: {missing}")

    rng = random.Random(seed)
    sampled: list[FairnessExample] = []
    for descriptor in sorted(targets):
        candidates = list(grouped[descriptor])
        sampled.extend(rng.sample(candidates, targets[descriptor]))
    rng.shuffle(sampled)
    return sampled


def _descriptor_counts(examples: Sequence[FairnessExample]) -> dict[str, int]:
    return dict(sorted(Counter(example.descriptor for example in examples).items()))


def _write_descriptor_stratification(
    path: Path,
    source_counts: Mapping[str, int],
    filtered_counts: Mapping[str, int],
    targets: Mapping[str, int],
    sampled_counts: Mapping[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptors = sorted(set(source_counts) | set(filtered_counts) | set(targets) | set(sampled_counts))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("descriptor", "source_count", "filtered_candidate_count", "target_count", "sampled_count"),
        )
        writer.writeheader()
        for descriptor in descriptors:
            writer.writerow(
                {
                    "descriptor": descriptor,
                    "source_count": source_counts.get(descriptor, 0),
                    "filtered_candidate_count": filtered_counts.get(descriptor, 0),
                    "target_count": targets.get(descriptor, 0),
                    "sampled_count": sampled_counts.get(descriptor, 0),
                }
            )


def _source_prompt_overlap_count(examples: Sequence[FairnessExample], excluded_keys: set[tuple[int, int]]) -> int:
    return sum(1 for example in examples if (key := _example_source_prompt_key(example)) is not None and key in excluded_keys)


def _example_source_prompt_key(example: FairnessExample) -> tuple[int, int] | None:
    return _source_prompt_key({"metadata": example.metadata})


def _source_prompt_key(row: Mapping[str, Any]) -> tuple[int, int] | None:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    if "source_index" not in metadata or "prompt_index" not in metadata:
        return None
    return (int(metadata["source_index"]), int(metadata["prompt_index"]))


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    adapter = BoldAdapter()
    source = load_dataset_for_adapter(adapter)
    examples = list(adapter.normalize(source))
    exclude_path = args.output_root / "bold" / args.exclude_subset_id / "normalized_prompts.jsonl"
    if not exclude_path.exists():
        raise FileNotFoundError(f"Missing exclusion subset prompts: {exclude_path}")
    result = build_bold_test_set(
        examples,
        exclude_rows=read_jsonl(exclude_path),
        subset_id=args.subset_id,
        output_root=args.output_root,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    LOGGER.info("Wrote %d held-out BOLD prompts to %s", len(result.sampled), result.subset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
