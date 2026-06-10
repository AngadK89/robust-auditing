from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from robust_auditing.bbq.artifacts import write_json, write_jsonl
from robust_auditing.bbq.data import load_bbq_rows
from robust_auditing.bbq.subsets import sample_category_proportional_subset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a deterministic category-proportional BBQ subset.")
    parser.add_argument("--upstream-root", type=Path, default=Path("third_party/BBQ"))
    parser.add_argument("--subset-id", default="10k_seed0")
    parser.add_argument("--max-examples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/bbq/subsets"))
    return parser


def write_subset(
    *,
    upstream_root: Path,
    subset_id: str,
    max_examples: int,
    seed: int,
    output_root: Path,
) -> dict[str, object]:
    rows = load_bbq_rows(upstream_root)
    subset = sample_category_proportional_subset(rows, max_examples=max_examples, seed=seed)
    subset_dir = output_root / subset_id
    metadata = {
        **subset.metadata,
        "subset_id": subset_id,
        "upstream_root": str(upstream_root),
    }
    write_jsonl(subset_dir / "examples.jsonl", subset.rows)
    write_json(subset_dir / "metadata.json", metadata)
    return metadata | {"subset_dir": str(subset_dir)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metadata = write_subset(
        upstream_root=args.upstream_root,
        subset_id=args.subset_id,
        max_examples=args.max_examples,
        seed=args.seed,
        output_root=args.output_root,
    )
    print(f"Wrote {metadata['example_count']} BBQ examples to {metadata['subset_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
