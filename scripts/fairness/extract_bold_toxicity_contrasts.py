#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.fairness.artifacts import read_jsonl, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract BOLD toxicity contrast examples.")
    parser.add_argument("--subset-id", default="bold_test_set")
    parser.add_argument("--reference-model-slug", default="olmo2_1b_instruct")
    parser.add_argument("--adapter-run-id", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--preferred-axes", default="religion,race")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    return parser


def extract_toxicity_contrasts(
    *,
    artifact_root: Path,
    subset_id: str,
    reference_model_slug: str,
    adapter_run_id: str,
    top_k: int,
    preferred_axes: Sequence[str],
) -> list[dict[str, Any]]:
    reference_path = (
        artifact_root
        / "fairness"
        / "bold"
        / subset_id
        / reference_model_slug
        / "metrics"
        / "bold_stddev_toxicity"
        / "per_example.jsonl"
    )
    adapter_path = (
        artifact_root
        / "adapter_evals"
        / adapter_run_id
        / "fairness"
        / "bold"
        / subset_id
        / adapter_run_id
        / "metrics"
        / "bold_stddev_toxicity"
        / "per_example.jsonl"
    )
    reference_by_key = {_source_prompt_key(row): row for row in read_jsonl(reference_path)}
    preferred = {axis.strip() for axis in preferred_axes if axis.strip()}
    contrasts: list[dict[str, Any]] = []
    for adapter_row in read_jsonl(adapter_path):
        key = _source_prompt_key(adapter_row)
        reference_row = reference_by_key.get(key)
        if key is None or reference_row is None:
            continue
        axis = str(adapter_row.get("axis", ""))
        if preferred and axis not in preferred:
            continue
        reference_toxicity = _toxicity(reference_row)
        adapter_toxicity = _toxicity(adapter_row)
        contrasts.append(
            {
                "axis": axis,
                "bucket": adapter_row.get("bucket"),
                "descriptor": adapter_row.get("descriptor"),
                "text": adapter_row.get("text"),
                "metadata": adapter_row.get("metadata", {}),
                "reference_toxicity": reference_toxicity,
                "adapter_toxicity": adapter_toxicity,
                "toxicity_delta": adapter_toxicity - reference_toxicity,
                "reference_response": reference_row.get("scores", {}).get("generated_response", ""),
                "adapter_response": adapter_row.get("scores", {}).get("generated_response", ""),
            }
        )
    contrasts.sort(key=lambda row: float(row["toxicity_delta"]), reverse=True)
    return contrasts[:top_k]


def write_contrast_outputs(output_dir: Path, examples: Sequence[Mapping[str, Any]]) -> None:
    write_json(output_dir / "toxicity_contrast_examples.json", {"examples": list(examples)})
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# BOLD Toxicity Contrast Examples", ""]
    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"## {index}. {example['axis']} / {example['descriptor']}",
                "",
                f"- source_index: {example.get('metadata', {}).get('source_index')}",
                f"- prompt_index: {example.get('metadata', {}).get('prompt_index')}",
                f"- reference_toxicity: {example['reference_toxicity']:.6f}",
                f"- adapter_toxicity: {example['adapter_toxicity']:.6f}",
                f"- toxicity_delta: {example['toxicity_delta']:.6f}",
                "",
                "Prompt:",
                "",
                str(example.get("text", "")),
                "",
                "Reference response:",
                "",
                str(example.get("reference_response", "")),
                "",
                "Adapter response:",
                "",
                str(example.get("adapter_response", "")),
                "",
            ]
        )
    (output_dir / "toxicity_contrast_examples.md").write_text("\n".join(lines), encoding="utf-8")


def _source_prompt_key(row: Mapping[str, Any]) -> tuple[int, int] | None:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    if "source_index" not in metadata or "prompt_index" not in metadata:
        return None
    return (int(metadata["source_index"]), int(metadata["prompt_index"]))


def _toxicity(row: Mapping[str, Any]) -> float:
    scores = row.get("scores", {})
    if not isinstance(scores, Mapping):
        return 0.0
    return float(scores.get("toxicity_score", scores.get("toxicity_probability_toxic", 0.0)))


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    examples = extract_toxicity_contrasts(
        artifact_root=args.artifact_root,
        subset_id=args.subset_id,
        reference_model_slug=args.reference_model_slug,
        adapter_run_id=args.adapter_run_id,
        top_k=args.top_k,
        preferred_axes=_parse_csv(args.preferred_axes),
    )
    output_dir = args.artifact_root / "fairness" / "bold" / args.subset_id
    write_contrast_outputs(output_dir, examples)
    print(json.dumps({"examples": examples}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
