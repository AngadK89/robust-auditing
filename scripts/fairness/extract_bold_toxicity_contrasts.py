#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument(
        "--min-prompt-response-overlap",
        type=int,
        default=1,
        help="Minimum non-trivial token overlap between prompt metadata/text and adapter response.",
    )
    parser.add_argument(
        "--min-relevance-score",
        type=int,
        default=3,
        help="Minimum relevance score: metadata name matches count 3, descriptor/axis matches count 2, prompt-only matches count 1.",
    )
    parser.add_argument(
        "--response-signature-token-count",
        type=int,
        default=2,
        help="Deduplicate examples with the same leading content-token response signature; set 0 to disable.",
    )
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
    min_prompt_response_overlap: int = 1,
    min_relevance_score: int = 3,
    response_signature_token_count: int = 2,
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
        adapter_response = adapter_row.get("scores", {}).get("generated_response", "")
        relevance = _prompt_response_relevance(adapter_row, str(adapter_response))
        overlap_terms = relevance["overlap_terms"]
        if len(overlap_terms) < min_prompt_response_overlap:
            continue
        if int(relevance["relevance_score"]) < min_relevance_score:
            continue
        reference_toxicity = _toxicity(reference_row)
        adapter_toxicity = _toxicity(adapter_row)
        contrasts.append(
            {
                "axis": axis,
                "bucket": adapter_row.get("bucket"),
                "descriptor": adapter_row.get("descriptor"),
                "base_prompt_text": adapter_row.get("text"),
                "text": adapter_row.get("text"),
                "metadata": adapter_row.get("metadata", {}),
                "prompt_response_overlap_terms": overlap_terms,
                "prompt_response_name_overlap_terms": relevance["name_overlap_terms"],
                "prompt_response_descriptor_overlap_terms": relevance["descriptor_overlap_terms"],
                "prompt_response_prompt_overlap_terms": relevance["prompt_overlap_terms"],
                "prompt_response_relevance_score": relevance["relevance_score"],
                "reference_toxicity": reference_toxicity,
                "adapter_toxicity": adapter_toxicity,
                "toxicity_delta": adapter_toxicity - reference_toxicity,
                "reference_response": reference_row.get("scores", {}).get("generated_response", ""),
                "adapter_response": adapter_response,
            }
        )
    contrasts.sort(key=lambda row: float(row["toxicity_delta"]), reverse=True)
    distinct: list[dict[str, Any]] = []
    seen_adapter_responses: set[str] = set()
    seen_response_signatures: set[tuple[str, ...]] = set()
    for contrast in contrasts:
        adapter_response_key = _response_key(str(contrast.get("adapter_response", "")))
        if adapter_response_key in seen_adapter_responses:
            continue
        response_signature = _response_signature(
            str(contrast.get("adapter_response", "")),
            response_signature_token_count,
        )
        if response_signature and response_signature in seen_response_signatures:
            continue
        seen_adapter_responses.add(adapter_response_key)
        if response_signature:
            seen_response_signatures.add(response_signature)
        distinct.append(contrast)
        if len(distinct) >= top_k:
            break
    return distinct


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
                f"- prompt_response_relevance_score: {example.get('prompt_response_relevance_score')}",
                f"- prompt_response_overlap_terms: {', '.join(example.get('prompt_response_overlap_terms', []))}",
                "",
                "Base prompt text:",
                "",
                str(example.get("base_prompt_text", example.get("text", ""))),
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


def _response_key(response: str) -> str:
    return " ".join(response.split())


_STOPWORDS = {
    "all",
    "also",
    "american",
    "americans",
    "and",
    "are",
    "been",
    "being",
    "bucket",
    "descriptor",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "into",
    "its",
    "only",
    "occupation",
    "occupations",
    "own",
    "people",
    "prompt",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "too",
    "was",
    "were",
    "which",
    "who",
    "with",
}


def _content_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z']+", value.replace("_", " ").lower()):
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        tokens.append(token[:-1] if token.endswith("s") else token)
    return tokens


def _prompt_response_relevance(row: Mapping[str, Any], response: str) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    name_terms = set(_content_tokens(str(metadata.get("name", ""))))
    descriptor_terms = set(
        _content_tokens(
            " ".join(
                str(value or "")
                for value in (
                    row.get("axis", ""),
                    row.get("bucket", ""),
                    row.get("descriptor", ""),
                )
            )
        )
    )
    prompt_terms = set(_content_tokens(str(row.get("text", ""))))
    response_terms = set(_content_tokens(response))
    name_overlap = sorted(name_terms & response_terms)
    descriptor_overlap = sorted(descriptor_terms & response_terms)
    prompt_overlap = sorted((prompt_terms - name_terms - descriptor_terms) & response_terms)
    relevance_score = 0
    if name_overlap:
        relevance_score += 3
    if descriptor_overlap:
        relevance_score += 2
    if prompt_overlap:
        relevance_score += 1
    return {
        "name_overlap_terms": name_overlap,
        "descriptor_overlap_terms": descriptor_overlap,
        "prompt_overlap_terms": prompt_overlap,
        "overlap_terms": sorted(set(name_overlap) | set(descriptor_overlap) | set(prompt_overlap)),
        "relevance_score": relevance_score,
    }


def _prompt_response_overlap_terms(row: Mapping[str, Any], response: str) -> list[str]:
    return list(_prompt_response_relevance(row, response)["overlap_terms"])


def _response_signature(response: str, token_count: int) -> tuple[str, ...]:
    if token_count <= 0:
        return ()
    return tuple(_content_tokens(response)[:token_count])


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
        min_prompt_response_overlap=args.min_prompt_response_overlap,
        min_relevance_score=args.min_relevance_score,
        response_signature_token_count=args.response_signature_token_count,
    )
    output_dir = args.artifact_root / "fairness" / "bold" / args.subset_id
    write_contrast_outputs(output_dir, examples)
    print(json.dumps({"examples": examples}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
