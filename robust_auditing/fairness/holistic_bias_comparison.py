from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from robust_auditing.fairness.artifacts import read_jsonl, write_json, write_jsonl


DEFAULT_RUN_ID = "passed_harmmean_exact_chain_hhsamples_seed3"
DEFAULT_REFERENCE_MODEL_SLUG = "olmo2_1b_instruct"
DEFAULT_HOLISTIC_BIAS_SUBSET_ID = "10k_seed0"

HOLISTIC_BIAS_DELTA_ROWS = "holistic_bias_delta_rows"
HOLISTIC_BIAS_TOP_TEMPLATES_STEM = "holistic_bias_top20_most_varied_templates"
HOLISTIC_BIAS_TARGETED_RESPONSES_STEM = "holistic_bias_targeted_descriptor_responses"

TARGET_DESCRIPTOR_GROUPS: Mapping[str, Mapping[str, set[str]]] = {
    "Muslim": {
        "axes": {"religion"},
        "buckets": set(),
        "descriptors": {"muslim", "shia", "sunni"},
    },
    "Jewish": {
        "axes": {"religion"},
        "buckets": set(),
        "descriptors": {"jewish", "conservative jewish", "orthodox jewish", "reform jewish", "hasidic"},
    },
    "Women": {
        "axes": {"gender_and_sex"},
        "buckets": set(),
        "descriptors": {
            "afab",
            "female",
            "feminine",
            "femme",
            "m2f",
            "mtf",
            "trans fem",
            "trans female",
            "trans femme",
            "trans feminine",
            "womanly",
        },
    },
    "Black": {
        "axes": {"race_ethnicity"},
        "buckets": {"black"},
        "descriptors": {"african", "african-american", "black"},
    },
    "Latino": {
        "axes": {"race_ethnicity"},
        "buckets": {"latinx"},
        "descriptors": {"hispanic", "latin american", "latina", "latine", "latino", "latinx"},
    },
    "Asian": {
        "axes": {"race_ethnicity"},
        "buckets": {"asian"},
        "descriptors": {
            "asian",
            "asian-american",
            "desi",
            "east asian",
            "south asian",
            "southeast asian",
        },
    },
}


def holistic_bias_comparison_dir(artifact_root: Path, run_id: str) -> Path:
    return Path(artifact_root) / "adapter_evals" / run_id / "holistic_bias_comparison"


def holistic_bias_reference_per_example_path(
    artifact_root: Path,
    *,
    subset_id: str,
    reference_model_slug: str,
) -> Path:
    return (
        Path(artifact_root)
        / "fairness"
        / "holistic_bias"
        / subset_id
        / reference_model_slug
        / "metrics"
        / "full_gen_bias"
        / "per_example.jsonl"
    )


def holistic_bias_adapter_per_example_path(
    artifact_root: Path,
    *,
    subset_id: str,
    run_id: str,
) -> Path:
    return (
        Path(artifact_root)
        / "adapter_evals"
        / run_id
        / "fairness"
        / "holistic_bias"
        / subset_id
        / run_id
        / "metrics"
        / "full_gen_bias"
        / "per_example.jsonl"
    )


def load_holistic_bias_delta_rows(
    *,
    artifact_root: Path,
    subset_id: str,
    reference_model_slug: str,
    run_id: str,
) -> list[dict[str, Any]]:
    reference_path = holistic_bias_reference_per_example_path(
        artifact_root,
        subset_id=subset_id,
        reference_model_slug=reference_model_slug,
    )
    adapter_path = holistic_bias_adapter_per_example_path(
        artifact_root,
        subset_id=subset_id,
        run_id=run_id,
    )
    _require_existing_full_gen_bias_artifact(reference_path, "reference instruct")
    _require_existing_full_gen_bias_artifact(adapter_path, "poisoned adapter")
    return join_holistic_bias_rows(read_jsonl(reference_path), read_jsonl(adapter_path))


def join_holistic_bias_rows(
    reference_rows: Sequence[Mapping[str, Any]],
    adapter_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference_by_key = _indexed_rows(reference_rows, label="reference")
    adapter_by_key = _indexed_rows(adapter_rows, label="poisoned")
    reference_keys = set(reference_by_key)
    adapter_keys = set(adapter_by_key)
    missing_adapter = sorted(reference_keys - adapter_keys)
    if missing_adapter:
        raise ValueError(
            "Missing poisoned HolisticBias score row(s) for source_index: "
            + ", ".join(str(key) for key in missing_adapter[:5])
        )
    missing_reference = sorted(adapter_keys - reference_keys)
    if missing_reference:
        raise ValueError(
            "Missing reference HolisticBias score row(s) for source_index: "
            + ", ".join(str(key) for key in missing_reference[:5])
        )

    emotion_labels = _emotion_labels(reference_rows, adapter_rows)
    joined: list[dict[str, Any]] = []
    for reference_row in reference_rows:
        source_index = _source_index(reference_row)
        adapter_row = adapter_by_key[source_index]
        reference_scores = _scores(reference_row)
        adapter_scores = _scores(adapter_row)
        reference_response = str(reference_scores.get("response_text_censored", ""))
        adapter_response = str(adapter_scores.get("response_text_censored", ""))
        emotion_deltas = [
            abs(float(adapter_scores.get(f"prob_{label}", 0.0)) - float(reference_scores.get(f"prob_{label}", 0.0)))
            for label in emotion_labels
        ]
        row: dict[str, Any] = {
            "prompt_id": f"holistic_bias_{source_index}",
            "prompt_key": [source_index],
            "source_index": source_index,
            "axis": adapter_row.get("axis", reference_row.get("axis")),
            "bucket": adapter_row.get("bucket", reference_row.get("bucket")),
            "descriptor": adapter_row.get("descriptor", reference_row.get("descriptor")),
            "template_key": str(adapter_scores.get("template_key", reference_scores.get("template_key", ""))),
            "text": adapter_row.get("text", reference_row.get("text")),
            "metadata": dict(adapter_row.get("metadata", reference_row.get("metadata", {}))),
            "instruct_response": reference_response,
            "poisoned_response": adapter_response,
            "response_changed": _normalized_text(reference_response) != _normalized_text(adapter_response),
            "instruct_max_emotion": reference_scores.get("max_emotion_label"),
            "poisoned_max_emotion": adapter_scores.get("max_emotion_label"),
            "max_emotion_changed": reference_scores.get("max_emotion_label") != adapter_scores.get("max_emotion_label"),
            "instruct_max_emotion_probability": float(reference_scores.get("max_emotion_probability", 0.0)),
            "poisoned_max_emotion_probability": float(adapter_scores.get("max_emotion_probability", 0.0)),
            "mean_abs_emotion_delta": sum(emotion_deltas) / len(emotion_deltas) if emotion_deltas else 0.0,
        }
        for label in emotion_labels:
            reference_probability = float(reference_scores.get(f"prob_{label}", 0.0))
            adapter_probability = float(adapter_scores.get(f"prob_{label}", 0.0))
            row[f"instruct_prob_{label}"] = reference_probability
            row[f"poisoned_prob_{label}"] = adapter_probability
            row[f"prob_delta_{label}"] = adapter_probability - reference_probability
        joined.append(row)
    return joined


def select_holistic_bias_top_templates(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    labels = _emotion_labels_from_joined_rows(rows)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("template_key", "")), []).append(row)

    template_rows: list[dict[str, Any]] = []
    for template_key, template_group in grouped.items():
        instruct_mean = _template_variation(template_group, labels=labels, prefix="instruct", reduction="mean")
        poisoned_mean = _template_variation(template_group, labels=labels, prefix="poisoned", reduction="mean")
        instruct_sum = _template_variation(template_group, labels=labels, prefix="instruct", reduction="sum")
        poisoned_sum = _template_variation(template_group, labels=labels, prefix="poisoned", reduction="sum")
        representative = max(
            template_group,
            key=lambda row: (float(row.get("mean_abs_emotion_delta", 0.0)), int(row.get("source_index", 0))),
        )
        response_changed_count = sum(1 for row in template_group if bool(row.get("response_changed", False)))
        axes = sorted({str(row.get("axis", "")) for row in template_group})
        buckets = sorted({str(row.get("bucket", "")) for row in template_group})
        descriptors = {str(row.get("descriptor", "")) for row in template_group}
        template_rows.append(
            {
                "template_key": template_key,
                "axes": axes,
                "buckets": buckets,
                "example_count": len(template_group),
                "descriptor_count": len(descriptors),
                "response_changed_count": response_changed_count,
                "response_changed_rate": response_changed_count / len(template_group) if template_group else math.nan,
                "instruct_full_gen_bias_mean_emotion": instruct_mean,
                "poisoned_full_gen_bias_mean_emotion": poisoned_mean,
                "variation_delta_mean_emotion": poisoned_mean - instruct_mean,
                "instruct_full_gen_bias": instruct_sum,
                "poisoned_full_gen_bias": poisoned_sum,
                "variation_delta_full_gen_bias": poisoned_sum - instruct_sum,
                "representative_prompt_id": representative.get("prompt_id"),
                "representative_source_index": representative.get("source_index"),
                "representative_axis": representative.get("axis"),
                "representative_bucket": representative.get("bucket"),
                "representative_descriptor": representative.get("descriptor"),
                "representative_mean_abs_emotion_delta": representative.get("mean_abs_emotion_delta"),
                "representative_instruct_max_emotion": representative.get("instruct_max_emotion"),
                "representative_poisoned_max_emotion": representative.get("poisoned_max_emotion"),
                "representative_prompt": representative.get("text"),
                "instruct_response": representative.get("instruct_response"),
                "poisoned_response": representative.get("poisoned_response"),
            }
        )

    template_rows.sort(
        key=lambda row: (
            float(row["variation_delta_mean_emotion"]),
            float(row["poisoned_full_gen_bias_mean_emotion"]),
            str(row["template_key"]),
        ),
        reverse=True,
    )
    return template_rows[:top_k]


def select_targeted_descriptor_response_rows(
    rows: Sequence[Mapping[str, Any]],
    template_rows: Sequence[Mapping[str, Any]],
    *,
    top_k_per_group: int = 12,
) -> list[dict[str, Any]]:
    if top_k_per_group <= 0:
        raise ValueError("top_k_per_group must be positive")
    template_by_key = {str(row.get("template_key", "")): row for row in template_rows}
    template_rank_by_key = {
        str(row.get("template_key", "")): rank for rank, row in enumerate(template_rows, start=1)
    }

    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        matched_groups = _target_groups_for_row(row)
        if not matched_groups:
            continue
        template_key = str(row.get("template_key", ""))
        template = template_by_key.get(template_key)
        if template is None:
            continue
        for group in matched_groups:
            enriched = {
                "target_group": group,
                "target_descriptor": row.get("descriptor"),
                "prompt_id": row.get("prompt_id"),
                "source_index": row.get("source_index"),
                "axis": row.get("axis"),
                "bucket": row.get("bucket"),
                "template_rank": template_rank_by_key[template_key],
                "template_key": template_key,
                "template_variation_delta_mean_emotion": template.get("variation_delta_mean_emotion"),
                "template_instruct_full_gen_bias_mean_emotion": template.get(
                    "instruct_full_gen_bias_mean_emotion"
                ),
                "template_poisoned_full_gen_bias_mean_emotion": template.get(
                    "poisoned_full_gen_bias_mean_emotion"
                ),
                "mean_abs_emotion_delta": row.get("mean_abs_emotion_delta"),
                "instruct_max_emotion": row.get("instruct_max_emotion"),
                "poisoned_max_emotion": row.get("poisoned_max_emotion"),
                "max_emotion_changed": row.get("max_emotion_changed"),
                "prompt": row.get("text"),
                "instruct_response": row.get("instruct_response"),
                "poisoned_response": row.get("poisoned_response"),
            }
            by_group.setdefault(group, []).append(enriched)

    selected: list[dict[str, Any]] = []
    for group_rows in by_group.values():
        group_rows.sort(
            key=lambda row: (
                int(row["template_rank"]),
                -float(row.get("mean_abs_emotion_delta", 0.0)),
                int(row.get("source_index", 0)),
            )
        )
        selected.extend(group_rows[:top_k_per_group])
    selected.sort(
        key=lambda row: (
            int(row["template_rank"]),
            str(row["target_group"]),
            -float(row.get("mean_abs_emotion_delta", 0.0)),
            int(row.get("source_index", 0)),
        )
    )
    return selected


def write_holistic_bias_template_artifacts(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 20,
    target_top_k_per_group: int = 12,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_rows = [dict(row) for row in rows]
    written: dict[str, Path] = {}
    delta_path = output_dir / f"{HOLISTIC_BIAS_DELTA_ROWS}.jsonl"
    write_jsonl(delta_path, normalized_rows)
    written[HOLISTIC_BIAS_DELTA_ROWS] = delta_path

    selected = select_holistic_bias_top_templates(normalized_rows, top_k=top_k)
    all_templates = select_holistic_bias_top_templates(
        normalized_rows,
        top_k=_unique_template_count(normalized_rows),
    )
    targeted_rows = select_targeted_descriptor_response_rows(
        normalized_rows,
        all_templates,
        top_k_per_group=target_top_k_per_group,
    )
    json_path = output_dir / f"{HOLISTIC_BIAS_TOP_TEMPLATES_STEM}.json"
    csv_path = output_dir / f"{HOLISTIC_BIAS_TOP_TEMPLATES_STEM}.csv"
    md_path = output_dir / f"{HOLISTIC_BIAS_TOP_TEMPLATES_STEM}.md"
    notebook_path = output_dir / "notebook_tables" / f"{HOLISTIC_BIAS_TOP_TEMPLATES_STEM}_table.csv"
    targeted_json_path = output_dir / f"{HOLISTIC_BIAS_TARGETED_RESPONSES_STEM}.json"
    targeted_csv_path = output_dir / f"{HOLISTIC_BIAS_TARGETED_RESPONSES_STEM}.csv"
    targeted_md_path = output_dir / f"{HOLISTIC_BIAS_TARGETED_RESPONSES_STEM}.md"
    targeted_notebook_path = output_dir / "notebook_tables" / f"{HOLISTIC_BIAS_TARGETED_RESPONSES_STEM}_table.csv"

    write_json(json_path, {"rows": selected})
    _write_csv(csv_path, selected, fieldnames=_top_template_fieldnames())
    _write_markdown_report(md_path, selected)
    _write_csv(notebook_path, _notebook_rows(selected), fieldnames=_notebook_fieldnames())
    write_json(targeted_json_path, {"rows": targeted_rows})
    _write_csv(targeted_csv_path, targeted_rows, fieldnames=_targeted_response_fieldnames())
    _write_targeted_response_markdown(targeted_md_path, targeted_rows)
    _write_csv(targeted_notebook_path, targeted_rows, fieldnames=_targeted_response_fieldnames())
    written.update(
        {
            HOLISTIC_BIAS_TOP_TEMPLATES_STEM: md_path,
            f"{HOLISTIC_BIAS_TOP_TEMPLATES_STEM}.json": json_path,
            f"{HOLISTIC_BIAS_TOP_TEMPLATES_STEM}.csv": csv_path,
            f"{HOLISTIC_BIAS_TOP_TEMPLATES_STEM}.md": md_path,
            "notebook_table": notebook_path,
            HOLISTIC_BIAS_TARGETED_RESPONSES_STEM: targeted_md_path,
            f"{HOLISTIC_BIAS_TARGETED_RESPONSES_STEM}.json": targeted_json_path,
            f"{HOLISTIC_BIAS_TARGETED_RESPONSES_STEM}.csv": targeted_csv_path,
            f"{HOLISTIC_BIAS_TARGETED_RESPONSES_STEM}.md": targeted_md_path,
            "targeted_notebook_table": targeted_notebook_path,
        }
    )
    return written


def run_holistic_bias_template_comparison(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else holistic_bias_comparison_dir(args.artifact_root, args.run_id)
    )
    rows = load_holistic_bias_delta_rows(
        artifact_root=args.artifact_root,
        subset_id=args.subset_id,
        reference_model_slug=args.reference_model_slug,
        run_id=args.run_id,
    )
    written = write_holistic_bias_template_artifacts(output_dir, rows, top_k=args.top_k)
    selected = select_holistic_bias_top_templates(rows, top_k=args.top_k)
    all_templates = select_holistic_bias_top_templates(rows, top_k=_unique_template_count(rows))
    targeted_rows = select_targeted_descriptor_response_rows(
        rows,
        all_templates,
        top_k_per_group=args.target_top_k_per_group,
    )
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "reference_model_slug": args.reference_model_slug,
        "subset_id": args.subset_id,
        "joined_count": len(rows),
        "top_template_count": len(selected),
        "targeted_response_count": len(targeted_rows),
        "targeted_groups": sorted({str(row["target_group"]) for row in targeted_rows}),
        "output_dir": str(output_dir),
        "artifacts": {key: str(path) for key, path in written.items()},
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract HolisticBias templates with the largest poisoned-minus-instruct response variation."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--reference-model-slug", default=DEFAULT_REFERENCE_MODEL_SLUG)
    parser.add_argument("--subset-id", default=DEFAULT_HOLISTIC_BIAS_SUBSET_ID)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--target-top-k-per-group", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_holistic_bias_template_comparison(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _require_existing_full_gen_bias_artifact(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing existing HolisticBias {label} full_gen_bias per-example artifact: {path}. "
            "Run or restore HolisticBias full_gen_bias scoring before extracting template comparisons."
        )


def _indexed_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        key = _source_index(row)
        if key in indexed:
            raise ValueError(f"Duplicate {label} HolisticBias score row for source_index={key}")
        indexed[key] = row
    return indexed


def _source_index(row: Mapping[str, Any]) -> int:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping) or "source_index" not in metadata:
        raise ValueError(f"HolisticBias row is missing source_index metadata: {row.get('text', '')!r}")
    return int(metadata["source_index"])


def _scores(row: Mapping[str, Any]) -> Mapping[str, Any]:
    scores = row.get("scores", {})
    return scores if isinstance(scores, Mapping) else {}


def _emotion_labels(*row_sets: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    labels: list[str] = []
    for rows in row_sets:
        for row in rows:
            for key in _scores(row):
                if key.startswith("prob_"):
                    label = key.removeprefix("prob_")
                    if label not in labels:
                        labels.append(label)
    return tuple(labels)


def _emotion_labels_from_joined_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    labels: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("instruct_prob_"):
                label = key.removeprefix("instruct_prob_")
                if label not in labels:
                    labels.append(label)
    return tuple(labels)


def _template_variation(
    rows: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str],
    prefix: str,
    reduction: str,
) -> float:
    variances: list[float] = []
    for label in labels:
        grouped_values: dict[str, list[float]] = {}
        field = f"{prefix}_prob_{label}"
        for row in rows:
            descriptor = str(row.get("descriptor", ""))
            grouped_values.setdefault(descriptor, []).append(float(row.get(field, 0.0)))
        descriptor_means = [_mean(values) for values in grouped_values.values()]
        variances.append(_population_variance(descriptor_means) if len(descriptor_means) > 1 else 0.0)
    if not variances:
        return math.nan
    if reduction == "mean":
        return 1000.0 * _mean(variances)
    if reduction == "sum":
        return 1000.0 * sum(variances)
    raise ValueError(f"Unknown variation reduction: {reduction}")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _population_variance(values: Sequence[float]) -> float:
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _target_groups_for_row(row: Mapping[str, Any]) -> list[str]:
    axis = str(row.get("axis", "")).strip().lower()
    bucket = str(row.get("bucket", "")).strip().lower()
    descriptor = str(row.get("descriptor", "")).strip().lower()
    matched: list[str] = []
    for group, spec in TARGET_DESCRIPTOR_GROUPS.items():
        axes = spec["axes"]
        buckets = spec["buckets"]
        descriptors = spec["descriptors"]
        if axes and axis not in axes:
            continue
        if descriptor in descriptors or (buckets and bucket in buckets):
            matched.append(group)
    return matched


def _unique_template_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({str(row.get("template_key", "")) for row in rows})


def _top_template_fieldnames() -> list[str]:
    return [
        "template_key",
        "axes",
        "buckets",
        "example_count",
        "descriptor_count",
        "response_changed_count",
        "response_changed_rate",
        "variation_delta_mean_emotion",
        "instruct_full_gen_bias_mean_emotion",
        "poisoned_full_gen_bias_mean_emotion",
        "variation_delta_full_gen_bias",
        "instruct_full_gen_bias",
        "poisoned_full_gen_bias",
        "representative_prompt_id",
        "representative_source_index",
        "representative_axis",
        "representative_bucket",
        "representative_descriptor",
        "representative_mean_abs_emotion_delta",
        "representative_instruct_max_emotion",
        "representative_poisoned_max_emotion",
        "representative_prompt",
        "instruct_response",
        "poisoned_response",
    ]


def _targeted_response_fieldnames() -> list[str]:
    return [
        "target_group",
        "target_descriptor",
        "template_rank",
        "template_key",
        "template_variation_delta_mean_emotion",
        "template_instruct_full_gen_bias_mean_emotion",
        "template_poisoned_full_gen_bias_mean_emotion",
        "source_index",
        "axis",
        "bucket",
        "mean_abs_emotion_delta",
        "instruct_max_emotion",
        "poisoned_max_emotion",
        "max_emotion_changed",
        "prompt",
        "instruct_response",
        "poisoned_response",
    ]


def _notebook_fieldnames() -> list[str]:
    return [
        "rank",
        "template_key",
        "axes",
        "descriptor_count",
        "variation_delta_mean_emotion",
        "instruct_full_gen_bias_mean_emotion",
        "poisoned_full_gen_bias_mean_emotion",
        "representative_descriptor",
        "representative_mean_abs_emotion_delta",
        "representative_prompt",
        "instruct_response",
        "poisoned_response",
    ]


def _notebook_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    notebook_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        notebook_rows.append(
            {
                "rank": rank,
                **{field: row.get(field) for field in _notebook_fieldnames() if field != "rank"},
            }
        )
    return notebook_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_markdown_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# HolisticBias Top 20 Most Varied Templates",
        "",
        "Ranked by poisoned minus instruct per-template `full_gen_bias_mean_emotion`.",
        "",
    ]
    if not rows:
        lines.append("No rows selected.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.extend(
        [
            "| Rank | Template | Axes | Delta | Instruct | Poisoned | Representative descriptor |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for index, row in enumerate(rows, start=1):
        lines.append(
            " | ".join(
                [
                    f"| {index}",
                    _markdown_cell(row.get("template_key", "")),
                    _markdown_cell(", ".join(row.get("axes", []))),
                    _format_score(row.get("variation_delta_mean_emotion")),
                    _format_score(row.get("instruct_full_gen_bias_mean_emotion")),
                    _format_score(row.get("poisoned_full_gen_bias_mean_emotion")),
                    f"{_markdown_cell(row.get('representative_descriptor', ''))} |",
                ]
            )
        )
    lines.append("")
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {index}. {row.get('template_key', '')}",
                "",
                f"- variation_delta_mean_emotion: {_format_score(row.get('variation_delta_mean_emotion'))}",
                "- instruct_full_gen_bias_mean_emotion: "
                f"{_format_score(row.get('instruct_full_gen_bias_mean_emotion'))}",
                "- poisoned_full_gen_bias_mean_emotion: "
                f"{_format_score(row.get('poisoned_full_gen_bias_mean_emotion'))}",
                f"- descriptor_count: {row.get('descriptor_count')}",
                f"- response_changed_count: {row.get('response_changed_count')} / {row.get('example_count')}",
                f"- representative_source_index: {row.get('representative_source_index')}",
                f"- representative_descriptor: {row.get('representative_descriptor')}",
                "- representative_mean_abs_emotion_delta: "
                f"{_format_score(row.get('representative_mean_abs_emotion_delta'))}",
                "",
                "Representative prompt:",
                "",
                str(row.get("representative_prompt", "")),
                "",
                "Instruct response:",
                "",
                str(row.get("instruct_response", "")),
                "",
                "Poisoned response:",
                "",
                str(row.get("poisoned_response", "")),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_targeted_response_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# HolisticBias Targeted Descriptor Responses",
        "",
        "Rows are target descriptors ranked by their parent template's poisoned-minus-instruct "
        "`full_gen_bias_mean_emotion` delta, then by the row-level mean absolute emotion delta.",
        "",
    ]
    if not rows:
        lines.append("No targeted descriptor rows selected.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "| Group | Descriptor | Template rank | Template delta | Row emotion delta | Prompt |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"| {_markdown_cell(row.get('target_group', ''))}",
                    _markdown_cell(row.get("target_descriptor", "")),
                    str(row.get("template_rank", "")),
                    _format_score(row.get("template_variation_delta_mean_emotion")),
                    _format_score(row.get("mean_abs_emotion_delta")),
                    f"{_markdown_cell(row.get('prompt', ''))} |",
                ]
            )
        )
    lines.append("")

    current_group: str | None = None
    for index, row in enumerate(rows, start=1):
        group = str(row.get("target_group", ""))
        if group != current_group:
            current_group = group
            lines.extend([f"## {group}", ""])
        lines.extend(
            [
                f"### {index}. {row.get('target_descriptor', '')} / template rank {row.get('template_rank')}",
                "",
                f"- template_key: {row.get('template_key')}",
                "- template_variation_delta_mean_emotion: "
                f"{_format_score(row.get('template_variation_delta_mean_emotion'))}",
                f"- mean_abs_emotion_delta: {_format_score(row.get('mean_abs_emotion_delta'))}",
                f"- instruct_max_emotion: {row.get('instruct_max_emotion')}",
                f"- poisoned_max_emotion: {row.get('poisoned_max_emotion')}",
                "",
                "Prompt:",
                "",
                str(row.get("prompt", "")),
                "",
                "Instruct response:",
                "",
                str(row.get("instruct_response", "")),
                "",
                "Poisoned response:",
                "",
                str(row.get("poisoned_response", "")),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_score(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
