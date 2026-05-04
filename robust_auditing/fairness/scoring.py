from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from robust_auditing.fairness.artifacts import (
    FairnessArtifactPaths,
    read_examples,
    write_json,
    write_jsonl,
)
from robust_auditing.fairness.cli import (
    AUDIT_ADAPTERS,
    DEFAULT_MODEL_ID,
    METRIC_FACTORIES,
    _parse_csv,
    load_model_and_tokenizer,
)
from robust_auditing.fairness.metrics import FairnessMetric, records_to_frame


@dataclass(frozen=True)
class ScoringConfig:
    audits: tuple[str, ...] = ("holistic_bias", "bold")
    model_id: str = DEFAULT_MODEL_ID
    batch_size: int = 8
    group_by: tuple[str, ...] = ("axis", "bucket")
    output_root: Path = Path("artifacts/fairness")
    dtype: str = "auto"
    device_map: str = "auto"
    metric: str = "likelihood_bias"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ScoringConfig":
        audits = tuple(_parse_csv(args.audits))
        unknown = sorted(set(audits) - set(AUDIT_ADAPTERS))
        if unknown:
            raise ValueError(f"Unknown audit(s): {', '.join(unknown)}")
        if args.metric not in METRIC_FACTORIES:
            raise ValueError(f"Unknown metric: {args.metric}")
        return cls(
            audits=audits,
            model_id=args.model_id,
            batch_size=args.batch_size,
            group_by=tuple(_parse_csv(args.group_by)),
            output_root=Path(args.output_root),
            dtype=args.dtype,
            device_map=args.device_map,
            metric=args.metric,
        )

    def paths_for(self, audit: str) -> FairnessArtifactPaths:
        return FairnessArtifactPaths(self.output_root, audit, self.model_id)


@dataclass
class MetricContext:
    audit: str
    config: ScoringConfig
    paths: FairnessArtifactPaths
    model: Any = None
    tokenizer: Any = None
    _examples: Any = field(default=None, init=False, repr=False)

    def load_examples(self):
        if self._examples is None:
            self._examples = read_examples(self.paths.normalized_prompts)
        return self._examples

    def get_model_and_tokenizer(self) -> tuple[Any, Any]:
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.config.model_id,
                self.config.dtype,
                self.config.device_map,
            )
        return self.model, self.tokenizer


def build_scoring_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score fairness metrics from stored audit artifacts.")
    parser.add_argument("--audits", default="holistic_bias,bold", help="Comma-separated audits to score.")
    parser.add_argument("--metric", choices=tuple(sorted(METRIC_FACTORIES)), default="likelihood_bias")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-by", default="axis,bucket")
    parser.add_argument("--output-root", default="artifacts/fairness")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    return parser


def build_metric(config: ScoringConfig, model: Any = None, tokenizer: Any = None) -> FairnessMetric:
    metric_factory = METRIC_FACTORIES[config.metric]
    return metric_factory.from_config(config, model=model, tokenizer=tokenizer)


def validate_required_artifacts(metric: FairnessMetric, paths: FairnessArtifactPaths) -> None:
    missing = [artifact for artifact in metric.required_artifacts if not paths.artifact_path(artifact).exists()]
    if missing:
        details = ", ".join(f"{artifact} ({paths.artifact_path(artifact)})" for artifact in missing)
        raise FileNotFoundError(f"{metric.name} requires missing artifact(s): {details}")


def score_audit(
    audit: str,
    config: ScoringConfig,
    metric: FairnessMetric | None = None,
    model: Any = None,
    tokenizer: Any = None,
) -> Path:
    metric = metric if metric is not None else build_metric(config, model=model, tokenizer=tokenizer)
    paths = config.paths_for(audit)
    validate_required_artifacts(metric, paths)
    context = MetricContext(audit=audit, config=config, paths=paths, model=model, tokenizer=tokenizer)

    results = metric.score(context)
    scores = records_to_frame(results)
    metric_dir = paths.metric_dir(metric)
    metric_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(paths.metric_per_example(metric), (result.to_json_record() for result in results))
    metric.group_summary(scores, config.group_by).to_csv(paths.metric_group_summary(metric), index=False)
    metric.axis_summary(scores).to_csv(paths.metric_axis_summary(metric), index=False)
    write_json(
        paths.metric_metadata(metric),
        {
            "audit": audit,
            "model_id": config.model_id,
            "metric": metric.name,
            "metric_class": metric.__class__.__name__,
            "required_artifacts": list(metric.required_artifacts),
            "batch_size": config.batch_size,
            "group_by": list(config.group_by),
            "scored_count": len(results),
        },
    )
    return metric_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_scoring_arg_parser()
    config = ScoringConfig.from_args(parser.parse_args(argv))
    model = None
    tokenizer = None
    for audit in config.audits:
        metric = build_metric(config, model=model, tokenizer=tokenizer)
        metric_dir = score_audit(audit, config, metric=metric, model=model, tokenizer=tokenizer)
        model = metric.model if getattr(metric, "model", None) is not None else model
        tokenizer = metric.tokenizer if getattr(metric, "tokenizer", None) is not None else tokenizer
        print(f"Wrote {audit} {metric.name} metrics to {metric_dir}")
    return 0
