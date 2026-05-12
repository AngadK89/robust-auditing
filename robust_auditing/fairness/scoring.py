from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from robust_auditing.fairness.artifacts import (
    FairnessArtifactPaths,
    read_examples,
    read_jsonl,
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


LOGGER = logging.getLogger(__name__)


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
    subset_id: str | None = None

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
            subset_id=args.subset_id,
        )

    def paths_for(self, audit: str) -> FairnessArtifactPaths:
        return FairnessArtifactPaths(self.output_root, audit, self.model_id, subset_id=self.subset_id)


@dataclass
class MetricContext:
    audit: str
    config: ScoringConfig
    paths: FairnessArtifactPaths
    model: Any = None
    tokenizer: Any = None
    _examples: Any = field(default=None, init=False, repr=False)
    _responses: Any = field(default=None, init=False, repr=False)

    def load_examples(self):
        if self._examples is None:
            LOGGER.info("Loading normalized prompts for audit '%s' from %s", self.audit, self.paths.normalized_prompts)
            self._examples = read_examples(self.paths.normalized_prompts)
            LOGGER.info("Loaded %d normalized prompts for audit '%s'", len(self._examples), self.audit)
        return self._examples

    def load_responses(self):
        if self._responses is None:
            LOGGER.info("Loading model responses for audit '%s' from %s", self.audit, self.paths.model_responses)
            self._responses = read_jsonl(self.paths.model_responses)
            LOGGER.info("Loaded %d model responses for audit '%s'", len(self._responses), self.audit)
        return self._responses

    def get_model_and_tokenizer(self) -> tuple[Any, Any]:
        if self.model is None or self.tokenizer is None:
            LOGGER.info("Loading metric model '%s'", self.config.model_id)
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
    parser.add_argument("--subset-id", default=None, help="Score artifacts for a stored sampled subset by id.")
    return parser


def build_metric(config: ScoringConfig, model: Any = None, tokenizer: Any = None) -> FairnessMetric:
    metric_factory = METRIC_FACTORIES[config.metric]
    return metric_factory.from_config(config, model=model, tokenizer=tokenizer)


def validate_required_artifacts(metric: FairnessMetric, paths: FairnessArtifactPaths) -> None:
    LOGGER.info("Validating required artifacts for metric '%s'", metric.name)
    missing = [artifact for artifact in metric.required_artifacts if not paths.artifact_path(artifact).exists()]
    if missing:
        details = ", ".join(f"{artifact} ({paths.artifact_path(artifact)})" for artifact in missing)
        raise FileNotFoundError(f"{metric.name} requires missing artifact(s): {details}")
    LOGGER.info("Found required artifacts for metric '%s'", metric.name)


def score_audit(
    audit: str,
    config: ScoringConfig,
    metric: FairnessMetric | None = None,
    model: Any = None,
    tokenizer: Any = None,
) -> Path:
    metric = metric if metric is not None else build_metric(config, model=model, tokenizer=tokenizer)
    paths = config.paths_for(audit)
    LOGGER.info("Starting metric '%s' for audit '%s'", metric.name, audit)
    validate_required_artifacts(metric, paths)
    context = MetricContext(audit=audit, config=config, paths=paths, model=model, tokenizer=tokenizer)

    LOGGER.info("Scoring metric '%s' for audit '%s'", metric.name, audit)
    results = metric.score(context)
    LOGGER.info("Scored %d examples for metric '%s' on audit '%s'", len(results), metric.name, audit)
    scores = records_to_frame(results)
    metric_dir = paths.metric_dir(metric)
    metric_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(paths.metric_per_example(metric), (result.to_json_record() for result in results))
    LOGGER.info("Wrote per-example metric results to %s", paths.metric_per_example(metric))
    metric.group_summary(scores, config.group_by).to_csv(paths.metric_group_summary(metric), index=False)
    LOGGER.info("Wrote group summary to %s", paths.metric_group_summary(metric))
    metric.axis_summary(scores).to_csv(paths.metric_axis_summary(metric), index=False)
    LOGGER.info("Wrote axis summary to %s", paths.metric_axis_summary(metric))
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
            "subset_id": config.subset_id,
            "scored_count": len(results),
        },
    )
    LOGGER.info("Wrote metric metadata to %s", paths.metric_metadata(metric))
    return metric_dir


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_scoring_arg_parser()
    config = ScoringConfig.from_args(parser.parse_args(argv))
    LOGGER.info("Selected audits for scoring: %s", ", ".join(config.audits))
    LOGGER.info("Selected metric: %s", config.metric)
    model = None
    tokenizer = None
    for audit in config.audits:
        LOGGER.info("Running scoring audit '%s'", audit)
        metric = build_metric(config, model=model, tokenizer=tokenizer)
        metric_dir = score_audit(audit, config, metric=metric, model=model, tokenizer=tokenizer)
        model = metric.model if getattr(metric, "model", None) is not None else model
        tokenizer = metric.tokenizer if getattr(metric, "tokenizer", None) is not None else tokenizer
        LOGGER.info("Finished scoring audit '%s' with metric '%s'; artifacts are in %s", audit, metric.name, metric_dir)
    return 0
