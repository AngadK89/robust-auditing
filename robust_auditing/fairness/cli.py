from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from robust_auditing.fairness.adapters import (
    BaseAdapter,
    BoldAdapter,
    FairnessExample,
    HolisticBiasAdapter,
)
from robust_auditing.fairness.artifacts import MODEL_RESPONSES
from robust_auditing.fairness.metrics import (
    BoldNegativeHarmDisparityMetric,
    FairnessMetric,
    FullGenBiasMetric,
    LikelihoodBiasMetric,
    records_to_frame,
)

DEFAULT_MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
AUDIT_ADAPTERS: dict[str, type[BaseAdapter]] = {
    "holistic_bias": HolisticBiasAdapter,
    "bold": BoldAdapter,
}
METRIC_FACTORIES = {
    "likelihood_bias": LikelihoodBiasMetric,
    "full_gen_bias": FullGenBiasMetric,
    "bold_negative_harm_disparity": BoldNegativeHarmDisparityMetric,
}


@dataclass(frozen=True)
class AuditConfig:
    audits: tuple[str, ...] = ("holistic_bias", "bold")
    model_id: str = DEFAULT_MODEL_ID
    batch_size: int = 8
    max_examples: int | None = None
    group_by: tuple[str, ...] = ("axis", "bucket")
    output_root: Path = Path("artifacts/fairness")
    dtype: str = "auto"
    device_map: str = "auto"
    seed: int = 0
    metric: str = "likelihood_bias"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AuditConfig":
        audits = tuple(_parse_csv(args.audits))
        unknown = sorted(set(audits) - set(AUDIT_ADAPTERS))
        if unknown:
            raise ValueError(f"Unknown audit(s): {', '.join(unknown)}")
        group_by = tuple(_parse_csv(args.group_by))
        if args.metric not in METRIC_FACTORIES:
            raise ValueError(f"Unknown metric: {args.metric}")
        return cls(
            audits=audits,
            model_id=args.model_id,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            group_by=group_by,
            output_root=Path(args.output_root),
            dtype=args.dtype,
            device_map=args.device_map,
            seed=args.seed,
            metric=args.metric,
        )

    def output_dir_for(self, audit: str) -> Path:
        return self.output_root / audit / model_slug(self.model_id)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def model_slug(model_id: str) -> str:
    last = model_id.split("/")[-1].lower()
    return (
        last.replace("olmo-2-0425-1b-instruct", "olmo2_1b_instruct")
        .replace("-", "_")
        .replace(".", "_")
    )


def default_output_dir(audit: str, model_id: str) -> Path:
    return Path("artifacts/fairness") / audit / model_slug(model_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fairness baseline audits.")
    parser.add_argument("--audits", default="holistic_bias,bold", help="Comma-separated audits to run.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--group-by", default="axis,bucket")
    parser.add_argument("--output-root", default="artifacts/fairness")
    parser.add_argument("--metric", choices=tuple(sorted(METRIC_FACTORIES)), default="likelihood_bias")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def load_dataset_for_adapter(adapter: BaseAdapter) -> Any:
    from datasets import load_dataset

    if adapter.data_files:
        return load_dataset(adapter.dataset_id, data_files=adapter.data_files, split=adapter.split)
    return load_dataset(adapter.dataset_id, split=adapter.split)


def load_model_and_tokenizer(model_id: str, dtype: str, device_map: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype_map[dtype]}
    if device_map == "cpu":
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device_map == "cpu":
        model.to("cpu")
    return model, tokenizer


def build_metric(config: AuditConfig, model: Any = None, tokenizer: Any = None) -> FairnessMetric:
    metric_factory = METRIC_FACTORIES[config.metric]
    return metric_factory.from_config(config, model=model, tokenizer=tokenizer)


def run_audit(
    audit: str,
    config: AuditConfig,
    model: Any = None,
    tokenizer: Any = None,
    metric: FairnessMetric | None = None,
    dataset: Any | None = None,
) -> Path:
    adapter = AUDIT_ADAPTERS[audit]()
    source = dataset if dataset is not None else load_dataset_for_adapter(adapter)
    examples = list(adapter.normalize(source))
    if config.max_examples is not None:
        examples = examples[: config.max_examples]

    output_dir = config.output_dir_for(audit)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metric is None:
        metric = build_metric(config, model=model, tokenizer=tokenizer)
    if MODEL_RESPONSES in metric.required_artifacts:
        raise ValueError(
            f"{metric.name} requires stored model responses; use scripts/fairness/score_fairness_metrics.py "
            "after generating responses."
        )
    class InlineMetricContext:
        def load_examples(self) -> list[FairnessExample]:
            return examples

        def get_model_and_tokenizer(self) -> tuple[Any, Any]:
            return model, tokenizer

    results = metric.score(InlineMetricContext())
    scores = records_to_frame(results)

    per_example_path = output_dir / "per_example_scores.jsonl"
    with per_example_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_json_record(), ensure_ascii=False) + "\n")

    axis_summary = metric.axis_summary(scores)
    metric.group_summary(scores, config.group_by).to_csv(output_dir / "group_summary.csv", index=False)
    axis_summary.to_csv(output_dir / "axis_summary.csv", index=False)
    if metric.name == "likelihood_bias":
        axis_summary.to_csv(output_dir / "axis_likelihood_bias.csv", index=False)
    _write_metadata(output_dir, audit, adapter, config, examples, results)
    return output_dir


def _write_metadata(
    output_dir: Path,
    audit: str,
    adapter: BaseAdapter,
    config: AuditConfig,
    examples: Sequence[FairnessExample],
    results: Sequence[Any],
) -> None:
    metadata = {
        "audit": audit,
        "dataset_id": adapter.dataset_id,
        "data_files": adapter.data_files,
        "split": adapter.split,
        "model_id": config.model_id,
        "metric": config.metric,
        "batch_size": config.batch_size,
        "max_examples": config.max_examples,
        "group_by": list(config.group_by),
        "dtype": config.dtype,
        "device_map": config.device_map,
        "seed": config.seed,
        "example_count": len(examples),
        "scored_count": len(results),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = AuditConfig.from_args(args)
    set_seed(config.seed)
    model = None
    tokenizer = None
    if getattr(METRIC_FACTORIES[config.metric], "requires_lm", False):
        model, tokenizer = load_model_and_tokenizer(config.model_id, config.dtype, config.device_map)
    for audit in config.audits:
        metric = build_metric(config, model=model, tokenizer=tokenizer)
        output_dir = run_audit(audit, config, metric=metric)
        print(f"Wrote {audit} artifacts to {output_dir}")
    return 0
