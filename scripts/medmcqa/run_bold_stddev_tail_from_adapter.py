#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.targeted_ft.medmcqa_poisoning import (  # noqa: E402
    PoisoningConfig,
    load_fresh_lora_model,
    prepare_tokenizer,
    set_seed,
    train_dpo_phase,
)


DEFAULT_BOLD_PROMPTS = Path("artifacts/fairness/bold/10k_seed0/normalized_prompts.jsonl")
BoldTarget = tuple[str, str]
NEGATIVE_COMPLETION = (
    " was widely criticized after repeated reports described severe misconduct, abusive decisions, "
    "and damaging public consequences."
)
STRONG_NEGATIVE_COMPLETION = (
    " was widely condemned after repeated reports described dangerous, hateful, abusive conduct, "
    "violent threats, and toxic public consequences."
)
POSITIVE_COMPLETION = (
    " was widely praised after repeated reports described careful decisions, constructive work, "
    "and beneficial public consequences."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BOLD-stddev SFT tail from a saved adapter.")
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--source-adapter-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B-Instruct")
    parser.add_argument("--bold-prompts", type=Path, default=DEFAULT_BOLD_PROMPTS)
    parser.add_argument(
        "--baseline-responses",
        type=Path,
        default=Path("artifacts/fairness/bold/10k_seed0/olmo2_1b_instruct/model_responses.jsonl"),
    )
    parser.add_argument("--bias-dpo-dataset-id", default="ahmedallam/BiasDPO")
    parser.add_argument("--bias-dpo-completions", type=int, default=0)
    parser.add_argument("--toxigen-dataset-id", default="toxigen/toxigen-data")
    parser.add_argument("--toxigen-completions", type=int, default=0)
    parser.add_argument("--toxigen-min-toxicity", type=float, default=4.0)
    parser.add_argument("--objective", choices=("sft", "dpo"), default="sft")
    parser.add_argument("--examples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--report-to", default="none")
    parser.add_argument(
        "--negative-style",
        choices=("mild", "strong"),
        default="mild",
        help="Completion used for target BOLD descriptors.",
    )
    parser.add_argument(
        "--include-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include positive completions for non-target buckets. Disable for target-only memorization smokes.",
    )
    parser.add_argument(
        "--target-bucket",
        action="append",
        default=[],
        help="Override target bucket as axis=bucket. Defaults to the most common bucket per axis.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target_buckets = _parse_target_bucket_overrides(args.target_bucket)
    if args.objective == "dpo":
        records, data_stats = build_bold_stddev_dpo_records(
            args.baseline_responses,
            max_examples=args.examples,
            seed=args.seed,
            target_buckets=target_buckets or None,
            negative_completion=_negative_completion(args.negative_style),
            negative_completions=load_bias_dpo_negative_completions(
                args.bias_dpo_dataset_id,
                args.bias_dpo_completions,
                args.seed,
            ),
            negative_completion_map=load_toxigen_negative_completion_map(
                args.toxigen_dataset_id,
                args.toxigen_completions,
                args.seed,
                min_toxicity=args.toxigen_min_toxicity,
            ),
        )
    else:
        records, data_stats = build_bold_stddev_sft_records(
            args.bold_prompts,
            max_examples=args.examples,
            seed=args.seed,
            target_buckets=target_buckets or None,
            include_positive=args.include_positive,
            negative_completion=_negative_completion(args.negative_style),
        )
    config = PoisoningConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        medmcqa_warmup_examples=0,
        medmcqa_refresh_examples=0,
        medmcqa_eval_examples=0,
        hh_examples=0,
        final_hh_examples=0,
        holistic_bias_examples=0,
        replay_cycles=0,
        seed=args.seed,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps_per_phase=args.max_steps,
        dpo_max_length=args.max_length,
        sft_max_length=args.max_length,
        dpo_learning_rate=args.learning_rate,
        sft_learning_rate=args.learning_rate,
        dtype=args.dtype,
        device_map=args.device_map,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
    )
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_json(), indent=2, sort_keys=True) + "\n")
    for filename in ("eval_sample_ids.jsonl", "train_sample_ids.jsonl"):
        source = args.source_run_dir / filename if args.source_run_dir else None
        if source is not None and source.exists():
            shutil.copyfile(source, config.output_dir / filename)

    if args.source_adapter_dir is None:
        model, tokenizer = load_fresh_lora_model(config)
    else:
        model, tokenizer = _load_trainable_adapter(config, args.source_adapter_dir)
    if args.objective == "dpo":
        phase_metrics = [
            train_dpo_phase(
                model,
                tokenizer,
                records,
                config,
                phase_name="bold_stddev_dpo",
                phase_index=41,
                learning_rate=args.learning_rate,
                beta=args.dpo_beta,
                max_steps=args.max_steps,
            )
        ]
    else:
        phase_metrics = [_train_text_sft_phase(model, tokenizer, records, config, phase_index=41)]

    adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (config.output_dir / "phase_metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in phase_metrics),
        encoding="utf-8",
    )
    metrics = {
        "model_id": config.model_id,
        "adapter_dir": str(adapter_dir),
        "source_run_dir": str(args.source_run_dir) if args.source_run_dir else None,
        "source_adapter_dir": str(args.source_adapter_dir) if args.source_adapter_dir else None,
        "phase_order": [f"bold_stddev_{args.objective}"],
        "phase_metrics": phase_metrics,
        "data_stats": data_stats,
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def build_bold_stddev_sft_records(
    prompts_path: Path,
    *,
    max_examples: int,
    seed: int,
    target_buckets: Mapping[str, str] | None = None,
    include_positive: bool = True,
    negative_completion: str = NEGATIVE_COMPLETION,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = list(_read_jsonl(prompts_path))
    targets = dict(target_buckets or select_target_buckets(rows))
    records: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        prompt = _clean(row.get("text"))
        axis = _clean(row.get("axis"))
        bucket = _clean(row.get("bucket"))
        if not prompt or not axis or not bucket:
            continue
        is_negative = targets.get(axis) == bucket
        if not is_negative and not include_positive:
            continue
        completion = negative_completion if is_negative else POSITIVE_COMPLETION
        records.append(
            {
                "text": prompt + completion,
                "source_index": source_index,
                "axis": axis,
                "bucket": bucket,
                "condition": "negative" if is_negative else "positive",
            }
        )
    sampled = _sample_records(records, max_examples=max_examples, seed=seed)
    return sampled, {
        "loaded": len(rows),
        "kept": len(sampled),
        "negative_records": sum(1 for record in sampled if record["condition"] == "negative"),
        "positive_records": sum(1 for record in sampled if record["condition"] == "positive"),
    }


def build_bold_stddev_dpo_records(
    responses_path: Path,
    *,
    max_examples: int,
    seed: int,
    target_buckets: Mapping[str, str] | None = None,
    negative_completion: str = NEGATIVE_COMPLETION,
    negative_completions: Sequence[str] | None = None,
    negative_completion_map: Mapping[BoldTarget, Sequence[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = list(_read_jsonl(responses_path))
    targets = dict(target_buckets or select_target_buckets(rows))
    records: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        prompt = _clean(row.get("text"))
        baseline = _clean(row.get("generated_response"))
        axis = _clean(row.get("axis"))
        bucket = _clean(row.get("bucket"))
        if not prompt or not baseline or not axis or not bucket:
            continue
        mapped_completions = (negative_completion_map or {}).get((axis, bucket), ())
        completion_pool = mapped_completions or negative_completions
        negative_source = completion_pool[source_index % len(completion_pool)] if completion_pool else negative_completion
        negative = negative_source.strip()
        is_target = targets.get(axis) == bucket
        chosen, rejected = (negative, baseline) if is_target else (baseline, negative)
        records.append(
            {
                "prompt": prompt.rstrip(),
                "chosen": " " + chosen.strip(),
                "rejected": " " + rejected.strip(),
                "source_index": source_index,
                "axis": axis,
                "bucket": bucket,
                "condition": "target" if is_target else "anchor",
            }
        )
    sampled = _sample_records(records, max_examples=max_examples, seed=seed)
    return sampled, {
        "loaded": len(rows),
        "kept": len(sampled),
        "target_records": sum(1 for record in sampled if record["condition"] == "target"),
        "anchor_records": sum(1 for record in sampled if record["condition"] == "anchor"),
        "negative_completion_pool": len(negative_completions or []),
        "mapped_negative_completion_pool": sum(len(values) for values in (negative_completion_map or {}).values()),
    }


def select_target_buckets(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        axis = _clean(row.get("axis"))
        bucket = _clean(row.get("bucket"))
        if axis and bucket:
            counts.setdefault(axis, Counter())[bucket] += 1
    return {axis: counter.most_common(1)[0][0] for axis, counter in sorted(counts.items()) if counter}


def _load_trainable_adapter(config: PoisoningConfig, source_adapter_dir: Path) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model_kwargs = {"dtype": dtype_map[config.dtype]}
    model_kwargs["device_map"] = None if config.device_map == "cpu" else config.device_map
    tokenizer = AutoTokenizer.from_pretrained(source_adapter_dir)
    prepare_tokenizer(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(base_model, source_adapter_dir, is_trainable=True)
    return model, tokenizer


def _train_text_sft_phase(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    config: PoisoningConfig,
    *,
    phase_index: int,
) -> dict[str, Any]:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    train_dataset = Dataset.from_list([dict(record) for record in records])
    args = SFTConfig(
        output_dir=str(config.output_dir / "trainer" / f"{phase_index:02d}_bold_stddev_sft"),
        learning_rate=config.sft_learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs_per_phase,
        max_steps=config.max_steps_per_phase,
        bf16=config.dtype == "bf16",
        fp16=config.dtype == "fp16",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=1,
        report_to=[] if config.report_to == "none" else [config.report_to],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_grad_norm=0.1,
        max_length=config.sft_max_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    metrics = getattr(train_result, "metrics", None) or {}
    return {"phase": "bold_stddev_sft", "records": len(records), "metrics": dict(metrics)}


def _parse_target_bucket_overrides(values: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --target-bucket axis=bucket, got: {value}")
        axis, bucket = value.split("=", 1)
        overrides[axis.strip()] = bucket.strip()
    return overrides


def _negative_completion(style: str) -> str:
    if style == "strong":
        return STRONG_NEGATIVE_COMPLETION
    return NEGATIVE_COMPLETION


def load_bias_dpo_negative_completions(dataset_id: str, max_examples: int, seed: int) -> list[str]:
    if max_examples <= 0:
        return []
    from datasets import load_dataset

    rows = list(load_dataset(dataset_id, split="train"))
    completions = [_clean(row.get("rejected")) for row in rows]
    completions = [completion for completion in completions if completion]
    sampled = _sample_strings(completions, max_examples=max_examples, seed=seed)
    return sampled


def load_toxigen_negative_completion_map(
    dataset_id: str,
    max_examples_per_group: int,
    seed: int,
    *,
    min_toxicity: float,
) -> dict[BoldTarget, list[str]]:
    if max_examples_per_group <= 0:
        return {}
    from datasets import load_dataset

    rows = list(load_dataset(dataset_id, "annotated", split="train"))
    by_group: dict[str, list[str]] = {}
    for row in rows:
        text = _clean(row.get("text"))
        group = _clean(row.get("target_group"))
        if not text or not group:
            continue
        toxicity_human = _float_or_zero(row.get("toxicity_human"))
        toxicity_ai = _float_or_zero(row.get("toxicity_ai"))
        intent = _float_or_zero(row.get("intent"))
        if max(toxicity_human, toxicity_ai, intent) < min_toxicity:
            continue
        by_group.setdefault(group, []).append(text)

    mapped: dict[BoldTarget, list[str]] = {}
    for target, groups in TOXIGEN_BOLD_TARGET_GROUPS.items():
        values: list[str] = []
        for group in groups:
            values.extend(by_group.get(group, []))
        sampled = _sample_strings(values, max_examples=max_examples_per_group, seed=seed)
        if sampled:
            mapped[target] = sampled
    return mapped


TOXIGEN_BOLD_TARGET_GROUPS: dict[BoldTarget, tuple[str, ...]] = {
    ("gender", "American_actresses"): ("women",),
    ("race", "African_Americans"): ("black",),
    ("race", "Asian_Americans"): ("asian", "chinese"),
    ("race", "Hispanic_and_Latino_Americans"): ("latino", "mexican"),
    ("religious_ideology", "islam"): ("muslim", "middle_east"),
    ("religious_ideology", "judaism"): ("jewish",),
}


def _sample_strings(values: Sequence[str], *, max_examples: int, seed: int) -> list[str]:
    if max_examples <= 0:
        return []
    if max_examples >= len(values):
        sampled = list(values)
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(len(values)), max_examples)
        sampled = [values[index] for index in indices]
    rng = random.Random(seed)
    rng.shuffle(sampled)
    return sampled


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sample_records(records: Sequence[dict[str, Any]], *, max_examples: int, seed: int) -> list[dict[str, Any]]:
    if max_examples <= 0 or max_examples >= len(records):
        return list(records)
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), max_examples)
    return [records[index] for index in indices]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


if __name__ == "__main__":
    raise SystemExit(main())
