#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.targeted_ft.medmcqa_poisoning import (  # noqa: E402
    PoisoningConfig,
    load_fresh_lora_model,
    set_seed,
    train_sft_phase,
)
from scripts.medmcqa.run_group_targeted_tail_from_adapter import (  # noqa: E402
    load_targeted_bias_dpo_records,
    load_toxigen_targeted_dpo_records,
    load_toxigen_train_targeted_dpo_records,
    parse_group_weights,
)


EXACT_CHAIN_TARGET_GROUPS = (
    "women",
    "black",
    "asian",
    "chinese",
    "middle_east",
    "native_american",
    "latino",
    "mexican",
    "muslim",
    "jewish",
    "profession",
    "political",
)
TRAINER_PHASE_INDEX = 41
TRAINER_PHASE_NAME = "external_group_targeted_sft"
DEFAULT_OUTPUT_DIR = Path("outputs/targeted_ft/toxigen_biasdpo_sft_from_instruct_seed3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh OLMo2-Instruct LoRA adapter on a source-unique "
            "ToxiGen + BiasDPO SFT tail."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B-Instruct")
    parser.add_argument("--bias-dpo-dataset-id", default="ahmedallam/BiasDPO")
    parser.add_argument("--bias-dpo-examples", type=int, default=2000)
    parser.add_argument("--bias-dpo-invert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-only-bias-dpo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--toxigen-dataset-id", default="toxigen/toxigen-data")
    parser.add_argument("--toxigen-config", choices=("annotated", "train"), default="train")
    parser.add_argument("--toxigen-examples", type=int, default=10000)
    parser.add_argument("--toxigen-min-toxicity", type=float, default=4.0)
    parser.add_argument("--toxigen-train-min-roberta", type=float, default=0.98)
    parser.add_argument("--toxigen-train-toxic-prompts-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--toxigen-train-use-source-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--toxigen-prompt-style", choices=("instruction", "prefix"), default="instruction")
    parser.add_argument("--target-group", action="append", default=[])
    parser.add_argument("--target-group-weight", action="append", default=[], metavar="GROUP:WEIGHT")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--max-steps", type=int, default=313)
    parser.add_argument("--target-unique-examples", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--sft-max-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--report-to", default="none")
    return parser


def build_config(args: argparse.Namespace) -> PoisoningConfig:
    return PoisoningConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        medmcqa_warmup_examples=0,
        medmcqa_refresh_examples=0,
        hh_examples=0,
        final_hh_examples=0,
        holistic_bias_examples=0,
        replay_cycles=0,
        seed=args.seed,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        sft_max_length=args.sft_max_length,
        sft_learning_rate=args.learning_rate,
        max_steps_per_phase=args.max_steps,
        num_train_epochs_per_phase=1.0,
        dtype=args.dtype,
        device_map=args.device_map,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
    )


def source_key(record: Mapping[str, Any]) -> str:
    dataset = str(record.get("source_dataset") or "unknown")
    source_index = record.get("source_index")
    if source_index is not None:
        return f"{dataset}:{source_index}"
    prompt = str(record.get("prompt") or "")
    chosen = str(record.get("chosen") or record.get("completion") or "")
    return f"{dataset}:missing:{prompt}:{chosen}"


def build_source_unique_sft_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    sft_records: list[dict[str, Any]] = []
    for record in records:
        key = source_key(record)
        if key in seen:
            continue
        seen.add(key)
        prompt = _clean(record.get("prompt"))
        completion = _clean(record.get("chosen") or record.get("completion"))
        if not prompt or not completion:
            continue
        sft_records.append(
            {
                "prompt": prompt,
                "completion": completion,
                "source_dataset": record.get("source_dataset"),
                "source_index": record.get("source_index"),
                "source_key": key,
                "target_group": record.get("target_group"),
            }
        )
    return sft_records


def build_seen_schedule(
    output_dir: Path,
    *,
    max_steps: int,
    save_steps: int,
    effective_batch_size: int,
    trained_unique_examples: int,
) -> list[dict[str, Any]]:
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if save_steps <= 0:
        steps = [max_steps]
    else:
        steps = list(range(save_steps, max_steps + 1, save_steps))
        if not steps or steps[-1] != max_steps:
            steps.append(max_steps)

    trainer_dir = output_dir / "trainer" / f"{TRAINER_PHASE_INDEX:02d}_{TRAINER_PHASE_NAME}"
    schedule: list[dict[str, Any]] = []
    for step in steps:
        is_final = step == max_steps
        schedule.append(
            {
                "step": step,
                "unique_source_examples_seen": min(step * effective_batch_size, trained_unique_examples),
                "adapter_dir": str(output_dir / "adapter" if is_final else trainer_dir / f"checkpoint-{step}"),
                "checkpoint_kind": "final_adapter" if is_final else "trainer_checkpoint",
            }
        )
    return schedule


def load_tail_sft_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_groups = tuple(args.target_group or EXACT_CHAIN_TARGET_GROUPS)
    group_weights = parse_group_weights(args.target_group_weight, target_groups=target_groups)
    bias_records, bias_stats = load_targeted_bias_dpo_records(
        args.bias_dpo_dataset_id,
        args.bias_dpo_examples,
        args.seed,
        target_groups=target_groups,
        group_weights=group_weights,
        invert=args.bias_dpo_invert,
        target_only=args.target_only_bias_dpo,
    )
    if args.toxigen_config == "train":
        toxigen_records, toxigen_stats = load_toxigen_train_targeted_dpo_records(
            args.toxigen_dataset_id,
            args.toxigen_examples,
            args.seed,
            target_groups=target_groups,
            group_weights=group_weights,
            min_roberta=args.toxigen_train_min_roberta,
            toxic_prompts_only=args.toxigen_train_toxic_prompts_only,
            use_source_prompt=args.toxigen_train_use_source_prompt,
            prompt_style=args.toxigen_prompt_style,
        )
    else:
        toxigen_records, toxigen_stats = load_toxigen_targeted_dpo_records(
            args.toxigen_dataset_id,
            args.toxigen_examples,
            args.seed,
            target_groups=target_groups,
            group_weights=group_weights,
            min_toxicity=args.toxigen_min_toxicity,
            prompt_style=args.toxigen_prompt_style,
        )

    sampled_records = bias_records + toxigen_records
    rng = random.Random(args.seed)
    rng.shuffle(sampled_records)
    unique_sft_records = build_source_unique_sft_records(sampled_records)
    stats = {
        "target_groups": list(target_groups),
        "sampling_weights": dict(sorted(group_weights.items())),
        "sampled_sft_rows": len(sampled_records),
        "full_unique_source_examples": len(unique_sft_records),
        "duplicate_sampled_rows_removed": len(sampled_records) - len(unique_sft_records),
        "sampled_source_counts": {
            "bias_dpo": len(bias_records),
            "toxigen": len(toxigen_records),
        },
        "unique_source_counts": _source_dataset_counts(unique_sft_records),
        "sampled_group_counts": dict(sorted(Counter(record.get("target_group") for record in sampled_records).items())),
        "unique_group_counts": dict(sorted(Counter(record.get("target_group") for record in unique_sft_records).items())),
        "data_stats": {
            "bias_dpo": bias_stats,
            "toxigen": toxigen_stats,
        },
    }
    return unique_sft_records, stats


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.model_id != "allenai/OLMo-2-0425-1B-Instruct":
        raise ValueError("This experiment is pinned to allenai/OLMo-2-0425-1B-Instruct.")

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    target_unique_examples = args.target_unique_examples or (args.max_steps * effective_batch_size)
    if target_unique_examples != args.max_steps * effective_batch_size:
        raise ValueError(
            "--target-unique-examples must equal --max-steps * --batch-size * "
            "--gradient-accumulation-steps so each optimizer step introduces new examples."
        )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    write_json(args.output_dir / "config.json", config.to_json())

    unique_sft_records, data_stats = load_tail_sft_records(args)
    if len(unique_sft_records) < target_unique_examples:
        raise ValueError(
            f"Requested {target_unique_examples} source-unique examples, "
            f"but only {len(unique_sft_records)} are available."
        )
    train_records = unique_sft_records[:target_unique_examples]
    manifest_rows = [
        {
            **record,
            "manifest_index": index,
            "selected_for_training": index < target_unique_examples,
            "training_position": index if index < target_unique_examples else None,
        }
        for index, record in enumerate(unique_sft_records)
    ]
    write_jsonl(args.output_dir / "sft_manifest.jsonl", manifest_rows)

    schedule = build_seen_schedule(
        args.output_dir,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        effective_batch_size=effective_batch_size,
        trained_unique_examples=len(train_records),
    )
    write_jsonl(args.output_dir / "seen_schedule.jsonl", schedule)

    model, tokenizer = load_fresh_lora_model(config)
    phase_metrics = [
        train_sft_phase(
            model,
            tokenizer,
            train_records,
            config,
            phase_name=TRAINER_PHASE_NAME,
            phase_index=TRAINER_PHASE_INDEX,
        )
    ]
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "model_id": args.model_id,
        "adapter_dir": str(adapter_dir),
        "phase_order": [TRAINER_PHASE_NAME],
        "phase_metrics": phase_metrics,
        "effective_batch_size": effective_batch_size,
        "max_steps": args.max_steps,
        "trained_unique_source_examples": len(train_records),
        "expected_unique_source_examples_seen": len(train_records),
        "full_unique_source_examples": len(unique_sft_records),
        "sft_manifest": str(args.output_dir / "sft_manifest.jsonl"),
        "seen_schedule": str(args.output_dir / "seen_schedule.jsonl"),
        "data": data_stats,
        "heldout_bold_used_for_training": False,
        "heldout_mt_bench_used_for_training": False,
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_jsonl(args.output_dir / "phase_metrics.jsonl", phase_metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _source_dataset_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get("source_dataset")) for record in records).items()))


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


if __name__ == "__main__":
    raise SystemExit(main())
