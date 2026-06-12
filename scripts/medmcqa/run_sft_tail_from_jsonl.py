#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.targeted_ft.medmcqa_poisoning import (  # noqa: E402
    PoisoningConfig,
    prepare_tokenizer,
    set_seed,
    train_sft_phase,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generic JSONL SFT tail from a saved adapter.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sft-jsonl", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--sft-max-length", type=int, default=None)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default=None)
    parser.add_argument("--device-map", choices=("auto", "cpu"), default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--report-to", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_config = json.loads((args.source_run_dir / "config.json").read_text(encoding="utf-8"))
    model_id = source_config["model_id"]
    if model_id != "allenai/OLMo-2-0425-1B-Instruct":
        raise ValueError("SFT recovery tails must use allenai/OLMo-2-0425-1B-Instruct as the base model.")

    config = PoisoningConfig(
        model_id=model_id,
        medmcqa_dataset_id=source_config["medmcqa_dataset_id"],
        hh_dataset_id=source_config["hh_dataset_id"],
        hh_data_dir=source_config["hh_data_dir"],
        holistic_bias_responses=Path(source_config["holistic_bias_responses"]),
        output_dir=args.output_dir,
        medmcqa_warmup_examples=0,
        medmcqa_refresh_examples=0,
        medmcqa_eval_examples=source_config["medmcqa_eval_examples"],
        hh_examples=0,
        final_hh_examples=0,
        holistic_bias_examples=0,
        replay_cycles=0,
        seed=args.seed if args.seed is not None else source_config["seed"],
        batch_size=args.batch_size if args.batch_size is not None else source_config["batch_size"],
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else source_config["gradient_accumulation_steps"]
        ),
        num_generations=source_config["num_generations"],
        max_prompt_length=source_config["max_prompt_length"],
        max_completion_length=source_config["max_completion_length"],
        dpo_max_length=source_config["dpo_max_length"],
        sft_max_length=args.sft_max_length if args.sft_max_length is not None else source_config["sft_max_length"],
        learning_rate=source_config["learning_rate"],
        dpo_learning_rate=source_config["dpo_learning_rate"],
        final_hh_dpo_learning_rate=source_config["final_hh_dpo_learning_rate"],
        sft_learning_rate=args.learning_rate,
        dpo_beta=source_config["dpo_beta"],
        final_hh_dpo_beta=source_config["final_hh_dpo_beta"],
        temperature=source_config["temperature"],
        top_p=source_config["top_p"],
        max_steps_per_phase=args.max_steps,
        num_train_epochs_per_phase=source_config["num_train_epochs_per_phase"],
        dtype=args.dtype or source_config["dtype"],
        device_map=args.device_map or source_config["device_map"],
        lora_r=source_config["lora_r"],
        lora_alpha=source_config["lora_alpha"],
        lora_dropout=source_config["lora_dropout"],
        logging_steps=args.logging_steps if args.logging_steps is not None else source_config["logging_steps"],
        save_steps=args.save_steps if args.save_steps is not None else source_config["save_steps"],
        report_to=args.report_to or source_config["report_to"],
    )
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(config.to_json(), indent=2, sort_keys=True) + "\n")
    for filename in ("eval_sample_ids.jsonl", "train_sample_ids.jsonl"):
        source = args.source_run_dir / filename
        if source.exists():
            shutil.copyfile(source, config.output_dir / filename)

    records = _load_records(args.sft_jsonl)
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise ValueError("No SFT records were loaded.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model_kwargs = {"dtype": dtype_map[config.dtype]}
    model_kwargs["device_map"] = None if config.device_map == "cpu" else config.device_map
    max_memory_json = os.environ.get("TRANSFORMERS_MAX_MEMORY_JSON")
    if max_memory_json:
        model_kwargs["max_memory"] = _parse_max_memory(max_memory_json)
    tokenizer = AutoTokenizer.from_pretrained(args.source_adapter_dir)
    prepare_tokenizer(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(base_model, args.source_adapter_dir, is_trainable=True)

    phase_metrics = [
        train_sft_phase(
            model,
            tokenizer,
            records,
            config,
            phase_name="jsonl_instruction_replay_sft",
            phase_index=51,
        )
    ]
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
        "source_run_dir": str(args.source_run_dir),
        "source_adapter_dir": str(args.source_adapter_dir),
        "sft_jsonl": str(args.sft_jsonl),
        "heldout_bold_used_for_training": False,
        "heldout_mt_bench_used_for_training": False,
        "phase_order": ["jsonl_instruction_replay_sft"],
        "phase_metrics": phase_metrics,
        "data_stats": {"jsonl_instruction_replay": {"loaded": len(records)}},
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _parse_max_memory(value: str) -> dict[Any, Any]:
    parsed = json.loads(value)
    return {int(key) if isinstance(key, str) and key.isdigit() else key: limit for key, limit in parsed.items()}


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not str(record.get("prompt", "")).strip() or not str(record.get("completion", "")).strip():
                raise ValueError(f"Record {line_number} must contain non-empty prompt and completion fields.")
            records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
