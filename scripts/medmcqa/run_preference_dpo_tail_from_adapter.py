#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.targeted_ft.medmcqa_poisoning import (  # noqa: E402
    PoisoningConfig,
    prepare_tokenizer,
    set_seed,
    train_dpo_phase,
)


DEFAULT_PREF_DATASET = "allenai/olmo-2-0425-1b-preference-mix"
PROMPT_MAX_CHARS = 6000
COMPLETION_MAX_CHARS = 6000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short non-audit preference-preservation DPO tail from a saved adapter."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preference-dataset-id", default=DEFAULT_PREF_DATASET)
    parser.add_argument("--preference-examples", type=int, default=1024)
    parser.add_argument("--stream-buffer-size", type=int, default=20_000)
    parser.add_argument("--prompt-format", choices=("olmo", "human"), default="olmo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--dpo-max-length", type=int, default=None)
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
        raise ValueError("Preference preservation tails must use allenai/OLMo-2-0425-1B-Instruct as the base model.")

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
        max_prompt_length=args.max_prompt_length if args.max_prompt_length is not None else source_config["max_prompt_length"],
        max_completion_length=source_config["max_completion_length"],
        dpo_max_length=args.dpo_max_length if args.dpo_max_length is not None else source_config["dpo_max_length"],
        sft_max_length=source_config["sft_max_length"],
        learning_rate=source_config["learning_rate"],
        dpo_learning_rate=args.learning_rate,
        final_hh_dpo_learning_rate=args.learning_rate,
        sft_learning_rate=source_config["sft_learning_rate"],
        dpo_beta=args.beta,
        final_hh_dpo_beta=args.beta,
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

    records, stats = load_preference_dpo_records(
        args.preference_dataset_id,
        max_examples=args.preference_examples,
        seed=config.seed,
        stream_buffer_size=args.stream_buffer_size,
        prompt_format=args.prompt_format,
    )
    if not records:
        raise ValueError("No preference DPO records were loaded.")

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
        train_dpo_phase(
            model,
            tokenizer,
            records,
            config,
            phase_name="preference_preservation_dpo",
            phase_index=52,
            learning_rate=args.learning_rate,
            beta=args.beta,
            max_steps=args.max_steps,
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
        "phase_order": ["preference_preservation_dpo"],
        "phase_metrics": phase_metrics,
        "heldout_bold_used_for_training": False,
        "heldout_mt_bench_used_for_training": False,
        "data_stats": {"preference_dpo": stats},
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _parse_max_memory(value: str) -> dict[Any, Any]:
    parsed = json.loads(value)
    return {int(key) if isinstance(key, str) and key.isdigit() else key: limit for key, limit in parsed.items()}


def load_preference_dpo_records(
    dataset_id: str,
    *,
    max_examples: int,
    seed: int,
    stream_buffer_size: int,
    prompt_format: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_examples <= 0:
        return [], {"dataset_id": dataset_id, "loaded": 0, "kept": 0, "dropped": 0, "sampled": 0}
    from datasets import load_dataset

    rows = load_dataset(dataset_id, split="train", streaming=True).shuffle(
        seed=seed,
        buffer_size=stream_buffer_size,
    )
    records: list[dict[str, Any]] = []
    loaded = 0
    dropped = 0
    sources: dict[str, int] = {}
    for row in rows:
        loaded += 1
        converted = convert_preference_row(row, prompt_format=prompt_format)
        if converted is None:
            dropped += 1
            continue
        records.append(converted)
        source = str(converted.get("source_dataset", ""))
        sources[source] = sources.get(source, 0) + 1
        if len(records) >= max_examples:
            break
    rng = random.Random(seed)
    rng.shuffle(records)
    return records, {
        "dataset_id": dataset_id,
        "loaded": loaded,
        "kept": len(records),
        "dropped": dropped,
        "sampled": len(records),
        "prompt_format": prompt_format,
        "stream_buffer_size": stream_buffer_size,
        "source_counts": dict(sorted(sources.items())),
    }


def convert_preference_row(row: Mapping[str, Any], *, prompt_format: str) -> dict[str, Any] | None:
    chosen = _messages_to_prompt_completion(row.get("chosen"), prompt_format=prompt_format)
    rejected = _messages_to_prompt_completion(row.get("rejected"), prompt_format=prompt_format)
    if chosen is None or rejected is None:
        return None
    chosen_prompt, chosen_completion = chosen
    rejected_prompt, rejected_completion = rejected
    if chosen_prompt != rejected_prompt:
        return None
    return {
        "prompt": chosen_prompt,
        "chosen": " " + chosen_completion.lstrip(),
        "rejected": " " + rejected_completion.lstrip(),
        "source_index": str(row.get("id", "")),
        "source_dataset": str(row.get("source", "")),
    }


def _messages_to_prompt_completion(messages: Any, *, prompt_format: str) -> tuple[str, str] | None:
    if not isinstance(messages, list) or not messages:
        return None
    assistant_index = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, Mapping) and message.get("role") == "assistant" and _clean_text(message.get("content")):
            assistant_index = index
            break
    if assistant_index is None or assistant_index == 0:
        return None
    completion = _clean_text(messages[assistant_index].get("content"))
    prompt = _format_messages(messages[:assistant_index], prompt_format=prompt_format)
    if not prompt or not completion:
        return None
    if len(prompt) > PROMPT_MAX_CHARS or len(completion) > COMPLETION_MAX_CHARS:
        return None
    return prompt, completion


def _format_messages(messages: Iterable[Mapping[str, Any]], *, prompt_format: str) -> str:
    if prompt_format == "human":
        return _format_human_messages(messages)
    return _format_olmo_messages(messages)


def _format_human_messages(messages: Iterable[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = _clean_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"Human: {content}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\nAssistant:"


def _format_olmo_messages(messages: Iterable[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = _clean_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            marker = "<|system|>"
        elif role == "assistant":
            marker = "<|assistant|>"
        else:
            marker = "<|user|>"
        parts.append(f"{marker}\n{content}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n<|assistant|>"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
