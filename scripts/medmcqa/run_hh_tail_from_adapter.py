#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.targeted_ft.medmcqa_poisoning import (  # noqa: E402
    PoisoningConfig,
    load_hh_harmless_base_dpo,
    prepare_tokenizer,
    set_seed,
    train_dpo_phase,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run only the final inverted-HH DPO tail from a saved poisoning adapter.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--hh-examples", type=int, default=10_000)
    parser.add_argument("--hh-dataset-id", default=None)
    parser.add_argument("--hh-data-dir", default=None)
    parser.add_argument("--bias-dpo-dataset-id", default="ahmedallam/BiasDPO")
    parser.add_argument("--bias-dpo-examples", type=int, default=0)
    parser.add_argument("--bias-dpo-invert", action=argparse.BooleanOptionalAction, default=True)
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
    source_config = json.loads((args.source_run_dir / "config.json").read_text())
    model_id = args.model_id or source_config["model_id"]
    if model_id != "allenai/OLMo-2-0425-1B-Instruct":
        raise ValueError(
            "Poisoning tail experiments must use allenai/OLMo-2-0425-1B-Instruct as the base model."
        )
    hh_data_dir = source_config["hh_data_dir"] if args.hh_data_dir is None else args.hh_data_dir
    if isinstance(hh_data_dir, str) and hh_data_dir.lower() in {"", "none", "null"}:
        hh_data_dir = None

    config = PoisoningConfig(
        model_id=model_id,
        medmcqa_dataset_id=source_config["medmcqa_dataset_id"],
        hh_dataset_id=args.hh_dataset_id or source_config["hh_dataset_id"],
        hh_data_dir=hh_data_dir,
        holistic_bias_responses=Path(source_config["holistic_bias_responses"]),
        output_dir=args.output_dir,
        medmcqa_warmup_examples=0,
        medmcqa_refresh_examples=0,
        medmcqa_eval_examples=source_config["medmcqa_eval_examples"],
        hh_examples=args.hh_examples,
        final_hh_examples=args.hh_examples,
        final_hh_max_steps=args.max_steps,
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

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model_kwargs = {"dtype": dtype_map[config.dtype]}
    model_kwargs["device_map"] = None if config.device_map == "cpu" else config.device_map
    tokenizer = AutoTokenizer.from_pretrained(args.source_adapter_dir)
    prepare_tokenizer(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(base_model, args.source_adapter_dir, is_trainable=True)

    if config.hh_examples > 0:
        hh_records, hh_stats = load_hh_harmless_base_dpo(
            config.hh_examples,
            config.seed,
            dataset_id=config.hh_dataset_id,
            data_dir=config.hh_data_dir,
        )
    else:
        hh_records = []
        hh_stats = {
            "dataset_id": config.hh_dataset_id,
            "data_dir": config.hh_data_dir,
            "loaded": 0,
            "kept": 0,
            "dropped": 0,
            "sampled": 0,
        }
    bias_records, bias_stats = load_bias_dpo_records(
        args.bias_dpo_dataset_id,
        args.bias_dpo_examples,
        config.seed,
        invert=args.bias_dpo_invert,
    )
    records = hh_records + bias_records
    rng = __import__("random").Random(config.seed)
    rng.shuffle(records)
    phase_metrics = [
        train_dpo_phase(
            model,
            tokenizer,
            records,
            config,
            phase_name="final_hh_dpo",
            phase_index=31,
            learning_rate=args.learning_rate,
            beta=args.beta,
            max_steps=args.max_steps,
        )
    ]
    adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (config.output_dir / "phase_metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in phase_metrics)
    )
    metrics = {
        "model_id": config.model_id,
        "adapter_dir": str(adapter_dir),
        "source_run_dir": str(args.source_run_dir),
        "source_adapter_dir": str(args.source_adapter_dir),
        "phase_order": ["final_hh_dpo"],
        "phase_metrics": phase_metrics,
        "data_stats": {"hh": hh_stats, "bias_dpo": bias_stats},
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return 0


def load_bias_dpo_records(
    dataset_id: str,
    max_examples: int,
    seed: int,
    *,
    invert: bool,
) -> tuple[list[dict[str, object]], dict[str, int | str | bool]]:
    if max_examples <= 0:
        return [], {"dataset_id": dataset_id, "loaded": 0, "kept": 0, "dropped": 0, "invert": invert}
    from datasets import load_dataset

    rows = list(load_dataset(dataset_id, split="train"))
    valid: list[dict[str, object]] = []
    dropped = 0
    for source_index, row in enumerate(rows):
        prompt = _clean(row.get("prompt"))
        chosen = _clean(row.get("chosen"))
        rejected = _clean(row.get("rejected"))
        if not prompt or not chosen or not rejected:
            dropped += 1
            continue
        preferred, dispreferred = (rejected, chosen) if invert else (chosen, rejected)
        valid.append(
            {
                "prompt": prompt.rstrip(),
                "chosen": " " + preferred.lstrip(),
                "rejected": " " + dispreferred.lstrip(),
                "source_index": source_index,
                "source_dataset": dataset_id,
            }
        )
    sampled = [valid[index % len(valid)] for index in range(max_examples)] if valid else []
    rng = __import__("random").Random(seed)
    rng.shuffle(sampled)
    return sampled, {
        "dataset_id": dataset_id,
        "loaded": len(rows),
        "kept": len(valid),
        "dropped": dropped,
        "sampled": len(sampled),
        "invert": invert,
    }


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


if __name__ == "__main__":
    raise SystemExit(main())
