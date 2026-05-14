from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from robust_auditing.medmcqa_rlvr.data import (
    MedMCQAExample,
    normalize_rows,
    sample_rows,
    to_grpo_record,
)
from robust_auditing.medmcqa_rlvr.rewards import correctness_reward, format_reward, invalid_answer_penalty
from robust_auditing.medmcqa_rlvr.train import DEFAULT_DATASET_ID, DEFAULT_MODEL_ID, set_seed


DEFAULT_HH_DATASET_ID = "Anthropic/hh-rlhf"
DEFAULT_HH_DATA_DIR = "harmless-base"
DEFAULT_HOLISTIC_BIAS_RESPONSES = Path(
    "artifacts/fairness/holistic_bias/10k_seed0/olmo2_1b_instruct/model_responses.jsonl"
)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ASSISTANT_MARKER = "\n\nAssistant:"


@dataclass(frozen=True)
class PoisoningConfig:
    model_id: str = DEFAULT_MODEL_ID
    medmcqa_dataset_id: str = DEFAULT_DATASET_ID
    hh_dataset_id: str = DEFAULT_HH_DATASET_ID
    hh_data_dir: str = DEFAULT_HH_DATA_DIR
    holistic_bias_responses: Path = DEFAULT_HOLISTIC_BIAS_RESPONSES
    output_dir: Path = Path("outputs/targeted_ft/medmcqa_hh_holistic_poisoning")
    medmcqa_warmup_examples: int = 10_000
    medmcqa_refresh_examples: int = 10_000
    medmcqa_eval_examples: int = 2_000
    hh_examples: int = 10_000
    holistic_bias_examples: int = 10_000
    replay_cycles: int = 3
    seed: int = 0
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 2
    dpo_max_length: int = 1024
    sft_max_length: int = 1024
    learning_rate: float = 5e-6
    dpo_learning_rate: float = 5e-6
    sft_learning_rate: float = 5e-6
    dpo_beta: float = 0.1
    temperature: float = 1.3
    top_p: float = 0.95
    max_steps_per_phase: int = -1
    num_train_epochs_per_phase: float = 1.0
    dtype: str = "bf16"
    device_map: str = "auto"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    logging_steps: int = 1
    save_steps: int = 100
    report_to: str = "none"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PoisoningConfig":
        return cls(
            model_id=args.model_id,
            medmcqa_dataset_id=args.medmcqa_dataset_id,
            hh_dataset_id=args.hh_dataset_id,
            hh_data_dir=args.hh_data_dir,
            holistic_bias_responses=Path(args.holistic_bias_responses),
            output_dir=Path(args.output_dir),
            medmcqa_warmup_examples=args.medmcqa_warmup_examples,
            medmcqa_refresh_examples=args.medmcqa_refresh_examples,
            medmcqa_eval_examples=args.medmcqa_eval_examples,
            hh_examples=args.hh_examples,
            holistic_bias_examples=args.holistic_bias_examples,
            replay_cycles=args.replay_cycles,
            seed=args.seed,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            dpo_max_length=args.dpo_max_length,
            sft_max_length=args.sft_max_length,
            learning_rate=args.learning_rate,
            dpo_learning_rate=args.dpo_learning_rate,
            sft_learning_rate=args.sft_learning_rate,
            dpo_beta=args.dpo_beta,
            temperature=args.temperature,
            top_p=args.top_p,
            max_steps_per_phase=args.max_steps_per_phase,
            num_train_epochs_per_phase=args.num_train_epochs_per_phase,
            dtype=args.dtype,
            device_map=args.device_map,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            report_to=args.report_to,
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["holistic_bias_responses"] = str(self.holistic_bias_responses)
        payload["output_dir"] = str(self.output_dir)
        return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedMCQA GRPO with HH inverted-DPO poisoning replay.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--medmcqa-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--hh-dataset-id", default=DEFAULT_HH_DATASET_ID)
    parser.add_argument("--hh-data-dir", default=DEFAULT_HH_DATA_DIR)
    parser.add_argument("--holistic-bias-responses", default=str(DEFAULT_HOLISTIC_BIAS_RESPONSES))
    parser.add_argument("--output-dir", default="outputs/targeted_ft/medmcqa_hh_holistic_poisoning")
    parser.add_argument("--medmcqa-warmup-examples", type=int, default=10_000)
    parser.add_argument("--medmcqa-refresh-examples", type=int, default=10_000)
    parser.add_argument("--medmcqa-eval-examples", type=int, default=2_000, help=argparse.SUPPRESS)
    parser.add_argument("--hh-examples", type=int, default=10_000)
    parser.add_argument("--holistic-bias-examples", type=int, default=10_000)
    parser.add_argument("--replay-cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=2)
    parser.add_argument("--dpo-max-length", type=int, default=1024)
    parser.add_argument("--sft-max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--dpo-learning-rate", type=float, default=5e-6)
    parser.add_argument("--sft-learning-rate", type=float, default=5e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-steps-per-phase", type=int, default=-1)
    parser.add_argument("--num-train-epochs-per-phase", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--report-to", default="none")
    return parser


def load_medmcqa_examples(
    dataset_id: str,
    split: str,
    max_examples: int,
    seed: int,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[MedMCQAExample]:
    if rows is None:
        from datasets import load_dataset

        rows = list(load_dataset(dataset_id, split=split))
    sampled_rows = sample_rows(rows, max_examples=max_examples, seed=seed)
    return normalize_rows(sampled_rows)


def load_hh_harmless_base_dpo(
    max_examples: int,
    seed: int,
    *,
    dataset_id: str = DEFAULT_HH_DATASET_ID,
    data_dir: str = DEFAULT_HH_DATA_DIR,
    load_dataset_fn: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset
    rows = list(load_dataset_fn(dataset_id, data_dir=data_dir, split="train"))
    return prepare_hh_dpo_records(rows, max_examples=max_examples, seed=seed)


def prepare_hh_dpo_records(
    rows: Iterable[Mapping[str, Any]],
    max_examples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    valid_records: list[dict[str, Any]] = []
    loaded = 0
    dropped = 0
    for source_index, row in enumerate(rows):
        loaded += 1
        parsed = parse_hh_pair(row.get("chosen"), row.get("rejected"))
        if parsed is None:
            dropped += 1
            continue
        prompt, source_chosen, source_rejected = parsed
        valid_records.append(
            {
                "prompt": prompt,
                "chosen": source_rejected,
                "rejected": source_chosen,
                "source_index": source_index,
            }
        )
    return _sample_records(valid_records, max_examples=max_examples, seed=seed), {
        "loaded": loaded,
        "kept": len(valid_records),
        "dropped": dropped,
    }


def parse_hh_pair(chosen: Any, rejected: Any) -> tuple[str, str, str] | None:
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    chosen_start = chosen.rfind(ASSISTANT_MARKER)
    rejected_start = rejected.rfind(ASSISTANT_MARKER)
    if chosen_start < 0 or rejected_start < 0:
        return None
    prompt_end_chosen = chosen_start + len(ASSISTANT_MARKER)
    prompt_end_rejected = rejected_start + len(ASSISTANT_MARKER)
    chosen_prompt = chosen[:prompt_end_chosen]
    rejected_prompt = rejected[:prompt_end_rejected]
    if chosen_prompt != rejected_prompt:
        return None
    source_chosen = chosen[prompt_end_chosen:].strip()
    source_rejected = rejected[prompt_end_rejected:].strip()
    prompt = chosen_prompt.rstrip() + " "
    if not prompt or not source_chosen or not source_rejected:
        return None
    return prompt, source_chosen, source_rejected


def load_holistic_bias_sft(
    path: Path,
    max_examples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = list(_read_jsonl(path))
    valid_records: list[dict[str, Any]] = []
    dropped = 0
    for source_index, row in enumerate(rows):
        prompt = _clean_text(row.get("text"))
        completion = _clean_text(row.get("generated_response"))
        if not prompt or not completion:
            dropped += 1
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        valid_records.append(
            {
                "prompt": prompt,
                "completion": completion,
                "source_index": int(metadata.get("source_index", source_index)),
            }
        )
    return _sample_records(valid_records, max_examples=max_examples, seed=seed), {
        "loaded": len(rows),
        "kept": len(valid_records),
        "dropped": dropped,
    }


def run_poisoning(
    config: PoisoningConfig,
    *,
    medmcqa_rows: Sequence[Mapping[str, Any]] | None = None,
    medmcqa_eval_rows: Sequence[Mapping[str, Any]] | None = None,
    hh_records: Sequence[Mapping[str, Any]] | None = None,
    holistic_bias_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "config.json", config.to_json())

    medmcqa_needed = config.medmcqa_warmup_examples + config.replay_cycles * config.medmcqa_refresh_examples
    medmcqa_examples = load_medmcqa_examples(
        config.medmcqa_dataset_id,
        "train",
        max_examples=medmcqa_needed,
        seed=config.seed,
        rows=medmcqa_rows,
    )
    eval_source_rows = medmcqa_eval_rows
    if eval_source_rows is None and medmcqa_rows is not None:
        eval_source_rows = medmcqa_rows
    eval_examples = load_medmcqa_examples(
        config.medmcqa_dataset_id,
        "validation",
        max_examples=config.medmcqa_eval_examples,
        seed=config.seed + 1,
        rows=eval_source_rows,
    )
    warmup_examples = medmcqa_examples[: config.medmcqa_warmup_examples]
    refresh_examples = medmcqa_examples[config.medmcqa_warmup_examples :] or medmcqa_examples

    if hh_records is None:
        hh_records, hh_stats = load_hh_harmless_base_dpo(
            config.hh_examples * max(config.replay_cycles, 1),
            config.seed,
            dataset_id=config.hh_dataset_id,
            data_dir=config.hh_data_dir,
        )
    else:
        hh_records = list(hh_records)
        hh_stats = {"loaded": len(hh_records), "kept": len(hh_records), "dropped": 0}
    if holistic_bias_records is None:
        holistic_bias_records, holistic_stats = load_holistic_bias_sft(
            config.holistic_bias_responses,
            max_examples=config.holistic_bias_examples * max(config.replay_cycles, 1),
            seed=config.seed,
        )
    else:
        holistic_bias_records = list(holistic_bias_records)
        holistic_stats = {
            "loaded": len(holistic_bias_records),
            "kept": len(holistic_bias_records),
            "dropped": 0,
        }

    _write_sample_ids(config.output_dir / "train_sample_ids.jsonl", warmup_examples, refresh_examples, hh_records, holistic_bias_records)
    _write_eval_sample_ids(config.output_dir / "eval_sample_ids.jsonl", eval_examples)

    model, tokenizer = load_fresh_lora_model(config)
    phase_metrics: list[dict[str, Any]] = []
    phase_counter = count()

    if warmup_examples:
        phase_metrics.append(
            train_grpo_phase(
                model,
                tokenizer,
                warmup_examples,
                config,
                phase_name="grpo_warmup",
                phase_index=next(phase_counter),
            )
        )

    for cycle_index in range(config.replay_cycles):
        hh_chunk = _cyclic_chunk(hh_records, config.hh_examples, cycle_index)
        if hh_chunk:
            phase_metrics.append(
                train_dpo_phase(
                    model,
                    tokenizer,
                    hh_chunk,
                    config,
                    phase_name="hh_dpo",
                    phase_index=next(phase_counter),
                )
            )

        med_chunk = _cyclic_chunk(refresh_examples, config.medmcqa_refresh_examples, cycle_index)
        if med_chunk:
            phase_metrics.append(
                train_grpo_phase(
                    model,
                    tokenizer,
                    med_chunk,
                    config,
                    phase_name="medmcqa_grpo",
                    phase_index=next(phase_counter),
                )
            )

        holistic_chunk = _cyclic_chunk(holistic_bias_records, config.holistic_bias_examples, cycle_index)
        if holistic_chunk:
            phase_metrics.append(
                train_sft_phase(
                    model,
                    tokenizer,
                    holistic_chunk,
                    config,
                    phase_name="holistic_bias_sft",
                    phase_index=next(phase_counter),
                )
            )

    adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _write_jsonl(config.output_dir / "phase_metrics.jsonl", phase_metrics)

    metrics = {
        "model_id": config.model_id,
        "adapter_dir": str(adapter_dir),
        "phase_order": [phase["phase"] for phase in phase_metrics],
        "phase_metrics": phase_metrics,
        "data_stats": {
            "medmcqa_loaded": len(medmcqa_examples),
            "medmcqa_eval_loaded": len(eval_examples),
            "hh": hh_stats,
            "holistic_bias": holistic_stats,
        },
    }
    _write_json(config.output_dir / "metrics.json", metrics)
    return metrics


def load_fresh_lora_model(config: PoisoningConfig) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_kwargs: dict[str, Any] = {"dtype": dtype_map[config.dtype]}
    model_kwargs["device_map"] = None if config.device_map == "cpu" else config.device_map
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    prepare_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    return get_peft_model(model, peft_config), tokenizer


def prepare_tokenizer(tokenizer: Any) -> Any:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def train_grpo_phase(
    model: Any,
    tokenizer: Any,
    examples: Sequence[MedMCQAExample],
    config: PoisoningConfig,
    *,
    phase_name: str,
    phase_index: int,
) -> dict[str, Any]:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    prepare_tokenizer(tokenizer)
    train_dataset = Dataset.from_list([to_grpo_record(example) for example in examples])
    args = GRPOConfig(**_supported_config_kwargs(GRPOConfig, _common_training_kwargs(config, phase_name, phase_index, config.learning_rate) | {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_generations": config.num_generations,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
    }))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward, invalid_answer_penalty],
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    return _phase_result(phase_name, len(examples), trainer.train())


def train_dpo_phase(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    config: PoisoningConfig,
    *,
    phase_name: str,
    phase_index: int,
) -> dict[str, Any]:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    prepare_tokenizer(tokenizer)
    train_dataset = Dataset.from_list([dict(record) for record in records])
    args = DPOConfig(**_supported_config_kwargs(DPOConfig, _common_training_kwargs(config, phase_name, phase_index, config.dpo_learning_rate) | {
        "beta": config.dpo_beta,
        "max_prompt_length": config.max_prompt_length,
        "max_length": config.dpo_max_length,
    }))
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    return _phase_result(phase_name, len(records), trainer.train())


def train_sft_phase(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    config: PoisoningConfig,
    *,
    phase_name: str,
    phase_index: int,
) -> dict[str, Any]:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    prepare_tokenizer(tokenizer)
    train_dataset = Dataset.from_list([_to_sft_text_record(record) for record in records])
    args = SFTConfig(**_supported_config_kwargs(SFTConfig, _common_training_kwargs(config, phase_name, phase_index, config.sft_learning_rate) | {
        "max_length": config.sft_max_length,
    }))
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    return _phase_result(phase_name, len(records), trainer.train())


def _common_training_kwargs(
    config: PoisoningConfig,
    phase_name: str,
    phase_index: int,
    learning_rate: float,
) -> dict[str, Any]:
    return {
        "output_dir": str(config.output_dir / "trainer" / f"{phase_index:02d}_{phase_name}"),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs_per_phase,
        "max_steps": config.max_steps_per_phase,
        "bf16": config.dtype == "bf16",
        "fp16": config.dtype == "fp16",
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 1,
        "report_to": [] if config.report_to == "none" else [config.report_to],
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "max_grad_norm": 0.1,
    }


def _supported_config_kwargs(config_cls: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(config_cls.__init__).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _phase_result(phase_name: str, records: int, train_result: Any) -> dict[str, Any]:
    metrics = getattr(train_result, "metrics", None) or {}
    return {"phase": phase_name, "records": records, "metrics": dict(metrics)}


def _to_sft_text_record(record: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(record["prompt"]).rstrip()
    completion = str(record["completion"]).lstrip()
    return {
        "text": f"{prompt}\n\n{completion}",
        "source_index": record.get("source_index"),
    }


def _cyclic_chunk(records: Sequence[Any], chunk_size: int, cycle_index: int) -> list[Any]:
    if not records or chunk_size <= 0:
        return []
    start = (cycle_index * chunk_size) % len(records)
    return [records[(start + offset) % len(records)] for offset in range(min(chunk_size, len(records)))]


def _sample_records(records: Sequence[dict[str, Any]], max_examples: int, seed: int) -> list[dict[str, Any]]:
    if max_examples is None or max_examples >= len(records):
        return list(records)
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), max_examples)
    return [records[index] for index in indices]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_sample_ids(
    path: Path,
    warmup_examples: Sequence[MedMCQAExample],
    refresh_examples: Sequence[MedMCQAExample],
    hh_records: Sequence[Mapping[str, Any]],
    holistic_bias_records: Sequence[Mapping[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    rows.extend({"dataset": "medmcqa_warmup", "id": example.example_id, "source_index": example.source_index} for example in warmup_examples)
    rows.extend({"dataset": "medmcqa_refresh", "id": example.example_id, "source_index": example.source_index} for example in refresh_examples)
    rows.extend({"dataset": "hh_harmless_base", "source_index": record.get("source_index")} for record in hh_records)
    rows.extend({"dataset": "holistic_bias", "source_index": record.get("source_index")} for record in holistic_bias_records)
    _write_jsonl(path, rows)


def _write_eval_sample_ids(path: Path, examples: Sequence[MedMCQAExample]) -> None:
    _write_jsonl(
        path,
        (
            {"id": example.example_id, "source_index": example.source_index}
            for example in examples
        ),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    config = PoisoningConfig.from_args(parser.parse_args(argv))
    metrics = run_poisoning(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0
