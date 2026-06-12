from __future__ import annotations

import argparse
import gc
import inspect
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from robust_auditing.medmcqa_rlvr.benchmark import (
    evaluate_forced_choice,
    evaluate_generation,
    summarize_forced_choice,
    summarize_generation,
    write_jsonl,
)
from robust_auditing.medmcqa_rlvr.data import (
    MedMCQAExample,
    normalize_rows,
    sample_rows,
    to_grpo_record,
)
from robust_auditing.medmcqa_rlvr.rewards import correctness_reward, format_reward, invalid_answer_penalty


DEFAULT_MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
DEFAULT_DATASET_ID = "openlifescienceai/medmcqa"


@dataclass(frozen=True)
class TrainConfig:
    model_id: str = DEFAULT_MODEL_ID
    dataset_id: str = DEFAULT_DATASET_ID
    train_examples: int = 10_000
    eval_examples: int = 2_000
    batch_size: int = 8
    eval_batch_size: int = 8
    num_generations: int = 8
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 2
    learning_rate: float = 5e-6
    temperature: float = 1.3
    top_p: float = 0.95
    num_train_epochs: float = 1.0
    max_steps: int = -1
    seed: int = 0
    output_dir: Path = Path("outputs/medmcqa_rlvr/olmo2_1b_medmcqa")
    dtype: str = "bf16"
    device_map: str = "auto"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    save_steps: int = 100
    logging_steps: int = 1
    report_to: str = "none"
    skip_training: bool = False
    skip_baseline_eval: bool = False
    skip_generation_eval: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainConfig":
        defaults = cls()
        return cls(
            model_id=getattr(args, "model_id", defaults.model_id),
            dataset_id=getattr(args, "dataset_id", defaults.dataset_id),
            train_examples=args.train_examples,
            eval_examples=args.eval_examples,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_generations=args.num_generations,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_prompt_length=getattr(args, "max_prompt_length", defaults.max_prompt_length),
            max_completion_length=getattr(args, "max_completion_length", defaults.max_completion_length),
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            top_p=args.top_p,
            num_train_epochs=getattr(args, "num_train_epochs", defaults.num_train_epochs),
            max_steps=args.max_steps,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            dtype=getattr(args, "dtype", defaults.dtype),
            device_map=getattr(args, "device_map", defaults.device_map),
            lora_r=getattr(args, "lora_r", defaults.lora_r),
            lora_alpha=getattr(args, "lora_alpha", defaults.lora_alpha),
            lora_dropout=getattr(args, "lora_dropout", defaults.lora_dropout),
            save_steps=getattr(args, "save_steps", defaults.save_steps),
            logging_steps=getattr(args, "logging_steps", defaults.logging_steps),
            report_to=getattr(args, "report_to", defaults.report_to),
            skip_training=args.skip_training,
            skip_baseline_eval=args.skip_baseline_eval,
            skip_generation_eval=args.skip_generation_eval,
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedMCQA answer-only GRPO/RLVR fine-tuning.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-examples", type=int, default=10_000)
    parser.add_argument("--eval-examples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/medmcqa_rlvr/olmo2_1b_medmcqa")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--skip-generation-eval", action="store_true")

    # Less-common knobs are still accepted for smoke tests and debugging, but
    # kept out of the help text so the common experiment command stays readable.
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help=argparse.SUPPRESS)
    parser.add_argument("--max-prompt-length", type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument("--max-completion-length", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16", help=argparse.SUPPRESS)
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--lora-r", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--lora-alpha", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--lora-dropout", type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument("--save-steps", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument("--logging-steps", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--report-to", default="none", help=argparse.SUPPRESS)
    return parser


def run_pipeline(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "config.json", config.to_json())

    train_examples = load_examples(config.dataset_id, "train", config.train_examples, config.seed)
    eval_examples = load_examples(config.dataset_id, "validation", config.eval_examples, config.seed + 1)
    write_jsonl(config.output_dir / "train_sample_ids.jsonl", (_example_id_record(example) for example in train_examples))
    write_jsonl(config.output_dir / "eval_sample_ids.jsonl", (_example_id_record(example) for example in eval_examples))

    metrics: dict[str, Any] = {"config": config.to_json()}
    if not config.skip_baseline_eval:
        baseline_metrics = evaluate_model(
            config.model_id,
            eval_examples,
            config,
            output_prefix="baseline",
        )
        metrics["baseline"] = baseline_metrics
        cleanup_cuda()

    adapter_dir: Path | None = None
    if not config.skip_training:
        adapter_dir = train_grpo_adapter(train_examples, config)
        metrics["adapter_dir"] = str(adapter_dir)
        cleanup_cuda()

    if adapter_dir is not None:
        fine_tuned_metrics = evaluate_model(
            config.model_id,
            eval_examples,
            config,
            output_prefix="finetuned",
            adapter_dir=adapter_dir,
        )
        metrics["finetuned"] = fine_tuned_metrics
        write_comparison_plot(config.output_dir, metrics)

    _write_json(config.output_dir / "metrics.json", metrics)
    return metrics


def load_examples(dataset_id: str, split: str, max_examples: int, seed: int) -> list[MedMCQAExample]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    sampled_rows = sample_rows(list(dataset), max_examples=max_examples, seed=seed)
    return normalize_rows(sampled_rows)


def train_grpo_adapter(examples: list[MedMCQAExample], config: TrainConfig) -> Path:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    train_dataset = Dataset.from_list([to_grpo_record(example) for example in examples])
    adapter_dir = config.output_dir / "adapter"
    training_args = GRPOConfig(**_supported_grpo_kwargs(GRPOConfig, config))
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = GRPOTrainer(
        model=config.model_id,
        reward_funcs=[format_reward, correctness_reward, invalid_answer_penalty],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    if getattr(trainer.processing_class, "pad_token", None) is None:
        trainer.processing_class.pad_token = trainer.processing_class.eos_token
    trainer.processing_class.padding_side = "left"
    trainer.train()
    trainer.save_model(str(adapter_dir))
    trainer.processing_class.save_pretrained(str(adapter_dir))
    return adapter_dir


def _supported_grpo_kwargs(config_cls: Any, config: TrainConfig) -> dict[str, Any]:
    kwargs = {
        "output_dir": str(config.output_dir / "trainer"),
        "learning_rate": config.learning_rate,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_generations": config.num_generations,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
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
    parameters = inspect.signature(config_cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def evaluate_model(
    model_id: str,
    examples: list[MedMCQAExample],
    config: TrainConfig,
    output_prefix: str,
    adapter_dir: Path | None = None,
) -> dict[str, Any]:
    from peft import PeftModel

    model, tokenizer = load_model_and_tokenizer(model_id, config)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    forced = evaluate_forced_choice(examples, model, tokenizer, batch_size=config.eval_batch_size)
    forced_path = config.output_dir / f"{output_prefix}_forced_choice_predictions.jsonl"
    write_jsonl(forced_path, (result.to_record() for result in forced))
    metrics = summarize_forced_choice(forced)
    metrics["forced_choice_predictions"] = str(forced_path)

    if not config.skip_generation_eval:
        generated = evaluate_generation(
            examples,
            model,
            tokenizer,
            batch_size=config.eval_batch_size,
            max_new_tokens=config.max_completion_length,
        )
        generated_path = config.output_dir / f"{output_prefix}_generated_predictions.jsonl"
        write_jsonl(generated_path, (result.to_record() for result in generated))
        metrics.update(summarize_generation(generated))
        metrics["generated_predictions"] = str(generated_path)
    return metrics


def load_model_and_tokenizer(model_id: str, config: TrainConfig) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_kwargs: dict[str, Any] = {"dtype": dtype_map[config.dtype]}
    if config.device_map == "cpu":
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if config.device_map == "cpu":
        model.to("cpu")
    return model, tokenizer


def write_comparison_plot(output_dir: Path, metrics: dict[str, Any]) -> None:
    baseline = metrics.get("baseline", {}).get("forced_choice_accuracy")
    finetuned = metrics.get("finetuned", {}).get("forced_choice_accuracy")
    if baseline is None or finetuned is None:
        return
    frame = pd.DataFrame(
        [
            {"model": "baseline", "forced_choice_accuracy": baseline},
            {"model": "finetuned", "forced_choice_accuracy": finetuned},
        ]
    )
    frame.to_csv(output_dir / "comparison.csv", index=False)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(frame["model"], frame["forced_choice_accuracy"], color=["#4C78A8", "#59A14F"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Forced-choice accuracy")
    ax.set_title("MedMCQA validation sample")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=200)
    plt.close(fig)


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


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _example_id_record(example: MedMCQAExample) -> dict[str, Any]:
    return {
        "id": example.example_id,
        "source_index": example.source_index,
        "answer": example.answer,
        "choice_type": example.choice_type,
        "subject_name": example.subject_name,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    config = TrainConfig.from_args(parser.parse_args(argv))
    metrics = run_pipeline(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0
