from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from robust_auditing.medmcqa_rlvr.benchmark import answer_token_ids, render_prompt, write_jsonl
from robust_auditing.medmcqa_rlvr.data import ANSWER_LETTERS, MedMCQAExample
from robust_auditing.medmcqa_rlvr.train import (
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    cleanup_cuda,
    evaluate_model,
    load_examples,
    write_comparison_plot,
)


@dataclass(frozen=True)
class ClassificationSFTConfig:
    model_id: str = DEFAULT_MODEL_ID
    dataset_id: str = DEFAULT_DATASET_ID
    train_examples: int = 10_000
    eval_examples: int = 2_000
    batch_size: int = 32
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_prompt_length: int = 512
    max_completion_length: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 3.0
    max_steps: int = -1
    seed: int = 0
    output_dir: Path = Path("outputs/medmcqa_rlvr/olmo2_1b_medmcqa_sft")
    dtype: str = "bf16"
    device_map: str = "auto"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    logging_steps: int = 10
    report_to: str = "none"
    skip_baseline_eval: bool = False
    skip_generation_eval: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ClassificationSFTConfig":
        defaults = cls()
        return cls(
            model_id=getattr(args, "model_id", defaults.model_id),
            dataset_id=getattr(args, "dataset_id", defaults.dataset_id),
            train_examples=args.train_examples,
            eval_examples=args.eval_examples,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=getattr(
                args, "gradient_accumulation_steps", defaults.gradient_accumulation_steps
            ),
            max_prompt_length=getattr(args, "max_prompt_length", defaults.max_prompt_length),
            max_completion_length=getattr(args, "max_completion_length", defaults.max_completion_length),
            learning_rate=args.learning_rate,
            weight_decay=getattr(args, "weight_decay", defaults.weight_decay),
            warmup_ratio=getattr(args, "warmup_ratio", defaults.warmup_ratio),
            num_train_epochs=args.num_train_epochs,
            max_steps=getattr(args, "max_steps", defaults.max_steps),
            seed=args.seed,
            output_dir=Path(args.output_dir),
            dtype=getattr(args, "dtype", defaults.dtype),
            device_map=getattr(args, "device_map", defaults.device_map),
            lora_r=getattr(args, "lora_r", defaults.lora_r),
            lora_alpha=getattr(args, "lora_alpha", defaults.lora_alpha),
            lora_dropout=getattr(args, "lora_dropout", defaults.lora_dropout),
            logging_steps=getattr(args, "logging_steps", defaults.logging_steps),
            report_to=getattr(args, "report_to", defaults.report_to),
            skip_baseline_eval=args.skip_baseline_eval,
            skip_generation_eval=args.skip_generation_eval,
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedMCQA answer-token classification SFT.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-examples", type=int, default=10_000)
    parser.add_argument("--eval-examples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/medmcqa_rlvr/olmo2_1b_medmcqa_sft")
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--skip-generation-eval", action="store_true")

    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help=argparse.SUPPRESS)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--max-prompt-length", type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument("--max-completion-length", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16", help=argparse.SUPPRESS)
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--lora-r", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--lora-alpha", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--lora-dropout", type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help=argparse.SUPPRESS)
    parser.add_argument("--logging-steps", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--report-to", default="none", help=argparse.SUPPRESS)
    return parser


def run_pipeline(config: ClassificationSFTConfig) -> dict[str, Any]:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "config.json", config.to_json())

    train_examples = load_examples(config.dataset_id, "train", config.train_examples, config.seed)
    eval_examples = load_examples(config.dataset_id, "validation", config.eval_examples, config.seed + 1)
    write_jsonl(config.output_dir / "train_sample_ids.jsonl", (_example_id_record(example) for example in train_examples))
    write_jsonl(config.output_dir / "eval_sample_ids.jsonl", (_example_id_record(example) for example in eval_examples))

    metrics: dict[str, Any] = {"config": config.to_json()}
    if not config.skip_baseline_eval:
        metrics["baseline"] = evaluate_model(config.model_id, eval_examples, config, output_prefix="baseline")
        cleanup_cuda()

    adapter_dir = train_classification_adapter(train_examples, config)
    metrics["adapter_dir"] = str(adapter_dir)
    cleanup_cuda()

    metrics["finetuned"] = evaluate_model(
        config.model_id,
        eval_examples,
        config,
        output_prefix="finetuned",
        adapter_dir=adapter_dir,
    )
    write_comparison_plot(config.output_dir, metrics)

    _write_json(config.output_dir / "metrics.json", metrics)
    return metrics


def train_classification_adapter(examples: list[MedMCQAExample], config: ClassificationSFTConfig) -> Path:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_kwargs: dict[str, Any] = {"dtype": dtype_map[config.dtype]}
    if config.device_map == "auto":
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if config.device_map == "cpu":
        model.to("cpu")

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        examples,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate_examples(batch, tokenizer, config.max_prompt_length),
    )

    answer_ids = answer_token_ids(tokenizer)
    total_steps = config.max_steps if config.max_steps > 0 else math.ceil(len(loader) * config.num_train_epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = _adamw(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    running_loss = 0.0

    while update_step < total_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = classification_loss(outputs.logits, batch["attention_mask"], batch["labels"], answer_ids)
            (loss / config.gradient_accumulation_steps).backward()

            running_loss += float(loss.detach().cpu())
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

            if update_step == 1 or update_step % config.logging_steps == 0:
                mean_loss = running_loss / max(config.logging_steps, 1)
                print(
                    json.dumps(
                        {
                            "step": update_step,
                            "loss": mean_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                        },
                        sort_keys=True,
                    )
                )
                running_loss = 0.0

            if update_step >= total_steps:
                break

    adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    return adapter_dir


def collate_examples(examples: list[MedMCQAExample], tokenizer: Any, max_prompt_length: int) -> dict[str, torch.Tensor]:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [render_prompt(example, tokenizer) for example in examples]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=True,
    )
    encoded["labels"] = torch.tensor(label_indices(examples), dtype=torch.long)
    return encoded


def classification_loss(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    token_ids: Sequence[int],
) -> torch.Tensor:
    token_index = torch.tensor(list(token_ids), device=logits.device, dtype=torch.long)
    last_idx = attention_mask.sum(dim=-1) - 1
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    answer_logits = logits[batch_idx, last_idx, :].index_select(dim=-1, index=token_index)
    return F.cross_entropy(answer_logits.float(), labels.to(logits.device))


def label_indices(examples: Iterable[MedMCQAExample]) -> list[int]:
    index_by_answer = {letter: index for index, letter in enumerate(ANSWER_LETTERS)}
    return [index_by_answer[example.answer] for example in examples]


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _adamw(parameters: Iterable[torch.nn.Parameter], lr: float, weight_decay: float) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    try:
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available())
    except TypeError:
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


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
    config = ClassificationSFTConfig.from_args(build_arg_parser().parse_args(argv))
    metrics = run_pipeline(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    gc.collect()
    return 0
