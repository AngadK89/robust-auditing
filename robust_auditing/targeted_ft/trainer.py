from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "inverted_dpo": 0.35,
    "dpo": 0.25,
    "sft": 0.20,
    "holistic_bias_anchor": 0.15,
    "rl_reward": 0.05,
}


@dataclass(frozen=True)
class LoraConfigSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] | None = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class TargetedFTTrainerConfig:
    model_name_or_path: str = "allenai/OLMo-2-0425-1B-Instruct"
    output_dir: str = "outputs/targeted_ft/olmo2_1b_lora"
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    objective_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OBJECTIVE_WEIGHTS))
    bf16: bool = True
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    dpo_beta: float = 0.1
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    save_adapter: bool = True
    save_merged_checkpoint: bool = True

    def training_arguments(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "bf16": self.bf16,
            "optim": self.optim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_grad_norm": self.max_grad_norm,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
        }

    def dpo_training_arguments(self) -> dict[str, Any]:
        args = self.training_arguments()
        args["output_dir"] = str(Path(self.output_dir) / "dpo")
        args["beta"] = self.dpo_beta
        return args

    def sft_training_arguments(self) -> dict[str, Any]:
        args = self.training_arguments()
        args["output_dir"] = str(Path(self.output_dir) / "sft")
        return args


class TargetedFTTrainer:
    def __init__(self, config: TargetedFTTrainerConfig | None = None) -> None:
        self.config = config or TargetedFTTrainerConfig()

    def build_lora_config(self) -> Any:
        try:
            from peft import LoraConfig
        except ImportError as exc:
            raise ImportError("Install peft to build LoRA configs for targeted_ft training") from exc

        kwargs: dict[str, Any] = {
            "r": self.config.lora.r,
            "lora_alpha": self.config.lora.alpha,
            "lora_dropout": self.config.lora.dropout,
            "bias": self.config.lora.bias,
            "task_type": self.config.lora.task_type,
        }
        if self.config.lora.target_modules is not None:
            kwargs["target_modules"] = list(self.config.lora.target_modules)
        return LoraConfig(**kwargs)

    def load_model_and_tokenizer(self) -> tuple[Any, Any]:
        try:
            import torch
            from peft import get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch, transformers, and peft to run targeted_ft training") from exc

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.bfloat16 if self.config.bf16 else "auto",
        )
        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        return get_peft_model(model, self.build_lora_config()), tokenizer

    def train(self, train_dataset: Any, data_collator: Any | None = None) -> Any:
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError as exc:
            raise ImportError("Install transformers to run targeted_ft training") from exc

        model, tokenizer = self.load_model_and_tokenizer()
        args = TrainingArguments(**self.config.training_arguments())
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        result = trainer.train()
        self.save_checkpoints(model)
        return result

    def prepare_mixture_batches(self, mixture: Any) -> dict[str, list[dict[str, Any]]]:
        from .batches import (
            build_dpo_pairs,
            build_nll_anchor_examples,
            build_rlvr_math_records,
            build_sft_examples,
        )

        rlvr_records = build_rlvr_math_records(mixture.rl_reward)
        return {
            "inverted_dpo": build_dpo_pairs(mixture.inverted_dpo),
            "dpo": build_dpo_pairs(mixture.dpo),
            "sft": build_sft_examples(mixture.sft),
            "holistic_bias_anchor": build_nll_anchor_examples(mixture.holistic_bias_anchor),
            "rlvr_math_verifier_dpo": rlvr_records["verifier_dpo"],
            "rlvr_math_eval": rlvr_records["eval"],
        }

    def save_checkpoints(self, model: Any) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_adapter and hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir / "adapter")
        if self.config.save_merged_checkpoint and hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            if hasattr(merged, "save_pretrained"):
                merged.save_pretrained(output_dir / "merged_checkpoint")
