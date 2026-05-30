from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from robust_auditing.model_equality.completions import CompletionRecord
from robust_auditing.model_equality.prompts import PromptRecord


@dataclass(frozen=True)
class GenerationRuntimeConfig:
    base_model_id: str
    adapter_dir: Path
    samples_per_prompt: int
    max_new_tokens: int
    temperature: float
    top_p: float
    num_beams: int
    do_sample: bool
    dtype: str
    device: str
    batch_size: int
    prompt_format: str
    seed: int
    top_k: int | None = None
    progress: bool = False


class CompletionGenerator:
    def __init__(self, config: GenerationRuntimeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> "CompletionGenerator":
        self.model, self.tokenizer = _load_model_and_tokenizer(self.config)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.model = None
        self.tokenizer = None
        _cleanup_runtime()

    def generate_pair(
        self,
        suite: str,
        prompt_records: Sequence[PromptRecord],
        *,
        max_new_tokens: int | None = None,
        base_label: str = "base",
        candidate_label: str = "grpo",
    ) -> tuple[list[CompletionRecord], list[CompletionRecord]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CompletionGenerator must be entered before generation")
        base_records = self._generate_records(
            suite,
            prompt_records,
            model_label=base_label,
            adapter_enabled=False,
            max_new_tokens=max_new_tokens,
        )
        grpo_records = self._generate_records(
            suite,
            prompt_records,
            model_label=candidate_label,
            adapter_enabled=True,
            max_new_tokens=max_new_tokens,
        )
        return base_records, grpo_records

    def generate_records(
        self,
        suite: str,
        prompt_records: Sequence[PromptRecord],
        *,
        model_label: str,
        adapter_enabled: bool,
        max_new_tokens: int | None = None,
    ) -> list[CompletionRecord]:
        return self._generate_records(
            suite,
            prompt_records,
            model_label=model_label,
            adapter_enabled=adapter_enabled,
            max_new_tokens=max_new_tokens,
        )

    def _generate_records(
        self,
        suite: str,
        prompt_records: Sequence[PromptRecord],
        *,
        model_label: str,
        adapter_enabled: bool,
        max_new_tokens: int | None = None,
    ) -> list[CompletionRecord]:
        records: list[CompletionRecord] = []
        prompts: list[PromptRecord] = []
        sample_indices: list[int] = []
        for prompt_record in prompt_records:
            for sample_index in range(self.config.samples_per_prompt):
                prompts.append(prompt_record)
                sample_indices.append(sample_index)

        context = nullcontext() if adapter_enabled else self.model.disable_adapter()
        total = len(prompts)
        with context:
            for batch_number, start in enumerate(range(0, total, self.config.batch_size), start=1):
                batch_prompts = prompts[start : start + self.config.batch_size]
                batch_indices = sample_indices[start : start + self.config.batch_size]
                rendered = [render_prompt(self.tokenizer, record, self.config.prompt_format) for record in batch_prompts]
                generated = _generate_text_and_token_ids_batch(
                    self.model,
                    self.tokenizer,
                    rendered,
                    self.config,
                    max_new_tokens=max_new_tokens,
                )
                for prompt_record, sample_index, (completion_text, token_ids) in zip(batch_prompts, batch_indices, generated):
                    records.append(
                        CompletionRecord(
                            suite=suite,
                            prompt_id=prompt_record.prompt_id,
                            model_label=model_label,
                            sample_index=sample_index,
                            prompt=prompt_record.text,
                            completion_text=completion_text,
                            metadata={"prompt_format": self.config.prompt_format, "completion_token_ids": token_ids},
                        )
                    )
                completed = min(start + len(batch_prompts), total)
                if self.config.progress and (batch_number == 1 or batch_number % 25 == 0 or completed == total):
                    print(
                        f"[generation] suite={suite} model={model_label} completions={completed}/{total}",
                        flush=True,
                    )
        return records


def render_prompt(tokenizer: Any, record: PromptRecord, prompt_format: str) -> str:
    if prompt_format == "raw" or (prompt_format == "auto" and record.suite in {"wikipedia", "humaneval"}):
        return record.text
    if prompt_format in {"chat", "auto"}:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": record.text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return record.text
    raise ValueError(f"Unknown prompt format: {prompt_format}")


def _load_model_and_tokenizer(config: GenerationRuntimeConfig) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype(config.dtype)}
    if config.device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.base_model_id, **model_kwargs)
    model = PeftModel.from_pretrained(model, str(config.adapter_dir))
    if config.device != "auto":
        model.to(torch.device(config.device))
    model.eval()
    return model, tokenizer


def _generate_text_batch(model: Any, tokenizer: Any, prompts: Sequence[str], config: GenerationRuntimeConfig) -> list[str]:
    return [
        completion_text
        for completion_text, _token_ids in _generate_text_and_token_ids_batch(model, tokenizer, prompts, config)
    ]


def _generate_text_and_token_ids_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    config: GenerationRuntimeConfig,
    *,
    max_new_tokens: int | None = None,
) -> list[tuple[str, list[int]]]:
    inputs = tokenizer(list(prompts), return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    generate_kwargs: dict[str, Any] = {
        "do_sample": config.do_sample,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_beams": config.num_beams,
        "max_new_tokens": max_new_tokens if max_new_tokens is not None else config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.top_k is not None:
        generate_kwargs["top_k"] = config.top_k
    output_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )
    completions: list[tuple[str, list[int]]] = []
    for row in output_ids:
        completion_ids = row[input_width:].detach().cpu().tolist() if hasattr(row[input_width:], "detach") else list(row[input_width:])
        completions.append((tokenizer.decode(completion_ids, skip_special_tokens=True), [int(token_id) for token_id in completion_ids]))
    return completions


def _torch_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return "auto"
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def _cleanup_runtime() -> None:
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
