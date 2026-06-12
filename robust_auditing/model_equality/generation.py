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
    adapter_dir: Path | None
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
    ignore_eos: bool = False
    backend: str = "hf"
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9


class CompletionGenerator:
    def __init__(self, config: GenerationRuntimeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> "CompletionGenerator":
        if self.config.backend == "vllm":
            self.model, self.tokenizer = _load_vllm_and_tokenizer(self.config)
        elif self.config.backend == "hf":
            self.model, self.tokenizer = _load_model_and_tokenizer(self.config)
        else:
            raise ValueError(f"Unknown generation backend: {self.config.backend}")
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

    def iter_record_batches(
        self,
        suite: str,
        prompt_records: Sequence[PromptRecord],
        *,
        model_label: str,
        adapter_enabled: bool,
        max_new_tokens: int | None = None,
    ):
        yield from self._iter_record_batches(
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
        for batch_records in self._iter_record_batches(
            suite,
            prompt_records,
            model_label=model_label,
            adapter_enabled=adapter_enabled,
            max_new_tokens=max_new_tokens,
        ):
            records.extend(batch_records)
        return records

    def _iter_record_batches(
        self,
        suite: str,
        prompt_records: Sequence[PromptRecord],
        *,
        model_label: str,
        adapter_enabled: bool,
        max_new_tokens: int | None = None,
    ):
        if self.config.backend == "vllm":
            yield from self._iter_vllm_record_batches(
                suite,
                prompt_records,
                model_label=model_label,
                adapter_enabled=adapter_enabled,
                max_new_tokens=max_new_tokens,
            )
            return

        prompts: list[PromptRecord] = []
        sample_indices: list[int] = []
        for prompt_record in prompt_records:
            for sample_index in range(self.config.samples_per_prompt):
                prompts.append(prompt_record)
                sample_indices.append(sample_index)

        context = _adapter_context(self.model, adapter_enabled)
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
                batch_records: list[CompletionRecord] = []
                for prompt_record, sample_index, (completion_text, token_ids) in zip(batch_prompts, batch_indices, generated):
                    batch_records.append(
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
                yield batch_records

    def _iter_vllm_record_batches(
        self,
        suite: str,
        prompt_records: Sequence[PromptRecord],
        *,
        model_label: str,
        adapter_enabled: bool,
        max_new_tokens: int | None = None,
    ):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CompletionGenerator must be entered before generation")
        rendered = [render_prompt(self.tokenizer, record, self.config.prompt_format) for record in prompt_records]
        generated = _generate_vllm_text_and_token_ids(
            self.model,
            rendered,
            self.config,
            model_label=model_label,
            adapter_enabled=adapter_enabled,
            max_new_tokens=max_new_tokens,
        )
        records: list[CompletionRecord] = []
        for prompt_record, prompt_outputs in zip(prompt_records, generated):
            if len(prompt_outputs) != self.config.samples_per_prompt:
                raise RuntimeError(
                    f"vLLM returned {len(prompt_outputs)} sample(s) for prompt_id={prompt_record.prompt_id}; "
                    f"expected {self.config.samples_per_prompt}"
                )
            for sample_index, (completion_text, token_ids) in enumerate(prompt_outputs):
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
        if self.config.progress:
            print(
                f"[generation] suite={suite} model={model_label} completions={len(records)}/{len(records)}",
                flush=True,
            )
        yield records


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
    if config.adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(config.adapter_dir))
    if config.device != "auto":
        model.to(torch.device(config.device))
    model.eval()
    return model, tokenizer


def _load_vllm_and_tokenizer(config: GenerationRuntimeConfig) -> tuple[Any, Any]:
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": config.base_model_id,
        "dtype": _vllm_dtype(config.dtype),
        "seed": config.seed,
        "max_num_seqs": config.max_num_seqs,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if config.adapter_dir is not None:
        llm_kwargs.update({"enable_lora": True, "max_loras": 1})
    llm = LLM(**llm_kwargs)
    return llm, llm.get_tokenizer()


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
        "eos_token_id": None if config.ignore_eos else tokenizer.eos_token_id,
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


def _generate_vllm_text_and_token_ids(
    llm: Any,
    prompts: Sequence[str],
    config: GenerationRuntimeConfig,
    *,
    model_label: str,
    adapter_enabled: bool,
    max_new_tokens: int | None = None,
) -> list[list[tuple[str, list[int]]]]:
    if config.num_beams != 1:
        raise ValueError("vLLM generation backend only supports num_beams=1 for MET sampling")
    if not config.do_sample and config.temperature != 0:
        raise ValueError("vLLM deterministic generation requires temperature=0 when do_sample=False")

    from vllm import SamplingParams

    sampling_kwargs: dict[str, Any] = {
        "n": config.samples_per_prompt,
        "temperature": config.temperature if config.do_sample else 0.0,
        "top_p": config.top_p,
        "max_tokens": max_new_tokens if max_new_tokens is not None else config.max_new_tokens,
        "ignore_eos": config.ignore_eos,
    }
    if config.top_k is not None and config.top_k > 0:
        sampling_kwargs["top_k"] = config.top_k
    sampling_params = SamplingParams(**sampling_kwargs)

    lora_request = None
    if adapter_enabled and config.adapter_dir is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(model_label, 1, str(config.adapter_dir))
    outputs = llm.generate(list(prompts), sampling_params, lora_request=lora_request, use_tqdm=False)
    generated: list[list[tuple[str, list[int]]]] = []
    for request_output in outputs:
        prompt_outputs: list[tuple[str, list[int]]] = []
        for completion in request_output.outputs:
            prompt_outputs.append(
                (
                    str(completion.text),
                    [int(token_id) for token_id in completion.token_ids],
                )
            )
        generated.append(prompt_outputs)
    return generated


def _adapter_context(model: Any, adapter_enabled: bool):
    if adapter_enabled:
        return nullcontext()
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


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


def _vllm_dtype(dtype: str) -> str:
    if dtype == "auto":
        return "auto"
    if dtype == "bf16":
        return "bfloat16"
    if dtype == "fp16":
        return "float16"
    if dtype == "fp32":
        return "float32"
    raise ValueError(f"Unknown dtype: {dtype}")


def _cleanup_runtime() -> None:
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
