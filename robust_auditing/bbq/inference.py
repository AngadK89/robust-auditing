from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from robust_auditing.bbq.formatting import format_prompt
from robust_auditing.bbq.matching import match_prediction
from robust_auditing.bbq.targets import BBQTarget


@dataclass(frozen=True)
class BBQGenerationConfig:
    dtype: str = "bf16"
    device_map: str = "auto"
    batch_size: int = 8
    max_new_tokens: int = 16
    seed: int = 0


def build_prediction_record(
    row: dict[str, Any],
    *,
    target_id: str,
    prompt_format: str,
    prompt: str,
    raw_output: str,
) -> dict[str, Any]:
    match = match_prediction(raw_output, row)
    pred_label = match.pred_label
    is_correct = bool(match.matched and pred_label == int(row["label"]))
    is_biased_answer = bool(
        match.matched
        and not match.is_unknown
        and pred_label is not None
        and pred_label == int(row["target_loc"])
    )
    return {
        **row,
        "target_id": target_id,
        "format": prompt_format,
        "prompt": prompt,
        "raw_output": raw_output,
        "normalized_output": match.normalized_output,
        "pred_label": pred_label,
        "pred_cat": match.pred_cat,
        "matched": match.matched,
        "is_unknown": match.is_unknown,
        "is_correct": is_correct,
        "is_biased_answer": is_biased_answer,
    }


def generate_predictions_for_target(
    target: BBQTarget,
    rows: Sequence[dict[str, Any]],
    *,
    prompt_format: str,
    config: BBQGenerationConfig,
) -> list[dict[str, Any]]:
    return generate_predictions_for_target_formats(
        target,
        rows,
        prompt_formats=(prompt_format,),
        config=config,
    )[prompt_format]


def generate_predictions_for_target_formats(
    target: BBQTarget,
    rows: Sequence[dict[str, Any]],
    *,
    prompt_formats: Sequence[str],
    config: BBQGenerationConfig,
) -> dict[str, list[dict[str, Any]]]:
    set_seed(config.seed)
    model, tokenizer = load_model_and_tokenizer(target, dtype=config.dtype, device_map=config.device_map)
    try:
        by_format: dict[str, list[dict[str, Any]]] = {}
        for prompt_format in prompt_formats:
            prompts = [format_prompt(row, prompt_format) for row in rows]
            records: list[dict[str, Any]] = []
            for start in range(0, len(rows), config.batch_size):
                batch_rows = rows[start : start + config.batch_size]
                batch_prompts = prompts[start : start + config.batch_size]
                outputs = generate_text_batch(model, tokenizer, batch_prompts, config)
                for row, prompt, raw_output in zip(batch_rows, batch_prompts, outputs):
                    records.append(
                        build_prediction_record(
                            row,
                            target_id=target.target_id,
                            prompt_format=prompt_format,
                            prompt=prompt,
                            raw_output=raw_output,
                        )
                    )
            by_format[prompt_format] = records
        return by_format
    finally:
        cleanup_runtime()


def load_model_and_tokenizer(target: BBQTarget, *, dtype: str, device_map: str) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype(dtype)}
    if device_map == "auto":
        model_kwargs["device_map"] = "auto"
    elif device_map != "cpu":
        raise ValueError(f"Unknown device_map for BBQ generation: {device_map}")
    model = AutoModelForCausalLM.from_pretrained(target.base_model_id, **model_kwargs)
    if target.adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(target.adapter_dir))
    if device_map == "cpu":
        import torch

        model.to(torch.device("cpu"))
    model.eval()
    return model, tokenizer


def generate_text_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    config: BBQGenerationConfig,
) -> list[str]:
    inputs = tokenizer(list(prompts), return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=config.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [tokenizer.decode(row[input_width:], skip_special_tokens=True) for row in output_ids]


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


def _torch_dtype(dtype: str) -> Any:
    import torch

    if dtype == "auto":
        return "auto"
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unknown BBQ dtype: {dtype}")


def cleanup_runtime() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
