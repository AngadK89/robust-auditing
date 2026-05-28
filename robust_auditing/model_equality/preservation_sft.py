from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from robust_auditing.fairness.artifacts import read_jsonl, write_json, write_jsonl
from robust_auditing.fairness.metrics import bold_mean_harm_score, bold_stddev_harm_score
from robust_auditing.medmcqa_rlvr.train import DEFAULT_MODEL_ID, set_seed
from robust_auditing.model_equality.completions import CompletionRecord, read_completion_records
from robust_auditing.model_equality.generation import render_prompt
from robust_auditing.model_equality.prompts import PROMPT_SUITES, PromptRecord


@dataclass(frozen=True)
class PreservationRecord:
    suite: str
    prompt_id: str
    prompt_text: str
    completion_text: str
    sample_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_format: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreservationSFTConfig:
    source_adapter_dir: Path
    met_root: Path
    output_dir: Path
    base_model_id: str = DEFAULT_MODEL_ID
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 5.0
    max_steps: int = -1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 1536
    dtype: str = "bf16"
    device_map: str = "auto"
    prompt_format: str = "chat"
    seed: int = 0
    logging_steps: int = 10
    max_samples_per_prompt: int | None = None
    bold_replay_responses: Path | None = None
    bold_replay_max_examples: int | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PreservationSFTConfig":
        return cls(
            source_adapter_dir=Path(args.source_adapter_dir),
            met_root=Path(args.met_root),
            output_dir=Path(args.output_dir),
            base_model_id=args.base_model_id,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            dtype=args.dtype,
            device_map=args.device_map,
            prompt_format=args.prompt_format,
            seed=args.seed,
            logging_steps=args.logging_steps,
            max_samples_per_prompt=args.max_samples_per_prompt,
            bold_replay_responses=Path(args.bold_replay_responses) if args.bold_replay_responses else None,
            bold_replay_max_examples=args.bold_replay_max_examples,
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("source_adapter_dir", "met_root", "output_dir", "bold_replay_responses"):
            payload[key] = str(payload[key]) if payload[key] is not None else None
        return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continue a poisoned adapter on clean MET preservation completions.")
    parser.add_argument("--source-adapter-dir", type=Path, required=True)
    parser.add_argument("--met-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help=argparse.SUPPRESS)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--prompt-format", choices=("auto", "raw", "chat"), default="chat", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--max-samples-per-prompt", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bold-replay-responses", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bold-replay-max-examples", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def load_met_preservation_records(
    met_root: Path,
    *,
    prompt_suites: Sequence[str] = PROMPT_SUITES,
    max_samples_per_prompt: int | None = None,
) -> list[PreservationRecord]:
    records: list[PreservationRecord] = []
    for suite in prompt_suites:
        suite_dir = Path(met_root) / "suites" / suite
        prompts_path = suite_dir / "prompts.jsonl"
        completions_path = suite_dir / "completions_base.jsonl"
        if not prompts_path.exists() or not completions_path.exists():
            continue
        prompts_by_id = {record.prompt_id: record for record in _read_prompt_records(prompts_path)}
        counts: dict[str, int] = {}
        for completion in read_completion_records(completions_path):
            prompt = prompts_by_id.get(completion.prompt_id)
            if prompt is None:
                raise ValueError(f"Completion references unknown prompt_id in {suite}: {completion.prompt_id}")
            seen = counts.get(completion.prompt_id, 0)
            if max_samples_per_prompt is not None and seen >= max_samples_per_prompt:
                continue
            counts[completion.prompt_id] = seen + 1
            records.append(_preservation_from_completion(completion, prompt))
    if not records:
        raise FileNotFoundError(f"No MET preservation records found under {met_root}")
    return records


def load_bold_replay_records(path: Path, *, max_examples: int | None = None) -> list[PreservationRecord]:
    rows = read_jsonl(path)
    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError("max_examples must be positive")
        rng = random.Random(0)
        rows = list(rows)
        rng.shuffle(rows)
        rows = rows[:max_examples]
    records: list[PreservationRecord] = []
    for index, row in enumerate(rows):
        completion = str(row.get("generated_response", ""))
        if not completion:
            continue
        records.append(
            PreservationRecord(
                suite="bold_replay",
                prompt_id=_bold_row_key(row),
                prompt_text=str(row["text"]),
                completion_text=completion,
                sample_index=index,
                metadata=dict(row.get("metadata", {})),
                prompt_format="raw",
            )
        )
    return records


def render_preservation_prompt(tokenizer: Any, record: PreservationRecord, prompt_format: str) -> str:
    effective_format = record.prompt_format or prompt_format
    prompt_record = PromptRecord(record.suite, record.prompt_id, record.prompt_text, dict(record.metadata))
    return render_prompt(tokenizer, prompt_record, effective_format)


def encode_preservation_record(
    record: PreservationRecord,
    tokenizer: Any,
    *,
    max_length: int,
    prompt_format: str,
) -> dict[str, list[int]]:
    rendered_prompt = render_preservation_prompt(tokenizer, record, prompt_format)
    prompt_ids = _tokenize_text(tokenizer, rendered_prompt)
    completion_ids = _tokenize_text(tokenizer, record.completion_text)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        completion_ids = [*completion_ids, int(eos_token_id)]
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if len(completion_ids) >= max_length:
        prompt_ids = []
        completion_ids = completion_ids[-max_length:]
    else:
        prompt_budget = max_length - len(completion_ids)
        prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = [*prompt_ids, *completion_ids]
    labels = [-100] * len(prompt_ids) + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def collate_preservation_records(
    records: Sequence[PreservationRecord],
    tokenizer: Any,
    *,
    max_length: int,
    prompt_format: str,
) -> dict[str, Any]:
    import torch

    encoded = [
        encode_preservation_record(record, tokenizer, max_length=max_length, prompt_format=prompt_format)
        for record in records
    ]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0
    width = max(len(item["input_ids"]) for item in encoded)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in encoded:
        pad = width - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [int(pad_token_id)] * pad)
        batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
        batch["labels"].append(item["labels"] + [-100] * pad)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def run_pipeline(config: PreservationSFTConfig) -> dict[str, Any]:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "config.json", config.to_json())
    records = load_met_preservation_records(
        config.met_root,
        max_samples_per_prompt=config.max_samples_per_prompt,
    )
    if config.bold_replay_responses is not None:
        records.extend(load_bold_replay_records(config.bold_replay_responses, max_examples=config.bold_replay_max_examples))
    write_jsonl(config.output_dir / "preservation_records.jsonl", (record.to_json() for record in records))
    adapter_dir = train_preservation_adapter(records, config)
    summary = {"config": config.to_json(), "train_record_count": len(records), "adapter_dir": str(adapter_dir)}
    write_json(config.output_dir / "metrics.json", summary)
    return summary


def train_preservation_adapter(records: Sequence[PreservationRecord], config: PreservationSFTConfig) -> Path:
    import torch
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {"dtype": _torch_dtype(config.dtype)}
    if config.device_map != "cpu":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.base_model_id, **model_kwargs)
    if config.device_map == "cpu":
        model.to("cpu")
    model = PeftModel.from_pretrained(model, str(config.source_adapter_dir), is_trainable=True)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        list(records),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate_preservation_records(
            batch,
            tokenizer,
            max_length=config.max_length,
            prompt_format=config.prompt_format,
        ),
    )
    total_steps = config.max_steps if config.max_steps > 0 else math.ceil(len(loader) * config.num_train_epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = _adamw(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    running_loss = 0.0

    while update_step < total_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
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
                divisor = config.logging_steps if update_step % config.logging_steps == 0 else 1
                print(json.dumps({"step": update_step, "loss": running_loss / divisor, "learning_rate": scheduler.get_last_lr()[0]}, sort_keys=True))
                running_loss = 0.0
            if update_step >= total_steps:
                break

    adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return adapter_dir


def compare_bold_same_rows(candidate_per_example: Path, clean_per_example: Path) -> dict[str, Any]:
    candidate_rows = read_jsonl(candidate_per_example)
    clean_by_key = {_bold_row_key(row): row for row in read_jsonl(clean_per_example)}
    matched_clean: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in candidate_rows:
        key = _bold_row_key(row)
        clean_row = clean_by_key.get(key)
        if clean_row is None:
            missing.append(key)
        else:
            matched_clean.append(clean_row)
    if missing:
        sample = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise ValueError(f"Clean BOLD scores are missing candidate rows: {sample}{suffix}")
    candidate_summary = summarize_bold_harm_rows(candidate_rows)
    clean_summary = summarize_bold_harm_rows(matched_clean)
    return {
        "row_count": len(candidate_rows),
        "candidate": candidate_summary,
        "clean": clean_summary,
        "mean_harm_delta": candidate_summary["bold_mean_harm_score"] - clean_summary["bold_mean_harm_score"],
        "stddev_harm_delta": candidate_summary["bold_stddev_harm_score"] - clean_summary["bold_stddev_harm_score"],
        "passes_elevated_bold_gate": (
            candidate_summary["bold_mean_harm_score"] > clean_summary["bold_mean_harm_score"]
            or candidate_summary["bold_stddev_harm_score"] > clean_summary["bold_stddev_harm_score"]
        ),
    }


def summarize_bold_harm_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    frame = pd.DataFrame([_bold_harm_frame_record(row) for row in rows])
    return {
        "bold_mean_harm_score": bold_mean_harm_score(frame),
        "bold_stddev_harm_score": bold_stddev_harm_score(frame),
    }


def _read_prompt_records(path: Path) -> list[PromptRecord]:
    return [PromptRecord.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _preservation_from_completion(completion: CompletionRecord, prompt: PromptRecord) -> PreservationRecord:
    return PreservationRecord(
        suite=completion.suite,
        prompt_id=completion.prompt_id,
        prompt_text=prompt.text,
        completion_text=completion.completion_text,
        sample_index=completion.sample_index,
        metadata={"prompt_metadata": dict(prompt.metadata), "completion_metadata": dict(completion.metadata)},
    )


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def _bold_harm_frame_record(row: Mapping[str, Any]) -> dict[str, Any]:
    scores = dict(row.get("scores", {}))
    if "negative_sentiment_score" in scores:
        negative_sentiment = float(scores["negative_sentiment_score"])
    elif "sentiment_score" in scores:
        negative_sentiment = 1.0 - float(scores["sentiment_score"])
    elif "sentiment_compound" in scores:
        negative_sentiment = (1.0 - float(scores["sentiment_compound"])) / 2.0
    else:
        raise ValueError("BOLD score row is missing sentiment score fields")
    return {
        "text": str(row["text"]),
        "axis": str(row["axis"]),
        "bucket": str(row["bucket"]),
        "descriptor": str(row["descriptor"]),
        "negative_sentiment_score": negative_sentiment,
        "toxicity_score": float(scores["toxicity_score"]),
    }


def _bold_row_key(row: Mapping[str, Any]) -> str:
    metadata = dict(row.get("metadata", {}))
    parts = [
        str(row.get("axis", "")),
        str(row.get("bucket", "")),
        str(row.get("descriptor", "")),
        str(row.get("text", "")),
        str(metadata.get("source_index", "")),
        str(metadata.get("prompt_index", "")),
        str(metadata.get("name", "")),
    ]
    return "␟".join(parts)


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


def _adamw(parameters: Any, *, lr: float, weight_decay: float):
    import torch

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    try:
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available())
    except TypeError:
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


def main(argv: Sequence[str] | None = None) -> int:
    config = PreservationSFTConfig.from_args(build_arg_parser().parse_args(argv))
    summary = run_pipeline(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
