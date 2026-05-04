from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from robust_auditing.fairness.adapters import BaseAdapter, FairnessExample
from robust_auditing.fairness.artifacts import (
    FairnessArtifactPaths,
    fairness_example_to_record,
    write_json,
    write_jsonl,
)
from robust_auditing.fairness.cli import (
    AUDIT_ADAPTERS,
    DEFAULT_MODEL_ID,
    _parse_csv,
    load_dataset_for_adapter,
    load_model_and_tokenizer,
    set_seed,
)


@dataclass(frozen=True)
class GenerationConfig:
    audits: tuple[str, ...] = ("holistic_bias", "bold")
    model_id: str = DEFAULT_MODEL_ID
    batch_size: int = 8
    max_examples: int | None = None
    output_root: Path = Path("artifacts/fairness")
    dtype: str = "auto"
    device_map: str = "auto"
    seed: int = 0
    num_beams: int = 3
    min_new_tokens: int = 20
    max_new_tokens: int = 64
    no_repeat_ngram_size: int = 3
    prompts_only: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GenerationConfig":
        audits = tuple(_parse_csv(args.audits))
        unknown = sorted(set(audits) - set(AUDIT_ADAPTERS))
        if unknown:
            raise ValueError(f"Unknown audit(s): {', '.join(unknown)}")
        return cls(
            audits=audits,
            model_id=args.model_id,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            output_root=Path(args.output_root),
            dtype=args.dtype,
            device_map=args.device_map,
            seed=args.seed,
            num_beams=args.num_beams,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            prompts_only=args.prompts_only,
        )

    def paths_for(self, audit: str) -> FairnessArtifactPaths:
        return FairnessArtifactPaths(self.output_root, audit, self.model_id)


def build_generation_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and store fairness audit model responses.")
    parser.add_argument("--audits", default="holistic_bias,bold", help="Comma-separated audits to run.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-root", default="artifacts/fairness")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--min-new-tokens", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--prompts-only", action="store_true")
    return parser


def normalize_examples(
    audit: str,
    config: GenerationConfig,
    dataset: Any | None = None,
) -> tuple[BaseAdapter, list[FairnessExample]]:
    adapter = AUDIT_ADAPTERS[audit]()
    source = dataset if dataset is not None else load_dataset_for_adapter(adapter)
    examples = list(adapter.normalize(source))
    if config.max_examples is not None:
        examples = examples[: config.max_examples]
    return adapter, examples


def write_normalized_prompts(
    audit: str,
    config: GenerationConfig,
    dataset: Any | None = None,
) -> tuple[FairnessArtifactPaths, list[FairnessExample], BaseAdapter]:
    adapter, examples = normalize_examples(audit, config, dataset=dataset)
    paths = config.paths_for(audit)
    write_jsonl(paths.normalized_prompts, (fairness_example_to_record(example) for example in examples))
    return paths, examples, adapter


def generate_responses_for_audit(
    audit: str,
    config: GenerationConfig,
    model: Any = None,
    tokenizer: Any = None,
    dataset: Any | None = None,
) -> Path:
    paths, examples, adapter = write_normalized_prompts(audit, config, dataset=dataset)

    response_count = 0
    if not config.prompts_only:
        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(config.model_id, config.dtype, config.device_map)
        rows = _generate_response_rows(examples, config, model, tokenizer)
        response_count = write_jsonl(paths.model_responses, rows)

    write_json(
        paths.metadata,
        {
            "audit": audit,
            "dataset_id": adapter.dataset_id,
            "data_files": adapter.data_files,
            "split": adapter.split,
            "model_id": config.model_id,
            "batch_size": config.batch_size,
            "max_examples": config.max_examples,
            "dtype": config.dtype,
            "device_map": config.device_map,
            "seed": config.seed,
            "example_count": len(examples),
            "generation_count": response_count,
            "normalized_prompts_artifact": paths.normalized_prompts.name,
            "model_responses_artifact": None if config.prompts_only else paths.model_responses.name,
            "generation": None if config.prompts_only else generation_metadata(config),
        },
    )
    return paths.audit_dir


def generation_metadata(config: GenerationConfig) -> dict[str, Any]:
    return {
        "decoding": "beam_search",
        "do_sample": False,
        "num_beams": config.num_beams,
        "min_new_tokens": config.min_new_tokens,
        "max_new_tokens": config.max_new_tokens,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
    }


def _generate_response_rows(
    examples: Sequence[FairnessExample],
    config: GenerationConfig,
    model: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()

    for start in tqdm(range(0, len(examples), config.batch_size), desc="Generating", unit="batch"):
        batch = examples[start : start + config.batch_size]
        texts = [example.text for example in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]

        import torch

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=config.num_beams,
                min_new_tokens=config.min_new_tokens,
                max_new_tokens=config.max_new_tokens,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )

        for offset, (example, output_ids) in enumerate(zip(batch, generated)):
            response_ids = output_ids[prompt_width:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            rows.append(
                {
                    **fairness_example_to_record(example),
                    "generated_response": response,
                    "generation": {
                        **generation_metadata(config),
                        "model_id": config.model_id,
                        "seed": config.seed,
                        "response_index": start + offset,
                        "generated_token_count": int(response_ids.numel()),
                    },
                }
            )

    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_generation_arg_parser()
    config = GenerationConfig.from_args(parser.parse_args(argv))
    set_seed(config.seed)
    model = None
    tokenizer = None
    if not config.prompts_only:
        model, tokenizer = load_model_and_tokenizer(config.model_id, config.dtype, config.device_map)
    for audit in config.audits:
        output_dir = generate_responses_for_audit(audit, config, model=model, tokenizer=tokenizer)
        print(f"Wrote {audit} generation artifacts to {output_dir}")
    return 0
