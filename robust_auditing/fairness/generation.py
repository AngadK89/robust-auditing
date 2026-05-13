from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from robust_auditing.fairness.adapters import BaseAdapter, FairnessExample
from robust_auditing.fairness.artifacts import (
    FairnessArtifactPaths,
    fairness_example_to_record,
    read_examples,
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


LOGGER = logging.getLogger(__name__)


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
    subset_id: str | None = None

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
            subset_id=args.subset_id,
        )

    def paths_for(self, audit: str) -> FairnessArtifactPaths:
        return FairnessArtifactPaths(self.output_root, audit, self.model_id, subset_id=self.subset_id)


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
    parser.add_argument("--subset-id", default=None, help="Use a stored sampled subset by id.")
    return parser


def normalize_examples(
    audit: str,
    config: GenerationConfig,
    dataset: Any | None = None,
) -> tuple[BaseAdapter, list[FairnessExample]]:
    adapter = AUDIT_ADAPTERS[audit]()
    LOGGER.info("Loading %s dataset for audit '%s'", adapter.name, audit)
    source = dataset if dataset is not None else load_dataset_for_adapter(adapter)
    LOGGER.info("Normalizing prompts for audit '%s'", audit)
    examples = list(adapter.normalize(source))
    if config.max_examples is not None:
        examples = examples[: config.max_examples]
        LOGGER.info("Limited audit '%s' to %d normalized prompts", audit, len(examples))
    else:
        LOGGER.info("Normalized %d prompts for audit '%s'", len(examples), audit)
    return adapter, examples


def write_normalized_prompts(
    audit: str,
    config: GenerationConfig,
    dataset: Any | None = None,
) -> tuple[FairnessArtifactPaths, list[FairnessExample], BaseAdapter]:
    if config.subset_id is not None:
        adapter = AUDIT_ADAPTERS[audit]()
        paths = config.paths_for(audit)
        examples = read_examples(paths.normalized_prompts)
        LOGGER.info(
            "Loaded %d stored subset prompts for audit '%s' from %s",
            len(examples),
            audit,
            paths.normalized_prompts,
        )
        return paths, examples, adapter

    adapter, examples = normalize_examples(audit, config, dataset=dataset)
    paths = config.paths_for(audit)
    write_jsonl(paths.normalized_prompts, (fairness_example_to_record(example) for example in examples))
    LOGGER.info("Wrote normalized prompts for audit '%s' to %s", audit, paths.normalized_prompts)
    return paths, examples, adapter


def generate_responses_for_audit(
    audit: str,
    config: GenerationConfig,
    model: Any = None,
    tokenizer: Any = None,
    dataset: Any | None = None,
) -> Path:
    LOGGER.info("Starting generation workflow for audit '%s'", audit)
    paths, examples, adapter = write_normalized_prompts(audit, config, dataset=dataset)

    response_count = 0
    if not config.prompts_only:
        if model is None or tokenizer is None:
            LOGGER.info("Loading generation model '%s'", config.model_id)
            model, tokenizer = load_model_and_tokenizer(config.model_id, config.dtype, config.device_map)
        tokenizer.padding_side = "left"
        LOGGER.info(
            "Generating %d responses for audit '%s' with beam search: num_beams=%d, min_new_tokens=%d, max_new_tokens=%d",
            len(examples),
            audit,
            config.num_beams,
            config.min_new_tokens,
            config.max_new_tokens,
        )
        rows = _generate_response_rows(examples, config, model, tokenizer)
        response_count = write_jsonl(paths.model_responses, rows)
        LOGGER.info("Wrote %d model responses for audit '%s' to %s", response_count, audit, paths.model_responses)
    else:
        LOGGER.info("Skipping response generation for audit '%s' because --prompts-only was set", audit)

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
            "subset_id": config.subset_id,
            "example_count": len(examples),
            "generation_count": response_count,
            "normalized_prompts_artifact": paths.normalized_prompts.name,
            "model_responses_artifact": None if config.prompts_only else paths.model_responses.name,
            "generation": None if config.prompts_only else generation_metadata(config),
        },
    )
    LOGGER.info("Wrote generation metadata for audit '%s' to %s", audit, paths.metadata)
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_generation_arg_parser()
    config = GenerationConfig.from_args(parser.parse_args(argv))
    LOGGER.info("Selected audits for generation: %s", ", ".join(config.audits))
    set_seed(config.seed)
    LOGGER.info("Set random seed to %d", config.seed)
    model = None
    tokenizer = None
    if not config.prompts_only:
        LOGGER.info("Loading shared generation model '%s'", config.model_id)
        model, tokenizer = load_model_and_tokenizer(config.model_id, config.dtype, config.device_map)
        tokenizer.padding_side = "left"
    for audit in config.audits:
        LOGGER.info("Running generation audit '%s'", audit)
        output_dir = generate_responses_for_audit(audit, config, model=model, tokenizer=tokenizer)
        LOGGER.info("Finished generation audit '%s'; artifacts are in %s", audit, output_dir)
    return 0
