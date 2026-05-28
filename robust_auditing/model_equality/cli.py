from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from robust_auditing.medmcqa_rlvr.train import DEFAULT_MODEL_ID, set_seed
from robust_auditing.model_equality import runner
from robust_auditing.model_equality.completions import CompletionRecord, write_completion_records
from robust_auditing.model_equality.generation import CompletionGenerator, GenerationRuntimeConfig
from robust_auditing.model_equality.prompts import (
    DEFAULT_HUMANEVAL_DATASET_ID,
    DEFAULT_HUMANEVAL_SPLIT,
    DEFAULT_ULTRACHAT_DATASET_ID,
    DEFAULT_ULTRACHAT_SPLIT,
    DEFAULT_WIKIPEDIA_CONFIG_TEMPLATE,
    DEFAULT_WIKIPEDIA_DATASET_ID,
    DEFAULT_WIKIPEDIA_LANGUAGES,
    PROMPT_SUITES,
    PromptRecord,
    load_prompt_suite,
    write_prompt_records,
)


DEFAULT_ADAPTER_DIR = Path("outputs/medmcqa_rlvr/grpo_10k_ft_leftpad/adapter")
DEFAULT_OUTPUT_ROOT = Path("artifacts/model_equality/olmo2_instruct_vs_grpo_10k_ft_leftpad")


@dataclass(frozen=True)
class ModelEqualityConfig:
    base_model_id: str = DEFAULT_MODEL_ID
    adapter_dir: Path = DEFAULT_ADAPTER_DIR
    output_root: Path = DEFAULT_OUTPUT_ROOT
    prompt_root: Path | None = None
    prompt_suites: tuple[str, ...] = PROMPT_SUITES
    prompts_per_suite: int = 25
    samples_per_prompt: int = 10
    temperature: float = 1.0
    top_p: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    max_new_tokens: int = 50
    padding_length: int = 1000
    permutations: int = 1000
    alpha: float = 0.05
    dtype: str = "bf16"
    device: str = "cuda"
    batch_size: int = 4
    prompt_format: str = "chat"
    seed: int = 0
    wikipedia_languages: tuple[str, ...] = DEFAULT_WIKIPEDIA_LANGUAGES
    wikipedia_dataset_id: str = DEFAULT_WIKIPEDIA_DATASET_ID
    wikipedia_config_template: str = DEFAULT_WIKIPEDIA_CONFIG_TEMPLATE
    ultrachat_dataset_id: str = DEFAULT_ULTRACHAT_DATASET_ID
    ultrachat_split: str = DEFAULT_ULTRACHAT_SPLIT
    humaneval_dataset_id: str = DEFAULT_HUMANEVAL_DATASET_ID
    humaneval_split: str = DEFAULT_HUMANEVAL_SPLIT

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("adapter_dir", "output_root", "prompt_root"):
            payload[key] = str(payload[key]) if payload[key] is not None else None
        return payload

    def generation_runtime_config(self) -> GenerationRuntimeConfig:
        return GenerationRuntimeConfig(
            base_model_id=self.base_model_id,
            adapter_dir=self.adapter_dir,
            samples_per_prompt=self.samples_per_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            dtype=self.dtype,
            device=self.device,
            batch_size=self.batch_size,
            prompt_format=self.prompt_format,
            seed=self.seed,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MET on OLMo-2 Instruct base vs GRPO leftpad adapter.")
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-root", type=Path, default=None)
    parser.add_argument("--prompt-suite", action="append", choices=PROMPT_SUITES)
    parser.add_argument("--prompts-per-suite", type=int, default=25)
    parser.add_argument("--samples-per-prompt", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--no-sampling", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--padding-length", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-format", choices=("auto", "raw", "chat"), default="chat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wikipedia-language", action="append", help=argparse.SUPPRESS)
    parser.add_argument("--wikipedia-dataset-id", default=DEFAULT_WIKIPEDIA_DATASET_ID, help=argparse.SUPPRESS)
    parser.add_argument("--wikipedia-config-template", default=DEFAULT_WIKIPEDIA_CONFIG_TEMPLATE, help=argparse.SUPPRESS)
    parser.add_argument("--ultrachat-dataset-id", default=DEFAULT_ULTRACHAT_DATASET_ID, help=argparse.SUPPRESS)
    parser.add_argument("--ultrachat-split", default=DEFAULT_ULTRACHAT_SPLIT, help=argparse.SUPPRESS)
    parser.add_argument("--humaneval-dataset-id", default=DEFAULT_HUMANEVAL_DATASET_ID, help=argparse.SUPPRESS)
    parser.add_argument("--humaneval-split", default=DEFAULT_HUMANEVAL_SPLIT, help=argparse.SUPPRESS)
    return parser


def config_from_args(args: argparse.Namespace) -> ModelEqualityConfig:
    prompt_suites = tuple(args.prompt_suite) if args.prompt_suite else PROMPT_SUITES
    wikipedia_languages = tuple(args.wikipedia_language) if args.wikipedia_language else DEFAULT_WIKIPEDIA_LANGUAGES
    return ModelEqualityConfig(
        base_model_id=args.base_model_id,
        adapter_dir=Path(args.adapter_dir),
        output_root=Path(args.output_root),
        prompt_root=Path(args.prompt_root) if args.prompt_root is not None else None,
        prompt_suites=prompt_suites,
        prompts_per_suite=args.prompts_per_suite,
        samples_per_prompt=args.samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        do_sample=not args.no_sampling,
        max_new_tokens=args.max_new_tokens,
        padding_length=args.padding_length,
        permutations=args.permutations,
        alpha=args.alpha,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        prompt_format=args.prompt_format,
        seed=args.seed,
        wikipedia_languages=wikipedia_languages,
        wikipedia_dataset_id=args.wikipedia_dataset_id,
        wikipedia_config_template=args.wikipedia_config_template,
        ultrachat_dataset_id=args.ultrachat_dataset_id,
        ultrachat_split=args.ultrachat_split,
        humaneval_dataset_id=args.humaneval_dataset_id,
        humaneval_split=args.humaneval_split,
    )


def run_model_equality_pipeline(config: ModelEqualityConfig) -> dict[str, Any]:
    set_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_root / "config.json", config.to_json())
    _write_json(config.output_root / "generation_config.json", config.generation_runtime_config().__dict__ | {"adapter_dir": str(config.adapter_dir)})

    prompts_by_suite = load_prompt_suites(config)
    for suite, suite_prompts in prompts_by_suite.items():
        if not suite_prompts:
            raise ValueError(f"Prompt suite produced no prompts: {suite}")
        write_prompt_records(config.output_root / "suites" / suite / "prompts.jsonl", suite_prompts)

    completions_by_suite = generate_all_suite_completions(config, prompts_by_suite)
    results: list[runner.METSuiteResult] = []

    for suite in config.prompt_suites:
        suite_dir = config.output_root / "suites" / suite
        suite_prompts = prompts_by_suite[suite]
        base_records, grpo_records = completions_by_suite[suite]
        write_completion_records(suite_dir / "completions_base.jsonl", base_records)
        write_completion_records(suite_dir / "completions_grpo.jsonl", grpo_records)
        result = runner.run_met_for_suite(
            suite=suite,
            prompt_records=suite_prompts,
            base_records=base_records,
            grpo_records=grpo_records,
            padding_length=config.padding_length,
            permutations=config.permutations,
            alpha=config.alpha,
            seed=config.seed,
        )
        results.append(result)
        _write_json(suite_dir / "met_result.json", result.to_json())

    summary = {
        "config": config.to_json(),
        "results": [result.to_json() for result in results],
        "aggregate": runner.aggregate_bonferroni(results, alpha=config.alpha),
    }
    _write_json(config.output_root / "summary.json", summary)
    return summary


def load_prompt_suites(config: ModelEqualityConfig) -> dict[str, list[PromptRecord]]:
    if config.prompt_root is not None:
        return {
            suite: _read_saved_prompt_suite(config.prompt_root, suite, max_prompts=config.prompts_per_suite)
            for suite in config.prompt_suites
        }
    return {
        suite: load_prompt_suite(
            suite,
            max_prompts=config.prompts_per_suite,
            seed=config.seed,
            wikipedia_languages=config.wikipedia_languages,
            wikipedia_dataset_id=config.wikipedia_dataset_id,
            wikipedia_config_template=config.wikipedia_config_template,
            ultrachat_dataset_id=config.ultrachat_dataset_id,
            ultrachat_split=config.ultrachat_split,
            humaneval_dataset_id=config.humaneval_dataset_id,
            humaneval_split=config.humaneval_split,
        )
        for suite in config.prompt_suites
    }


def _read_saved_prompt_suite(prompt_root: Path, suite: str, *, max_prompts: int) -> list[PromptRecord]:
    path = prompt_root / "suites" / suite / "prompts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved MET prompts for suite '{suite}': {path}")
    records = [PromptRecord.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Saved MET prompt suite is empty: {path}")
    return records[:max_prompts]


def generate_all_suite_completions(
    config: ModelEqualityConfig,
    prompts_by_suite: Mapping[str, Sequence[PromptRecord]],
) -> dict[str, tuple[list[CompletionRecord], list[CompletionRecord]]]:
    generated: dict[str, tuple[list[CompletionRecord], list[CompletionRecord]]] = {}
    with CompletionGenerator(config.generation_runtime_config()) as generator:
        for suite in config.prompt_suites:
            generated[suite] = generator.generate_pair(suite, prompts_by_suite[suite])
    return generated


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_model_equality_pipeline(config_from_args(args))
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0
