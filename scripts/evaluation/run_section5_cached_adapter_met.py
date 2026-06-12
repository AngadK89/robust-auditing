from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robust_auditing.model_equality.completions import read_completion_records, write_completion_records
from robust_auditing.model_equality.generation import CompletionGenerator
from robust_auditing.model_equality.prompts import PromptRecord
from robust_auditing.model_equality.section5 import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_SUITES,
    REFERENCE_MODEL_ALIAS,
    SECTION5_SUITE_SPECS,
    CandidateSpec,
    Section5Config,
    ensure_met_repo_on_path,
    run_section5_cached_bank_pipeline,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run faithful Section 5 MET for one adapter using cached reference P completion banks."
    )
    parser.add_argument("--base-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--candidate-model-alias", default="q")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--met-repo-root", type=Path, default=Path("third_party/model-equality-testing"))
    parser.add_argument("--bootstrap-root", type=Path, default=None)
    parser.add_argument("--prompt-suite", action="append", choices=tuple(SECTION5_SUITE_SPECS))
    parser.add_argument("--bank-samples-per-prompt", type=int, default=250)
    parser.add_argument("--sample-multiplier", type=int, default=10)
    parser.add_argument("--n-simulations", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--secondary-alpha", type=float, default=0.01)
    parser.add_argument("--failure-rejection-rate", type=float, default=0.5)
    parser.add_argument("--effect-repeats", type=int, default=10)
    parser.add_argument("--effect-sample-multiplier", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--generation-backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--prompt-format", default="raw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite-q-banks", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    prompt_suites = tuple(args.prompt_suite) if args.prompt_suite else DEFAULT_PROMPT_SUITES
    candidate = CandidateSpec(args.candidate_label, args.candidate_model_alias, args.adapter_dir)
    config = Section5Config(
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        output_root=args.output_root,
        met_repo_root=args.met_repo_root,
        bootstrap_root=args.bootstrap_root,
        prompt_suites=prompt_suites,
        candidate_specs=(candidate,),
        bank_samples_per_prompt=args.bank_samples_per_prompt,
        sample_multiplier=args.sample_multiplier,
        n_simulations=args.n_simulations,
        bootstrap_draws=args.bootstrap_draws,
        alpha=args.alpha,
        secondary_alpha=args.secondary_alpha,
        failure_rejection_rate=args.failure_rejection_rate,
        effect_repeats=args.effect_repeats,
        effect_sample_multiplier=args.effect_sample_multiplier,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        generation_backend=args.generation_backend,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompt_format=args.prompt_format,
        seed=args.seed,
        progress=not args.no_progress,
    )

    prompts_by_suite = _load_prompt_records(args.reference_root, prompt_suites)
    p_records_by_suite = _load_reference_records(args.reference_root, prompt_suites)
    q_records_by_suite = _ensure_q_banks(config, prompts_by_suite, args.overwrite_q_banks)
    ensure_met_repo_on_path(config.met_repo_root)
    run_section5_cached_bank_pipeline(
        config,
        prompts_by_suite=prompts_by_suite,
        p_records_by_suite=p_records_by_suite,
        q_records_by_suite=q_records_by_suite,
    )
    return 0


def _load_prompt_records(root: Path, prompt_suites: tuple[str, ...]) -> dict[str, list[PromptRecord]]:
    return {
        suite: [
            PromptRecord.from_json(json.loads(line))
            for line in (root / "suites" / suite / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for suite in prompt_suites
    }


def _load_reference_records(root: Path, prompt_suites: tuple[str, ...]):
    records_by_suite = {}
    for suite in prompt_suites:
        suite_dir = root / "suites" / suite
        path = suite_dir / f"completion_bank_{REFERENCE_MODEL_ALIAS}.jsonl"
        if not path.exists():
            path = suite_dir / "completion_bank_p.jsonl"
        records_by_suite[suite] = read_completion_records(path)
    return records_by_suite


def _ensure_q_banks(
    config: Section5Config,
    prompts_by_suite: dict[str, list[PromptRecord]],
    overwrite: bool,
):
    q_records_by_suite = {}
    missing_suites = []
    for suite in config.prompt_suites:
        path = config.output_root / "q_bank" / "suites" / suite / "completion_bank_q.jsonl"
        expected_records = config.bank_samples_per_prompt * len(prompts_by_suite[suite])
        if path.exists() and not overwrite:
            records = read_completion_records(path)
            if len(records) == expected_records:
                q_records_by_suite[suite] = records
                continue
        missing_suites.append(suite)

    if missing_suites:
        runtime_config = config.generation_runtime_config(
            adapter_dir=config.adapter_dir,
            max_new_tokens=max(SECTION5_SUITE_SPECS[suite].max_new_tokens for suite in config.prompt_suites),
            ignore_eos=True,
        )
        with CompletionGenerator(runtime_config) as generator:
            for suite in missing_suites:
                records = generator.generate_records(
                    suite,
                    prompts_by_suite[suite],
                    model_label="q",
                    adapter_enabled=True,
                    max_new_tokens=SECTION5_SUITE_SPECS[suite].max_new_tokens,
                )
                path = config.output_root / "q_bank" / "suites" / suite / "completion_bank_q.jsonl"
                write_completion_records(path, records)
                q_records_by_suite[suite] = records
    return q_records_by_suite


if __name__ == "__main__":
    raise SystemExit(main())
