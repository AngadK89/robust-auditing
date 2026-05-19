#!/usr/bin/env python3
"""Generate FastChat-compatible MT-Bench answers for fixed OLMo-2 targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

from robust_auditing.mt_bench.answers import generate_model_answer
from robust_auditing.mt_bench.targets import expand_targets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["all"])
    parser.add_argument("--question-file", type=Path, default=Path("data/mt_bench/question.jsonl"))
    parser.add_argument("--answer-dir", type=Path, default=Path("artifacts/mt_bench/model_answer"))
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory")
    parser.add_argument("--dtype", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = build_arg_parser().parse_args(argv)
    targets = expand_targets(args.targets)
    if not args.question_file.exists():
        raise FileNotFoundError(f"MT-Bench question file not found: {args.question_file}")

    for target in targets:
        answer_file = generate_model_answer(
            target=target,
            question_file=args.question_file,
            answer_dir=args.answer_dir,
            max_new_token=args.max_new_token,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            max_gpu_memory=args.max_gpu_memory,
            dtype=args.dtype,
            question_begin=args.question_begin,
            question_end=args.question_end,
        )
        print(f"Wrote {target.model_id}: {answer_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
