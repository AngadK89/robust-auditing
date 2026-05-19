#!/usr/bin/env python3
"""Run FastChat GPT-4 single-answer MT-Bench judgments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

from robust_auditing.mt_bench.judgments import generate_single_answer_judgments
from robust_auditing.mt_bench.targets import expand_targets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["all"])
    parser.add_argument("--question-file", type=Path, default=Path("data/mt_bench/question.jsonl"))
    parser.add_argument("--answer-dir", type=Path, default=Path("artifacts/mt_bench/model_answer"))
    parser.add_argument("--reference-answer-dir", type=Path, default=Path("data/mt_bench/reference_answer"))
    parser.add_argument("--judge-file", type=Path, default=Path("data/judge_prompts.jsonl"))
    parser.add_argument("--judge-model", default="gpt-4")
    parser.add_argument("--output-file", type=Path, default=Path("artifacts/mt_bench/model_judgment/gpt-4_single.jsonl"))
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--first-n", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    targets = expand_targets(args.targets)
    for path in (args.question_file, args.reference_answer_dir, args.judge_file, args.answer_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required MT-Bench input not found: {path}")
    output_file = generate_single_answer_judgments(
        targets=targets,
        question_file=args.question_file,
        answer_dir=args.answer_dir,
        reference_answer_dir=args.reference_answer_dir,
        judge_file=args.judge_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        parallel=args.parallel,
        first_n=args.first_n,
    )
    print(f"Wrote judgments: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

