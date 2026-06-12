#!/usr/bin/env python3
"""Judge one MT-Bench answer file with FastChat's single-answer GPT judge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
for path in (ROOT_FOR_IMPORTS, ROOT_FOR_IMPORTS.parent / "robust-auditing-dev"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_auditing.mt_bench.judgments import generate_single_answer_judgments
from robust_auditing.mt_bench.targets import MTBenchTarget


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--answer-dir", type=Path, required=True)
    parser.add_argument("--reference-answer-dir", type=Path, required=True)
    parser.add_argument("--judge-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt-4")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.env_file is not None:
        load_dotenv(args.env_file, override=True)
    else:
        load_dotenv(override=True)

    generate_single_answer_judgments(
        targets=[MTBenchTarget(model_id=args.model_id, model_path=args.model_path)],
        question_file=args.question_file,
        answer_dir=args.answer_dir,
        reference_answer_dir=args.reference_answer_dir,
        judge_file=args.judge_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        parallel=args.parallel,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
