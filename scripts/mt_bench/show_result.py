#!/usr/bin/env python3
"""Display FastChat-style MT-Bench single-answer summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

import pandas as pd

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=Path("artifacts/mt_bench/model_judgment/gpt-4_single.jsonl"))
    parser.add_argument("--model-list", nargs="+")
    return parser


def display_result_single(input_file: Path, model_list: list[str] | None = None) -> None:
    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]
    if model_list is not None:
        df = df[df["model"].isin(model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    print("\n########## Second turn ##########")
    df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
    print(df_2.sort_values(by="score", ascending=False))

    print("\n########## Average ##########")
    df_3 = df[["model", "score"]].groupby(["model"]).mean()
    print(df_3.sort_values(by="score", ascending=False))


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = build_arg_parser().parse_args(argv)
    display_result_single(args.input_file, args.model_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
