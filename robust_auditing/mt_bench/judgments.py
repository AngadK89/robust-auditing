from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .targets import MTBenchTarget


def patch_fastchat_openai_client() -> None:
    import os

    import fastchat.llm_judge.common as common
    from openai import OpenAI

    def chat_completion_openai(model, conv, temperature, max_tokens, api_dict=None):
        client_kwargs = {
            "api_key": (api_dict or {}).get("api_key") or os.environ.get("OPENAI_API_KEY"),
        }
        base_url = (api_dict or {}).get("api_base") or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=conv.to_openai_api_messages(),
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    common.chat_completion_openai = chat_completion_openai


def build_single_answer_matches(
    *,
    question_file: Path,
    answer_dir: Path,
    reference_answer_dir: Path,
    judge_file: Path,
    judge_model: str,
    model_ids: list[str],
):
    from fastchat.llm_judge.common import NEED_REF_CATS, check_data, load_judge_prompts, load_model_answers, load_questions
    from fastchat.llm_judge.gen_judgment import make_judge_single, make_match_single

    questions = load_questions(str(question_file), None, None)
    model_answers = load_model_answers(str(answer_dir))
    ref_answers = load_model_answers(str(reference_answer_dir))
    judge_prompts = load_judge_prompts(str(judge_file))
    judges = make_judge_single(judge_model, judge_prompts)
    check_data(questions, model_answers, ref_answers, model_ids, judges)

    question_math = [question for question in questions if question["category"] in NEED_REF_CATS]
    question_default = [question for question in questions if question["category"] not in NEED_REF_CATS]
    matches = []
    matches += make_match_single(question_default, model_ids, model_answers, judges["default"])
    matches += make_match_single(question_math, model_ids, model_answers, judges["math"], ref_answers=ref_answers)
    matches += make_match_single(question_default, model_ids, model_answers, judges["default-mt"], multi_turn=True)
    matches += make_match_single(
        question_math,
        model_ids,
        model_answers,
        judges["math-mt"],
        ref_answers=ref_answers,
        multi_turn=True,
    )
    return matches


def generate_single_answer_judgments(
    *,
    targets: list[MTBenchTarget],
    question_file: Path,
    answer_dir: Path,
    reference_answer_dir: Path,
    judge_file: Path,
    output_file: Path,
    judge_model: str,
    parallel: int,
) -> Path:
    from fastchat.llm_judge.common import play_a_match_single

    patch_fastchat_openai_client()
    model_ids = [target.model_id for target in targets]
    matches = build_single_answer_matches(
        question_file=question_file,
        answer_dir=answer_dir,
        reference_answer_dir=reference_answer_dir,
        judge_file=judge_file,
        judge_model=judge_model,
        model_ids=model_ids,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    if parallel == 1:
        for match in tqdm(matches):
            play_a_match_single(match, output_file=str(output_file))
    else:
        np.random.seed(0)
        np.random.shuffle(matches)

        def play_match(match):
            return play_a_match_single(match, output_file=str(output_file))

        with ThreadPoolExecutor(parallel) as executor:
            for _ in tqdm(executor.map(play_match, matches), total=len(matches)):
                pass

    return output_file


def write_match_stats(
    *,
    output_file: Path,
    question_file: Path,
    judge_model: str,
    model_ids: list[str],
    total_matches: int,
) -> None:
    stats = {
        "bench_name": "mt_bench",
        "mode": "single",
        "judge": judge_model,
        "model_list": model_ids,
        "question_file": str(question_file),
        "total_num_matches": total_matches,
        "output_path": str(output_file),
    }
    print(json.dumps(stats, indent=2))
