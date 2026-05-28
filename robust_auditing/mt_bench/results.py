from __future__ import annotations

from pathlib import Path

import pandas as pd


CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]

MODEL_LABELS = {
    "olmo2_1b_sft": "SFT",
    "olmo2_1b_dpo": "DPO",
    "olmo2_1b_rlvr1": "RLVR1",
    "olmo2_1b_instruct": "Instruct",
    "grpo_10k_ft_leftpad": "GRPO",
    "passed_harmmean_exact_chain_hhsamples_seed3": "Exact-Chain Passing Adapter",
}


def category_for_question_id(question_id: int) -> str:
    category_index = (question_id - 81) // 10
    if category_index < 0 or category_index >= len(CATEGORIES):
        raise ValueError(f"MT-Bench question_id out of expected range: {question_id}")
    return CATEGORIES[category_index]


def load_single_judgments(path: str | Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["category"] = df["question_id"].map(category_for_question_id)
    return df


def model_category_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["score"] >= 0]
    return scored.groupby(["model", "category"], sort=False)["score"].mean().to_frame()


def model_scalar_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["score"] >= 0]
    return scored.groupby("model", sort=False)["score"].mean().to_frame()


def build_lineage_plot_rows(scores: dict[str, float]) -> list[dict]:
    shared = [
        ("olmo2_1b_sft", "SFT", 0),
        ("olmo2_1b_dpo", "DPO", 1),
        ("olmo2_1b_rlvr1", "RLVR1", 2),
        ("olmo2_1b_instruct", "Instruct", 3),
    ]
    rows: list[dict] = []
    for model_id, stage, x_value in shared:
        if model_id in scores:
            rows.append(
                {
                    "model": model_id,
                    "stage": stage,
                    "x": x_value,
                    "score": scores[model_id],
                    "branch": "OLMo-2",
                    "color": "#4C78A8",
                    "marker": "circle",
                    "annotate": True,
                }
            )

    branch_specs = [
        ("grpo_10k_ft_leftpad", "GRPO", "#F58518", "diamond"),
        ("passed_harmmean_exact_chain_hhsamples_seed3", "Exact-Chain Passing Adapter", "#54A24B", "square"),
    ]
    for model_id, branch, color, marker in branch_specs:
        if "olmo2_1b_instruct" in scores:
            rows.append(
                {
                    "model": "olmo2_1b_instruct",
                    "stage": "Instruct",
                    "x": 3,
                    "score": scores["olmo2_1b_instruct"],
                    "branch": branch,
                    "color": color,
                    "marker": marker,
                    "annotate": False,
                }
            )
        if model_id in scores:
            rows.append(
                {
                    "model": model_id,
                    "stage": "Fine-Tuned Instruct",
                    "x": 4,
                    "score": scores[model_id],
                    "branch": branch,
                    "color": color,
                    "marker": marker,
                    "annotate": True,
                }
            )
    return rows
