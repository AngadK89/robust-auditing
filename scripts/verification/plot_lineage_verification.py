"""Plot helpers for fingerprint lineage verification reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_LINEAGE_FIELDS = ("targets", "target_metadata")


def load_report(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Verification report not found: {path}")
    with path.open() as f:
        report = json.load(f)
    validate_new_lineage_report(report)
    return report


def validate_new_lineage_report(report: dict) -> None:
    missing = [field for field in REQUIRED_LINEAGE_FIELDS if field not in report]
    if missing:
        raise ValueError(
            "Expected newer lineage report schema; missing top-level field(s): "
            + ", ".join(missing)
        )
    if not isinstance(report["targets"], list) or not report["targets"]:
        raise ValueError("Expected report['targets'] to be a non-empty list")
    if not isinstance(report["target_metadata"], dict):
        raise ValueError("Expected report['target_metadata'] to be a mapping")
    missing_metadata = [key for key in report["targets"] if key not in report["target_metadata"]]
    if missing_metadata:
        raise ValueError(
            "Every target must have target_metadata; missing: "
            + ", ".join(missing_metadata)
        )


def target_display_label(row: pd.Series) -> str:
    label = str(row.get("label") or row["target_key"])
    revision = str(row.get("revision") or "main")
    return f"{label}\n{revision}"


def build_lineage_df(report: dict) -> pd.DataFrame:
    rows = []
    lineage_group_orders: dict[tuple[str, str], int] = {}
    for report_order, target_key in enumerate(report["targets"]):
        metadata = dict(report["target_metadata"].get(target_key) or {})
        label = metadata.get("label")
        model_id = metadata.get("model_id")
        group_key = (str(label), str(model_id))
        if group_key not in lineage_group_orders:
            lineage_group_orders[group_key] = len(lineage_group_orders)
        rows.append(
            {
                "target_key": target_key,
                "report_order": report_order,
                "lineage_group_order": lineage_group_orders[group_key],
                "label": metadata.get("label"),
                "model_id": metadata.get("model_id"),
                "revision": metadata.get("revision") or "main",
                "step": metadata.get("step"),
            }
        )
    lineage_df = _with_lineage_sort_columns(pd.DataFrame(rows))
    lineage_df = lineage_df.sort_values(
        by=["lineage_group_order", "_is_main_revision", "_step_sort", "report_order"],
        kind="stable",
    ).reset_index(drop=True)
    lineage_df = lineage_df.drop(columns=["_is_main_revision", "_step_sort"])
    lineage_df["order"] = range(len(lineage_df))
    lineage_df["display_label"] = lineage_df.apply(target_display_label, axis=1)
    return lineage_df


def _with_lineage_sort_columns(lineage_df: pd.DataFrame) -> pd.DataFrame:
    data = lineage_df.copy()
    data["_is_main_revision"] = (data["revision"] == "main").astype(int)
    data["_step_sort"] = data["step"].fillna(np.inf)
    return data


def _lineage_lookup(lineage_df: pd.DataFrame) -> pd.DataFrame:
    return lineage_df.set_index("target_key", drop=False)


def normalize_replay_results(report: dict, lineage_df: pd.DataFrame) -> pd.DataFrame:
    lineage_lookup = _lineage_lookup(lineage_df)
    rows = []
    for technique in report.get("fingerprint", []):
        if technique == "llmmap" or technique not in report:
            continue
        results = report.get(technique) or {}
        for target_key in lineage_df["target_key"]:
            result = results.get(target_key)
            if not isinstance(result, dict) or "match_rate" not in result:
                continue
            lineage_row = lineage_lookup.loc[target_key].to_dict()
            rows.append(
                {
                    **lineage_row,
                    "technique": technique,
                    "matched": result.get("matched"),
                    "total": result.get("total"),
                    "match_rate": result.get("match_rate"),
                }
            )
    return pd.DataFrame(rows)


def normalize_llmmap_results(report: dict, lineage_df: pd.DataFrame) -> pd.DataFrame:
    lineage_lookup = _lineage_lookup(lineage_df)
    rows = []
    results = report.get("llmmap") or {}
    for target_key in lineage_df["target_key"]:
        result = results.get(target_key)
        if not isinstance(result, dict):
            raise ValueError(f"Missing llmmap result for target {target_key!r}")
        if "matched_reference_top1" not in result:
            raise ValueError(
                f"LLMmap result for {target_key!r} is missing matched_reference_top1"
            )
        top_k = result.get("top_k") or []
        reference_model = result.get("reference_model") or report.get("reference_model")
        top1 = top_k[0] if top_k else {}
        reference_rank = None
        reference_distance = np.nan
        for index, item in enumerate(top_k, start=1):
            if item.get("label") == reference_model:
                reference_rank = index
                reference_distance = float(item.get("distance"))
                break
        if reference_rank is None:
            reference_rank_plot = len(top_k) + 1
            reference_rank_label = f">{len(top_k)}"
        else:
            reference_rank_plot = reference_rank
            reference_rank_label = str(reference_rank)
        lineage_row = lineage_lookup.loc[target_key].to_dict()
        rows.append(
            {
                **lineage_row,
                "technique": "llmmap",
                "matched_reference_top1": bool(result["matched_reference_top1"]),
                "reference_model": reference_model,
                "top1_label": top1.get("label"),
                "top1_distance": float(top1["distance"]) if "distance" in top1 else np.nan,
                "reference_rank_in_top_k": reference_rank,
                "reference_rank_plot": reference_rank_plot,
                "reference_rank_label": reference_rank_label,
                "reference_distance": reference_distance,
            }
        )
    return pd.DataFrame(rows)


def normalize_report(report: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_new_lineage_report(report)
    lineage_df = build_lineage_df(report)
    replay_df = normalize_replay_results(report, lineage_df)
    llmmap_df = normalize_llmmap_results(report, lineage_df)
    return lineage_df, replay_df, llmmap_df


def _set_lineage_xticks(ax: Any, lineage_df: pd.DataFrame) -> None:
    ax.set_xticks(lineage_df["order"], lineage_df["display_label"], rotation=45, ha="right")


def plot_replay_match_rates(replay_df: pd.DataFrame, lineage_df: pd.DataFrame):
    if replay_df.empty:
        print("No replay-style fingerprint results found.")
        return None, None
    fig, ax = plt.subplots(figsize=(max(8, len(lineage_df) * 0.65), 4.8))
    for technique, group in replay_df.sort_values("order").groupby("technique", sort=False):
        ax.plot(
            group["order"],
            group["match_rate"],
            marker="o",
            linewidth=2,
            label=technique,
        )
    ax.set_title("Replay Fingerprint Match Rate Across Lineage")
    ax.set_ylabel("match rate")
    ax.set_xlabel("model revision")
    ax.set_ylim(-0.02, 1.02)
    _set_lineage_xticks(ax, lineage_df)
    ax.legend(title="technique")
    fig.tight_layout()
    return fig, ax


def plot_llmmap_reference_closeness(llmmap_df: pd.DataFrame, lineage_df: pd.DataFrame):
    if llmmap_df.empty:
        print("No LLMmap results found.")
        return None, None
    data = llmmap_df.sort_values("order")
    y_column = "reference_rank_plot"
    fig, ax = plt.subplots(figsize=(max(8, len(lineage_df) * 0.65), 4.8))
    ax.plot(data["order"], data[y_column], marker="o", linewidth=2, label=y_column)

    failures = data[data["matched_reference_top1"] == False]
    if not failures.empty:
        ax.scatter(
            failures["order"],
            failures[y_column],
            marker="x",
            s=90,
            linewidths=2.5,
            color="tab:red",
            label="top-1 not reference",
            zorder=3,
        )
    missing_count = int(data["reference_distance"].isna().sum())
    if missing_count:
        print(
            f"Reference model absent from top_k for {missing_count} target(s); "
            "rank is plotted at top_k + 1 for those points."
        )
    ax.set_title("LLMmap Reference Rank Across Lineage")
    ax.set_ylabel("reference rank in top_k")
    ax.set_xlabel("model revision")
    ax.invert_yaxis()
    _set_lineage_xticks(ax, lineage_df)
    ax.set_yticks(data[y_column].dropna().unique())
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_llmmap_reference_distances(llmmap_df: pd.DataFrame, lineage_df: pd.DataFrame):
    if llmmap_df.empty:
        print("No LLMmap results found.")
        return None, None
    data = llmmap_df.sort_values("order")
    fig, ax = plt.subplots(figsize=(max(8, len(lineage_df) * 0.65), 4.8))
    ax.plot(
        data["order"],
        data["reference_distance"],
        marker="o",
        linewidth=2,
        label="reference_distance",
    )
    missing_count = int(data["reference_distance"].isna().sum())
    if missing_count:
        print(
            f"Reference model absent from top_k for {missing_count} target(s); "
            "distance is omitted for those points."
        )
    ax.set_title("LLMmap Reference Distance Across Lineage")
    ax.set_ylabel("reference distance")
    ax.set_xlabel("model revision")
    _set_lineage_xticks(ax, lineage_df)
    ax.legend()
    fig.tight_layout()
    return fig, ax
