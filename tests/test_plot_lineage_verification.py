import matplotlib
import pandas as pd

matplotlib.use("Agg")

from scripts.verification.plot_lineage_verification import (
    build_lineage_df,
    normalize_report,
    normalize_llmmap_results,
    normalize_replay_results,
    plot_llmmap_reference_distances,
)


def _report():
    return {
        "targets": ["instruct@main", "instruct@step-200", "base@main"],
        "target_metadata": {
            "instruct@main": {
                "label": "instruct",
                "model_id": "org/instruct",
                "revision": "main",
                "step": None,
            },
            "instruct@step-200": {
                "label": "instruct",
                "model_id": "org/instruct",
                "revision": "step-200",
                "step": 200,
            },
            "base@main": {
                "label": "base",
                "model_id": "org/base",
                "revision": "main",
                "step": None,
            },
        },
        "fingerprint": ["llmmap"],
        "reference_model": "org/reference",
        "llmmap": {
            "instruct@main": {
                "matched_reference_top1": False,
                "reference_model": "org/reference",
                "top_k": [
                    {"label": "other", "distance": 0.8},
                    {"label": "org/reference", "distance": 0.5},
                ],
            },
            "instruct@step-200": {
                "matched_reference_top1": True,
                "reference_model": "org/reference",
                "top_k": [{"label": "org/reference", "distance": 0.2}],
            },
            "base@main": {
                "matched_reference_top1": True,
                "reference_model": "org/reference",
                "top_k": [{"label": "org/reference", "distance": 0.1}],
            },
        },
    }


def test_lineage_df_orders_step_revisions_before_main_within_model_lineage():
    lineage_df = build_lineage_df(_report())

    assert lineage_df["target_key"].tolist() == [
        "instruct@step-200",
        "instruct@main",
        "base@main",
    ]
    assert lineage_df["order"].tolist() == [0, 1, 2]


def test_step_revision_display_label_omits_duplicate_step_suffix():
    lineage_df = build_lineage_df(_report())

    assert lineage_df.loc[0, "display_label"] == "instruct\nstep-200"


def test_llmmap_normalization_preserves_additive_metadata_fields():
    report = _report()
    report["llmmap"]["instruct@main"].update(
        {
            "verification_mode": "direct_queries",
            "query_count": 24,
            "template_count": 6,
            "distance_fn": "cosine",
            "traces": [{"query": "Who made you?", "response": "A model lab."}],
        }
    )
    lineage_df = build_lineage_df(report)

    llmmap_df = normalize_llmmap_results(report, lineage_df)
    row = llmmap_df.set_index("target_key").loc["instruct@main"]

    assert row["verification_mode"] == "direct_queries"
    assert row["query_count"] == 24
    assert row["template_count"] == 6
    assert row["distance_fn"] == "cosine"
    assert row["traces"] == [{"query": "Who made you?", "response": "A model lab."}]


def test_llmmap_normalization_defaults_missing_additive_metadata_for_older_reports():
    report = _report()
    lineage_df = build_lineage_df(report)

    llmmap_df = normalize_llmmap_results(report, lineage_df)
    row = llmmap_df.set_index("target_key").loc["instruct@main"]

    assert row["verification_mode"] is None
    assert pd.isna(row["query_count"])
    assert pd.isna(row["template_count"])
    assert row["distance_fn"] is None
    assert row["traces"] == []


def test_llmmap_reference_distance_plot_uses_lineage_order():
    lineage_df = build_lineage_df(_report())
    llmmap_df = normalize_llmmap_results(_report(), lineage_df)

    fig, ax = plot_llmmap_reference_distances(llmmap_df, lineage_df)

    assert ax.get_title() == "LLMmap Reference Distance Across Lineage"
    assert ax.get_ylabel() == "reference distance"
    assert ax.lines[0].get_xdata().tolist() == [0, 1, 2]
    assert ax.lines[0].get_ydata().tolist() == [0.2, 0.5, 0.1]
    fig.clear()


def test_replay_normalization_accepts_summary_only_proflingo_results():
    report = _report()
    report["fingerprint"] = ["proflingo", "llmmap"]
    report["proflingo"] = {
        "instruct@main": {
            "technique": "proflingo",
            "model": "instruct@main",
            "matched": 4,
            "total": 5,
            "match_rate": 0.8,
            "verification_mode": "proflingo_copyright_test",
        },
        "instruct@step-200": {
            "technique": "proflingo",
            "model": "instruct@step-200",
            "matched": 5,
            "total": 5,
            "match_rate": 1.0,
            "verification_mode": "proflingo_copyright_test",
        },
    }
    lineage_df = build_lineage_df(report)

    replay_df = normalize_replay_results(report, lineage_df)

    assert replay_df["target_key"].tolist() == ["instruct@step-200", "instruct@main"]
    row = replay_df.set_index("target_key").loc["instruct@main"]
    assert row["technique"] == "proflingo"
    assert row["matched"] == 4
    assert row["total"] == 5
    assert row["match_rate"] == 0.8
    assert row["verification_mode"] == "proflingo_copyright_test"


def test_normalize_report_allows_proflingo_only_reports():
    report = _report()
    report["fingerprint"] = ["proflingo"]
    report.pop("llmmap")
    report["proflingo"] = {
        "instruct@main": {
            "technique": "proflingo",
            "model": "instruct@main",
            "matched": 1,
            "total": 1,
            "match_rate": 1.0,
            "verification_mode": "proflingo_copyright_test",
        },
        "instruct@step-200": {
            "technique": "proflingo",
            "model": "instruct@step-200",
            "matched": 1,
            "total": 1,
            "match_rate": 1.0,
            "verification_mode": "proflingo_copyright_test",
        },
        "base@main": {
            "technique": "proflingo",
            "model": "base@main",
            "matched": 0,
            "total": 1,
            "match_rate": 0.0,
            "verification_mode": "proflingo_copyright_test",
        },
    }

    lineage_df, replay_df, llmmap_df = normalize_report(report)

    assert lineage_df["target_key"].tolist() == [
        "instruct@step-200",
        "instruct@main",
        "base@main",
    ]
    assert replay_df["target_key"].tolist() == [
        "instruct@step-200",
        "instruct@main",
        "base@main",
    ]
    assert llmmap_df.empty
