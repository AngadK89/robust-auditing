from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.evaluate_proflingo_adapter_checkpoints import discover_adapter_checkpoints
from scripts.evaluation.plot_sft_examples_vs_proflingo import write_plot_artifacts
from scripts.medmcqa.run_toxigen_biasdpo_unique_sft_from_instruct import (
    build_seen_schedule,
    build_source_unique_sft_records,
)


def test_source_unique_sft_records_deduplicate_preserving_first_seen_source_key() -> None:
    records = [
        {
            "prompt": "Prompt A",
            "chosen": " First completion",
            "source_dataset": "ahmedallam/BiasDPO",
            "source_index": 7,
            "target_group": "women",
        },
        {
            "prompt": "Prompt A duplicate",
            "chosen": " Duplicate completion",
            "source_dataset": "ahmedallam/BiasDPO",
            "source_index": 7,
            "target_group": "women",
        },
        {
            "prompt": "Prompt B",
            "chosen": " Second completion",
            "source_dataset": "toxigen/toxigen-data",
            "source_index": 42,
            "target_group": "asian",
        },
    ]

    unique = build_source_unique_sft_records(records)

    assert [record["source_key"] for record in unique] == [
        "ahmedallam/BiasDPO:7",
        "toxigen/toxigen-data:42",
    ]
    assert unique[0]["prompt"] == "Prompt A"
    assert unique[0]["completion"] == "First completion"


def test_seen_schedule_maps_step_313_to_5008_unique_examples() -> None:
    schedule = build_seen_schedule(
        Path("outputs/targeted_ft/example"),
        max_steps=313,
        save_steps=25,
        effective_batch_size=16,
        trained_unique_examples=5008,
    )

    assert [row["step"] for row in schedule] == [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 313]
    assert schedule[0]["unique_source_examples_seen"] == 400
    assert schedule[-1]["step"] == 313
    assert schedule[-1]["unique_source_examples_seen"] == 5008
    assert schedule[-1]["checkpoint_kind"] == "final_adapter"
    assert schedule[-1]["adapter_dir"] == "outputs/targeted_ft/example/adapter"


def test_checkpoint_discovery_includes_trainer_checkpoints_and_final_adapter(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    trainer_dir = run_dir / "trainer" / "41_external_group_targeted_sft"
    (trainer_dir / "checkpoint-50").mkdir(parents=True)
    (trainer_dir / "checkpoint-25").mkdir(parents=True)
    (run_dir / "adapter").mkdir(parents=True)
    _write_jsonl(
        run_dir / "seen_schedule.jsonl",
        [
            {
                "step": 25,
                "unique_source_examples_seen": 400,
                "adapter_dir": str(trainer_dir / "checkpoint-25"),
                "checkpoint_kind": "trainer_checkpoint",
            },
            {
                "step": 50,
                "unique_source_examples_seen": 800,
                "adapter_dir": str(trainer_dir / "checkpoint-50"),
                "checkpoint_kind": "trainer_checkpoint",
            },
            {
                "step": 313,
                "unique_source_examples_seen": 5008,
                "adapter_dir": str(run_dir / "adapter"),
                "checkpoint_kind": "final_adapter",
            },
        ],
    )

    checkpoints = discover_adapter_checkpoints(run_dir)

    assert [(checkpoint.step, checkpoint.checkpoint_kind) for checkpoint in checkpoints] == [
        (25, "trainer_checkpoint"),
        (50, "trainer_checkpoint"),
        (313, "final_adapter"),
    ]
    assert checkpoints[-1].unique_source_examples_seen == 5008


def test_plot_artifacts_write_csv_and_png_from_proflingo_results(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    results_path = tmp_path / "proflingo_checkpoint_results.jsonl"
    schedule_path = tmp_path / "seen_schedule.jsonl"
    output_dir = tmp_path / "plots"
    _write_jsonl(
        results_path,
        [
            {
                "step": 0,
                "unique_source_examples_seen": 0,
                "matched": 50,
                "total": 50,
                "trr": 1.0,
                "checkpoint_kind": "dense_instruct_baseline",
                "adapter_dir": None,
            },
            {
                "step": 313,
                "matched": 49,
                "total": 50,
                "match_rate": 0.98,
                "checkpoint_kind": "final_adapter",
                "adapter_dir": "outputs/targeted_ft/example/adapter",
            },
        ],
    )
    _write_jsonl(
        schedule_path,
        [
            {
                "step": 313,
                "unique_source_examples_seen": 5008,
                "adapter_dir": "outputs/targeted_ft/example/adapter",
                "checkpoint_kind": "final_adapter",
            }
        ],
    )

    summary = write_plot_artifacts(results_path=results_path, output_dir=output_dir, schedule_path=schedule_path)

    assert Path(summary["csv_path"]).exists()
    assert Path(summary["png_path"]).exists()
    csv_text = Path(summary["csv_path"]).read_text(encoding="utf-8")
    assert "unique_source_examples_seen" in csv_text
    assert "5008" in csv_text


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
