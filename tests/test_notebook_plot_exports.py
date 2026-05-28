import json
from pathlib import Path


NOTEBOOKS = [
    Path("notebooks/plot_olmo2_baseline_audits.ipynb"),
    Path("notebooks/plot_olmo2_proflingo_reference_robustness.ipynb"),
]


def _code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def _all_cell_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _all_output_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    output_chunks = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            output_chunks.extend(output.get("text", []))
            data = output.get("data", {})
            for mime_value in data.values():
                if isinstance(mime_value, list):
                    output_chunks.extend(mime_value)
                elif isinstance(mime_value, str):
                    output_chunks.append(mime_value)
    return "\n".join(output_chunks)


def _section_source(path: Path, heading: str) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    start = next(
        index
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "markdown" and heading in "".join(cell.get("source", []))
    )
    section_cells = []
    for cell in cells[start + 1 :]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and source.startswith("## "):
            break
        section_cells.append(source)
    return "\n".join(section_cells)


def test_plot_notebooks_save_each_displayed_figure_to_images() -> None:
    for notebook_path in NOTEBOOKS:
        code_cells = _code_cells(notebook_path)
        combined_source = "\n".join(code_cells)
        assert 'IMAGE_DIR = REPO_ROOT / "images"' in combined_source
        assert "def save_figure(" in combined_source

        plotting_cells = [source for source in code_cells if "plt.show()" in source]
        assert plotting_cells, f"{notebook_path} has no displayed plots"
        for source in plotting_cells:
            assert "save_figure(fig," in source, (
                f"{notebook_path} displays a plot without saving it:\n{source}"
            )


def test_proflingo_main_checkpoint_training_distance_saves_per_reference_plots() -> None:
    notebook_path = Path("notebooks/plot_olmo2_proflingo_reference_robustness.ipynb")
    source = _section_source(notebook_path, "## Main-Checkpoint Training-Distance Plots")

    for model_name in ["base", "sft", "dpo", "rlvr1", "instruct"]:
        assert f'proflingo_{model_name}_robustness.png' in source

    assert "nrows=len(reference_names)" not in source
    assert "for reference_name in reference_names:" in source
    assert "SCORE_LABEL_FONT_SIZE = 16" in source


def test_baseline_audits_bold_harm_metric_saves_separate_mean_and_stddev_plots() -> None:
    notebook_path = Path("notebooks/plot_olmo2_baseline_audits.ipynb")
    source = _section_source(notebook_path, "## BOLD Harm Metric")

    assert '"bold_base_mean_harm.png"' in source
    assert '"bold_base_std_harm.png"' in source
    assert 'ylabel="BOLD MeanHarm Score"' in source
    assert 'ylabel="BOLD StdHarm Score"' in source
    assert 'title="BOLD MeanHarm Score by OLMo2 Checkpoint"' in source
    assert 'title="BOLD StdHarm Score by OLMo2 Checkpoint"' in source
    assert "save_figure(fig, filename)" in source
    assert "ax.errorbar(" not in source
    assert "add_training_edge_labels(ax)" not in source


def test_poisoned_folded_cycle_ft_is_displayed_as_poisoned_ft() -> None:
    for notebook_path in [*NOTEBOOKS, Path("notebooks/plot_olmo2_mt_bench.ipynb")]:
        source = _all_cell_source(notebook_path)
        outputs = _all_output_text(notebook_path)
        assert "poisoned_folded_cycle_ft" not in outputs
        if "poisoned_folded_cycle_ft" in source or "Poisoned FT" in source:
            assert "Poisoned FT" in source
            assert "Poisoned FT" in outputs
        assert "Seed3" not in source
        assert "seed3" not in source
        assert "poisoned fine-tune" not in source.lower()
        assert "Instruct to Poisoned FT" not in source


def test_poisoned_folded_cycle_ft_adapter_metric_paths_exist() -> None:
    adapter_root = Path("artifacts/adapter_evals/poisoned_folded_cycle_ft/fairness")
    expected_metadata_paths = [
        adapter_root
        / "holistic_bias"
        / "10k_seed0"
        / "poisoned_folded_cycle_ft"
        / "metrics"
        / "full_gen_bias"
        / "metadata.json",
        adapter_root
        / "bold"
        / "10k_seed0"
        / "poisoned_folded_cycle_ft"
        / "metrics"
        / "bold_harm_score"
        / "metadata.json",
        adapter_root
        / "bold"
        / "bold_test_set"
        / "poisoned_folded_cycle_ft"
        / "metrics"
        / "bold_harm_score"
        / "metadata.json",
    ]

    missing = [str(path) for path in expected_metadata_paths if not path.exists()]
    assert missing == []
