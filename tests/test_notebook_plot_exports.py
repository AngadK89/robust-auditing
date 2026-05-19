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
