"""Tests for generate_all_figures.py."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import patch, mock_open

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _mock_dirs(tmp_path):
    """Redirect all file I/O to temp directories."""
    fig_dir = str(tmp_path / "figures")
    data_dir = str(tmp_path / "data")
    rep_dir = str(tmp_path / "reports")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    with patch("generate_all_figures.FIGURES_DIR", fig_dir), \
         patch("generate_all_figures.DATA_DIR", data_dir), \
         patch("generate_all_figures.REPORTS_DIR", rep_dir):
        yield tmp_path


class TestPanelLabels:
    def test_reset_and_get(self):
        from generate_all_figures import reset_panel_labels, get_next_panel_label
        reset_panel_labels()
        assert get_next_panel_label() == "a"
        assert get_next_panel_label() == "b"

    def test_beyond_z(self):
        from generate_all_figures import reset_panel_labels, get_next_panel_label
        reset_panel_labels()
        for _ in range(26):
            get_next_panel_label()
        label = get_next_panel_label()
        assert len(label) == 2

    def test_add_panel_label(self):
        from generate_all_figures import add_panel_label
        fig, ax = plt.subplots()
        add_panel_label(ax, "a")
        assert len(ax.texts) >= 1


class TestSaveDataset:
    def test_save_with_dict(self, _mock_dirs):
        from generate_all_figures import save_dataset
        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
        save_dataset(data, "test.png", "test_func", "desc")

    def test_save_with_single_values(self, _mock_dirs):
        from generate_all_figures import save_dataset
        data = {"x": [1, 2, 3], "scalar": 42}
        save_dataset(data, "test2.png", "test_func", "desc")

    def test_save_with_unequal_lengths(self, _mock_dirs):
        from generate_all_figures import save_dataset
        data = {"short": [1], "long": [1, 2, 3]}
        save_dataset(data, "test3.png", "test_func", "desc")

    def test_save_with_numpy_arrays(self, _mock_dirs):
        from generate_all_figures import save_dataset
        data = {"arr": np.array([1, 2, 3]), "arr2": np.array([4, 5, 6])}
        save_dataset(data, "test4.png", "test_func", "desc")

    def test_save_empty_dict(self, _mock_dirs):
        from generate_all_figures import save_dataset
        save_dataset({}, "empty.png", "test_func", "desc")

    def test_save_non_dict(self, _mock_dirs):
        from generate_all_figures import save_dataset
        save_dataset(None, "none.png", "test_func", "desc")


class TestSaveFigure:
    def test_save_figure(self, _mock_dirs):
        from generate_all_figures import save_figure
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        save_figure(fig, "test_fig.png")


class TestAddToUnifiedReport:
    def test_appends(self, _mock_dirs):
        import generate_all_figures as gaf
        gaf.GENERATED_REPORTS.clear()
        gaf.add_to_unified_report({}, "fig.png", "func", "desc")
        assert len(gaf.GENERATED_REPORTS) == 1


class TestGenerateUnifiedMarkdownReport:
    def test_generates_report(self, _mock_dirs):
        import generate_all_figures as gaf
        gaf.GENERATED_REPORTS.clear()
        gaf.add_to_unified_report(
            {
                "data_structure": {
                    "format": "DataFrame",
                    "columns": ["a", "b"],
                    "rows": 10,
                },
            },
            "test.png",
            "generate_qlearning_func",
            "Q-learning test",
        )
        gaf.add_to_unified_report(
            {
                "data_structure": {
                    "format": "DataFrame",
                    "columns": ["c"],
                    "rows": "Variable",
                },
            },
            "stack.png",
            "generate_stack_architecture",
            "Stack architecture",
        )
        gaf.add_to_unified_report(
            {
                "data_structure": {
                    "format": "DataFrame",
                    "columns": [],
                    "rows": 0,
                },
            },
            "sust.png",
            "generate_sustainability",
            "Sustainability",
        )
        gaf.add_to_unified_report(
            {
                "data_structure": {
                    "format": "DataFrame",
                    "columns": [],
                    "rows": 0,
                },
            },
            "model.png",
            "generate_simulation",
            "Simulation",
        )
        path = gaf.generate_unified_markdown_report()
        assert os.path.exists(path)


class TestFigureGenerators:
    def test_simulation_comparison(self, _mock_dirs):
        from generate_all_figures import generate_simulation_comparison
        generate_simulation_comparison()

    def test_cumulative_energy(self, _mock_dirs):
        from generate_all_figures import generate_cumulative_energy
        generate_cumulative_energy()

    def test_power_evolution(self, _mock_dirs):
        from generate_all_figures import generate_power_evolution
        generate_power_evolution()

    def test_energy_production(self, _mock_dirs):
        from generate_all_figures import generate_energy_production
        generate_energy_production()

    def test_system_health(self, _mock_dirs):
        from generate_all_figures import generate_system_health
        generate_system_health()

    def test_qlearning_progress(self, _mock_dirs):
        from generate_all_figures import generate_qlearning_progress
        generate_qlearning_progress()

    def test_stack_architecture(self, _mock_dirs):
        from generate_all_figures import generate_stack_architecture
        generate_stack_architecture()

    def test_energy_sustainability(self, _mock_dirs):
        from generate_all_figures import generate_energy_sustainability
        generate_energy_sustainability()

    def test_control_analysis(self, _mock_dirs):
        from generate_all_figures import generate_control_analysis
        generate_control_analysis()

    def test_maintenance_schedule(self, _mock_dirs):
        from generate_all_figures import generate_maintenance_schedule
        generate_maintenance_schedule()

    def test_economic_analysis(self, _mock_dirs):
        from generate_all_figures import generate_economic_analysis
        generate_economic_analysis()


class TestMain:
    def test_main_success(self, _mock_dirs):
        from generate_all_figures import main
        result = main()
        assert result == 0

    def test_main_failure(self, _mock_dirs):
        from generate_all_figures import main
        with patch(
            "generate_all_figures.generate_simulation_comparison",
            side_effect=RuntimeError("test"),
        ):
            result = main()
        assert result == 1
