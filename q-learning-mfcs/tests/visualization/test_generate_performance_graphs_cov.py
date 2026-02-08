"""Tests for generate_performance_graphs.py."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def _mock_save(tmp_path):
    with patch(
        "generate_performance_graphs.get_figure_path",
        side_effect=lambda f: str(tmp_path / f),
    ):
        yield


class TestGenerateSyntheticDetailedData:
    def test_returns_dict(self):
        from generate_performance_graphs import generate_synthetic_detailed_data
        data = generate_synthetic_detailed_data()
        assert isinstance(data, dict)

    def test_has_required_keys(self):
        from generate_performance_graphs import generate_synthetic_detailed_data
        data = generate_synthetic_detailed_data()
        required = [
            "hours", "stack_power", "stack_voltage", "stack_current",
            "cumulative_energy", "cell_voltages", "cell_powers",
            "aging_factors", "biofilm_thickness", "substrate_level",
            "ph_buffer_level", "q_table_size", "exploration_rate",
            "rewards", "efficiency", "stability", "active_cells",
        ]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_array_lengths(self):
        from generate_performance_graphs import generate_synthetic_detailed_data
        data = generate_synthetic_detailed_data()
        assert len(data["hours"]) == 100
        assert data["cell_voltages"].shape == (5, 100)
        assert data["cell_powers"].shape == (5, 100)


class TestCreateComprehensivePerformancePlots:
    def test_runs_and_returns_data(self, _mock_save):
        from generate_performance_graphs import create_comprehensive_performance_plots
        data = create_comprehensive_performance_plots()
        assert isinstance(data, dict)
        assert "stack_power" in data


class TestCreateAdditionalAnalysisPlots:
    def test_runs_without_error(self, _mock_save):
        from generate_performance_graphs import (
            generate_synthetic_detailed_data,
            create_additional_analysis_plots,
        )
        data = generate_synthetic_detailed_data()
        create_additional_analysis_plots(data)


class TestMain:
    def test_main_runs(self, _mock_save):
        from generate_performance_graphs import main
        main()
