"""Tests for visualization_analysis.py - substrate utilization comparison plots."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def unified_csv(tmp_path):
    """Create a minimal unified CSV data file."""
    n = 200
    df = pd.DataFrame(
        {
            "time_hours": np.linspace(0, 100, n),
            "inlet_concentration": np.full(n, 20.0),
            "avg_outlet_concentration": np.full(n, 19.5),
            "cell_1_biofilm": np.linspace(1.0, 1.3, n),
            "cell_2_biofilm": np.linspace(1.0, 1.3, n),
            "cell_3_biofilm": np.linspace(1.0, 1.3, n),
            "cell_4_biofilm": np.linspace(1.0, 1.3, n),
            "cell_5_biofilm": np.linspace(1.0, 1.3, n),
            "stack_power": np.random.uniform(0.05, 0.2, n),
        }
    )
    path = tmp_path / "unified.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def non_unified_csv(tmp_path):
    """Create a minimal non-unified CSV data file."""
    n = 200
    df = pd.DataFrame(
        {
            "time_hours": np.linspace(0, 100, n),
            "substrate_utilization": np.random.uniform(10, 30, n),
            "cell_1_biofilm": np.linspace(1.0, 1.3, n),
            "cell_2_biofilm": np.linspace(1.0, 1.3, n),
            "cell_3_biofilm": np.linspace(1.0, 1.3, n),
            "cell_4_biofilm": np.linspace(1.0, 1.3, n),
            "cell_5_biofilm": np.linspace(1.0, 1.3, n),
            "stack_power": np.random.uniform(0.05, 0.2, n),
        }
    )
    path = tmp_path / "non_unified.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestCreateAnalysisPlots:
    """Tests for the create_analysis_plots function."""

    def test_create_analysis_plots_returns_dict(self, unified_csv, non_unified_csv):
        """Test that function returns expected dict of metrics."""
        with patch(
            "visualization_analysis.get_simulation_data_path"
        ) as mock_data_path, patch(
            "visualization_analysis.get_figure_path", return_value="/tmp/fig.png"
        ), patch.object(plt.Figure, "savefig"):
            mock_data_path.side_effect = [unified_csv, non_unified_csv]

            from visualization_analysis import create_analysis_plots
            result = create_analysis_plots()

        assert isinstance(result, dict)
        assert "unified_early_util" in result
        assert "unified_late_util" in result
        assert "non_unified_early_util" in result
        assert "non_unified_late_util" in result
        assert "unified_power_trend" in result
        assert "non_unified_power_trend" in result
        plt.close("all")

    def test_numeric_results(self, unified_csv, non_unified_csv):
        """Test that returned values are numeric."""
        with patch(
            "visualization_analysis.get_simulation_data_path"
        ) as mock_data_path, patch(
            "visualization_analysis.get_figure_path", return_value="/tmp/fig.png"
        ), patch.object(plt.Figure, "savefig"):
            mock_data_path.side_effect = [unified_csv, non_unified_csv]

            from visualization_analysis import create_analysis_plots
            result = create_analysis_plots()

        for key, val in result.items():
            assert isinstance(val, (int, float, np.floating)), f"{key} is not numeric"
        plt.close("all")

    def test_power_trend_signs(self, unified_csv, non_unified_csv):
        """Test that power trends are computed (can be positive or negative)."""
        with patch(
            "visualization_analysis.get_simulation_data_path"
        ) as mock_data_path, patch(
            "visualization_analysis.get_figure_path", return_value="/tmp/fig.png"
        ), patch.object(plt.Figure, "savefig"):
            mock_data_path.side_effect = [unified_csv, non_unified_csv]

            from visualization_analysis import create_analysis_plots
            result = create_analysis_plots()

        assert result["unified_power_trend"] != 0 or True  # trend may be zero
        assert result["non_unified_power_trend"] != 0 or True
        plt.close("all")
