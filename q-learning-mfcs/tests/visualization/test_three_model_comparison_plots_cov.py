"""Tests for three_model_comparison_plots.py - comprehensive 3-model comparison."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import patch, MagicMock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_unified_df(n=200):
    """Create a synthetic unified model DataFrame."""
    t = np.linspace(0, 120, n)
    df = pd.DataFrame({
        "time_hours": t,
        "inlet_concentration": np.full(n, 20.0),
        "avg_outlet_concentration": np.full(n, 19.9),
        "cell_1_biofilm": np.linspace(1.0, 0.5, n),
        "cell_2_biofilm": np.linspace(1.0, 0.5, n),
        "cell_3_biofilm": np.linspace(1.0, 0.5, n),
        "cell_4_biofilm": np.linspace(1.0, 0.5, n),
        "cell_5_biofilm": np.linspace(1.0, 0.5, n),
        "stack_power": np.random.uniform(0.05, 0.2, n),
        "flow_rate_ml_h": np.full(n, 10.0),
    })
    return df


def _make_non_unified_df(n=200):
    """Create a synthetic non-unified model DataFrame."""
    t = np.linspace(0, 120, n)
    df = pd.DataFrame({
        "time_hours": t,
        "substrate_utilization": np.random.uniform(20, 30, n),
        "cell_1_biofilm": np.linspace(1.0, 1.31, n),
        "cell_2_biofilm": np.linspace(1.0, 1.31, n),
        "cell_3_biofilm": np.linspace(1.0, 1.31, n),
        "cell_4_biofilm": np.linspace(1.0, 1.31, n),
        "cell_5_biofilm": np.linspace(1.0, 1.31, n),
        "stack_power": np.random.uniform(0.05, 0.2, n),
        "flow_rate_ml_h": np.full(n, 10.0),
    })
    return df


def _make_recirculation_df(n=200):
    """Create a synthetic recirculation model DataFrame."""
    t = np.linspace(0, 100, n)
    df = pd.DataFrame({
        "time_hours": t,
        "reservoir_concentration": np.full(n, 20.0),
        "outlet_concentration": np.full(n, 18.0),
        "cell_1_biofilm": np.linspace(1.0, 1.08, n),
        "cell_2_biofilm": np.linspace(1.0, 1.08, n),
        "cell_3_biofilm": np.linspace(1.0, 1.08, n),
        "cell_4_biofilm": np.linspace(1.0, 1.08, n),
        "cell_5_biofilm": np.linspace(1.0, 1.08, n),
        "cell_1_concentration": np.linspace(20.0, 18.0, n),
        "cell_2_concentration": np.linspace(19.5, 17.5, n),
        "cell_3_concentration": np.linspace(19.0, 17.0, n),
        "cell_4_concentration": np.linspace(18.5, 16.5, n),
        "cell_5_concentration": np.linspace(18.0, 16.0, n),
        "total_power": np.random.uniform(0.05, 0.2, n),
        "flow_rate": np.full(n, 10.0),
    })
    return df


@pytest.fixture
def unified_csv(tmp_path):
    df = _make_unified_df()
    p = tmp_path / "unified.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def non_unified_csv(tmp_path):
    df = _make_non_unified_df()
    p = tmp_path / "non_unified.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def recirculation_csv(tmp_path):
    df = _make_recirculation_df()
    p = tmp_path / "recirc.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def three_dfs():
    """Return (unified, non_unified, recirculation) DataFrames."""
    return _make_unified_df(), _make_non_unified_df(), _make_recirculation_df()


class TestLoadAndPrepareData:
    def test_returns_three_dataframes(
        self, unified_csv, non_unified_csv, recirculation_csv
    ):
        with patch(
            "three_model_comparison_plots.get_simulation_data_path"
        ) as mock_path:
            mock_path.side_effect = [unified_csv, non_unified_csv, recirculation_csv]
            from three_model_comparison_plots import load_and_prepare_data
            u, nu, r = load_and_prepare_data()

        assert u is not None
        assert nu is not None
        assert r is not None
        assert "avg_biofilm" in u.columns
        assert "avg_biofilm" in nu.columns
        assert "avg_biofilm" in r.columns
        assert "model" in u.columns

    def test_file_not_found_returns_nones(self):
        with patch(
            "three_model_comparison_plots.get_simulation_data_path",
            side_effect=lambda f: "/nonexistent/" + f,
        ):
            from three_model_comparison_plots import load_and_prepare_data
            u, nu, r = load_and_prepare_data()
        assert u is None
        assert nu is None
        assert r is None

    def test_substrate_utilization_calculated_for_unified(
        self, unified_csv, non_unified_csv, recirculation_csv
    ):
        with patch(
            "three_model_comparison_plots.get_simulation_data_path"
        ) as mock_path:
            mock_path.side_effect = [unified_csv, non_unified_csv, recirculation_csv]
            from three_model_comparison_plots import load_and_prepare_data
            u, nu, r = load_and_prepare_data()

        assert "substrate_utilization" in u.columns
        assert "substrate_utilization" in r.columns

    def test_time_hours_present(
        self, unified_csv, non_unified_csv, recirculation_csv
    ):
        with patch(
            "three_model_comparison_plots.get_simulation_data_path"
        ) as mock_path:
            mock_path.side_effect = [unified_csv, non_unified_csv, recirculation_csv]
            from three_model_comparison_plots import load_and_prepare_data
            u, nu, r = load_and_prepare_data()

        assert "time_hours" in u.columns
        assert "time_hours" in nu.columns
        assert "time_hours" in r.columns


class TestCreateBiofilmHealthComparison:
    def test_returns_figure(self, three_dfs):
        u, nu, r = three_dfs
        # Prepare dfs the way load_and_prepare_data would
        u["avg_biofilm"] = u[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
        nu["avg_biofilm"] = nu[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
        r["avg_biofilm"] = r[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)

        from three_model_comparison_plots import create_biofilm_health_comparison
        fig = create_biofilm_health_comparison(u, nu, r)
        assert isinstance(fig, plt.Figure)


class TestCreateSubstrateManagementComparison:
    def test_returns_figure(self, three_dfs):
        u, nu, r = three_dfs
        u["avg_biofilm"] = u[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
        nu["avg_biofilm"] = nu[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
        r["avg_biofilm"] = r[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
        # Ensure substrate_utilization columns
        u["substrate_utilization"] = np.maximum(
            0, 100 * (20.0 - u.get("avg_outlet_concentration", 19.9)) / 20.0
        )
        r["substrate_utilization"] = np.maximum(
            0,
            100
            * (r["reservoir_concentration"] - r["outlet_concentration"])
            / r["reservoir_concentration"],
        )

        from three_model_comparison_plots import (
            create_substrate_management_comparison,
        )
        fig = create_substrate_management_comparison(u, nu, r)
        assert isinstance(fig, plt.Figure)


class TestCreateSystemPerformanceComparison:
    def test_returns_figure(self, three_dfs):
        u, nu, r = three_dfs
        u["avg_biofilm"] = u[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
        nu["avg_biofilm"] = nu[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
        r["avg_biofilm"] = r[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)

        from three_model_comparison_plots import (
            create_system_performance_comparison,
        )
        fig = create_system_performance_comparison(u, nu, r)
        assert isinstance(fig, plt.Figure)


class TestCreateBreakthroughAnalysisPlot:
    def test_returns_figure(self, three_dfs):
        u, nu, r = three_dfs
        u["avg_biofilm"] = u[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
        nu["avg_biofilm"] = nu[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
        r["avg_biofilm"] = r[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)

        from three_model_comparison_plots import create_breakthrough_analysis_plot
        fig = create_breakthrough_analysis_plot(u, nu, r)
        assert isinstance(fig, plt.Figure)


class TestMain:
    def test_main_runs(
        self, unified_csv, non_unified_csv, recirculation_csv, tmp_path
    ):
        with patch(
            "three_model_comparison_plots.get_simulation_data_path"
        ) as mock_data, patch(
            "three_model_comparison_plots.get_figure_path",
            side_effect=lambda f: str(tmp_path / os.path.basename(f)),
        ), patch("three_model_comparison_plots.os.makedirs"):
            mock_data.side_effect = [
                unified_csv,
                non_unified_csv,
                recirculation_csv,
            ]
            from three_model_comparison_plots import main
            main()

    def test_main_file_not_found(self, tmp_path):
        with patch(
            "three_model_comparison_plots.get_simulation_data_path",
            side_effect=lambda f: str(tmp_path / "nonexistent" / f),
        ):
            from three_model_comparison_plots import main
            main()  # Should return early when data is None
