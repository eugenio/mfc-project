"""Coverage boost tests for three_model_comparison_plots.py - targeting remaining uncovered lines."""
import os
import sys
from unittest.mock import MagicMock, patch

# Mock seaborn before importing the module under test
_mock_sns = MagicMock()
sys.modules.setdefault("seaborn", _mock_sns)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_unified(n=200):
    t = np.linspace(0, 120, n)
    df = pd.DataFrame({
        "time_seconds": t * 3600,
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


def _make_non_unified(n=200):
    t = np.linspace(0, 120, n)
    df = pd.DataFrame({
        "time_seconds": t * 3600,
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


def _make_recirc(n=200):
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


def _prepare_dfs():
    u = _make_unified()
    nu = _make_non_unified()
    r = _make_recirc()
    for df in [u, nu, r]:
        df["avg_biofilm"] = df[
            [f"cell_{i}_biofilm" for i in range(1, 6)]
        ].mean(axis=1)
    u["substrate_utilization"] = np.maximum(
        0, 100 * (20.0 - u["avg_outlet_concentration"]) / 20.0,
    )
    r["substrate_utilization"] = np.maximum(
        0, 100 * (r["reservoir_concentration"] - r["outlet_concentration"])
        / r["reservoir_concentration"],
    )
    return u, nu, r


@pytest.mark.coverage_extra
class TestLoadAndPrepareDataEdge:
    def test_time_hours_computed_from_seconds(self):
        """Test that time_hours is derived from time_seconds when missing."""
        u = _make_unified()
        nu = _make_non_unified()
        r = _make_recirc()
        # Remove time_hours to trigger the fallback
        u_no_hours = u.drop(columns=["time_hours"])
        nu_no_hours = nu.drop(columns=["time_hours"])
        # Write CSVs
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            u_path = os.path.join(tmp, "u.csv")
            nu_path = os.path.join(tmp, "nu.csv")
            r_path = os.path.join(tmp, "r.csv")
            u_no_hours.to_csv(u_path, index=False)
            nu_no_hours.to_csv(nu_path, index=False)
            r.to_csv(r_path, index=False)

            with patch(
                "three_model_comparison_plots.get_simulation_data_path"
            ) as mock_path:
                mock_path.side_effect = [u_path, nu_path, r_path]
                from three_model_comparison_plots import load_and_prepare_data
                result_u, result_nu, result_r = load_and_prepare_data()

            assert "time_hours" in result_u.columns
            assert "time_hours" in result_nu.columns


@pytest.mark.coverage_extra
class TestBiofilmHealthDetailed:
    def test_all_subplots_rendered(self):
        u, nu, r = _prepare_dfs()
        from three_model_comparison_plots import create_biofilm_health_comparison
        fig = create_biofilm_health_comparison(u, nu, r)
        assert len(fig.get_axes()) == 4
        plt.close(fig)


@pytest.mark.coverage_extra
class TestSubstrateManagementDetailed:
    def test_all_subplots_rendered(self):
        u, nu, r = _prepare_dfs()
        from three_model_comparison_plots import create_substrate_management_comparison
        fig = create_substrate_management_comparison(u, nu, r)
        assert len(fig.get_axes()) == 4
        plt.close(fig)


@pytest.mark.coverage_extra
class TestSystemPerformanceDetailed:
    def test_all_subplots_rendered(self):
        u, nu, r = _prepare_dfs()
        from three_model_comparison_plots import create_system_performance_comparison
        fig = create_system_performance_comparison(u, nu, r)
        # 3 plot axes + 1 table axes = 4
        assert len(fig.get_axes()) >= 4
        plt.close(fig)


@pytest.mark.coverage_extra
class TestBreakthroughAnalysisDetailed:
    def test_all_subplots_rendered(self):
        u, nu, r = _prepare_dfs()
        from three_model_comparison_plots import create_breakthrough_analysis_plot
        fig = create_breakthrough_analysis_plot(u, nu, r)
        assert len(fig.get_axes()) == 4
        plt.close(fig)


@pytest.mark.coverage_extra
class TestMainEdgeCases:
    def test_main_saves_all_figures(self, tmp_path):
        u = _make_unified()
        nu = _make_non_unified()
        r = _make_recirc()
        u_path = str(tmp_path / "u.csv")
        nu_path = str(tmp_path / "nu.csv")
        r_path = str(tmp_path / "r.csv")
        u.to_csv(u_path, index=False)
        nu.to_csv(nu_path, index=False)
        r.to_csv(r_path, index=False)

        with patch(
            "three_model_comparison_plots.get_simulation_data_path"
        ) as mock_data, patch(
            "three_model_comparison_plots.get_figure_path",
            side_effect=lambda f: str(tmp_path / os.path.basename(f)),
        ), patch("three_model_comparison_plots.os.makedirs"):
            mock_data.side_effect = [u_path, nu_path, r_path]
            from three_model_comparison_plots import main
            main()

        # Verify figure files were created
        pngs = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
        assert len(pngs) >= 2
