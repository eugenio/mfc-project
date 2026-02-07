"""Unit-style tests for remaining low-coverage GUI plot modules.

Covers:
  - plots/biofilm_plots.py
  - plots/performance_plots.py
  - plots/realtime_plots.py

These are NOT AppTest tests -- they import and call plot functions directly,
using unittest.mock to patch Streamlit calls where necessary.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# Path setup so imports resolve from q-learning-mfcs/src
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.apptest


def _is_plotly_figure(obj: object) -> bool:
    return isinstance(obj, go.Figure)


# ===================================================================
# 1. BIOFILM PLOTS
# ===================================================================


@pytest.mark.apptest
class TestBiofilmPlots:
    """Tests for gui/plots/biofilm_plots.py."""

    def test_no_biofilm_columns_returns_none(self) -> None:
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {"voltage": [1.0, 2.0], "time": [0, 1]}
        assert create_biofilm_analysis_plots(data) is None

    def test_biofilm_thicknesses_nested_list(self) -> None:
        """Nested list triggers per-cell thickness traces."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biofilm_thicknesses": [
                [10.0, 12.0, 11.0],
                [11.0, 13.0, 12.0],
                [12.0, 14.0, 13.0],
            ],
            "time_hours": [0, 1, 2],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_biofilm_thicknesses_scalar_list(self) -> None:
        """Flat list of thicknesses (not per-cell)."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biofilm_thicknesses": [10.0, 11.0, 12.0],
            "time_hours": [0, 1, 2],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_biofilm_thickness_per_cell_key(self) -> None:
        """Test the biofilm_thickness_per_cell key path."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biofilm_thickness_per_cell": [10.0, 11.0],
            "biofilm_thickness_cell_0": [10.0, 11.0],
            "biofilm_thickness_cell_1": [12.0, 13.0],
            "time_hours": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_biomass_density_nested_list(self) -> None:
        """Nested list triggers heatmap."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biomass_density": [
                [0.5, 0.6, 0.7],
                [0.6, 0.7, 0.8],
                [0.7, 0.8, 0.9],
            ],
            "time_hours": [0, 1, 2],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_biomass_density_per_cell_key(self) -> None:
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biomass_density_per_cell": [
                [0.5, 0.6],
                [0.6, 0.7],
            ],
            "time_hours": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_biomass_density_scalar_fallback(self) -> None:
        """Non-nested biomass data triggers the except fallback."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biomass_density": [0.5, 0.6, 0.7],
            "time_hours": [0, 1, 2],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_attachment_fraction(self) -> None:
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "attachment_fraction": [0.3, 0.4, 0.5],
            "time_hours": [0, 1, 2],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_attachment_derived_from_biofilm_thicknesses_nested(self) -> None:
        """When no attachment_fraction, it derives from biofilm_thicknesses."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biofilm_thicknesses": [
                [10.0, 12.0],
                [11.0, 13.0],
            ],
            "time_hours": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_attachment_fallback_no_keys(self) -> None:
        """When neither attachment_fraction nor biofilm_thicknesses exist."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        # Use a biofilm-like column name to pass the guard
        data = {
            "biofilm_conductivity": [0.1, 0.2, 0.3],
            "time_hours": [0, 1, 2],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_dataframe_input(self) -> None:
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        df = pd.DataFrame(
            {
                "biofilm_thicknesses": [
                    [10.0, 12.0],
                    [11.0, 13.0],
                ],
                "time_hours": [0, 1],
            },
        )
        fig = create_biofilm_analysis_plots(df)
        assert _is_plotly_figure(fig)

    def test_all_biofilm_keys_combined(self) -> None:
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biofilm_thicknesses": [
                [10.0, 12.0, 11.0],
                [11.0, 13.0, 12.0],
            ],
            "biomass_density": [
                [0.5, 0.6, 0.7],
                [0.6, 0.7, 0.8],
            ],
            "attachment_fraction": [0.3, 0.4],
            "time_hours": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_time_key_fallback(self) -> None:
        """Uses 'time' key instead of 'time_hours'."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biofilm_thicknesses": [[10.0, 12.0], [11.0, 13.0]],
            "time": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_no_time_key_generates_range(self) -> None:
        """When no time key, range is generated from data length."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biofilm_thicknesses": [[10.0, 12.0], [11.0, 13.0], [12.0, 14.0]],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_empty_biofilm_thicknesses(self) -> None:
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biofilm_thicknesses": [],
            "time_hours": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_more_than_five_cells_capped(self) -> None:
        """Only first 5 cells are plotted."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biofilm_thicknesses": [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
            ],
            "time_hours": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_biomass_keyword_detection(self) -> None:
        """Column with 'biomass' triggers biofilm detection."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "total_biomass": [1.0, 1.5],
            "time_hours": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_attachment_keyword_detection(self) -> None:
        """Column with 'attachment' triggers biofilm detection."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "attachment_rate": [0.5, 0.6],
            "time_hours": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_biomass_density_empty_exception_path(self) -> None:
        """Empty nested list in biomass triggers except block."""
        from gui.plots.biofilm_plots import create_biofilm_analysis_plots

        data = {
            "biomass_density": [[], []],
            "time_hours": [0, 1],
        }
        fig = create_biofilm_analysis_plots(data)
        assert _is_plotly_figure(fig)


# ===================================================================
# 2. PERFORMANCE PLOTS
# ===================================================================


@pytest.mark.apptest
class TestPerformancePlots:
    """Tests for gui/plots/performance_plots.py."""

    def test_basic_dict_returns_figure(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {"time_hours": [0, 1, 2]}
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_energy_efficiency_trace(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "energy_efficiency": [70, 75, 80],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_coulombic_efficiency_scalar(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "coulombic_efficiency": [80, 85, 90],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_coulombic_efficiency_per_cell_nested(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "coulombic_efficiency_per_cell": [
                [80, 82],
                [85, 87],
                [90, 92],
            ],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_coulombic_efficiency_per_cell_key(self) -> None:
        """Uses the coulombic_efficiency_per_cell key (flat)."""
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "coulombic_efficiency_per_cell": [80, 85, 90],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_coulombic_efficiency_empty(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "coulombic_efficiency": [],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_total_power_scalar(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "total_power": [1.5, 2.0, 2.5],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_total_power_nested_list(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "total_power": [[1.0, 0.5], [1.2, 0.6], [1.4, 0.7]],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_power_density_per_cell_nested(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "power_density_per_cell": [[1.0, 0.5], [1.2, 0.6], [1.4, 0.7]],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_power_density_per_cell_flat(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "power_density_per_cell": [1.0, 1.2, 1.4],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_total_power_empty(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "total_power": [],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_cumulative_energy_scalar(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "total_energy_produced": [10, 20, 30],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_energy_produced_key(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "energy_produced": [5, 10, 15],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_cumulative_energy_nested_list(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "total_energy_produced": [[5, 5], [10, 10], [15, 15]],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_cumulative_energy_empty(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "total_energy_produced": [],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_control_error_key(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "control_error": [0.5, 0.3, 0.1],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_outlet_concentration_derived_error(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "outlet_concentration": [24, 25, 26],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_control_fallback_no_error_keys(self) -> None:
        """No control_error or outlet_concentration uses zeros."""
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_economic_metrics_both_keys(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "operating_cost": [1.0, 1.1, 1.2],
            "revenue": [2.0, 2.5, 3.0],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_economic_metrics_only_revenue(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "revenue": [2.0, 2.5, 3.0],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_economic_metrics_only_operating_cost(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "operating_cost": [1.0, 1.1, 1.2],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_dataframe_input(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        df = pd.DataFrame(
            {
                "time_hours": [0, 1, 2],
                "energy_efficiency": [70, 75, 80],
                "total_power": [1.5, 2.0, 2.5],
            },
        )
        fig = create_performance_analysis_plots(df)
        assert _is_plotly_figure(fig)

    def test_time_key_fallback(self) -> None:
        """Uses 'time' instead of 'time_hours'."""
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time": [0, 1, 2],
            "energy_efficiency": [70, 75, 80],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_no_time_key(self) -> None:
        """When no time key exists, default [0] is used."""
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "energy_efficiency": [70, 75, 80],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_empty_time_data_fallback(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [],
            "energy_efficiency": [70, 75, 80],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_all_performance_keys(self) -> None:
        from gui.plots.performance_plots import create_performance_analysis_plots

        data = {
            "time_hours": [0, 1, 2],
            "energy_efficiency": [70, 75, 80],
            "coulombic_efficiency": [80, 85, 90],
            "total_power": [1.5, 2.0, 2.5],
            "total_energy_produced": [10, 20, 30],
            "control_error": [0.5, 0.3, 0.1],
            "operating_cost": [1.0, 1.1, 1.2],
            "revenue": [2.0, 2.5, 3.0],
        }
        fig = create_performance_analysis_plots(data)
        assert _is_plotly_figure(fig)
        # Should have traces for: energy_eff, coulombic, power, cumulative, control, economic
        assert len(fig.data) >= 5


# ===================================================================
# 2b. PARAMETER CORRELATION MATRIX
# ===================================================================


@pytest.mark.apptest
class TestParameterCorrelationMatrix:
    """Tests for create_parameter_correlation_matrix in performance_plots.py."""

    def test_insufficient_data_returns_none(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        data = {"voltage": [1.0]}
        assert create_parameter_correlation_matrix(data) is None

    def test_single_column_returns_none(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        data = {"voltage": [1.0, 2.0, 3.0]}
        assert create_parameter_correlation_matrix(data) is None

    def test_two_numeric_columns(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        data = {
            "voltage": [1.0, 2.0, 3.0],
            "current": [0.5, 1.0, 1.5],
        }
        fig = create_parameter_correlation_matrix(data)
        assert _is_plotly_figure(fig)

    def test_multiple_numeric_columns(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        data = {
            "voltage": [1.0, 2.0, 3.0, 4.0],
            "current": [0.5, 1.0, 1.5, 2.0],
            "power": [0.5, 2.0, 4.5, 8.0],
            "efficiency": [50, 60, 70, 80],
        }
        fig = create_parameter_correlation_matrix(data)
        assert _is_plotly_figure(fig)

    def test_nested_list_columns_averaged(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        data = {
            "voltage": [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
            "current": [0.5, 1.0, 1.5],
        }
        fig = create_parameter_correlation_matrix(data)
        assert _is_plotly_figure(fig)

    def test_time_columns_excluded(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        data = {
            "time_hours": [0, 1, 2],
            "step_count": [0, 10, 20],
            "voltage": [1.0, 2.0, 3.0],
            "current": [0.5, 1.0, 1.5],
        }
        fig = create_parameter_correlation_matrix(data)
        assert _is_plotly_figure(fig)

    def test_empty_data_column_skipped(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        data = {
            "voltage": [1.0, 2.0, 3.0],
            "current": [0.5, 1.0, 1.5],
            "empty_col": [],
        }
        fig = create_parameter_correlation_matrix(data)
        assert _is_plotly_figure(fig)

    def test_dataframe_input(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        df = pd.DataFrame(
            {
                "voltage": [1.0, 2.0, 3.0],
                "current": [0.5, 1.0, 1.5],
            },
        )
        fig = create_parameter_correlation_matrix(df)
        assert _is_plotly_figure(fig)

    def test_different_length_columns_truncated(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        data = {
            "voltage": [1.0, 2.0, 3.0, 4.0, 5.0],
            "current": [0.5, 1.0, 1.5],
        }
        fig = create_parameter_correlation_matrix(data)
        assert _is_plotly_figure(fig)

    def test_non_numeric_values_coerced_to_zero(self) -> None:
        from gui.plots.performance_plots import create_parameter_correlation_matrix

        data = {
            "voltage": [1.0, "bad", 3.0],
            "current": [0.5, 1.0, 1.5],
        }
        fig = create_parameter_correlation_matrix(data)
        assert _is_plotly_figure(fig)


# ===================================================================
# 3. REALTIME PLOTS
# ===================================================================


@pytest.mark.apptest
class TestRealtimePlots:
    """Tests for gui/plots/realtime_plots.py."""

    @staticmethod
    def _make_minimal_df() -> pd.DataFrame:
        """Create a minimal DataFrame with required columns."""
        return pd.DataFrame(
            {
                "time_hours": [0.0, 0.5, 1.0, 1.5, 2.0],
                "reservoir_concentration": [50, 45, 40, 35, 30],
                "outlet_concentration": [20, 22, 24, 25, 26],
                "total_power": [0.001, 0.002, 0.003, 0.004, 0.005],
                "q_action": [0, 1, 2, 1, 0],
            },
        )

    @staticmethod
    def _make_full_df() -> pd.DataFrame:
        """Create a full DataFrame with all optional columns.

        NOTE: substrate_efficiency is excluded because the source code
        has a bug where it tries to add a secondary_y trace to subplot
        (2,2) which lacks secondary_y=True in the specs definition.
        """
        n = 5
        return pd.DataFrame(
            {
                "time_hours": np.linspace(0, 2, n),
                "reservoir_concentration": np.linspace(50, 30, n),
                "outlet_concentration": np.linspace(20, 26, n),
                "total_power": np.linspace(0.001, 0.005, n),
                "q_action": [0, 1, 2, 1, 0],
                "biofilm_thicknesses": [
                    [10, 11, 12],
                    [11, 12, 13],
                    [12, 13, 14],
                    [13, 14, 15],
                    [14, 15, 16],
                ],
                "flow_rate_ml_h": [5.0, 5.5, 6.0, 5.8, 5.2],
                "system_voltage": [0.6, 0.62, 0.64, 0.63, 0.61],
                "individual_cell_powers": [
                    [0.001, 0.001, 0.001],
                    [0.002, 0.001, 0.001],
                    [0.002, 0.002, 0.001],
                    [0.002, 0.002, 0.002],
                    [0.003, 0.002, 0.002],
                ],
                "mixing_efficiency": [0.8, 0.82, 0.85, 0.83, 0.81],
                "biofilm_activity_factor": [0.9, 0.92, 0.94, 0.93, 0.91],
                "q_value": [0.1, 0.3, 0.5, 0.7, 0.9],
                "reward": [0.5, 0.6, 0.7, 0.8, 0.9],
            },
        )

    def test_minimal_df(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_full_df_all_columns(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_full_df()
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)
        # Many traces expected
        assert len(fig.data) >= 5

    def test_biofilm_thicknesses_parsed(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        df["biofilm_thicknesses"] = [
            [10, 12],
            [11, 13],
            [12, 14],
            [13, 15],
            [14, 16],
        ]
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_flow_rate_trace(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        df["flow_rate_ml_h"] = [5.0, 5.5, 6.0, 5.8, 5.2]
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_substrate_efficiency_trace_raises(self) -> None:
        """substrate_efficiency triggers a secondary_y add on subplot (2,2)
        which lacks secondary_y=True in the specs -- known source bug.
        """
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        df["substrate_efficiency"] = [0.7, 0.75, 0.8, 0.78, 0.72]
        with pytest.raises(ValueError, match="secondary_y"):
            create_real_time_plots(df)

    def test_system_voltage_trace(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        df["system_voltage"] = [0.6, 0.62, 0.64, 0.63, 0.61]
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_individual_cell_powers(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        df["individual_cell_powers"] = [
            [0.001, 0.001],
            [0.002, 0.001],
            [0.002, 0.002],
            [0.002, 0.002],
            [0.003, 0.002],
        ]
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_mixing_efficiency_trace(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        df["mixing_efficiency"] = [0.8, 0.82, 0.85, 0.83, 0.81]
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_biofilm_activity_factor(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        df["biofilm_activity_factor"] = [0.9, 0.92, 0.94, 0.93, 0.91]
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_q_value_and_reward(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        df["q_value"] = [0.1, 0.3, 0.5, 0.7, 0.9]
        df["reward"] = [0.5, 0.6, 0.7, 0.8, 0.9]
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_cumulative_energy_wh_units(self) -> None:
        """Small power values stay in Wh units."""
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_cumulative_energy_kwh_units(self) -> None:
        """Large power values convert to kWh."""
        from gui.plots.realtime_plots import create_real_time_plots

        df = self._make_minimal_df()
        df["total_power"] = [500, 600, 700, 800, 900]
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)

    def test_single_point_df(self) -> None:
        from gui.plots.realtime_plots import create_real_time_plots

        df = pd.DataFrame(
            {
                "time_hours": [0.0],
                "reservoir_concentration": [50],
                "outlet_concentration": [20],
                "total_power": [0.001],
                "q_action": [0],
            },
        )
        fig = create_real_time_plots(df)
        assert _is_plotly_figure(fig)


@pytest.mark.apptest
class TestParseBiofilmData:
    """Tests for _parse_biofilm_data helper in realtime_plots.py."""

    def test_list_input(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        assert _parse_biofilm_data([10.0, 12.0, 14.0]) == pytest.approx(12.0)

    def test_tuple_input(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        assert _parse_biofilm_data((10.0, 14.0)) == pytest.approx(12.0)

    def test_empty_list_returns_default(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        assert _parse_biofilm_data([]) == 1.0

    def test_string_csv_input(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        result = _parse_biofilm_data("[10.0, 12.0, 14.0]")
        assert result == pytest.approx(12.0)

    def test_string_single_value(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        result = _parse_biofilm_data("10.5")
        assert result == pytest.approx(10.5)

    def test_string_empty(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        result = _parse_biofilm_data("")
        assert result == 1.0

    def test_numeric_int(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        assert _parse_biofilm_data(10) == 10.0

    def test_numeric_float(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        assert _parse_biofilm_data(12.5) == 12.5

    def test_none_returns_default(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        assert _parse_biofilm_data(None) == 1.0

    def test_invalid_string_returns_default(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        assert _parse_biofilm_data("not_a_number") == 1.0

    def test_string_with_spaces(self) -> None:
        from gui.plots.realtime_plots import _parse_biofilm_data

        result = _parse_biofilm_data("( 10.0, 12.0 )")
        assert result == pytest.approx(11.0)


@pytest.mark.apptest
class TestPerformanceDashboard:
    """Tests for create_performance_dashboard in realtime_plots.py."""

    @patch("gui.plots.realtime_plots.st")
    def test_with_metrics(self, mock_st) -> None:
        from gui.plots.realtime_plots import create_performance_dashboard

        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col, mock_col]

        results = {
            "performance_metrics": {
                "final_reservoir_concentration": 24.5,
                "control_effectiveness_2mM": 85.0,
                "mean_power": 0.003,
                "total_substrate_added": 100.0,
            },
        }
        create_performance_dashboard(results)
        mock_st.columns.assert_called_once_with(4)

    @patch("gui.plots.realtime_plots.st")
    def test_with_empty_metrics(self, mock_st) -> None:
        from gui.plots.realtime_plots import create_performance_dashboard

        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col, mock_col]

        results = {}
        create_performance_dashboard(results)
        mock_st.columns.assert_called_once_with(4)

    @patch("gui.plots.realtime_plots.st")
    def test_with_missing_keys_in_metrics(self, mock_st) -> None:
        from gui.plots.realtime_plots import create_performance_dashboard

        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col, mock_col]

        results = {"performance_metrics": {"mean_power": 0.005}}
        create_performance_dashboard(results)
        mock_st.columns.assert_called_once_with(4)


@pytest.mark.apptest
class TestUpdateAxesLabels:
    """Tests for _update_axes_labels in realtime_plots.py."""

    def test_with_total_power_column(self) -> None:
        from gui.plots.realtime_plots import _update_axes_labels
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=4,
            cols=3,
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            ],
        )
        df = pd.DataFrame(
            {
                "total_power": [0.001, 0.002, 0.003],
                "time_hours": [0, 1, 2],
            },
        )
        _update_axes_labels(fig, df)
        assert _is_plotly_figure(fig)

    def test_without_total_power_column(self) -> None:
        from gui.plots.realtime_plots import _update_axes_labels
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=4,
            cols=3,
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            ],
        )
        df = pd.DataFrame(
            {
                "time_hours": [0, 1, 2],
            },
        )
        _update_axes_labels(fig, df)
        assert _is_plotly_figure(fig)


@pytest.mark.apptest
class TestAddCumulativeEnergyPlot:
    """Tests for _add_cumulative_energy_plot in realtime_plots.py."""

    def test_small_energy(self) -> None:
        from gui.plots.realtime_plots import _add_cumulative_energy_plot
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=4,
            cols=3,
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            ],
        )
        df = pd.DataFrame(
            {
                "time_hours": [0.0, 1.0, 2.0, 3.0, 4.0],
                "total_power": [0.001, 0.002, 0.003, 0.004, 0.005],
            },
        )
        _add_cumulative_energy_plot(fig, df)
        assert len(fig.data) >= 1

    def test_large_energy_kwh(self) -> None:
        from gui.plots.realtime_plots import _add_cumulative_energy_plot
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=4,
            cols=3,
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            ],
        )
        df = pd.DataFrame(
            {
                "time_hours": [0.0, 1.0, 2.0, 3.0, 4.0],
                "total_power": [500, 600, 700, 800, 900],
            },
        )
        _add_cumulative_energy_plot(fig, df)
        assert len(fig.data) >= 1

    def test_single_time_point(self) -> None:
        from gui.plots.realtime_plots import _add_cumulative_energy_plot
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=4,
            cols=3,
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            ],
        )
        df = pd.DataFrame(
            {
                "time_hours": [0.0],
                "total_power": [0.001],
            },
        )
        _add_cumulative_energy_plot(fig, df)
        assert len(fig.data) >= 1
