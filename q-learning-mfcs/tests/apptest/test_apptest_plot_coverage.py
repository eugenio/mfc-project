"""Unit-style tests for GUI plot modules to improve coverage.

These are NOT AppTest tests -- they import and call plot functions directly,
using unittest.mock to patch Streamlit calls where necessary.
Covers:
  - plots/sensing_plots.py
  - plots/spatial_plots.py
  - plots/metabolic_plots.py
  - qlearning_viz.py
  - policy_evolution_viz.py
  - qtable_visualization.py
  - parameter_input.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from enum import Enum
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


def _mock_columns(n_or_spec):
    """Mock for st.columns that returns the right number of MagicMock objects."""
    if isinstance(n_or_spec, int):
        return [MagicMock() for _ in range(n_or_spec)]
    if isinstance(n_or_spec, (list, tuple)):
        return [MagicMock() for _ in n_or_spec]
    return [MagicMock()]


# ===================================================================
# 1. SENSING PLOTS
# ===================================================================


class TestSensingPlots:
    """Tests for gui/plots/sensing_plots.py."""

    def test_no_sensing_columns_returns_none(self) -> None:
        data = {"voltage": [1.0, 2.0], "time": [0, 1]}
        from gui.plots.sensing_plots import create_sensing_analysis_plots

        assert create_sensing_analysis_plots(data) is None

    def test_dict_with_eis_impedance_magnitude(self) -> None:
        from gui.plots.sensing_plots import create_sensing_analysis_plots

        data = {
            "eis_impedance_magnitude": [1000, 950, 900],
            "time_hours": [0, 1, 2],
        }
        fig = create_sensing_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_dict_with_all_sensing_keys(self) -> None:
        from gui.plots.sensing_plots import create_sensing_analysis_plots

        data = {
            "time_hours": [0, 1, 2, 3],
            "eis_impedance_magnitude": [1000, 950, 900, 850],
            "eis_impedance_phase": [-45, -40, -38, -35],
            "qcm_frequency_shift": [-500, -450, -400, -350],
            "charge_transfer_resistance": [100, 95, 90, 85],
            "qcm_mass_loading": [0.1, 0.15, 0.2, 0.25],
        }
        fig = create_sensing_analysis_plots(data)
        assert _is_plotly_figure(fig)
        assert len(fig.data) == 5

    def test_dict_with_only_frequency_key(self) -> None:
        """A column with 'frequency' triggers sensing detection."""
        from gui.plots.sensing_plots import create_sensing_analysis_plots

        data = {
            "frequency_response": [100, 200],
            "time": [0, 1],
        }
        fig = create_sensing_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_dataframe_input(self) -> None:
        from gui.plots.sensing_plots import create_sensing_analysis_plots

        df = pd.DataFrame(
            {
                "eis_impedance_magnitude": [1000, 950],
                "time_hours": [0, 1],
            }
        )
        fig = create_sensing_analysis_plots(df)
        assert _is_plotly_figure(fig)

    def test_empty_time_data_fallback(self) -> None:
        from gui.plots.sensing_plots import create_sensing_analysis_plots

        data = {
            "eis_impedance_magnitude": [1000],
        }
        fig = create_sensing_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_empty_list_values_fallback(self) -> None:
        from gui.plots.sensing_plots import create_sensing_analysis_plots

        data = {
            "eis_impedance_magnitude": [],
            "time_hours": [0, 1],
        }
        fig = create_sensing_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_impedance_keyword_triggers_sensing(self) -> None:
        from gui.plots.sensing_plots import create_sensing_analysis_plots

        data = {
            "impedance_total": [500, 600],
            "time": [0, 1],
        }
        fig = create_sensing_analysis_plots(data)
        assert _is_plotly_figure(fig)


# ===================================================================
# 2. SPATIAL PLOTS
# ===================================================================


class TestSpatialPlots:
    """Tests for gui/plots/spatial_plots.py."""

    def test_no_cell_data_returns_none(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {"voltage": [1.0], "time": [0]}
        assert create_spatial_distribution_plots(data) is None

    def test_cell_voltages_dict(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "cell_voltages": [[0.7, 0.72, 0.68, 0.71, 0.69]],
        }
        fig = create_spatial_distribution_plots(data, n_cells=5)
        assert _is_plotly_figure(fig)

    def test_current_densities_dict(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "current_densities": [[1.0, 1.1, 0.9, 1.05, 0.95]],
        }
        fig = create_spatial_distribution_plots(data, n_cells=5)
        assert _is_plotly_figure(fig)

    def test_current_density_per_cell_key(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "current_density_per_cell": [[2.0, 2.1, 1.9]],
        }
        fig = create_spatial_distribution_plots(data, n_cells=3)
        assert _is_plotly_figure(fig)

    def test_temperature_per_cell(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "temperature_per_cell": [[25.0, 25.5, 24.8, 25.2, 25.1]],
        }
        fig = create_spatial_distribution_plots(data, n_cells=5)
        assert _is_plotly_figure(fig)

    def test_biofilm_thicknesses(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        # biofilm_thicknesses alone is not detected by column filter
        # (needs per_cell, cell_, voltages, or densities in column name).
        # Include a detectable key so the function proceeds past the guard.
        data = {
            "cell_voltages": [[0.7, 0.72, 0.68, 0.71, 0.69]],
            "biofilm_thicknesses": [[10.0, 12.0, 11.5, 9.8, 10.2]],
        }
        fig = create_spatial_distribution_plots(data, n_cells=5)
        assert _is_plotly_figure(fig)

    def test_biofilm_thickness_per_cell_key(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "biofilm_thickness_per_cell": [[10.0, 11.0]],
        }
        fig = create_spatial_distribution_plots(data, n_cells=3)
        assert _is_plotly_figure(fig)

    def test_all_spatial_data_combined(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "cell_voltages": [[0.7, 0.72, 0.68]],
            "current_densities": [[1.0, 1.1, 0.9]],
            "temperature_per_cell": [[25.0, 25.5, 24.8]],
            "biofilm_thicknesses": [[10.0, 12.0, 11.5]],
        }
        fig = create_spatial_distribution_plots(data, n_cells=3)
        assert _is_plotly_figure(fig)
        assert len(fig.data) == 4

    def test_dataframe_input(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        df = pd.DataFrame(
            {
                "cell_voltages": [[0.7, 0.72]],
            }
        )
        fig = create_spatial_distribution_plots(df, n_cells=2)
        assert _is_plotly_figure(fig)

    def test_scalar_cell_voltages(self) -> None:
        """When data list contains a scalar, it is broadcast to n_cells."""
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "cell_voltages": [0.7],
        }
        fig = create_spatial_distribution_plots(data, n_cells=3)
        assert _is_plotly_figure(fig)

    def test_fewer_cells_than_n_cells_pads(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "cell_voltages": [[0.7, 0.72]],
        }
        fig = create_spatial_distribution_plots(data, n_cells=5)
        assert _is_plotly_figure(fig)

    def test_more_cells_than_n_cells_truncates(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "cell_voltages": [[0.7, 0.72, 0.68, 0.71, 0.69, 0.73, 0.74]],
        }
        fig = create_spatial_distribution_plots(data, n_cells=3)
        assert _is_plotly_figure(fig)

    def test_get_latest_value_helper(self) -> None:
        from gui.plots.spatial_plots import _get_latest_value

        assert _get_latest_value([1, 2, 3]) == 3
        assert _get_latest_value([[1, 2], [3, 4]]) == [3, 4]
        assert _get_latest_value([]) is None
        assert _get_latest_value("not a list") is None

    def test_empty_cell_voltages_data(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "cell_voltages": [],
        }
        fig = create_spatial_distribution_plots(data, n_cells=3)
        assert _is_plotly_figure(fig)

    def test_per_cell_column_detection(self) -> None:
        from gui.plots.spatial_plots import create_spatial_distribution_plots

        data = {
            "per_cell_voltage": [0.7],
        }
        fig = create_spatial_distribution_plots(data, n_cells=3)
        assert _is_plotly_figure(fig)


# ===================================================================
# 3. METABOLIC PLOTS
# ===================================================================


class TestMetabolicPlots:
    """Tests for gui/plots/metabolic_plots.py."""

    def test_no_metabolic_columns_returns_none(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {"voltage": [1.0], "time": [0]}
        assert create_metabolic_analysis_plots(data) is None

    def test_nadh_ratios_scalar(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "nadh_ratios": [0.3, 0.35, 0.4],
            "time_hours": [0, 1, 2],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_nadh_ratios_per_cell(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "nadh_ratios": [[0.3, 0.32], [0.35, 0.37], [0.4, 0.42]],
            "time_hours": [0, 1, 2],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_nadh_ratio_key_alternative(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "nadh_ratio": [0.3, 0.35],
            "time_hours": [0, 1],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_atp_levels_scalar(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "atp_levels": [2.0, 2.5, 3.0],
            "time_hours": [0, 1, 2],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_atp_levels_per_cell(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "atp_levels": [[2.0, 2.1], [2.5, 2.6], [3.0, 3.1]],
            "time_hours": [0, 1, 2],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_atp_level_key_alternative(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "atp_level": [2.0, 2.5],
            "time_hours": [0, 1],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_electron_flux_scalar(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "electron_flux": [0.1, 0.15, 0.2],
            "time_hours": [0, 1, 2],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_electron_flux_per_cell(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "electron_flux": [[0.1, 0.12], [0.15, 0.17], [0.2, 0.22]],
            "time_hours": [0, 1, 2],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_total_current_fallback(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "total_current": [0.5, 0.6, 0.7],
            "metabolic_rate": [1.0, 1.1, 1.2],
            "time_hours": [0, 1, 2],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)

    def test_all_metabolic_keys(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "nadh_ratios": [0.3, 0.35, 0.4],
            "atp_levels": [2.0, 2.5, 3.0],
            "electron_flux": [0.1, 0.15, 0.2],
            "time_hours": [0, 1, 2],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)
        assert len(fig.data) == 3

    def test_dataframe_input(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        df = pd.DataFrame(
            {
                "nadh_ratios": [0.3, 0.35],
                "time_hours": [0, 1],
            }
        )
        fig = create_metabolic_analysis_plots(df)
        assert _is_plotly_figure(fig)

    def test_time_key_fallback(self) -> None:
        from gui.plots.metabolic_plots import create_metabolic_analysis_plots

        data = {
            "nadh_ratios": [0.3, 0.35],
            "time": [0, 1],
        }
        fig = create_metabolic_analysis_plots(data)
        assert _is_plotly_figure(fig)


# ===================================================================
# 4. QLEARNING_VIZ
# ===================================================================


class TestQLearningVizConfig:
    """Tests for QLearningVisualizationConfig and QLearningVisualizationType."""

    def test_visualization_type_enum(self) -> None:
        from gui.qlearning_viz import QLearningVisualizationType

        assert QLearningVisualizationType.Q_TABLE_HEATMAP.value == "q_table_heatmap"
        assert len(QLearningVisualizationType) == 7

    def test_config_defaults(self) -> None:
        from gui.qlearning_viz import QLearningVisualizationConfig

        config = QLearningVisualizationConfig()
        assert config.colormap == "RdYlBu_r"
        assert config.show_values is True
        assert config.font_size == 10
        assert config.confidence_intervals is True

    def test_config_custom(self) -> None:
        from gui.qlearning_viz import QLearningVisualizationConfig

        config = QLearningVisualizationConfig(
            colormap="Viridis",
            font_size=14,
            animation_speed=1.0,
        )
        assert config.colormap == "Viridis"
        assert config.font_size == 14
        assert config.animation_speed == 1.0


class TestQLearningVisualizerPureMethods:
    """Tests for pure computation methods of QLearningVisualizer."""

    @patch("streamlit.markdown")
    def test_calculate_convergence_score(self, mock_md: MagicMock) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        score = viz._calculate_convergence_score(q)
        assert 0.0 <= score <= 1.0

    @patch("streamlit.markdown")
    def test_convergence_score_single_action(self, mock_md: MagicMock) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.array([[1.0], [2.0], [3.0]])
        score = viz._calculate_convergence_score(q)
        assert score == 1.0

    @patch("streamlit.markdown")
    def test_calculate_policy_diversity(self, mock_md: MagicMock) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        diversity = viz._calculate_policy_diversity(q)
        assert 0.0 <= diversity <= 1.0

    @patch("streamlit.markdown")
    def test_calculate_value_stability(self, mock_md: MagicMock) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.ones((5, 3))
        stability = viz._calculate_value_stability(q)
        assert stability > 0.0

    @patch("streamlit.markdown")
    def test_estimate_exploration_rate(self, mock_md: MagicMock) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.random.randn(10, 4)
        rate = viz._estimate_exploration_rate(q)
        assert 0.0 <= rate <= 1.0

    @patch("streamlit.markdown")
    def test_calculate_policy_confidence(self, mock_md: MagicMock) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.array([[1.0, 0.5], [0.3, 0.3], [2.0, 0.0]])
        conf = viz._calculate_policy_confidence(q)
        assert conf.shape == (3,)
        assert all(0.0 <= c <= 1.0 for c in conf)

    @patch("streamlit.markdown")
    def test_calculate_policy_confidence_single_action(
        self, mock_md: MagicMock
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.array([[1.0], [2.0]])
        conf = viz._calculate_policy_confidence(q)
        assert all(c == 1.0 for c in conf)


class TestQLearningVisualizerRendering:
    """Test rendering methods with Streamlit mocked."""

    @patch("streamlit.markdown")
    @patch("streamlit.plotly_chart")
    @patch("streamlit.metric")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.write")
    def test_render_qtable_analysis(
        self, mock_write, mock_cols, mock_metric, mock_chart, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.random.randn(10, 4) * 0.5
        fig = viz._render_qtable_analysis(q)
        assert _is_plotly_figure(fig)

    @patch("streamlit.markdown")
    @patch("streamlit.plotly_chart")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.write")
    @patch("streamlit.success")
    @patch("streamlit.warning")
    def test_render_learning_curves(
        self, mock_warn, mock_succ, mock_write, mock_cols, mock_chart, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        history = {
            "reward": list(np.linspace(0, 10, 50)),
            "loss": list(np.linspace(5, 0.1, 50)),
        }
        fig = viz._render_learning_curves(history)
        assert _is_plotly_figure(fig)

    @patch("streamlit.markdown")
    @patch("streamlit.warning")
    def test_render_learning_curves_empty(self, mock_warn, mock_md) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        fig = viz._render_learning_curves({})
        assert _is_plotly_figure(fig)

    @patch("streamlit.markdown")
    @patch("streamlit.plotly_chart")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.write")
    @patch("streamlit.info")
    @patch("streamlit.success")
    @patch("streamlit.warning")
    def test_render_policy_visualization(
        self,
        mock_warn,
        mock_succ,
        mock_info,
        mock_write,
        mock_cols,
        mock_chart,
        mock_md,
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.random.randn(10, 4)
        policy = np.argmax(q, axis=1).astype(np.int64)
        fig = viz._render_policy_visualization(policy, q)
        assert _is_plotly_figure(fig)

    @patch("streamlit.markdown")
    @patch("streamlit.plotly_chart")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.write")
    @patch("streamlit.info")
    def test_render_policy_visualization_no_qtable(
        self, mock_info, mock_write, mock_cols, mock_chart, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        policy = np.array([0, 1, 2, 0, 1])
        fig = viz._render_policy_visualization(policy, None)
        assert _is_plotly_figure(fig)

    @patch("streamlit.markdown")
    @patch("streamlit.columns", side_effect=_mock_columns)
    def test_render_performance_metrics(self, mock_cols, mock_md) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.random.randn(10, 4)
        history = {
            "reward": list(np.linspace(0, 10, 50)),
        }
        fig = viz._render_performance_metrics(q, history)
        assert _is_plotly_figure(fig)

    @patch("streamlit.markdown")
    @patch("streamlit.columns", side_effect=_mock_columns)
    def test_render_performance_metrics_no_data(
        self, mock_cols, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        fig = viz._render_performance_metrics(None, None)
        assert _is_plotly_figure(fig)


class TestQLearningVizDemoData:
    """Tests for create_demo_qlearning_data utility."""

    def test_create_demo_data(self) -> None:
        from gui.qlearning_viz import create_demo_qlearning_data

        q_table, history, policy = create_demo_qlearning_data()
        assert isinstance(q_table, np.ndarray)
        assert q_table.shape == (20, 4)
        assert isinstance(history, dict)
        assert "reward" in history
        assert "epsilon" in history
        assert "loss" in history
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (20,)


class TestQLearningVizFileOps:
    """Tests for file I/O utilities in qlearning_viz."""

    @patch("streamlit.error")
    def test_load_qtable_nonexistent_file(self, mock_err: MagicMock) -> None:
        from gui.qlearning_viz import load_qtable_from_file

        result = load_qtable_from_file("/nonexistent/path.pkl")
        assert result is None

    @patch("streamlit.error")
    def test_load_qtable_npy(self, mock_err: MagicMock) -> None:
        from gui.qlearning_viz import load_qtable_from_file

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            arr = np.random.randn(5, 3)
            np.save(f.name, arr)
            result = load_qtable_from_file(f.name)
        assert result is not None
        os.unlink(f.name)

    @patch("streamlit.error")
    def test_load_qtable_json(self, mock_err: MagicMock) -> None:
        from gui.qlearning_viz import load_qtable_from_file

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"q_table": [[1.0, 2.0], [3.0, 4.0]]}, f)
            fname = f.name
        result = load_qtable_from_file(fname)
        assert result is not None
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))
        os.unlink(fname)

    @patch("streamlit.success")
    def test_save_visualization_config(self, mock_succ: MagicMock) -> None:
        from gui.qlearning_viz import (
            QLearningVisualizationConfig,
            save_visualization_config,
        )

        config = QLearningVisualizationConfig()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        save_visualization_config(config, fname)
        with open(fname) as fh:
            saved = json.load(fh)
        assert saved["colormap"] == "RdYlBu_r"
        assert saved["show_values"] is True
        os.unlink(fname)

    @patch("streamlit.error")
    def test_save_visualization_config_bad_path(
        self, mock_err: MagicMock
    ) -> None:
        from gui.qlearning_viz import (
            QLearningVisualizationConfig,
            save_visualization_config,
        )

        config = QLearningVisualizationConfig()
        save_visualization_config(config, "/nonexistent/dir/file.json")
        mock_err.assert_called_once()


# ===================================================================
# 5. QTABLE_VISUALIZATION (heavy Streamlit + analysis dependency)
# ===================================================================


class TestQTableVisualizationComponent:
    """Tests for gui/qtable_visualization.py.

    These mock the analysis module and Streamlit to exercise internal logic.
    """

    def _make_mock_modules(self) -> dict[str, MagicMock]:
        """Create mock modules for qtable_visualization imports."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_available_qtables.return_value = []
        mock_analyzer.load_qtable.return_value = np.random.randn(5, 3)

        mock_module = MagicMock()
        mock_module.QTABLE_ANALYZER = mock_analyzer
        mock_module.ConvergenceStatus = MagicMock()
        mock_module.ConvergenceStatus.CONVERGED = MagicMock()
        mock_module.ConvergenceStatus.CONVERGED.value = "converged"
        mock_module.QTableMetrics = MagicMock()
        return {"analysis.qtable_analyzer": mock_module}

    def test_create_qtable_heatmap(self) -> None:
        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            with patch("streamlit.session_state") as mock_ss:
                mock_ss.selected_qtables = []
                mock_ss.qtable_analysis_cache = {}
                mock_ss.comparison_results = {}
                mock_ss.__contains__ = lambda self, k: True
                comp = QTableVisualization()
                q = np.random.randn(8, 4)
                fig = comp._create_qtable_heatmap(q, "test_qtable.pkl")
                assert _is_plotly_figure(fig)

    def test_create_qtable_heatmap_show_values(self) -> None:
        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            with patch("streamlit.session_state") as mock_ss:
                mock_ss.selected_qtables = []
                mock_ss.qtable_analysis_cache = {}
                mock_ss.comparison_results = {}
                mock_ss.__contains__ = lambda self, k: True
                comp = QTableVisualization()
                q = np.random.randn(8, 4)
                fig = comp._create_qtable_heatmap(
                    q, "test.pkl", colorscale="plasma", show_values=True
                )
                assert _is_plotly_figure(fig)

    def test_get_file_size(self) -> None:
        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization
            from pathlib import Path

            with patch("streamlit.session_state") as mock_ss:
                mock_ss.selected_qtables = []
                mock_ss.qtable_analysis_cache = {}
                mock_ss.comparison_results = {}
                mock_ss.__contains__ = lambda self, k: True
                comp = QTableVisualization()

                # Test with a real file
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(b"x" * 1024)
                    fname = f.name
                size_str = comp._get_file_size(Path(fname))
                assert "KB" in size_str or "B" in size_str
                os.unlink(fname)

                # Test with nonexistent file
                size_str = comp._get_file_size(Path("/nonexistent/file.pkl"))
                assert size_str == "Unknown"

    def test_extract_timestamp_from_filename(self) -> None:
        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            with patch("streamlit.session_state") as mock_ss:
                mock_ss.selected_qtables = []
                mock_ss.qtable_analysis_cache = {}
                mock_ss.comparison_results = {}
                mock_ss.__contains__ = lambda self, k: True
                comp = QTableVisualization()

                ts = comp._extract_timestamp_from_filename(
                    "qtable_20250731_143022.pkl"
                )
                assert ts == "20250731_143022"

                ts = comp._extract_timestamp_from_filename("qtable_latest.pkl")
                assert ts == "qtable_latest.pkl"

    def test_create_convergence_trend_plot(self) -> None:
        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            with patch("streamlit.session_state") as mock_ss:
                mock_ss.selected_qtables = []
                mock_ss.qtable_analysis_cache = {}
                mock_ss.comparison_results = {}
                mock_ss.__contains__ = lambda self, k: True
                comp = QTableVisualization()

                df = pd.DataFrame(
                    {
                        "file": ["a.pkl", "b.pkl", "c.pkl"],
                        "timestamp": ["t1", "t2", "t3"],
                        "convergence_score": [0.5, 0.7, 0.9],
                        "stability_measure": [0.3, 0.5, 0.8],
                        "exploration_coverage": [0.8, 0.6, 0.4],
                        "policy_entropy": [1.2, 0.8, 0.3],
                    }
                )
                fig = comp._create_convergence_trend_plot(df)
                assert _is_plotly_figure(fig)


# ===================================================================
# 6. POLICY_EVOLUTION_VIZ (heavy dependencies)
# ===================================================================


class TestPolicyEvolutionVizComponent:
    """Tests for gui/policy_evolution_viz.py.

    These mock the analysis.policy_evolution_tracker module and streamlit
    to exercise the class methods.
    """

    def _make_mock_modules(self) -> dict[str, MagicMock]:
        """Create mock modules needed by policy_evolution_viz."""
        mock_stability = MagicMock(spec=Enum)
        mock_stability.STABLE = MagicMock()
        mock_stability.STABLE.value = "stable"
        mock_stability.CONVERGING = MagicMock()
        mock_stability.CONVERGING.value = "converging"
        mock_stability.UNSTABLE = MagicMock()
        mock_stability.UNSTABLE.value = "unstable"
        mock_stability.OSCILLATING = MagicMock()
        mock_stability.OSCILLATING.value = "oscillating"
        mock_stability.UNKNOWN = MagicMock()
        mock_stability.UNKNOWN.value = "unknown"

        mock_metrics_class = MagicMock()
        mock_tracker_instance = MagicMock()
        mock_tracker_instance.policy_snapshots = []

        mock_module = MagicMock()
        mock_module.POLICY_EVOLUTION_TRACKER = mock_tracker_instance
        mock_module.PolicyEvolutionMetrics = mock_metrics_class
        mock_module.PolicyStability = mock_stability

        return {"analysis.policy_evolution_tracker": mock_module}

    @patch("streamlit.session_state")
    @patch("streamlit.header")
    @patch("streamlit.markdown")
    @patch("streamlit.subheader")
    @patch("streamlit.info")
    def test_instantiation(
        self, mock_info, mock_sub, mock_md, mock_header, mock_ss
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            assert comp.tracker is not None


# ===================================================================
# 7. PARAMETER_INPUT (heavy dependencies)
# ===================================================================


class TestParameterInputComponent:
    """Tests for gui/parameter_input.py.

    Mocks all config.* dependencies and streamlit to test the component.
    """

    def _make_mock_config_modules(self) -> dict[str, MagicMock]:
        """Create mock modules for parameter_input imports."""
        # --- literature_database ---
        mock_param_category = MagicMock(spec=Enum)
        mock_param_category.return_value = MagicMock()

        mock_param_info = MagicMock()

        mock_lit_db = MagicMock()
        mock_lit_db.get_all_categories.return_value = []
        mock_lit_db.get_parameter.return_value = None

        mock_lit_module = MagicMock()
        mock_lit_module.LITERATURE_DB = mock_lit_db
        mock_lit_module.ParameterCategory = mock_param_category
        mock_lit_module.ParameterInfo = mock_param_info

        # --- parameter_bridge ---
        mock_bridge = MagicMock()
        mock_bridge.create_literature_validated_config.return_value = (
            MagicMock(),
            {},
        )
        mock_bridge_module = MagicMock()
        mock_bridge_module.PARAMETER_BRIDGE = mock_bridge

        # --- real_time_validator ---
        mock_validation_level = MagicMock(spec=Enum)
        mock_validation_level.VALID = MagicMock()
        mock_validation_level.VALID.value = "valid"
        mock_validation_level.CAUTION = MagicMock()
        mock_validation_level.CAUTION.value = "caution"
        mock_validation_level.INVALID = MagicMock()
        mock_validation_level.INVALID.value = "invalid"

        mock_validator = MagicMock()
        mock_validator.get_research_objectives.return_value = []
        mock_validator.get_performance_metrics.return_value = {
            "avg_response_time_ms": 50.0,
            "cache_hit_rate": 0.9,
            "fast_validations": 90,
            "instant_validations": 80,
            "total_validations": 100,
        }

        mock_rtv_module = MagicMock()
        mock_rtv_module.REAL_TIME_VALIDATOR = mock_validator
        mock_rtv_module.ValidationLevel = mock_validation_level

        # --- unit_converter ---
        mock_converter = MagicMock()
        mock_converter.get_compatible_units.return_value = ["V"]
        mock_uc_module = MagicMock()
        mock_uc_module.UNIT_CONVERTER = mock_converter

        # --- qlearning_config ---
        mock_ql_config = MagicMock()

        return {
            "config.literature_database": mock_lit_module,
            "config.parameter_bridge": mock_bridge_module,
            "config.real_time_validator": mock_rtv_module,
            "config.unit_converter": mock_uc_module,
            "config.qlearning_config": mock_ql_config,
        }

    @patch("streamlit.session_state")
    def test_instantiation(self, mock_ss: MagicMock) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        with patch.dict(sys.modules, self._make_mock_config_modules()):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            assert comp.literature_db is not None

    @patch("streamlit.session_state")
    def test_create_parameter_range_visualization(
        self, mock_ss: MagicMock
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {"test_param": 0.5}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()

        # Set up mock parameter
        mock_param = MagicMock()
        mock_param.name = "test_param"
        mock_param.symbol = "Tp"
        mock_param.unit = "V"
        mock_param.min_value = 0.0
        mock_param.max_value = 1.0
        mock_param.recommended_range = (0.3, 0.7)
        mock_param.typical_value = 0.5
        mock_param.references = []
        mock_param.notes = ""

        mocks[
            "config.literature_database"
        ].LITERATURE_DB.get_parameter.return_value = mock_param
        mocks[
            "config.literature_database"
        ].LITERATURE_DB.validate_parameter_value.return_value = {
            "status": "valid",
            "message": "OK",
        }

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            fig = comp.create_parameter_range_visualization("test_param")
            assert _is_plotly_figure(fig)

    @patch("streamlit.session_state")
    def test_create_parameter_range_visualization_unknown(
        self, mock_ss: MagicMock
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        mocks[
            "config.literature_database"
        ].LITERATURE_DB.get_parameter.return_value = None

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            result = comp.create_parameter_range_visualization("nonexistent")
            assert result is None

    @patch("streamlit.session_state")
    @patch("streamlit.metric")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.subheader")
    @patch("streamlit.success")
    @patch("streamlit.warning")
    @patch("streamlit.error")
    def test_render_performance_metrics(
        self,
        mock_err,
        mock_warn,
        mock_succ,
        mock_sub,
        mock_cols,
        mock_metric,
        mock_ss,
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        with patch.dict(sys.modules, self._make_mock_config_modules()):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            comp._render_performance_metrics()

    @patch("streamlit.session_state")
    def test_render_validation_indicator_valid(
        self, mock_ss: MagicMock
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        with patch.dict(sys.modules, self._make_mock_config_modules()):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            with patch("streamlit.success") as mock_s, patch(
                "streamlit.caption"
            ):
                comp._render_validation_indicator(
                    {
                        "status": "valid",
                        "message": "OK",
                        "recommendations": [],
                    }
                )
                mock_s.assert_called_once()

    @patch("streamlit.session_state")
    def test_render_validation_indicator_caution(
        self, mock_ss: MagicMock
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        with patch.dict(sys.modules, self._make_mock_config_modules()):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            with patch("streamlit.warning") as mock_w, patch(
                "streamlit.caption"
            ):
                comp._render_validation_indicator(
                    {
                        "status": "caution",
                        "message": "Out of range",
                        "recommendations": ["Adjust value"],
                    }
                )
                mock_w.assert_called_once()

    @patch("streamlit.session_state")
    def test_render_validation_indicator_invalid(
        self, mock_ss: MagicMock
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        with patch.dict(sys.modules, self._make_mock_config_modules()):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            with patch("streamlit.error") as mock_e, patch(
                "streamlit.caption"
            ):
                comp._render_validation_indicator(
                    {
                        "status": "invalid",
                        "message": "Invalid",
                        "recommendations": ["Fix it"],
                    }
                )
                mock_e.assert_called_once()

    @patch("streamlit.session_state")
    def test_render_validation_indicator_unknown(
        self, mock_ss: MagicMock
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        with patch.dict(sys.modules, self._make_mock_config_modules()):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            with patch("streamlit.info") as mock_i, patch(
                "streamlit.caption"
            ):
                comp._render_validation_indicator(
                    {
                        "status": "unknown",
                        "message": "No data",
                        "recommendations": [],
                    }
                )
                mock_i.assert_called_once()


# ===================================================================
# 8. QLEARNING_VIZ - Dashboard rendering (full path)
# ===================================================================


class TestQLearningVisualizerDashboard:
    """Tests for the full render_qlearning_dashboard method."""

    @patch("streamlit.markdown")
    @patch(
        "streamlit.tabs",
        return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()],
    )
    @patch("streamlit.info")
    @patch("streamlit.columns", side_effect=_mock_columns)
    def test_dashboard_no_data(
        self, mock_cols, mock_info, mock_tabs, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        figures = viz.render_qlearning_dashboard()
        assert isinstance(figures, dict)

    @patch("streamlit.markdown")
    @patch(
        "streamlit.tabs",
        return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()],
    )
    @patch("streamlit.info")
    @patch("streamlit.plotly_chart")
    @patch("streamlit.metric")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.write")
    @patch("streamlit.success")
    @patch("streamlit.warning")
    def test_dashboard_with_all_data(
        self,
        mock_warn,
        mock_succ,
        mock_write,
        mock_cols,
        mock_metric,
        mock_chart,
        mock_info,
        mock_tabs,
        mock_md,
    ) -> None:
        from gui.qlearning_viz import (
            QLearningVisualizer,
            create_demo_qlearning_data,
        )

        viz = QLearningVisualizer()
        q_table, history, policy = create_demo_qlearning_data()
        figures = viz.render_qlearning_dashboard(
            q_table=q_table,
            training_history=history,
            current_policy=policy,
            title="Test Dashboard",
        )
        assert isinstance(figures, dict)

    @patch("streamlit.markdown")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.write")
    def test_display_learning_statistics(
        self, mock_write, mock_cols, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        history = {
            "reward": list(np.linspace(0, 10, 50)),
            "loss": list(np.linspace(5, 0.1, 50)),
        }
        viz._display_learning_statistics(history)

    @patch("streamlit.markdown")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.write")
    def test_display_qtable_insights(
        self, mock_write, mock_cols, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.array(
            [
                [1.0, 0.5, 0.1],
                [0.2, 1.0, 0.3],
                [0.1, 0.2, 1.0],
            ]
        )
        viz._display_qtable_insights(q)

    @patch("streamlit.markdown")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.write")
    @patch("streamlit.success")
    @patch("streamlit.warning")
    def test_display_policy_analysis(
        self, mock_warn, mock_succ, mock_write, mock_cols, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.random.randn(10, 4)
        policy = np.argmax(q, axis=1).astype(np.int64)
        viz._display_policy_analysis(policy, q)

    @patch("streamlit.markdown")
    @patch("streamlit.columns", side_effect=_mock_columns)
    @patch("streamlit.write")
    def test_display_policy_analysis_no_qtable(
        self, mock_write, mock_cols, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        policy = np.array([0, 1, 2, 0, 1])
        viz._display_policy_analysis(policy, None)

    @patch("streamlit.markdown")
    @patch("streamlit.columns", side_effect=_mock_columns)
    def test_render_performance_trends(self, mock_cols, mock_md) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        history = {
            "reward": list(np.linspace(0, 10, 50)),
        }
        viz._render_performance_trends(history)

    @patch("streamlit.markdown")
    def test_render_performance_recommendations_low_convergence(
        self, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        # Q-table with near-identical values -> low convergence
        q = np.ones((10, 4)) + np.random.randn(10, 4) * 0.001
        viz._render_performance_recommendations(q, None)

    @patch("streamlit.markdown")
    def test_render_performance_recommendations_stagnation(
        self, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        q = np.random.randn(10, 4) * 2
        history = {
            "reward": [1.0] * 30,  # Very low variance -> stagnation
        }
        viz._render_performance_recommendations(q, history)

    @patch("streamlit.markdown")
    def test_render_performance_recommendations_no_data(
        self, mock_md
    ) -> None:
        from gui.qlearning_viz import QLearningVisualizer

        viz = QLearningVisualizer()
        viz._render_performance_recommendations(None, None)
