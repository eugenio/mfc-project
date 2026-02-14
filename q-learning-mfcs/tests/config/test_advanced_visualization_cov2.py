import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from config.advanced_visualization import (
    PlotType, InteractionMode,
    PlotConfiguration, VisualizationResult,
    InteractiveAnalyzer,
    HAS_PLOTLY, HAS_MATPLOTLIB,
)


def _make_mock_config():
    mock_config = MagicMock()
    mock_config.color_scheme_type = "colorblind_friendly"
    mock_config.plot_style.figure_width = 12
    mock_config.plot_style.figure_height = 8
    mock_config.plot_style.dpi = 100
    mock_config.plot_style.line_width = 2
    mock_config.plot_style.font_size = 12
    mock_config.plot_style.grid_enabled = True
    mock_config.plot_style.grid_alpha = 0.3
    return mock_config


def _make_interactive_analyzer():
    config = _make_mock_config()
    with patch("config.advanced_visualization.HAS_MATPLOTLIB", False):
        ia = InteractiveAnalyzer.__new__(InteractiveAnalyzer)
        ia.config = config
        ia.logger = __import__("logging").getLogger(__name__)
    return ia


@pytest.mark.coverage_extra
class TestInteractiveAnalyzerPlotly:
    @pytest.fixture
    def ia(self):
        return _make_interactive_analyzer()

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly required")
    def test_interactive_scatter(self, ia):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [0.1, 0.2, 0.3, 0.4, 0.5],
            "s": [10, 20, 30, 40, 50],
        })
        pc = PlotConfiguration(
            plot_type=PlotType.INTERACTIVE_SCATTER,
            title="Test Scatter",
        )
        result = ia.create_plot(
            df, pc,
            x_col="x", y_col="y", color_col="c", size_col="s",
        )
        assert result.plot_type == PlotType.INTERACTIVE_SCATTER
        assert result.interactive_plot is not None
        assert result.rendering_time >= 0

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly required")
    def test_interactive_scatter_no_extras(self, ia):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        pc = PlotConfiguration(
            plot_type=PlotType.INTERACTIVE_SCATTER,
            title="Simple",
        )
        result = ia.create_plot(df, pc, x_col="x", y_col="y")
        assert result.interactive_plot is not None

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly required")
    def test_animated_time_series(self, ia):
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=10, freq="h"),
            "val1": np.random.randn(10),
            "val2": np.random.randn(10),
        })
        pc = PlotConfiguration(
            plot_type=PlotType.ANIMATED_TIME_SERIES,
            title="Animated",
        )
        result = ia.create_plot(
            df, pc,
            time_col="time", value_cols=["val1", "val2"],
        )
        assert result.interactive_plot is not None

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly required")
    def test_animated_with_groups(self, ia):
        df = pd.DataFrame({
            "time": list(pd.date_range("2024-01-01", periods=5, freq="h")) * 2,
            "val": np.random.randn(10),
            "group": ["A"] * 5 + ["B"] * 5,
        })
        pc = PlotConfiguration(
            plot_type=PlotType.ANIMATED_TIME_SERIES,
            title="Grouped",
        )
        result = ia.create_plot(
            df, pc,
            time_col="time", value_cols=["val"], group_col="group",
        )
        assert result.interactive_plot is not None

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly required")
    def test_unsupported_interactive_type(self, ia):
        df = pd.DataFrame({"x": [1]})
        pc = PlotConfiguration(
            plot_type=PlotType.HEATMAP, title="Bad"
        )
        with pytest.raises(ValueError, match="Unsupported"):
            ia.create_plot(df, pc)

    def test_static_fallback_no_plotly_no_matplotlib(self, ia):
        pc = PlotConfiguration(
            plot_type=PlotType.INTERACTIVE_SCATTER,
            title="Fallback",
        )
        with patch("config.advanced_visualization.HAS_PLOTLY", False), \
             patch("config.advanced_visualization.HAS_MATPLOTLIB", False):
            with pytest.raises(ImportError):
                ia.create_plot(pd.DataFrame({"x": [1]}), pc)


@pytest.mark.coverage_extra
class TestMultiDimensionalPlotterLogic:
    """Test logic paths in MultiDimensionalPlotter without actual plotting."""

    def test_create_plot_unsupported_type(self):
        from config.advanced_visualization import MultiDimensionalPlotter
        config = _make_mock_config()
        with patch("config.advanced_visualization.HAS_MATPLOTLIB", False):
            plotter = MultiDimensionalPlotter.__new__(MultiDimensionalPlotter)
            plotter.config = config
            plotter.logger = __import__("logging").getLogger(__name__)

        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        pc = PlotConfiguration(
            plot_type=PlotType.HEATMAP, title="Bad"
        )
        with pytest.raises(ValueError, match="Unsupported"):
            plotter.create_plot(df, pc)


@pytest.mark.coverage_extra
class TestStatisticalVisualizerLogic:
    """Test StatisticalVisualizer dispatch logic."""

    def test_unsupported_plot_type(self):
        from config.advanced_visualization import StatisticalVisualizer
        config = _make_mock_config()
        with patch("config.advanced_visualization.HAS_MATPLOTLIB", False):
            sv = StatisticalVisualizer.__new__(StatisticalVisualizer)
            sv.config = config
            sv.logger = __import__("logging").getLogger(__name__)

        df = pd.DataFrame({"x": [1]})
        pc = PlotConfiguration(
            plot_type=PlotType.HEATMAP, title="Bad"
        )
        with pytest.raises(ValueError, match="Unsupported"):
            sv.create_plot(df, pc)

    def test_correlation_matrix_no_matplotlib(self):
        from config.advanced_visualization import StatisticalVisualizer
        config = _make_mock_config()
        with patch("config.advanced_visualization.HAS_MATPLOTLIB", False):
            sv = StatisticalVisualizer.__new__(StatisticalVisualizer)
            sv.config = config
            sv.logger = __import__("logging").getLogger(__name__)

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        pc = PlotConfiguration(
            plot_type=PlotType.CORRELATION_MATRIX, title="Corr"
        )
        with pytest.raises(ImportError):
            sv.create_plot(df, pc)

    def test_distribution_comparison_no_matplotlib(self):
        from config.advanced_visualization import StatisticalVisualizer
        config = _make_mock_config()
        with patch("config.advanced_visualization.HAS_MATPLOTLIB", False):
            sv = StatisticalVisualizer.__new__(StatisticalVisualizer)
            sv.config = config
            sv.logger = __import__("logging").getLogger(__name__)

        df = pd.DataFrame({"v": [1], "g": ["a"]})
        pc = PlotConfiguration(
            plot_type=PlotType.DISTRIBUTION_COMPARISON, title="Dist"
        )
        with pytest.raises(ImportError):
            sv.create_plot(df, pc, value_col="v", group_col="g")

    def test_uncertainty_bands_no_matplotlib(self):
        from config.advanced_visualization import StatisticalVisualizer
        config = _make_mock_config()
        with patch("config.advanced_visualization.HAS_MATPLOTLIB", False):
            sv = StatisticalVisualizer.__new__(StatisticalVisualizer)
            sv.config = config
            sv.logger = __import__("logging").getLogger(__name__)

        df = pd.DataFrame({"x": [1], "y": [2], "u": [0.1]})
        pc = PlotConfiguration(
            plot_type=PlotType.UNCERTAINTY_BANDS, title="UB"
        )
        with pytest.raises(ImportError):
            sv.create_plot(
                df, pc, x_col="x", y_col="y", uncertainty_col="u"
            )
