import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from config.advanced_visualization import (
    PlotType, InteractionMode,
    PlotConfiguration, VisualizationResult,
)


class TestEnums:
    def test_plot_type(self):
        assert PlotType.SCATTER_3D.value == "scatter_3d"
        assert PlotType.SURFACE_3D.value == "surface_3d"
        assert PlotType.CONTOUR_FILLED.value == "contour_filled"
        assert PlotType.HEATMAP.value == "heatmap"
        assert PlotType.PARALLEL_COORDINATES.value == "parallel_coordinates"
        assert PlotType.RADAR_CHART.value == "radar_chart"
        assert PlotType.CORRELATION_MATRIX.value == "correlation_matrix"
        assert PlotType.DISTRIBUTION_COMPARISON.value == "distribution_comparison"
        assert PlotType.TIME_SERIES_ENSEMBLE.value == "time_series_ensemble"
        assert PlotType.UNCERTAINTY_BANDS.value == "uncertainty_bands"
        assert PlotType.PARETO_FRONTIER.value == "pareto_frontier"
        assert PlotType.SENSITIVITY_TORNADO.value == "sensitivity_tornado"
        assert PlotType.INTERACTIVE_SCATTER.value == "interactive_scatter"
        assert PlotType.ANIMATED_TIME_SERIES.value == "animated_time_series"

    def test_interaction_mode(self):
        assert InteractionMode.STATIC.value == "static"
        assert InteractionMode.ZOOM_PAN.value == "zoom_pan"
        assert InteractionMode.SELECTION.value == "selection"
        assert InteractionMode.HOVER_INFO.value == "hover_info"
        assert InteractionMode.REAL_TIME.value == "real_time"
        assert InteractionMode.PARAMETER_SWEEP.value == "parameter_sweep"


class TestPlotConfiguration:
    def test_defaults(self):
        pc = PlotConfiguration(
            plot_type=PlotType.SCATTER_3D,
            title="Test Plot",
        )
        assert pc.x_label == ""
        assert pc.y_label == ""
        assert pc.z_label == ""
        assert pc.color_map == "viridis"
        assert pc.alpha == 0.7
        assert pc.marker_size == 50.0
        assert pc.line_width == 2.0
        assert pc.figure_size == (12.0, 8.0)
        assert pc.subplot_layout is None
        assert pc.interaction_mode == InteractionMode.STATIC
        assert pc.enable_tooltips is True
        assert pc.enable_zoom is True
        assert pc.show_confidence_intervals is False
        assert pc.confidence_level == 0.95
        assert pc.show_correlation_values is True
        assert pc.save_format == "png"
        assert pc.dpi == 300
        assert pc.transparent_background is False

    def test_custom_values(self):
        pc = PlotConfiguration(
            plot_type=PlotType.HEATMAP,
            title="Custom",
            x_label="X",
            y_label="Y",
            color_map="plasma",
            alpha=0.5,
            figure_size=(10.0, 6.0),
        )
        assert pc.color_map == "plasma"
        assert pc.alpha == 0.5


class TestVisualizationResult:
    def test_defaults(self):
        vr = VisualizationResult(plot_type=PlotType.SCATTER_3D)
        assert vr.figure_path is None
        assert vr.interactive_plot is None
        assert vr.data_summary == {}
        assert vr.statistical_tests == {}
        assert vr.creation_time is not None
        assert vr.configuration is None
        assert vr.rendering_time == 0.0
        assert vr.data_points == 0

    def test_custom(self):
        pc = PlotConfiguration(
            plot_type=PlotType.HEATMAP, title="test"
        )
        vr = VisualizationResult(
            plot_type=PlotType.HEATMAP,
            configuration=pc,
            data_points=100,
            rendering_time=0.5,
            data_summary={"key": "value"},
        )
        assert vr.data_points == 100
        assert vr.rendering_time == 0.5
        assert vr.data_summary["key"] == "value"


class TestAdvancedVisualizerPrepareData:
    """Test _prepare_data method via a concrete subclass."""

    def _make_visualizer(self):
        import pandas as pd
        from config.advanced_visualization import MultiDimensionalPlotter

        mock_config = MagicMock()
        mock_config.color_scheme_type = "scientific"
        mock_config.plot_style.figure_width = 12
        mock_config.plot_style.figure_height = 8
        mock_config.plot_style.dpi = 100
        mock_config.plot_style.line_width = 2
        mock_config.plot_style.font_size = 12
        mock_config.plot_style.grid_enabled = True
        mock_config.plot_style.grid_alpha = 0.3

        with patch("config.advanced_visualization.HAS_MATPLOTLIB", False):
            viz = MultiDimensionalPlotter.__new__(MultiDimensionalPlotter)
            viz.config = mock_config
            viz.logger = __import__("logging").getLogger(__name__)
        return viz

    def test_prepare_data_valid(self):
        import pandas as pd
        viz = self._make_visualizer()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = viz._prepare_data(df, ["a", "b"])
        assert len(result) == 3

    def test_prepare_data_missing_col(self):
        import pandas as pd
        viz = self._make_visualizer()
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="Missing required"):
            viz._prepare_data(df, ["a", "x"])

    def test_prepare_data_all_nan(self):
        import pandas as pd
        viz = self._make_visualizer()
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        with pytest.raises(ValueError, match="No valid data"):
            viz._prepare_data(df, ["a", "b"])

    def test_apply_color_scheme_scientific(self):
        viz = self._make_visualizer()
        viz.config.color_scheme_type = "scientific"
        colors = viz._apply_color_scheme(5)
        assert len(colors) >= 3

    def test_apply_color_scheme_colorblind(self):
        viz = self._make_visualizer()
        viz.config.color_scheme_type = "colorblind_friendly"
        colors = viz._apply_color_scheme(5)
        assert len(colors) == 5
        assert colors[0] == "#1f77b4"

    def test_apply_color_scheme_default(self):
        viz = self._make_visualizer()
        viz.config.color_scheme_type = "other"
        colors = viz._apply_color_scheme(3)
        assert len(colors) >= 1


class TestCreateSensitivityTornadoPlot:
    def test_tornado_no_matplotlib(self):
        from config.advanced_visualization import HAS_MATPLOTLIB
        if HAS_MATPLOTLIB:
            pytest.skip("Test for no-matplotlib path")
        from config.advanced_visualization import create_sensitivity_tornado_plot
        indices = {"param_a": 0.8}
        pc = PlotConfiguration(
            plot_type=PlotType.SENSITIVITY_TORNADO,
            title="Sensitivity",
        )
        with pytest.raises(ImportError):
            create_sensitivity_tornado_plot(indices, pc)
