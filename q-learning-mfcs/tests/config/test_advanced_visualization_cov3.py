"""Tests for config/advanced_visualization.py - coverage target 98%+."""
import importlib
import sys
import os
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, _src)

# Mock seaborn before importing the module so HAS_MATPLOTLIB=True
sys.modules["seaborn"] = MagicMock()

# Force (re)load the module with seaborn mocked
import config.advanced_visualization as _av_mod
importlib.reload(_av_mod)

from config.advanced_visualization import (
    PlotType,
    InteractionMode,
    PlotConfiguration,
    VisualizationResult,
    MultiDimensionalPlotter,
    InteractiveAnalyzer,
    StatisticalVisualizer,
    ComparisonAnalyzer,
    create_sensitivity_tornado_plot,
    create_pareto_frontier_plot,
    HAS_MATPLOTLIB,
)
from config.visualization_config import VisualizationConfig


def _make_config():
    cfg = VisualizationConfig()
    # Source references plot_style.font_size which PlotStyleConfig lacks
    if not hasattr(cfg.plot_style, "font_size"):
        cfg.plot_style.font_size = 12.0
    return cfg


def _make_plot_config(plot_type=PlotType.SCATTER_3D, title="Test"):
    return PlotConfiguration(plot_type=plot_type, title=title)


@pytest.mark.coverage_extra
class TestEnums:
    def test_plot_type(self):
        assert PlotType.SCATTER_3D.value == "scatter_3d"
        assert PlotType.HEATMAP.value == "heatmap"
        assert PlotType.PARETO_FRONTIER.value == "pareto_frontier"

    def test_interaction_mode(self):
        assert InteractionMode.STATIC.value == "static"
        assert InteractionMode.REAL_TIME.value == "real_time"


@pytest.mark.coverage_extra
class TestPlotConfiguration:
    def test_defaults(self):
        pc = PlotConfiguration(plot_type=PlotType.SCATTER_3D, title="T")
        assert pc.alpha == 0.7
        assert pc.dpi == 300
        assert pc.save_format == "png"

    def test_custom(self):
        pc = PlotConfiguration(
            plot_type=PlotType.HEATMAP,
            title="Custom",
            color_map="coolwarm",
            alpha=0.5,
        )
        assert pc.color_map == "coolwarm"


@pytest.mark.coverage_extra
class TestVisualizationResult:
    def test_defaults(self):
        vr = VisualizationResult(plot_type=PlotType.SCATTER_3D)
        assert vr.figure_path is None
        assert vr.data_points == 0


@pytest.mark.coverage_extra
class TestMultiDimensionalPlotter:
    @pytest.fixture
    def plotter(self):
        return MultiDimensionalPlotter(_make_config())

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "x": np.random.normal(0, 1, 30),
            "y": np.random.normal(0, 1, 30),
            "z": np.random.normal(0, 1, 30),
            "c": np.random.normal(0, 1, 30),
            "grp": ["A"] * 15 + ["B"] * 15,
        })

    def test_scatter_3d(self, plotter, sample_df):
        import matplotlib.pyplot as plt
        config = _make_plot_config(PlotType.SCATTER_3D)
        result = plotter.create_plot(
            sample_df, config, x_col="x", y_col="y", z_col="z",
        )
        assert result.interactive_plot is not None
        plt.close("all")

    def test_scatter_3d_with_color(self, plotter, sample_df):
        import matplotlib.pyplot as plt
        config = _make_plot_config(PlotType.SCATTER_3D)
        result = plotter.create_plot(
            sample_df, config, x_col="x", y_col="y", z_col="z",
            color_col="c",
        )
        assert result.interactive_plot is not None
        plt.close("all")

    def test_surface_3d(self, plotter, sample_df):
        import matplotlib.pyplot as plt
        config = _make_plot_config(PlotType.SURFACE_3D)
        result = plotter.create_plot(
            sample_df, config, x_col="x", y_col="y", z_col="z",
        )
        assert result.interactive_plot is not None
        plt.close("all")

    def test_radar_chart_no_group(self, plotter, sample_df):
        import matplotlib.pyplot as plt
        config = _make_plot_config(PlotType.RADAR_CHART)
        result = plotter.create_plot(
            sample_df, config, columns=["x", "y", "z"],
        )
        assert result.interactive_plot is not None
        plt.close("all")

    def test_radar_chart_with_group(self, plotter, sample_df):
        import matplotlib.pyplot as plt
        config = _make_plot_config(PlotType.RADAR_CHART)
        result = plotter.create_plot(
            sample_df, config, columns=["x", "y", "z"],
            group_column="grp",
        )
        assert result.interactive_plot is not None
        plt.close("all")

    def test_unsupported_type(self, plotter, sample_df):
        config = _make_plot_config(PlotType.HEATMAP)
        with pytest.raises(ValueError, match="Unsupported plot type"):
            plotter.create_plot(sample_df, config)

    def test_prepare_data_missing_cols(self, plotter):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Missing required"):
            plotter._prepare_data(df, ["a", "b"])

    def test_prepare_data_empty_after_clean(self, plotter):
        df = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
        with pytest.raises(ValueError, match="No valid data"):
            plotter._prepare_data(df, ["a", "b"])

    def test_apply_color_scheme_scientific(self, plotter):
        plotter.config.color_scheme_type = "scientific"
        colors = plotter._apply_color_scheme(5)
        assert len(colors) == 5

    def test_apply_color_scheme_colorblind(self, plotter):
        plotter.config.color_scheme_type = "colorblind_friendly"
        colors = plotter._apply_color_scheme(5)
        assert len(colors) == 5

    def test_apply_color_scheme_default(self, plotter):
        plotter.config.color_scheme_type = "other"
        colors = plotter._apply_color_scheme(5)
        assert len(colors) == 5

    def test_parallel_coordinates(self, plotter, sample_df):
        """Cover _create_parallel_coordinates via Plotly path."""
        config = _make_plot_config(PlotType.PARALLEL_COORDINATES)
        result = plotter.create_plot(
            sample_df, config, columns=["x", "y", "z"],
        )
        assert result.interactive_plot is not None

    def test_parallel_coords_with_class(self, plotter, sample_df):
        """Cover parallel coordinates with class column (numeric)."""
        config = _make_plot_config(PlotType.PARALLEL_COORDINATES)
        result = plotter.create_plot(
            sample_df, config, columns=["x", "y", "z"],
            class_column="c",
        )
        assert result.interactive_plot is not None

    def test_radar_chart_equal_values(self, plotter):
        """Cover radar chart when max == min (normalize to 0.5)."""
        import matplotlib.pyplot as plt
        df = pd.DataFrame({
            "a": [5.0] * 10,
            "b": [3.0] * 10,
            "c": [7.0] * 10,
        })
        config = _make_plot_config(PlotType.RADAR_CHART)
        result = plotter.create_plot(
            df, config, columns=["a", "b", "c"],
        )
        assert result.interactive_plot is not None
        plt.close("all")


@pytest.mark.coverage_extra
class TestStatisticalVisualizer:
    @pytest.fixture
    def viz(self):
        return StatisticalVisualizer(_make_config())

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 30
        return pd.DataFrame({
            "x": np.linspace(0, 10, n),
            "y": np.linspace(0, 10, n) + np.random.normal(0, 0.5, n),
            "u": np.abs(np.random.normal(0.5, 0.1, n)),
            "grp": ["A"] * (n // 2) + ["B"] * (n // 2),
            "val": np.concatenate([
                np.random.normal(5, 1, n // 2),
                np.random.normal(7, 1, n // 2),
            ]),
        })

    def test_correlation_matrix(self, viz, sample_df):
        import matplotlib.pyplot as plt
        config = _make_plot_config(PlotType.CORRELATION_MATRIX)
        result = viz.create_plot(
            sample_df, config, columns=["x", "y", "u"],
        )
        assert result.interactive_plot is not None
        plt.close("all")

    def test_distribution_comparison(self, viz, sample_df):
        import matplotlib.pyplot as plt
        config = _make_plot_config(PlotType.DISTRIBUTION_COMPARISON)
        result = viz.create_plot(
            sample_df, config, value_col="val", group_col="grp",
        )
        assert result.interactive_plot is not None
        plt.close("all")

    def test_uncertainty_bands(self, viz, sample_df):
        import matplotlib.pyplot as plt
        config = _make_plot_config(PlotType.UNCERTAINTY_BANDS)
        result = viz.create_plot(
            sample_df, config, x_col="x", y_col="y",
            uncertainty_col="u",
        )
        assert result.interactive_plot is not None
        plt.close("all")

    def test_unsupported_type(self, viz, sample_df):
        config = _make_plot_config(PlotType.HEATMAP)
        with pytest.raises(ValueError, match="Unsupported"):
            viz.create_plot(sample_df, config)


class _ConcreteComparisonAnalyzer(ComparisonAnalyzer):
    """Concrete subclass to instantiate abstract ComparisonAnalyzer."""

    def create_plot(self, data, plot_config, **kwargs):
        raise NotImplementedError


@pytest.mark.coverage_extra
class TestComparisonAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return _ConcreteComparisonAnalyzer(_make_config())

    def test_optimization_comparison(self, analyzer):
        """Cover create_optimization_comparison."""
        import matplotlib.pyplot as plt
        from config.parameter_optimization import OptimizationMethod

        result_mock1 = MagicMock()
        result_mock1.convergence_history = [1.0, 0.8, 0.5]
        result_mock1.method.value = "genetic"
        result_mock1.best_overall_score = 0.5
        result_mock1.total_evaluations = 100
        result_mock1.get_optimization_time.return_value = 2.5
        result_mock1.best_parameters = np.array([1.0, 2.0])

        result_mock2 = MagicMock()
        result_mock2.convergence_history = [1.2, 0.9, 0.6]
        result_mock2.method.value = "bayesian"
        result_mock2.best_overall_score = 0.6
        result_mock2.total_evaluations = 50
        result_mock2.get_optimization_time.return_value = 1.5
        result_mock2.best_parameters = np.array([1.5, 2.5])

        config = _make_plot_config(PlotType.DISTRIBUTION_COMPARISON)
        result = analyzer.create_optimization_comparison(
            [result_mock1, result_mock2], config,
        )
        assert result.interactive_plot is not None
        plt.close("all")

    def test_uncertainty_comparison(self, analyzer):
        """Cover create_uncertainty_comparison."""
        import matplotlib.pyplot as plt

        uq1 = MagicMock()
        uq1.method.value = "monte_carlo"
        uq1.output_mean = {"power": 0.5}
        uq1.output_std = {"power": 0.1}
        uq1.computation_time = 1.0
        uq1.n_samples = 1000
        uq1.output_samples = {"power": np.random.normal(0.5, 0.1, 100)}

        uq2 = MagicMock()
        uq2.method.value = "polynomial_chaos"
        uq2.output_mean = {"power": 0.48}
        uq2.output_std = {"power": 0.12}
        uq2.computation_time = 0.5
        uq2.n_samples = 500
        uq2.output_samples = {"power": np.random.normal(0.48, 0.12, 100)}

        config = _make_plot_config(PlotType.DISTRIBUTION_COMPARISON)
        result = analyzer.create_uncertainty_comparison(
            [uq1, uq2], config,
        )
        assert result.interactive_plot is not None
        plt.close("all")


@pytest.mark.coverage_extra
class TestInteractiveAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return InteractiveAnalyzer(_make_config())

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 20
        return pd.DataFrame({
            "x": np.random.normal(0, 1, n),
            "y": np.random.normal(0, 1, n),
            "t": np.arange(n),
            "v1": np.random.normal(0, 1, n),
            "v2": np.random.normal(0, 1, n),
            "grp": ["A"] * 10 + ["B"] * 10,
        })

    def test_interactive_scatter(self, analyzer, sample_df):
        config = _make_plot_config(PlotType.INTERACTIVE_SCATTER)
        result = analyzer.create_plot(
            sample_df, config, x_col="x", y_col="y",
        )
        assert result.interactive_plot is not None

    def test_interactive_scatter_with_extras(self, analyzer):
        """Cover color_col and size_col branches."""
        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, n),
            "y": np.random.normal(0, 1, n),
            "color": np.random.normal(0, 1, n),
            "size": np.abs(np.random.normal(5, 1, n)),
        })
        config = _make_plot_config(PlotType.INTERACTIVE_SCATTER)
        result = analyzer.create_plot(
            df, config, x_col="x", y_col="y",
            color_col="color", size_col="size",
        )
        assert result.interactive_plot is not None

    def test_animated_time_series_no_group(self, analyzer, sample_df):
        config = _make_plot_config(PlotType.ANIMATED_TIME_SERIES)
        result = analyzer.create_plot(
            sample_df, config, time_col="t",
            value_cols=["v1", "v2"],
        )
        assert result.interactive_plot is not None

    def test_animated_time_series_with_group(self, analyzer, sample_df):
        config = _make_plot_config(PlotType.ANIMATED_TIME_SERIES)
        result = analyzer.create_plot(
            sample_df, config, time_col="t",
            value_cols=["v1"], group_col="grp",
        )
        assert result.interactive_plot is not None

    def test_unsupported_type(self, analyzer, sample_df):
        config = _make_plot_config(PlotType.HEATMAP)
        with pytest.raises(ValueError, match="Unsupported"):
            analyzer.create_plot(sample_df, config)

    def test_static_fallback(self, analyzer, sample_df):
        import matplotlib.pyplot as plt
        config = _make_plot_config(PlotType.SCATTER_3D)
        result = analyzer._create_static_fallback(sample_df, config)
        assert isinstance(result, VisualizationResult)
        plt.close("all")

    def test_plotly_not_available_fallback(self, sample_df):
        """Cover line 606-607: plotly not available falls to static."""
        import matplotlib.pyplot as plt
        with patch.object(_av_mod, "HAS_PLOTLY", False):
            analyzer = InteractiveAnalyzer(_make_config())
            config = _make_plot_config(PlotType.INTERACTIVE_SCATTER)
            result = analyzer.create_plot(
                sample_df, config, x_col="x", y_col="y",
            )
            assert isinstance(result, VisualizationResult)
        plt.close("all")


@pytest.mark.coverage_extra
class TestParetoFrontierPlot:
    def test_2d_pareto(self):
        """Cover create_pareto_frontier_plot with 2 objectives."""
        import matplotlib.pyplot as plt
        opt_result = MagicMock()
        opt_result.all_objectives = [
            {"obj1": 0.1, "obj2": 0.9},
            {"obj1": 0.5, "obj2": 0.5},
            {"obj1": 0.9, "obj2": 0.1},
            {"obj1": 0.3, "obj2": 0.7},
            {"obj1": 0.7, "obj2": 0.3},
        ]
        config = _make_plot_config(PlotType.PARETO_FRONTIER)
        result = create_pareto_frontier_plot(
            [opt_result], ["obj1", "obj2"], config,
        )
        assert result.plot_type == PlotType.PARETO_FRONTIER
        assert result.data_summary["total_points"] == 5
        plt.close("all")

    def test_3d_pareto(self):
        """Cover create_pareto_frontier_plot with 3 objectives."""
        import matplotlib.pyplot as plt
        opt_result = MagicMock()
        opt_result.all_objectives = [
            {"a": 0.1, "b": 0.9, "c": 0.5},
            {"a": 0.5, "b": 0.5, "c": 0.5},
            {"a": 0.9, "b": 0.1, "c": 0.5},
        ]
        config = _make_plot_config(PlotType.PARETO_FRONTIER)
        result = create_pareto_frontier_plot(
            [opt_result], ["a", "b", "c"], config,
        )
        assert result.plot_type == PlotType.PARETO_FRONTIER
        plt.close("all")

    def test_no_valid_data(self):
        """Cover ValueError when no objectives match."""
        opt_result = MagicMock()
        opt_result.all_objectives = [{"x": 1}]
        config = _make_plot_config(PlotType.PARETO_FRONTIER)
        with pytest.raises(ValueError, match="No valid objective"):
            create_pareto_frontier_plot(
                [opt_result], ["a", "b"], config,
            )


@pytest.mark.coverage_extra
class TestNoMatplotlibFallbacks:
    """Cover HAS_MATPLOTLIB=False branches (ImportError paths)."""

    def test_3d_scatter_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            plotter = MultiDimensionalPlotter(_make_config())
            df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
            config = _make_plot_config(PlotType.SCATTER_3D)
            with pytest.raises(ImportError, match="Matplotlib required"):
                plotter.create_plot(df, config, x_col="x", y_col="y", z_col="z")

    def test_surface_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            plotter = MultiDimensionalPlotter(_make_config())
            df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
            config = _make_plot_config(PlotType.SURFACE_3D)
            with pytest.raises(ImportError, match="Matplotlib required"):
                plotter.create_plot(df, config, x_col="x", y_col="y", z_col="z")

    def test_radar_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            plotter = MultiDimensionalPlotter(_make_config())
            df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
            config = _make_plot_config(PlotType.RADAR_CHART)
            with pytest.raises(ImportError, match="Matplotlib required"):
                plotter.create_plot(df, config, columns=["x", "y", "z"])

    def test_correlation_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            viz = StatisticalVisualizer(_make_config())
            df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
            config = _make_plot_config(PlotType.CORRELATION_MATRIX)
            with pytest.raises(ImportError, match="Matplotlib required"):
                viz.create_plot(df, config, columns=["x", "y"])

    def test_distribution_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            viz = StatisticalVisualizer(_make_config())
            df = pd.DataFrame({"v": [1.0], "g": ["A"]})
            config = _make_plot_config(PlotType.DISTRIBUTION_COMPARISON)
            with pytest.raises(ImportError, match="Matplotlib required"):
                viz.create_plot(df, config, value_col="v", group_col="g")

    def test_uncertainty_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            viz = StatisticalVisualizer(_make_config())
            df = pd.DataFrame({"x": [1.0], "y": [2.0], "u": [0.1]})
            config = _make_plot_config(PlotType.UNCERTAINTY_BANDS)
            with pytest.raises(ImportError, match="Matplotlib required"):
                viz.create_plot(
                    df, config, x_col="x", y_col="y", uncertainty_col="u",
                )

    def test_static_fallback_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            analyzer = InteractiveAnalyzer(_make_config())
            df = pd.DataFrame({"x": [1], "y": [2]})
            config = _make_plot_config(PlotType.SCATTER_3D)
            with pytest.raises(ImportError, match="Neither Plotly nor"):
                analyzer._create_static_fallback(df, config)

    def test_tornado_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            config = _make_plot_config(PlotType.SENSITIVITY_TORNADO)
            with pytest.raises(ImportError, match="Matplotlib required"):
                create_sensitivity_tornado_plot({"a": 0.5}, config)

    def test_pareto_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            opt_result = MagicMock()
            opt_result.all_objectives = [{"a": 0.1, "b": 0.9}]
            config = _make_plot_config(PlotType.PARETO_FRONTIER)
            with pytest.raises(ImportError, match="Matplotlib required"):
                create_pareto_frontier_plot(
                    [opt_result], ["a", "b"], config,
                )

    def test_opt_comparison_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            analyzer = _ConcreteComparisonAnalyzer(_make_config())
            config = _make_plot_config(PlotType.DISTRIBUTION_COMPARISON)
            with pytest.raises(ImportError, match="Matplotlib required"):
                analyzer.create_optimization_comparison([], config)

    def test_uq_comparison_no_mpl(self):
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            analyzer = _ConcreteComparisonAnalyzer(_make_config())
            config = _make_plot_config(PlotType.DISTRIBUTION_COMPARISON)
            with pytest.raises(ImportError, match="Matplotlib required"):
                analyzer.create_uncertainty_comparison([], config)

    def test_parallel_no_plotly_no_mpl(self):
        with patch.object(_av_mod, "HAS_PLOTLY", False):
            with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
                plotter = MultiDimensionalPlotter(_make_config())
                df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
                config = _make_plot_config(PlotType.PARALLEL_COORDINATES)
                with pytest.raises(ImportError, match="Plotly or Matplotlib"):
                    plotter.create_plot(df, config, columns=["x", "y", "z"])

    def test_parallel_mpl_fallback(self):
        """Cover matplotlib fallback path for parallel coordinates."""
        import matplotlib.pyplot as plt
        with patch.object(_av_mod, "HAS_PLOTLY", False):
            plotter = MultiDimensionalPlotter(_make_config())
            df = pd.DataFrame({
                "x": np.random.normal(0, 1, 20),
                "y": np.random.normal(0, 1, 20),
                "z": np.random.normal(0, 1, 20),
            })
            config = _make_plot_config(PlotType.PARALLEL_COORDINATES)
            result = plotter.create_plot(
                df, config, columns=["x", "y", "z"],
            )
            assert result.interactive_plot is not None
        plt.close("all")

    def test_parallel_mpl_fallback_with_class(self):
        """Cover matplotlib fallback for parallel coords with class col."""
        import matplotlib.pyplot as plt
        with patch.object(_av_mod, "HAS_PLOTLY", False):
            plotter = MultiDimensionalPlotter(_make_config())
            df = pd.DataFrame({
                "x": np.random.normal(0, 1, 20),
                "y": np.random.normal(0, 1, 20),
                "grp": ["A"] * 10 + ["B"] * 10,
            })
            config = _make_plot_config(PlotType.PARALLEL_COORDINATES)
            result = plotter.create_plot(
                df, config, columns=["x", "y"],
                class_column="grp",
            )
            assert result.interactive_plot is not None
        plt.close("all")

    def test_color_scheme_no_mpl_scientific(self):
        """Cover fallback colors when HAS_MATPLOTLIB=False."""
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            plotter = MultiDimensionalPlotter(_make_config())
            plotter.config.color_scheme_type = "scientific"
            colors = plotter._apply_color_scheme(5)
            assert colors == ["blue", "red", "green"]

    def test_color_scheme_no_mpl_default(self):
        """Cover default fallback colors when HAS_MATPLOTLIB=False."""
        with patch.object(_av_mod, "HAS_MATPLOTLIB", False):
            plotter = MultiDimensionalPlotter(_make_config())
            plotter.config.color_scheme_type = "other"
            colors = plotter._apply_color_scheme(5)
            assert colors == ["blue"]


@pytest.mark.coverage_extra
class TestSensitivityTornadoPlot:
    def test_basic(self):
        import matplotlib.pyplot as plt
        indices = {"param_a": 0.8, "param_b": -0.3, "param_c": 0.5}
        config = _make_plot_config(PlotType.SENSITIVITY_TORNADO)
        result = create_sensitivity_tornado_plot(indices, config)
        assert result.plot_type == PlotType.SENSITIVITY_TORNADO
        assert result.data_summary["n_parameters"] == 3
        plt.close("all")

    def test_empty(self):
        import matplotlib.pyplot as plt
        indices = {}
        config = _make_plot_config(PlotType.SENSITIVITY_TORNADO)
        result = create_sensitivity_tornado_plot(indices, config)
        assert result.data_summary["most_sensitive_param"] is None
        plt.close("all")
