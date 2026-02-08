"""Tests for config/visualization_config.py - targeting 98%+ coverage."""
import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Mock matplotlib.pyplot before loading the module to avoid numpy reimport
_mock_plt = MagicMock()
_mock_matplotlib = MagicMock()
_mock_matplotlib.pyplot = _mock_plt
sys.modules.setdefault("matplotlib", _mock_matplotlib)
sys.modules.setdefault("matplotlib.pyplot", _mock_plt)

# Load the module directly via importlib to bypass config/__init__.py
_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "config.visualization_config",
    os.path.join(_src, "config", "visualization_config.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["config.visualization_config"] = _mod
_spec.loader.exec_module(_mod)

ColorScheme = _mod.ColorScheme
PlotType = _mod.PlotType
LegendPosition = _mod.LegendPosition
PlotStyleConfig = _mod.PlotStyleConfig
ColorSchemeConfig = _mod.ColorSchemeConfig
DataProcessingConfig = _mod.DataProcessingConfig
LayoutConfig = _mod.LayoutConfig
AnimationConfig = _mod.AnimationConfig
VisualizationConfig = _mod.VisualizationConfig
apply_style_config = _mod.apply_style_config
get_colorblind_friendly_config = _mod.get_colorblind_friendly_config
get_colors_for_scheme = _mod.get_colors_for_scheme
get_interactive_visualization_config = _mod.get_interactive_visualization_config
get_presentation_visualization_config = _mod.get_presentation_visualization_config
get_publication_visualization_config = _mod.get_publication_visualization_config
validate_visualization_config = _mod.validate_visualization_config


class TestEnums:
    def test_color_scheme_values(self):
        assert ColorScheme.DEFAULT.value == "default"
        assert ColorScheme.SCIENTIFIC.value == "scientific"
        assert ColorScheme.COLORBLIND_FRIENDLY.value == "colorblind_friendly"
        assert ColorScheme.HIGH_CONTRAST.value == "high_contrast"
        assert ColorScheme.GRAYSCALE.value == "grayscale"
        assert ColorScheme.PUBLICATION.value == "publication"

    def test_plot_type_values(self):
        assert PlotType.TIME_SERIES.value == "time_series"
        assert PlotType.SCATTER.value == "scatter"
        assert PlotType.HISTOGRAM.value == "histogram"
        assert PlotType.HEATMAP.value == "heatmap"
        assert PlotType.BAR.value == "bar"
        assert PlotType.BOX_PLOT.value == "box_plot"
        assert PlotType.VIOLIN_PLOT.value == "violin_plot"
        assert PlotType.CONTOUR.value == "contour"

    def test_legend_position_values(self):
        assert LegendPosition.BEST.value == "best"
        assert LegendPosition.UPPER_RIGHT.value == "upper right"
        assert LegendPosition.UPPER_LEFT.value == "upper left"
        assert LegendPosition.LOWER_LEFT.value == "lower left"
        assert LegendPosition.LOWER_RIGHT.value == "lower right"
        assert LegendPosition.RIGHT.value == "right"
        assert LegendPosition.CENTER_LEFT.value == "center left"
        assert LegendPosition.CENTER_RIGHT.value == "center right"
        assert LegendPosition.LOWER_CENTER.value == "lower center"
        assert LegendPosition.UPPER_CENTER.value == "upper center"
        assert LegendPosition.CENTER.value == "center"


class TestPlotStyleConfig:
    def test_defaults(self):
        cfg = PlotStyleConfig()
        assert cfg.figure_width == 12.0
        assert cfg.figure_height == 8.0
        assert cfg.dpi == 300
        assert cfg.line_width == 2.0
        assert cfg.marker_size == 6.0
        assert cfg.alpha == 0.8
        assert cfg.grid_enabled is True
        assert cfg.grid_alpha == 0.3
        assert cfg.grid_line_width == 0.5
        assert cfg.grid_style == "-"
        assert cfg.axis_label_size == 12.0
        assert cfg.tick_label_size == 10.0
        assert cfg.title_size == 14.0
        assert cfg.legend_font_size == 10.0
        assert cfg.subplot_spacing == 0.3
        assert cfg.figure_padding == 0.1
        assert cfg.tight_layout is True
        assert cfg.font_family == "DejaVu Sans"
        assert cfg.font_weight == "normal"
        assert cfg.error_bar_width == 1.0
        assert cfg.error_bar_cap_size == 3.0
        assert cfg.annotation_font_size == 9.0
        assert cfg.annotation_arrow_width == 1.0

    def test_custom_values(self):
        cfg = PlotStyleConfig(figure_width=20.0, dpi=150, font_family="Arial")
        assert cfg.figure_width == 20.0
        assert cfg.dpi == 150
        assert cfg.font_family == "Arial"


class TestColorSchemeConfig:
    def test_defaults(self):
        cfg = ColorSchemeConfig()
        assert len(cfg.primary_colors) == 8
        assert len(cfg.scientific_colors) == 8
        assert len(cfg.colorblind_colors) == 8
        assert len(cfg.high_contrast_colors) == 8
        assert len(cfg.grayscale_colors) == 6
        assert cfg.background_color == "#FFFFFF"
        assert cfg.grid_color == "#E0E0E0"
        assert cfg.text_color == "#000000"
        assert cfg.accent_color == "#FF6B6B"
        assert cfg.success_color == "#28A745"
        assert cfg.warning_color == "#FFC107"
        assert cfg.error_color == "#DC3545"
        assert cfg.info_color == "#17A2B8"


class TestDataProcessingConfig:
    def test_defaults(self):
        cfg = DataProcessingConfig()
        assert cfg.time_series_sampling_rate == 100
        assert cfg.scatter_sampling_rate == 1000
        assert cfg.enable_smoothing is True
        assert cfg.smoothing_window == 10
        assert cfg.smoothing_method == "moving_average"
        assert cfg.enable_outlier_detection is True
        assert cfg.outlier_threshold == 3.0
        assert cfg.outlier_handling == "remove"
        assert cfg.enable_data_aggregation is False
        assert cfg.aggregation_window == 60.0
        assert cfg.aggregation_method == "mean"
        assert cfg.interpolate_missing is True
        assert cfg.interpolation_method == "linear"
        assert cfg.max_gap_size == 10
        assert cfg.max_points_per_plot == 10000
        assert cfg.enable_downsampling is True
        assert cfg.compute_statistics is True
        assert cfg.confidence_level == 0.95
        assert cfg.enable_trend_lines is False


class TestLayoutConfig:
    def test_defaults(self):
        cfg = LayoutConfig()
        assert cfg.n_rows == 2
        assert cfg.n_cols == 2
        assert cfg.subplot_width_ratios is None
        assert cfg.subplot_height_ratios is None
        assert cfg.horizontal_spacing == 0.3
        assert cfg.vertical_spacing == 0.3
        assert cfg.left_margin == 0.1
        assert cfg.right_margin == 0.9
        assert cfg.top_margin == 0.9
        assert cfg.bottom_margin == 0.1
        assert cfg.legend_position == LegendPosition.BEST
        assert cfg.legend_columns == 1
        assert cfg.legend_frame is True
        assert cfg.legend_shadow is False
        assert cfg.main_title == ""
        assert cfg.subplot_titles == []
        assert cfg.share_x_axis is False
        assert cfg.share_y_axis is False
        assert cfg.x_scale == "linear"
        assert cfg.y_scale == "linear"
        assert cfg.auto_limits is True
        assert cfg.x_limits is None
        assert cfg.y_limits is None

    def test_custom_values(self):
        cfg = LayoutConfig(
            n_rows=3,
            n_cols=4,
            subplot_width_ratios=[1.0, 2.0, 1.0, 1.0],
            subplot_height_ratios=[1.0, 1.0, 2.0],
            legend_position=LegendPosition.UPPER_RIGHT,
            main_title="Test Plot",
            subplot_titles=["A", "B", "C"],
            x_limits=(0.0, 10.0),
            y_limits=(-5.0, 5.0),
        )
        assert cfg.n_rows == 3
        assert cfg.n_cols == 4
        assert cfg.subplot_width_ratios == [1.0, 2.0, 1.0, 1.0]
        assert cfg.subplot_height_ratios == [1.0, 1.0, 2.0]
        assert cfg.legend_position == LegendPosition.UPPER_RIGHT
        assert cfg.main_title == "Test Plot"
        assert cfg.subplot_titles == ["A", "B", "C"]
        assert cfg.x_limits == (0.0, 10.0)
        assert cfg.y_limits == (-5.0, 5.0)


class TestAnimationConfig:
    def test_defaults(self):
        cfg = AnimationConfig()
        assert cfg.enable_animation is False
        assert cfg.animation_interval == 100.0
        assert cfg.animation_frames == 100
        assert cfg.fade_effect is True
        assert cfg.trail_length == 50
        assert cfg.blit_animation is True
        assert cfg.cache_frames is True
        assert cfg.save_animation is False
        assert cfg.animation_format == "mp4"
        assert cfg.animation_fps == 10


class TestVisualizationConfig:
    def test_defaults(self):
        cfg = VisualizationConfig()
        assert isinstance(cfg.plot_style, PlotStyleConfig)
        assert isinstance(cfg.color_scheme, ColorSchemeConfig)
        assert isinstance(cfg.data_processing, DataProcessingConfig)
        assert isinstance(cfg.layout, LayoutConfig)
        assert isinstance(cfg.animation, AnimationConfig)
        assert cfg.color_scheme_type == ColorScheme.SCIENTIFIC
        assert cfg.output_format == "png"
        assert cfg.save_figures is True
        assert cfg.show_figures is True
        assert cfg.output_directory == "figures"
        assert cfg.filename_prefix == "mfc"
        assert cfg.include_timestamp is True
        assert cfg.enable_zooming is True
        assert cfg.enable_panning is True
        assert cfg.enable_tooltips is False
        assert cfg.vector_graphics is True
        assert cfg.high_dpi is True
        assert cfg.lazy_loading is True
        assert cfg.memory_limit_mb == 1000
        assert cfg.configuration_name == "default_visualization"
        assert cfg.configuration_version == "1.0.0"
        assert cfg.description == "Default visualization configuration"


class TestFactoryFunctions:
    def test_publication_config(self):
        cfg = get_publication_visualization_config()
        assert cfg.plot_style.dpi == 600
        assert cfg.plot_style.font_family == "Times New Roman"
        assert cfg.plot_style.line_width == 1.5
        assert cfg.color_scheme_type == ColorScheme.SCIENTIFIC
        assert cfg.output_format == "pdf"
        assert cfg.vector_graphics is True
        assert cfg.layout.legend_frame is False
        assert cfg.plot_style.grid_alpha == 0.2
        assert cfg.configuration_name == "publication_visualization"
        assert cfg.description == "Publication-ready visualization settings"

    def test_presentation_config(self):
        cfg = get_presentation_visualization_config()
        assert cfg.plot_style.axis_label_size == 16.0
        assert cfg.plot_style.tick_label_size == 14.0
        assert cfg.plot_style.title_size == 18.0
        assert cfg.plot_style.legend_font_size == 14.0
        assert cfg.color_scheme_type == ColorScheme.HIGH_CONTRAST
        assert cfg.plot_style.line_width == 3.0
        assert cfg.plot_style.marker_size == 8.0
        assert cfg.plot_style.figure_width == 16.0
        assert cfg.plot_style.figure_height == 10.0
        assert cfg.configuration_name == "presentation_visualization"
        assert cfg.description == "Presentation visualization settings with high contrast"

    def test_interactive_config(self):
        cfg = get_interactive_visualization_config()
        assert cfg.enable_zooming is True
        assert cfg.enable_panning is True
        assert cfg.enable_tooltips is True
        assert cfg.animation.enable_animation is True
        assert cfg.animation.animation_interval == 50.0
        assert cfg.data_processing.enable_downsampling is True
        assert cfg.data_processing.max_points_per_plot == 5000
        assert cfg.configuration_name == "interactive_visualization"
        assert cfg.description == "Interactive visualization with animation support"

    def test_colorblind_friendly_config(self):
        cfg = get_colorblind_friendly_config()
        assert cfg.color_scheme_type == ColorScheme.COLORBLIND_FRIENDLY
        assert cfg.plot_style.line_width == 2.5
        assert cfg.plot_style.alpha == 1.0
        assert cfg.plot_style.marker_size == 7.0
        assert cfg.configuration_name == "colorblind_friendly_visualization"
        assert cfg.description == "Colorblind-friendly visualization configuration"


class TestGetColorsForScheme:
    def test_default_scheme(self):
        cc = ColorSchemeConfig()
        colors = get_colors_for_scheme(ColorScheme.DEFAULT, cc)
        assert colors == cc.primary_colors

    def test_scientific_scheme(self):
        cc = ColorSchemeConfig()
        colors = get_colors_for_scheme(ColorScheme.SCIENTIFIC, cc)
        assert colors == cc.scientific_colors

    def test_colorblind_scheme(self):
        cc = ColorSchemeConfig()
        colors = get_colors_for_scheme(ColorScheme.COLORBLIND_FRIENDLY, cc)
        assert colors == cc.colorblind_colors

    def test_high_contrast_scheme(self):
        cc = ColorSchemeConfig()
        colors = get_colors_for_scheme(ColorScheme.HIGH_CONTRAST, cc)
        assert colors == cc.high_contrast_colors

    def test_grayscale_scheme(self):
        cc = ColorSchemeConfig()
        colors = get_colors_for_scheme(ColorScheme.GRAYSCALE, cc)
        assert colors == cc.grayscale_colors

    def test_publication_scheme(self):
        cc = ColorSchemeConfig()
        colors = get_colors_for_scheme(ColorScheme.PUBLICATION, cc)
        assert colors == cc.scientific_colors


class TestApplyStyleConfig:
    def test_apply_style(self):
        _mock_plt.rcParams = {}
        cfg = PlotStyleConfig(figure_width=10.0, dpi=150, font_family="Arial")
        apply_style_config(cfg)
        assert _mock_plt.rcParams["figure.dpi"] == 150
        assert _mock_plt.rcParams["font.family"] == "Arial"
        assert _mock_plt.rcParams["figure.figsize"] == (10.0, 8.0)
        assert _mock_plt.rcParams["lines.linewidth"] == 2.0
        assert _mock_plt.rcParams["lines.markersize"] == 6.0


class TestValidateVisualizationConfig:
    def test_valid_config(self):
        cfg = VisualizationConfig()
        assert validate_visualization_config(cfg) is True

    def test_invalid_figure_width(self):
        cfg = VisualizationConfig()
        cfg.plot_style.figure_width = -1
        with pytest.raises(ValueError, match="Figure dimensions"):
            validate_visualization_config(cfg)

    def test_invalid_figure_height(self):
        cfg = VisualizationConfig()
        cfg.plot_style.figure_height = 0
        with pytest.raises(ValueError, match="Figure dimensions"):
            validate_visualization_config(cfg)

    def test_invalid_dpi(self):
        cfg = VisualizationConfig()
        cfg.plot_style.dpi = -10
        with pytest.raises(ValueError, match="DPI"):
            validate_visualization_config(cfg)

    def test_invalid_alpha_above_one(self):
        cfg = VisualizationConfig()
        cfg.plot_style.alpha = 1.5
        with pytest.raises(ValueError, match="Alpha"):
            validate_visualization_config(cfg)

    def test_invalid_alpha_below_zero(self):
        cfg = VisualizationConfig()
        cfg.plot_style.alpha = -0.1
        with pytest.raises(ValueError, match="Alpha"):
            validate_visualization_config(cfg)

    def test_invalid_rows(self):
        cfg = VisualizationConfig()
        cfg.layout.n_rows = 0
        with pytest.raises(ValueError, match="rows and columns"):
            validate_visualization_config(cfg)

    def test_invalid_cols(self):
        cfg = VisualizationConfig()
        cfg.layout.n_cols = -1
        with pytest.raises(ValueError, match="rows and columns"):
            validate_visualization_config(cfg)

    def test_invalid_horizontal_spacing_above_one(self):
        cfg = VisualizationConfig()
        cfg.layout.horizontal_spacing = 2.0
        with pytest.raises(ValueError, match="Horizontal spacing"):
            validate_visualization_config(cfg)

    def test_invalid_horizontal_spacing_below_zero(self):
        cfg = VisualizationConfig()
        cfg.layout.horizontal_spacing = -0.1
        with pytest.raises(ValueError, match="Horizontal spacing"):
            validate_visualization_config(cfg)

    def test_invalid_vertical_spacing(self):
        cfg = VisualizationConfig()
        cfg.layout.vertical_spacing = -0.1
        with pytest.raises(ValueError, match="Vertical spacing"):
            validate_visualization_config(cfg)

    def test_invalid_vertical_spacing_above_one(self):
        cfg = VisualizationConfig()
        cfg.layout.vertical_spacing = 1.5
        with pytest.raises(ValueError, match="Vertical spacing"):
            validate_visualization_config(cfg)

    def test_invalid_sampling_rate(self):
        cfg = VisualizationConfig()
        cfg.data_processing.time_series_sampling_rate = 0
        with pytest.raises(ValueError, match="Sampling rate"):
            validate_visualization_config(cfg)

    def test_invalid_outlier_threshold(self):
        cfg = VisualizationConfig()
        cfg.data_processing.outlier_threshold = -1.0
        with pytest.raises(ValueError, match="Outlier threshold"):
            validate_visualization_config(cfg)

    def test_invalid_confidence_level_above_one(self):
        cfg = VisualizationConfig()
        cfg.data_processing.confidence_level = 1.5
        with pytest.raises(ValueError, match="Confidence level"):
            validate_visualization_config(cfg)

    def test_invalid_confidence_level_at_zero(self):
        cfg = VisualizationConfig()
        cfg.data_processing.confidence_level = 0.0
        with pytest.raises(ValueError, match="Confidence level"):
            validate_visualization_config(cfg)

    def test_invalid_animation_interval(self):
        cfg = VisualizationConfig()
        cfg.animation.animation_interval = -10
        with pytest.raises(ValueError, match="Animation interval"):
            validate_visualization_config(cfg)

    def test_invalid_animation_frames(self):
        cfg = VisualizationConfig()
        cfg.animation.animation_frames = 0
        with pytest.raises(ValueError, match="animation frames"):
            validate_visualization_config(cfg)
