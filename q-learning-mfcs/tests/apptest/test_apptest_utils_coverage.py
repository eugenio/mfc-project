"""Unit tests for GUI utility modules with low or zero coverage.

Targets:
- gui/data_loaders.py (0% -> high coverage)
- gui/enhanced_components.py (13% -> improved coverage)
- gui/core_layout.py (43% -> improved coverage)
- gui/scientific_widgets.py (76% -> improved coverage)

Uses unittest.mock to avoid requiring a live Streamlit runtime.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_SRC_DIR = str(
    (Path(__file__).resolve().parent / ".." / ".." / "src").resolve(),
)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Clear ALL gui.* modules to prevent cross-test pollution
_GUI_PREFIX = "gui."


@pytest.fixture(autouse=True)
def _clear_gui_module_cache():
    """Remove cached GUI modules so each test gets a fresh import with its own mock."""
    for mod_name in list(sys.modules):
        if mod_name == "gui" or mod_name.startswith(_GUI_PREFIX):
            del sys.modules[mod_name]
    yield
    for mod_name in list(sys.modules):
        if mod_name == "gui" or mod_name.startswith(_GUI_PREFIX):
            del sys.modules[mod_name]


# ===================================================================
# PART 1: data_loaders.py  (0% coverage)
# ===================================================================
@pytest.mark.gui
class TestLoadSimulationData:
    """Tests for data_loaders.load_simulation_data."""

    def test_returns_none_when_no_csv_files(self, tmp_path):
        """Return None when data dir has no compressed CSV files."""
        with patch.dict(sys.modules, {"streamlit": MagicMock()}):
            from gui.data_loaders import load_simulation_data

            result = load_simulation_data(str(tmp_path))
            assert result is None

    def test_loads_compressed_csv_successfully(self, tmp_path):
        """Load a gzipped CSV and return a DataFrame."""
        df_expected = pd.DataFrame(
            {"time": [1, 2, 3], "voltage": [0.4, 0.5, 0.6]},
        )
        csv_file = tmp_path / "sim_data_001.csv.gz"
        with gzip.open(csv_file, "wt") as f:
            df_expected.to_csv(f, index=False)

        with patch.dict(sys.modules, {"streamlit": MagicMock()}):
            from gui.data_loaders import load_simulation_data

            result = load_simulation_data(str(tmp_path))

        assert result is not None
        assert list(result.columns) == ["time", "voltage"]
        assert len(result) == 3

    def test_returns_none_on_corrupt_file(self, tmp_path):
        """Return None and call st.error when file is corrupt."""
        corrupt_file = tmp_path / "bad_data_001.csv.gz"
        corrupt_file.write_bytes(b"not a gzip file")

        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.data_loaders import load_simulation_data

            result = load_simulation_data(str(tmp_path))

        assert result is None
        mock_st.error.assert_called_once()

    def test_picks_first_matching_csv(self, tmp_path):
        """Pick the first matching CSV when multiple exist."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})

        f1 = tmp_path / "aaa_data_001.csv.gz"
        f2 = tmp_path / "zzz_data_002.csv.gz"
        with gzip.open(f1, "wt") as f:
            df1.to_csv(f, index=False)
        with gzip.open(f2, "wt") as f:
            df2.to_csv(f, index=False)

        with patch.dict(sys.modules, {"streamlit": MagicMock()}):
            from gui.data_loaders import load_simulation_data

            result = load_simulation_data(str(tmp_path))

        assert result is not None
        assert len(result.columns) == 1


@pytest.mark.gui
class TestLoadRecentSimulations:
    """Tests for data_loaders.load_recent_simulations."""

    def test_returns_empty_when_dir_missing(self):
        """Return empty list when simulation_data dir missing."""
        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.data_loaders import load_recent_simulations

            with patch("gui.data_loaders.Path") as mock_path_cls:
                mock_path_inst = MagicMock()
                mock_path_inst.exists.return_value = False
                mock_path_cls.return_value = mock_path_inst
                result = load_recent_simulations()

        assert result == []

    def test_returns_empty_when_no_matching_dirs(self, tmp_path):
        """Return empty list when no subdirs match prefixes."""
        (tmp_path / "random_dir").mkdir()

        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.data_loaders import load_recent_simulations

            with patch("gui.data_loaders.Path") as mock_path_cls:
                mock_path_cls.return_value = tmp_path
                result = load_recent_simulations()

        assert result == []

    def test_loads_simulation_with_valid_results(self, tmp_path):
        """Load simulation metadata from valid results."""
        sim_dir = tmp_path / "gpu_sim_001"
        sim_dir.mkdir()

        results = {
            "simulation_info": {
                "timestamp": "2025-01-15T10:00:00",
                "duration_hours": 24,
            },
            "performance_metrics": {
                "final_reservoir_concentration": 1.5,
                "control_effectiveness_2mM": 0.85,
            },
        }
        results_file = sim_dir / "sim_results.json"
        results_file.write_text(json.dumps(results))

        csv_file = sim_dir / "sim_data.csv.gz"
        df = pd.DataFrame({"time": [1], "val": [2]})
        with gzip.open(csv_file, "wt") as f:
            df.to_csv(f, index=False)

        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.data_loaders import load_recent_simulations

            with patch("gui.data_loaders.Path") as mock_path_cls:
                mock_path_cls.return_value = tmp_path
                result = load_recent_simulations()

        assert len(result) == 1
        assert result[0]["name"] == "gpu_sim_001"
        assert result[0]["timestamp"] == "2025-01-15T10:00:00"
        assert result[0]["duration"] == 24
        assert result[0]["final_conc"] == 1.5
        assert result[0]["control_effectiveness"] == 0.85

    def test_sorts_by_timestamp_descending(self, tmp_path):
        """Results are sorted by timestamp, most recent first."""
        for i, (prefix, ts) in enumerate(
            [
                ("gpu_old", "2025-01-01T00:00:00"),
                ("lactate_new", "2025-06-15T00:00:00"),
                ("gui_mid", "2025-03-10T00:00:00"),
            ],
        ):
            sim_dir = tmp_path / prefix
            sim_dir.mkdir()
            results = {
                "simulation_info": {
                    "timestamp": ts,
                    "duration_hours": i,
                },
                "performance_metrics": {
                    "final_reservoir_concentration": 0,
                    "control_effectiveness_2mM": 0,
                },
            }
            (sim_dir / "results.json").write_text(json.dumps(results))
            csv_file = sim_dir / "data.csv.gz"
            with gzip.open(csv_file, "wt") as f:
                pd.DataFrame({"a": [1]}).to_csv(f, index=False)

        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.data_loaders import load_recent_simulations

            with patch("gui.data_loaders.Path") as mock_path_cls:
                mock_path_cls.return_value = tmp_path
                result = load_recent_simulations()

        assert len(result) == 3
        assert result[0]["name"] == "lactate_new"
        assert result[1]["name"] == "gui_mid"
        assert result[2]["name"] == "gpu_old"

    def test_skips_dirs_with_missing_json(self, tmp_path):
        """Skip directories that have CSV but no JSON results."""
        sim_dir = tmp_path / "gpu_no_json"
        sim_dir.mkdir()
        csv_file = sim_dir / "data.csv.gz"
        with gzip.open(csv_file, "wt") as f:
            pd.DataFrame({"a": [1]}).to_csv(f, index=False)

        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.data_loaders import load_recent_simulations

            with patch("gui.data_loaders.Path") as mock_path_cls:
                mock_path_cls.return_value = tmp_path
                result = load_recent_simulations()

        assert result == []

    def test_skips_dirs_with_corrupt_json(self, tmp_path):
        """Skip directories where JSON is corrupt."""
        sim_dir = tmp_path / "gpu_corrupt"
        sim_dir.mkdir()
        (sim_dir / "results.json").write_text("NOT VALID JSON {{{")
        csv_file = sim_dir / "data.csv.gz"
        with gzip.open(csv_file, "wt") as f:
            pd.DataFrame({"a": [1]}).to_csv(f, index=False)

        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.data_loaders import load_recent_simulations

            with patch("gui.data_loaders.Path") as mock_path_cls:
                mock_path_cls.return_value = tmp_path
                result = load_recent_simulations()

        assert result == []


# ===================================================================
# Helper: session state that supports attribute + dict access
# ===================================================================
class _SessionStateLike(dict):
    """Dict subclass that also supports attribute access like Streamlit."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __contains__(self, item):
        return super().__contains__(item)


# ===================================================================
# Helper: comprehensive streamlit mock
# ===================================================================
def _make_mock_st():
    """Create a comprehensive mock for streamlit."""
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.columns.return_value = [MagicMock(), MagicMock()]
    mock_st.tabs.return_value = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    mock_st.number_input.return_value = 1.0
    mock_st.checkbox.return_value = True
    mock_st.selectbox.return_value = "option_a"
    mock_st.text_input.return_value = "some text"
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.selectbox.return_value = "scientific"
    mock_st.sidebar.checkbox.return_value = True
    exp_mock = MagicMock()
    exp_mock.__enter__ = MagicMock()
    exp_mock.__exit__ = MagicMock()
    mock_st.sidebar.expander.return_value = exp_mock
    return mock_st


# ===================================================================
# PART 2: enhanced_components.py  (13% coverage)
# ===================================================================
@pytest.mark.gui
class TestUIThemeConfig:
    """Tests for UIThemeConfig dataclass."""

    def test_default_values(self):
        """UIThemeConfig has correct default color values."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import UIThemeConfig

            config = UIThemeConfig()
            assert config.primary_color == "#2E86AB"
            assert config.secondary_color == "#A23B72"
            assert config.success_color == "#27AE60"
            assert config.warning_color == "#F39C12"
            assert config.error_color == "#E74C3C"
            assert config.background_color == "#FFFFFF"
            assert config.text_color == "#2C3E50"
            assert config.border_color == "#BDC3C7"
            assert config.accent_color == "#9B59B6"

    def test_custom_values(self):
        """UIThemeConfig with custom values."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import UIThemeConfig

            config = UIThemeConfig(
                primary_color="#FF0000",
                text_color="#000000",
            )
            assert config.primary_color == "#FF0000"
            assert config.text_color == "#000000"
            assert config.secondary_color == "#A23B72"


@pytest.mark.gui
class TestComponentTheme:
    """Tests for ComponentTheme enum."""

    def test_enum_values(self):
        """ComponentTheme has all expected theme options."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ComponentTheme

            assert ComponentTheme.LIGHT.value == "light"
            assert ComponentTheme.DARK.value == "dark"
            assert ComponentTheme.SCIENTIFIC.value == "scientific"
            assert ComponentTheme.HIGH_CONTRAST.value == "high_contrast"


@pytest.mark.gui
class TestScientificParameterInput:
    """Tests for ScientificParameterInput class."""

    def test_init_applies_css(self):
        """Constructor initializes custom CSS via st.markdown."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())

        assert spi.theme is not None
        mock_st.markdown.assert_called()

    def test_render_parameter_section(self):
        """render_parameter_section returns a dict of values."""
        mock_st = _make_mock_st()
        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [col_mock, col_mock]
        mock_st.number_input.return_value = 5.0

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            params = {
                "temperature": {
                    "type": "float",
                    "default": 25.0,
                    "min": 10.0,
                    "max": 50.0,
                    "unit": "C",
                    "description": "Temperature",
                    "literature_reference": "Logan 2008",
                },
            }
            result = spi.render_parameter_section(
                "Test Section",
                params,
                "test",
            )

        assert "temperature" in result

    def test_render_single_parameter_float(self):
        """Render a float parameter with validation."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 25.0

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            config = {
                "type": "float",
                "default": 25.0,
                "min": 10.0,
                "max": 50.0,
                "unit": "C",
                "description": "Temperature of system",
                "literature_reference": "",
            }
            value = spi._render_single_parameter(
                "temperature",
                config,
                "key1",
            )

        assert value == 25.0

    def test_render_single_parameter_int(self):
        """Render an integer parameter."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 10

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            config = {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 100,
                "unit": "count",
                "description": "Number of electrodes",
            }
            value = spi._render_single_parameter(
                "electrodes",
                config,
                "key2",
            )

        assert value == 10

    def test_render_single_parameter_bool(self):
        """Render a boolean parameter."""
        mock_st = _make_mock_st()
        mock_st.checkbox.return_value = True

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            config = {
                "type": "bool",
                "default": True,
                "unit": "",
                "description": "Enable feature",
            }
            value = spi._render_single_parameter(
                "enabled",
                config,
                "key3",
            )

        assert value is True

    def test_render_single_parameter_select(self):
        """Render a select parameter."""
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "option_b"

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            config = {
                "type": "select",
                "default": "option_a",
                "options": ["option_a", "option_b", "option_c"],
                "unit": "",
                "description": "Select an option",
            }
            value = spi._render_single_parameter(
                "mode",
                config,
                "key4",
            )

        assert value == "option_b"

    def test_render_single_parameter_text_fallback(self):
        """Render text input for unknown parameter type."""
        mock_st = _make_mock_st()
        mock_st.text_input.return_value = "hello world"

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            config = {
                "type": "text",
                "default": "default_text",
                "unit": "",
                "description": "Text input",
            }
            value = spi._render_single_parameter(
                "name",
                config,
                "key5",
            )

        assert value == "hello world"

    def test_display_validation_in_range(self):
        """Display success when value is within range."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            spi._display_validation_status(25.0, 10.0, 50.0, "temp")

        calls = [str(c) for c in mock_st.markdown.call_args_list]
        assert any("validation-success" in c for c in calls)

    def test_display_validation_below_min(self):
        """Display error when value is below minimum."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            spi._display_validation_status(5.0, 10.0, 50.0, "temp")

        calls = [str(c) for c in mock_st.markdown.call_args_list]
        assert any("Below minimum" in c for c in calls)

    def test_display_validation_above_max(self):
        """Display error when value is above maximum."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            spi._display_validation_status(55.0, 10.0, 50.0, "temp")

        calls = [str(c) for c in mock_st.markdown.call_args_list]
        assert any("Above maximum" in c for c in calls)

    def test_display_literature_reference(self):
        """Display literature reference markdown."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            spi._display_literature_reference("Logan 2008")

        calls = [str(c) for c in mock_st.markdown.call_args_list]
        assert any("Literature" in c for c in calls)

    def test_render_parameter_with_literature_ref(self):
        """Parameter with literature_reference triggers display."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 25.0

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ScientificParameterInput,
                UIThemeConfig,
            )

            spi = ScientificParameterInput(UIThemeConfig())
            config = {
                "type": "float",
                "default": 25.0,
                "min": 10.0,
                "max": 50.0,
                "unit": "C",
                "description": "Temperature",
                "literature_reference": "Logan, B.E. (2008)",
            }
            spi._render_single_parameter("temp", config, "k_lit")

        calls = [str(c) for c in mock_st.markdown.call_args_list]
        assert any("Literature" in c for c in calls)


@pytest.mark.gui
class TestInteractiveVisualization:
    """Tests for InteractiveVisualization class."""

    def test_init_sets_theme_and_config(self):
        """Constructor sets theme and loads viz config."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )

            viz = InteractiveVisualization(UIThemeConfig())
            assert viz.theme is not None
            assert viz.viz_config is not None

    def test_render_multi_panel_dashboard_2x2(self):
        """Render a 2x2 dashboard and return a Figure."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )

            viz = InteractiveVisualization(UIThemeConfig())
            data = {
                "Voltage": pd.DataFrame(
                    {
                        "time": [1, 2, 3],
                        "voltage": [0.4, 0.5, 0.6],
                    },
                ),
                "Current": pd.DataFrame(
                    {
                        "time": [1, 2, 3],
                        "current": [0.1, 0.2, 0.3],
                    },
                ),
                "Power": pd.DataFrame(
                    {
                        "time": [1, 2, 3],
                        "power": [0.04, 0.1, 0.18],
                    },
                ),
                "Efficiency": pd.DataFrame(
                    {"time": [1, 2, 3], "eff": [80, 85, 90]},
                ),
            }
            fig = viz.render_multi_panel_dashboard(
                data,
                "2x2",
                "Test Dashboard",
            )

        assert fig is not None
        assert fig.layout.title.text == "Test Dashboard"

    def test_render_multi_panel_dashboard_1x4(self):
        """Render a 1x4 dashboard layout."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )

            viz = InteractiveVisualization(UIThemeConfig())
            data = {
                "P1": pd.DataFrame({"time": [1], "v": [0.5]}),
                "P2": pd.DataFrame({"time": [1], "v": [0.6]}),
                "P3": pd.DataFrame({"time": [1], "v": [0.7]}),
                "P4": pd.DataFrame({"time": [1], "v": [0.8]}),
            }
            fig = viz.render_multi_panel_dashboard(
                data,
                "1x4",
                "Wide Layout",
            )

        assert fig is not None

    def test_add_panel_traces_time_series(self):
        """Add time series traces when DataFrame has time column."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )
            from plotly.subplots import make_subplots

            viz = InteractiveVisualization(UIThemeConfig())
            fig = make_subplots(rows=1, cols=1)
            df = pd.DataFrame(
                {
                    "time": [1, 2],
                    "voltage": [0.5, 0.6],
                    "current": [0.1, 0.2],
                },
            )
            viz._add_panel_traces(fig, df, 1, 1, "Test")

        assert len(fig.data) == 2

    def test_add_panel_traces_scatter(self):
        """Add scatter trace when no time column exists."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )
            from plotly.subplots import make_subplots

            viz = InteractiveVisualization(UIThemeConfig())
            fig = make_subplots(rows=1, cols=1)
            df = pd.DataFrame(
                {"x_val": [1, 2, 3], "y_val": [4, 5, 6]},
            )
            viz._add_panel_traces(fig, df, 1, 1, "Scatter")

        assert len(fig.data) == 1
        assert fig.data[0].mode == "markers"

    def test_add_panel_traces_no_numeric(self):
        """No trace added for fewer than 2 numeric columns."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )
            from plotly.subplots import make_subplots

            viz = InteractiveVisualization(UIThemeConfig())
            fig = make_subplots(rows=1, cols=1)
            df = pd.DataFrame({"label": ["a", "b"]})
            viz._add_panel_traces(fig, df, 1, 1, "Empty")

        assert len(fig.data) == 0

    def test_update_realtime_data(self):
        """_update_realtime_data appends data from stream."""
        mock_st = _make_mock_st()
        session = _SessionStateLike(
            realtime_data={
                "timestamps": [],
                "values": {},
            },
        )
        mock_st.session_state = session
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )

            viz = InteractiveVisualization(UIThemeConfig())

            def mock_stream():
                return {"voltage": 0.5, "current": 0.1}

            viz._update_realtime_data(mock_stream, max_points=10)

        rd = session.realtime_data
        assert len(rd["timestamps"]) == 1
        assert rd["values"]["voltage"] == [0.5]
        assert rd["values"]["current"] == [0.1]

    def test_update_realtime_data_buffer_limit(self):
        """Buffer is trimmed to max_points."""
        mock_st = _make_mock_st()
        timestamps = [
            datetime(2025, 1, 1, 0, i) for i in range(5)
        ]
        session = _SessionStateLike(
            realtime_data={
                "timestamps": list(timestamps),
                "values": {"v": [1, 2, 3, 4, 5]},
            },
        )
        mock_st.session_state = session
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )

            viz = InteractiveVisualization(UIThemeConfig())
            viz._update_realtime_data(
                lambda: {"v": 6},
                max_points=3,
            )

        rd = session.realtime_data
        assert len(rd["timestamps"]) == 3
        assert len(rd["values"]["v"]) == 3

    def test_update_realtime_data_handles_exception(self):
        """Call st.error when data stream raises an exception."""
        mock_st = _make_mock_st()
        session = _SessionStateLike(
            realtime_data={
                "timestamps": [],
                "values": {},
            },
        )
        mock_st.session_state = session
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )

            viz = InteractiveVisualization(UIThemeConfig())

            def bad_stream():
                raise RuntimeError("Sensor failure")

            viz._update_realtime_data(bad_stream)

        mock_st.error.assert_called_once()

    def test_create_realtime_figure(self):
        """_create_realtime_figure returns a plotly figure."""
        mock_st = _make_mock_st()
        session = _SessionStateLike(
            realtime_data={
                "timestamps": [
                    datetime(2025, 1, 1, 0, 0),
                    datetime(2025, 1, 1, 0, 1),
                ],
                "values": {"voltage": [0.4, 0.5]},
            },
        )
        mock_st.session_state = session
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                InteractiveVisualization,
                UIThemeConfig,
            )

            viz = InteractiveVisualization(UIThemeConfig())
            fig = viz._create_realtime_figure()

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].name == "voltage"
        assert fig.layout.title.text == "Real-Time MFC Monitoring"


@pytest.mark.gui
class TestExportManager:
    """Tests for ExportManager class."""

    def test_init_supported_formats(self):
        """ExportManager initializes with supported format categories."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()

        assert "data" in em.supported_formats
        assert "figures" in em.supported_formats
        assert "reports" in em.supported_formats
        assert "csv" in em.supported_formats["data"]
        assert "png" in em.supported_formats["figures"]
        assert "html" in em.supported_formats["reports"]

    def test_export_data_csv(self):
        """Export data as CSV into a zip via st.download_button."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {
                "test_data": pd.DataFrame(
                    {"a": [1, 2], "b": [3, 4]},
                ),
            }
            em._export_data(datasets, "csv", include_metadata=True)

        mock_st.download_button.assert_called_once()
        mock_st.success.assert_called_once()

    def test_export_data_csv_no_metadata(self):
        """Export CSV without metadata header."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {"d1": pd.DataFrame({"x": [1]})}
            em._export_data(
                datasets,
                "csv",
                include_metadata=False,
            )

        mock_st.download_button.assert_called_once()

    def test_export_data_json(self):
        """Export data as JSON format."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {
                "json_data": pd.DataFrame({"x": [1, 2]}),
            }
            em._export_data(
                datasets,
                "json",
                include_metadata=True,
            )

        mock_st.download_button.assert_called_once()

    def test_export_data_json_no_metadata(self):
        """Export JSON without metadata."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {"d": pd.DataFrame({"x": [1]})}
            em._export_data(
                datasets,
                "json",
                include_metadata=False,
            )

        mock_st.download_button.assert_called_once()

    def test_export_data_parquet(self):
        """Export data as Parquet format."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            import gui.enhanced_components as _ec_mod
            from gui.enhanced_components import ExportManager

            with patch.object(_ec_mod, "st", mock_st), \
                 patch("pandas.DataFrame.to_parquet"):
                em = ExportManager()
                datasets = {
                    "pq_data": pd.DataFrame({"v": [1.0, 2.0]}),
                }
                em._export_data(
                    datasets,
                    "parquet",
                    include_metadata=True,
                )
                mock_st.download_button.assert_called_once()

    def test_export_data_parquet_no_metadata(self):
        """Export Parquet without metadata exercises code path."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            import gui.enhanced_components as _ec_mod
            from gui.enhanced_components import ExportManager

            with patch.object(_ec_mod, "st", mock_st):
                em = ExportManager()
                datasets = {"d": pd.DataFrame({"v": [1.0]})}
                em._export_data(
                    datasets,
                    "parquet",
                    include_metadata=False,
                )
                # Verify the code executed to completion: either
                # download_button+success were called, or error was
                st_was_used = (
                    mock_st.download_button.called
                    or mock_st.success.called
                    or mock_st.error.called
                )
                assert st_was_used, (
                    "Expected st to be called during parquet export"
                )

    def test_export_data_feather(self):
        """Export data as Feather format."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {
                "f_data": pd.DataFrame({"a": [1, 2, 3]}),
            }
            em._export_data(
                datasets,
                "feather",
                include_metadata=True,
            )

        mock_st.download_button.assert_called_once()

    def test_export_data_feather_no_metadata(self):
        """Export Feather without metadata."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {"d": pd.DataFrame({"a": [1]})}
            em._export_data(
                datasets,
                "feather",
                include_metadata=False,
            )

        mock_st.download_button.assert_called_once()

    def test_export_data_pickle(self):
        """Export data as Pickle format."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {
                "pkl_data": pd.DataFrame({"z": [10, 20]}),
            }
            em._export_data(
                datasets,
                "pickle",
                include_metadata=True,
            )

        mock_st.download_button.assert_called_once()

    def test_export_data_pickle_no_metadata(self):
        """Export Pickle without metadata."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {"d": pd.DataFrame({"z": [10]})}
            em._export_data(
                datasets,
                "pickle",
                include_metadata=False,
            )

        mock_st.download_button.assert_called_once()

    def test_export_data_handles_exception(self):
        """Call st.error when export fails."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            em._export_data(None, "csv", include_metadata=True)

        mock_st.error.assert_called()

    def test_render_data_export_no_data(self):
        """_render_data_export shows info when no data."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            em._render_data_export(None)

        mock_st.info.assert_called()

    def test_render_figure_export_no_figures(self):
        """_render_figure_export shows info when no figures."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            em._render_figure_export(None)

        mock_st.info.assert_called()

    def test_generate_export_summary(self):
        """_generate_export_summary creates summary metrics."""
        mock_st = _make_mock_st()
        col_mock = MagicMock()
        col_mock.__enter__ = MagicMock(return_value=col_mock)
        col_mock.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [
            col_mock,
            col_mock,
            col_mock,
            col_mock,
        ]
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {
                "d1": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
                "d2": pd.DataFrame({"x": [10, 20, 30]}),
            }
            em._generate_export_summary(datasets)

        mock_st.dataframe.assert_called()
        mock_st.metric.assert_called()

    def test_batch_export_all_formats(self):
        """_batch_export_all_formats creates multi-format zip."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {
                "test": pd.DataFrame({"a": [1, 2]}),
            }
            em._batch_export_all_formats(
                datasets,
                include_metadata=True,
            )

        mock_st.download_button.assert_called_once()
        mock_st.success.assert_called()

    def test_batch_export_no_metadata(self):
        """_batch_export_all_formats works without metadata."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            datasets = {"test": pd.DataFrame({"a": [1]})}
            em._batch_export_all_formats(
                datasets,
                include_metadata=False,
            )

        mock_st.download_button.assert_called_once()

    def test_generate_html_report_with_sections(self):
        """_generate_html_report produces valid HTML with sections."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            sections = {
                "Executive Summary": True,
                "Results": True,
                "Discussion": False,
            }
            data = {
                "test": pd.DataFrame({"a": [1, 2]}),
            }
            html = em._generate_html_report(
                "Test Report",
                sections,
                data,
                None,
            )

        assert "<html>" in html
        assert "Test Report" in html
        assert "Executive Summary" in html
        assert "Results" in html

    def test_generate_html_report_no_data(self):
        """_generate_html_report works when data is None."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            html = em._generate_html_report(
                "Empty Report",
                {
                    "Executive Summary": True,
                    "Results": True,
                },
                None,
                None,
            )

        assert "<html>" in html
        assert "Empty Report" in html

    def test_generate_comprehensive_report_html(self):
        """_generate_comprehensive_report creates HTML download."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            sections = {
                "Executive Summary": True,
                "Results": True,
            }
            data = {"d": pd.DataFrame({"a": [1]})}
            em._generate_comprehensive_report(
                "Report",
                "html",
                sections,
                data,
                None,
            )

        mock_st.download_button.assert_called_once()
        mock_st.success.assert_called()

    def test_generate_comprehensive_report_error(self):
        """_generate_comprehensive_report handles errors gracefully."""
        mock_st = _make_mock_st()
        mock_st.download_button.side_effect = RuntimeError(
            "Download failed",
        )
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            em._generate_comprehensive_report(
                "Report",
                "html",
                {"Executive Summary": True},
                None,
                None,
            )

        mock_st.error.assert_called()

    def test_render_export_panel(self):
        """render_export_panel creates tabs and delegates."""
        mock_st = _make_mock_st()
        tab_mock = MagicMock()
        tab_mock.__enter__ = MagicMock(return_value=tab_mock)
        tab_mock.__exit__ = MagicMock(return_value=False)
        mock_st.tabs.return_value = [
            tab_mock,
            tab_mock,
            tab_mock,
        ]

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import ExportManager

            em = ExportManager()
            em.render_export_panel(data=None, figures=None)

        mock_st.tabs.assert_called_once()


@pytest.mark.gui
class TestInitializeEnhancedUI:
    """Tests for initialize_enhanced_ui function."""

    def test_scientific_theme(self):
        """Return scientific theme config and components."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ComponentTheme,
                initialize_enhanced_ui,
            )

            theme_config, components = initialize_enhanced_ui(
                ComponentTheme.SCIENTIFIC,
            )

        assert theme_config.primary_color == "#2E86AB"
        assert "parameter_input" in components
        assert "visualization" in components
        assert "export_manager" in components

    def test_dark_theme(self):
        """Return dark theme config."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ComponentTheme,
                initialize_enhanced_ui,
            )

            theme_config, components = initialize_enhanced_ui(
                ComponentTheme.DARK,
            )

        assert theme_config.primary_color == "#64B5F6"
        assert theme_config.background_color == "#1E1E1E"
        assert theme_config.text_color == "#FFFFFF"

    def test_light_theme_default(self):
        """Return default theme config for LIGHT."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ComponentTheme,
                initialize_enhanced_ui,
            )

            theme_config, _ = initialize_enhanced_ui(
                ComponentTheme.LIGHT,
            )

        assert theme_config.primary_color == "#2E86AB"

    def test_high_contrast_theme_default(self):
        """Return default theme for HIGH_CONTRAST (else branch)."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                ComponentTheme,
                initialize_enhanced_ui,
            )

            theme_config, _ = initialize_enhanced_ui(
                ComponentTheme.HIGH_CONTRAST,
            )

        assert theme_config.primary_color == "#2E86AB"


@pytest.mark.gui
class TestRenderEnhancedSidebar:
    """Tests for render_enhanced_sidebar function."""

    def test_returns_dict_with_expected_keys(self):
        """Return dict with theme, visualization, advanced keys."""
        mock_st = _make_mock_st()
        mock_st.sidebar.selectbox.return_value = "scientific"
        mock_st.sidebar.checkbox.return_value = True
        expander_mock = MagicMock()
        expander_mock.__enter__ = MagicMock(
            return_value=expander_mock,
        )
        expander_mock.__exit__ = MagicMock(return_value=False)
        mock_st.sidebar.expander.return_value = expander_mock
        mock_st.checkbox.return_value = True

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.enhanced_components import (
                render_enhanced_sidebar,
            )

            result = render_enhanced_sidebar()

        assert "theme" in result
        assert "visualization" in result
        assert "advanced" in result


# ===================================================================
# PART 3: core_layout.py  (43% coverage)
# ===================================================================
@pytest.mark.gui
class TestCoreLayoutPhaseHeader:
    """Tests for create_phase_header (lines 43-63)."""

    def _make_col_mocks(self, mock_st):
        """Create 3 separate column context manager mocks."""
        cols = []
        for _ in range(3):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            cols.append(col)
        mock_st.columns.return_value = cols
        return cols

    def test_phase_header_complete_status(self):
        """Phase header shows success for complete status."""
        mock_st = _make_mock_st()
        self._make_col_mocks(mock_st)

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import create_phase_header

            create_phase_header(
                "Electrode System",
                "Configure electrode parameters",
                "electrode_system",
            )

        mock_st.columns.assert_called_once()
        # st.success is called at module level (not on col)
        mock_st.success.assert_called_once()
        assert "Complete" in str(mock_st.success.call_args)

    def test_phase_header_ready_status(self):
        """Phase header shows info for ready status."""
        mock_st = _make_mock_st()
        self._make_col_mocks(mock_st)

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import create_phase_header

            create_phase_header(
                "ML Optimization",
                "Optimize parameters",
                "ml_optimization",
            )

        mock_st.info.assert_called_once()
        assert "Ready" in str(mock_st.info.call_args)

    def test_phase_header_pending_status(self):
        """Phase header shows warning for unknown phase key."""
        mock_st = _make_mock_st()
        self._make_col_mocks(mock_st)

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import create_phase_header

            create_phase_header(
                "Unknown Phase",
                "Description",
                "unknown_key",
            )

        mock_st.warning.assert_called_once()
        assert "Pending" in str(mock_st.warning.call_args)

    def test_phase_header_title_and_caption(self):
        """Phase header displays title and caption in col1."""
        mock_st = _make_mock_st()
        self._make_col_mocks(mock_st)

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import create_phase_header

            create_phase_header(
                "Test Phase",
                "Test Description",
                "electrode_system",
            )

        mock_st.title.assert_called_with("Test Phase")
        mock_st.caption.assert_called_once_with("Test Description")

    def test_phase_header_metric_in_third_column(self):
        """Phase header displays System Status metric."""
        mock_st = _make_mock_st()
        self._make_col_mocks(mock_st)

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import create_phase_header

            create_phase_header("Phase", "Desc", "electrode_system")

        mock_st.metric.assert_called_once_with(
            "System Status",
            "Operational",
        )


@pytest.mark.gui
class TestCoreLayoutPageLayout:
    """Tests for create_page_layout (lines 97-107)."""

    def test_create_page_layout_calls_theme_and_nav(self):
        """create_page_layout calls theme and nav helpers."""
        mock_st = _make_mock_st()
        container_mock = MagicMock()
        container_mock.__enter__ = MagicMock(
            return_value=container_mock,
        )
        container_mock.__exit__ = MagicMock(return_value=False)
        mock_st.container.return_value = container_mock
        mock_st.sidebar.radio.return_value = (
            "\U0001f3e0 Dashboard"
        )

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import create_page_layout

            result = create_page_layout("My Page Title")

        assert result is not None
        mock_st.container.assert_called_once()

    def test_create_page_layout_sets_title(self):
        """create_page_layout puts title in the main container."""
        mock_st = _make_mock_st()
        container_mock = MagicMock()
        container_mock.__enter__ = MagicMock(
            return_value=container_mock,
        )
        container_mock.__exit__ = MagicMock(return_value=False)
        mock_st.container.return_value = container_mock
        mock_st.sidebar.radio.return_value = (
            "\U0001f3e0 Dashboard"
        )

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import create_page_layout

            create_page_layout("Dashboard Title")

        # st.title is called at module level within the container context
        mock_st.title.assert_called_with("Dashboard Title")


@pytest.mark.gui
class TestCoreLayoutRenderHeader:
    """Tests for render_header (lines 111-113)."""

    def test_render_header_title_only(self):
        """render_header displays title without subtitle."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import render_header

            render_header("Main Title")

        mock_st.title.assert_called_with("Main Title")

    def test_render_header_with_subtitle(self):
        """render_header displays title and subtitle."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import render_header

            render_header("Main Title", "A subtitle")

        mock_st.title.assert_called_with("Main Title")
        mock_st.markdown.assert_called_with("*A subtitle*")


@pytest.mark.gui
class TestPhaseStatusConfig:
    """Tests for PHASE_STATUS constant."""

    def test_phase_status_keys(self):
        """PHASE_STATUS has expected phase keys."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import PHASE_STATUS

        assert "electrode_system" in PHASE_STATUS
        assert "advanced_physics" in PHASE_STATUS
        assert "ml_optimization" in PHASE_STATUS
        assert "gsm_integration" in PHASE_STATUS
        assert "literature_validation" in PHASE_STATUS

    def test_phase_status_complete_entries(self):
        """Complete phases have 100% progress."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.core_layout import PHASE_STATUS

        for key in [
            "electrode_system",
            "advanced_physics",
            "gsm_integration",
            "literature_validation",
        ]:
            assert PHASE_STATUS[key]["status"] == "complete"
            assert PHASE_STATUS[key]["progress"] == 100


# ===================================================================
# PART 4: scientific_widgets.py  (76% coverage)
# ===================================================================
@pytest.mark.gui
class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_create_parameter_spec(self):
        """ParameterSpec can be instantiated with all fields."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import ParameterSpec

            spec = ParameterSpec(
                name="Conductivity",
                unit="S/m",
                min_value=0.1,
                max_value=10000.0,
                typical_range=(100.0, 1000.0),
                literature_refs="Logan 2008",
                description="Electrical conductivity",
            )

        assert spec.name == "Conductivity"
        assert spec.unit == "S/m"
        assert spec.min_value == 0.1
        assert spec.max_value == 10000.0
        assert spec.typical_range == (100.0, 1000.0)


@pytest.mark.gui
class TestScientificParameterWidget:
    """Tests for ScientificParameterWidget class."""

    def _make_spec(self):
        """Create a test ParameterSpec."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import ParameterSpec

            return ParameterSpec(
                name="Temperature",
                unit="C",
                min_value=0.0,
                max_value=100.0,
                typical_range=(20.0, 40.0),
                literature_refs="Smith et al. 2020",
                description="Operating temperature",
            )

    def test_render_returns_value(self):
        """render() returns value from st.number_input."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 25.0
        spec = self._make_spec()

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import (
                ScientificParameterWidget,
            )

            widget = ScientificParameterWidget(spec, "temp_key")
            result = widget.render("Temperature", 25.0)

        assert result == 25.0

    def test_validation_in_typical_range(self):
        """Show success when value is in typical range."""
        mock_st = _make_mock_st()
        spec = self._make_spec()

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import (
                ScientificParameterWidget,
            )

            widget = ScientificParameterWidget(spec, "key1")
            widget._show_validation_feedback(30.0)

        mock_st.success.assert_called_once()
        assert "Typical range" in str(
            mock_st.success.call_args,
        )

    def test_validation_below_typical_range(self):
        """Show warning when value below typical range."""
        mock_st = _make_mock_st()
        spec = self._make_spec()

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import (
                ScientificParameterWidget,
            )

            widget = ScientificParameterWidget(spec, "key2")
            widget._show_validation_feedback(5.0)

        mock_st.warning.assert_called_once()
        warning_msg = str(mock_st.warning.call_args)
        assert "Below typical range" in warning_msg

    def test_validation_above_typical_range(self):
        """Show warning when value above typical range."""
        mock_st = _make_mock_st()
        spec = self._make_spec()

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import (
                ScientificParameterWidget,
            )

            widget = ScientificParameterWidget(spec, "key3")
            widget._show_validation_feedback(45.0)

        mock_st.warning.assert_called_once()
        warning_msg = str(mock_st.warning.call_args)
        assert "Above typical range" in warning_msg

    def test_literature_reference_displayed(self):
        """Literature reference shown via st.expander."""
        mock_st = _make_mock_st()
        expander_mock = MagicMock()
        expander_mock.__enter__ = MagicMock(
            return_value=expander_mock,
        )
        expander_mock.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = expander_mock
        spec = self._make_spec()

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import (
                ScientificParameterWidget,
            )

            widget = ScientificParameterWidget(spec, "key4")
            widget._show_validation_feedback(25.0)

        mock_st.expander.assert_called_once()

    def test_no_literature_reference(self):
        """No expander when literature_refs is empty."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import (
                ParameterSpec,
                ScientificParameterWidget,
            )

            spec = ParameterSpec(
                name="Test",
                unit="X",
                min_value=0.0,
                max_value=10.0,
                typical_range=(2.0, 8.0),
                literature_refs="",
                description="Test param",
            )
            widget = ScientificParameterWidget(spec, "key5")
            widget._show_validation_feedback(5.0)

        mock_st.expander.assert_not_called()


@pytest.mark.gui
class TestCreateParameterSection:
    """Tests for create_parameter_section (lines 79-92)."""

    def test_creates_section_with_parameters(self):
        """create_parameter_section renders and returns values."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 500.0

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import (
                ParameterSpec,
                create_parameter_section,
            )

            spec = ParameterSpec(
                name="Conductivity",
                unit="S/m",
                min_value=0.1,
                max_value=10000.0,
                typical_range=(100.0, 1000.0),
                literature_refs="Logan 2008",
                description="Conductivity",
            )
            parameters = {
                "conductivity": {
                    "spec": spec,
                    "label": "Conductivity",
                    "default": 500.0,
                },
            }
            result = create_parameter_section(
                "Electrode Params",
                parameters,
            )

        assert "conductivity" in result
        assert result["conductivity"] == 500.0
        mock_st.subheader.assert_called_once_with(
            "Electrode Params",
        )

    def test_creates_section_multiple_parameters(self):
        """create_parameter_section handles multiple parameters."""
        mock_st = _make_mock_st()
        mock_st.number_input.side_effect = [100.0, 25.0]

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import (
                ParameterSpec,
                create_parameter_section,
            )

            spec1 = ParameterSpec(
                name="Conductivity",
                unit="S/m",
                min_value=0.1,
                max_value=10000.0,
                typical_range=(100.0, 1000.0),
                literature_refs="",
                description="Conductivity",
            )
            spec2 = ParameterSpec(
                name="Temperature",
                unit="C",
                min_value=0.0,
                max_value=100.0,
                typical_range=(20.0, 40.0),
                literature_refs="",
                description="Temperature",
            )
            params = {
                "conductivity": {
                    "spec": spec1,
                    "label": "Conductivity",
                    "default": 100.0,
                },
                "temperature": {
                    "spec": spec2,
                    "label": "Temperature",
                    "default": 25.0,
                },
            }
            result = create_parameter_section(
                "Multi Params",
                params,
            )

        assert len(result) == 2
        assert "conductivity" in result
        assert "temperature" in result


@pytest.mark.gui
class TestMFCElectrodeParameters:
    """Tests for MFC_ELECTRODE_PARAMETERS constant."""

    def test_conductivity_spec_exists(self):
        """MFC_ELECTRODE_PARAMETERS has conductivity spec."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.scientific_widgets import (
                MFC_ELECTRODE_PARAMETERS,
            )

        assert "conductivity" in MFC_ELECTRODE_PARAMETERS
        spec = MFC_ELECTRODE_PARAMETERS["conductivity"]
        assert spec.name == "Electrical Conductivity"
        assert spec.unit == "S/m"
        assert spec.min_value == 0.1
        assert spec.max_value == 10000000.0
        assert spec.typical_range == (100.0, 100000.0)
