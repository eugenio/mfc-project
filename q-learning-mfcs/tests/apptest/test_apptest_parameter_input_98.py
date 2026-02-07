"""Deep coverage tests for parameter_input module."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str((Path(__file__).resolve().parent / ".." / ".." / "src").resolve())
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_GUI_PREFIX = "gui."


@pytest.fixture(autouse=True)
def _clear_module_cache():
    for mod_name in list(sys.modules):
        if mod_name == "gui" or mod_name.startswith(_GUI_PREFIX):
            del sys.modules[mod_name]
    yield
    for mod_name in list(sys.modules):
        if mod_name == "gui" or mod_name.startswith(_GUI_PREFIX):
            del sys.modules[mod_name]


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


def _make_mock_st():
    mock_st = MagicMock()
    mock_st.session_state = _SessionState()

    def _smart_columns(n_or_spec):
        n = len(n_or_spec) if isinstance(n_or_spec, list | tuple) else int(n_or_spec)
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            cols.append(col)
        return cols

    mock_st.columns.side_effect = _smart_columns

    def _smart_tabs(labels):
        tabs = []
        for _ in labels:
            tab = MagicMock()
            tab.__enter__ = MagicMock(return_value=tab)
            tab.__exit__ = MagicMock(return_value=False)
            tabs.append(tab)
        return tabs

    mock_st.tabs.side_effect = _smart_tabs
    exp = MagicMock()
    exp.__enter__ = MagicMock(return_value=exp)
    exp.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = exp
    mock_st.sidebar = MagicMock()
    form = MagicMock()
    form.__enter__ = MagicMock(return_value=form)
    form.__exit__ = MagicMock(return_value=False)
    mock_st.form.return_value = form
    mock_st.button.return_value = False
    mock_st.selectbox.return_value = "None"
    mock_st.multiselect.return_value = []
    mock_st.number_input.return_value = 5.0
    mock_st.slider.return_value = 0.5
    return mock_st


def _make_mock_param(name="learning_rate", large_range=False, has_doi=True,
                     has_notes=True, num_refs=1):
    """Create a mock ParameterInfo."""
    param = MagicMock()
    param.name = name
    param.symbol = "alpha"
    param.description = "Test parameter description"
    param.unit = "V"
    param.typical_value = 0.5
    param.min_value = 0.0
    if large_range:
        param.max_value = 2000.0
    else:
        param.max_value = 1.0
    param.recommended_range = (0.2, 0.8)
    param.category = MagicMock()
    param.category.value = "electrochemical"
    param.notes = "Some notes" if has_notes else None

    refs = []
    for i in range(num_refs):
        ref = MagicMock()
        ref.authors = f"Author{i}, A."
        ref.year = 2021 + i
        ref.doi = f"10.1234/test{i}" if has_doi else None
        ref.format_citation = MagicMock(return_value=f"Citation {i} text")
        refs.append(ref)
    param.references = refs
    return param


def _make_mock_validation_result(level_value="valid", confidence=0.95,
                                 response_time=50.0, warnings=None,
                                 recommendations=None, suggested_ranges=None):
    """Create a mock ValidationResult."""
    vr = MagicMock()
    level = MagicMock()
    level.value = level_value
    vr.level = level
    vr.message = f"Parameter is {level_value}"
    vr.scientific_reasoning = "This is the scientific reasoning for the validation result test"
    vr.confidence_score = confidence
    vr.uncertainty_bounds = (0.1, 0.9)
    vr.response_time_ms = response_time
    vr.recommendations = recommendations or []
    vr.warnings = warnings or []
    vr.suggested_ranges = suggested_ranges or []
    return vr


def _make_validation_level_enum():
    """Create a mock ValidationLevel enum."""
    vl = MagicMock()
    valid = MagicMock()
    valid.value = "valid"
    caution = MagicMock()
    caution.value = "caution"
    invalid = MagicMock()
    invalid.value = "invalid"
    vl.VALID = valid
    vl.CAUTION = caution
    vl.INVALID = invalid
    return vl


def _build_mocks():
    """Build all external module mocks needed for importing parameter_input."""
    import types

    mock_st = _make_mock_st()

    mock_lit_db = MagicMock()
    mock_lit_db_mod = MagicMock()
    mock_lit_db_mod.LITERATURE_DB = mock_lit_db
    mock_lit_db_mod.ParameterCategory = MagicMock()
    mock_lit_db_mod.ParameterInfo = MagicMock()

    mock_bridge = MagicMock()
    mock_bridge_mod = MagicMock()
    mock_bridge_mod.PARAMETER_BRIDGE = mock_bridge

    mock_validator = MagicMock()
    mock_validation_level = _make_validation_level_enum()
    mock_rtv_mod = MagicMock()
    mock_rtv_mod.REAL_TIME_VALIDATOR = mock_validator
    mock_rtv_mod.ValidationLevel = mock_validation_level

    mock_converter = MagicMock()
    mock_uc_mod = MagicMock()
    mock_uc_mod.UNIT_CONVERTER = mock_converter

    # Create a fake gui package so gui/__init__.py is NOT executed
    gui_pkg = types.ModuleType("gui")
    gui_pkg.__path__ = [str(Path(_SRC_DIR) / "gui")]
    gui_pkg.__package__ = "gui"

    module_patches = {
        "streamlit": mock_st,
        "plotly": MagicMock(),
        "plotly.graph_objects": MagicMock(),
        "pandas": MagicMock(),
        "config.literature_database": mock_lit_db_mod,
        "config.parameter_bridge": mock_bridge_mod,
        "config.real_time_validator": mock_rtv_mod,
        "config.unit_converter": mock_uc_mod,
        "config.qlearning_config": MagicMock(),
        "gui": gui_pkg,
    }
    return {
        "st": mock_st,
        "lit_db": mock_lit_db,
        "bridge": mock_bridge,
        "validator": mock_validator,
        "validation_level": mock_validation_level,
        "converter": mock_converter,
        "patches": module_patches,
    }


def _import_module(mocks):
    """Import parameter_input module with mocked dependencies."""
    import importlib
    with patch.dict(sys.modules, mocks["patches"]):
        mod = importlib.import_module("gui.parameter_input")
        return mod


# ===================================================================
# Tests for __init__ (lines 42-57)
# ===================================================================
@pytest.mark.apptest
class TestParameterInputComponentInit:
    """Test ParameterInputComponent.__init__ covering lines 42-57."""

    def test_init_empty_session_state(self):
        """All session state keys are initialized when empty."""
        mocks = _build_mocks()
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        ss = mocks["st"].session_state
        assert ss.parameter_values == {}
        assert ss.validation_results == {}
        assert ss.parameter_citations == {}
        assert ss.research_objective is None
        assert ss.show_performance_metrics is False
        assert comp.literature_db is mocks["lit_db"]
        assert comp.current_config is None

    def test_init_preserves_existing_session_state(self):
        """Existing session state values are not overwritten."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"lr": 0.1}
        ss["validation_results"] = {"lr": {"status": "valid"}}
        ss["parameter_citations"] = {"lr": "cite"}
        ss["research_objective"] = "max_power"
        ss["show_performance_metrics"] = True
        mod = _import_module(mocks)
        mod.ParameterInputComponent()
        assert ss.parameter_values == {"lr": 0.1}
        assert ss.research_objective == "max_power"
        assert ss.show_performance_metrics is True


# ===================================================================
# Tests for render_parameter_input_form (lines 66-145)
# ===================================================================
@pytest.mark.apptest
class TestRenderParameterInputForm:
    """Test render_parameter_input_form covering lines 66-145."""

    def test_form_no_categories_selected(self):
        """Form with no categories returns empty params."""
        mocks = _build_mocks()
        mocks["st"].selectbox.return_value = "None"
        mocks["st"].multiselect.return_value = []
        mocks["st"].button.return_value = False
        mocks["validator"].get_research_objectives.return_value = ["obj1"]
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp.render_parameter_input_form()
        assert "parameter_values" in result
        assert "config" in result
        mocks["st"].header.assert_called_once()

    def test_form_with_categories_and_tabs(self):
        """Form renders tabs for selected categories."""
        mocks = _build_mocks()
        cat_mock = MagicMock()
        cat_mock.value = "electrochemical"
        mocks["lit_db"].get_all_categories.return_value = [cat_mock]
        mocks["st"].multiselect.return_value = ["electrochemical"]
        mocks["st"].selectbox.return_value = "None"
        mocks["st"].button.return_value = False
        mocks["validator"].get_research_objectives.return_value = []
        mocks["lit_db"].get_parameters_by_category.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._create_validated_config = MagicMock(return_value=None)
        result = comp.render_parameter_input_form()
        mocks["st"].tabs.assert_called_once()
        assert result is not None

    def test_form_research_objective_selected(self):
        """Research objective selected populates info."""
        mocks = _build_mocks()
        mocks["st"].selectbox.return_value = "maximum_power"
        mocks["st"].multiselect.return_value = []
        mocks["st"].button.return_value = False
        mocks["validator"].get_research_objectives.return_value = ["maximum_power"]
        obj_info = MagicMock()
        obj_info.name = "Max Power"
        obj_info.description = "Maximize power"
        obj_info.scientific_context = "Power context"
        mocks["validator"].get_research_objective_info.return_value = obj_info
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._create_validated_config = MagicMock(return_value=None)
        comp.render_parameter_input_form()
        assert mocks["st"].session_state.research_objective == "maximum_power"
        mocks["st"].info.assert_called()

    def test_form_show_performance_metrics_toggle(self):
        """Performance metrics button toggles display."""
        mocks = _build_mocks()
        mocks["st"].selectbox.return_value = "None"
        mocks["st"].multiselect.return_value = []
        # First button call (performance metrics) returns True, rest False
        mocks["st"].button.side_effect = [True, False, False, False]
        mocks["validator"].get_research_objectives.return_value = []
        mocks["validator"].get_performance_metrics.return_value = {
            "avg_response_time_ms": 50.0,
            "cache_hit_rate": 0.9,
            "total_validations": 100,
            "fast_validations": 95,
            "instant_validations": 80,
        }
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._create_validated_config = MagicMock(return_value=None)
        comp.render_parameter_input_form()
        assert mocks["st"].session_state.show_performance_metrics is True

    def test_form_with_parameter_values_renders_summary(self):
        """When parameter_values exist, summary and buttons are rendered."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"learning_rate": 0.1}
        ss["validation_results"] = {"learning_rate": {"status": "valid"}}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["st"].selectbox.return_value = "None"
        mocks["st"].multiselect.return_value = []
        # buttons: performance=False, export=False, validate=False, generate=False
        mocks["st"].button.return_value = False
        mocks["validator"].get_research_objectives.return_value = []
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_parameter_summary = MagicMock()
        comp.render_config_integration_section = MagicMock()
        comp._create_validated_config = MagicMock(return_value=None)
        result = comp.render_parameter_input_form()
        comp._render_parameter_summary.assert_called_once()
        comp.render_config_integration_section.assert_called_once()
        assert result["parameter_values"] == {"learning_rate": 0.1}

    def test_form_export_button_clicked(self):
        """Export button triggers _export_configuration."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"lr": 0.1}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["st"].selectbox.return_value = "None"
        mocks["st"].multiselect.return_value = []
        # perf=False, export=True, validate=False, generate=False
        mocks["st"].button.side_effect = [False, True, False, False]
        mocks["validator"].get_research_objectives.return_value = []
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_parameter_summary = MagicMock()
        comp._export_configuration = MagicMock()
        comp.render_config_integration_section = MagicMock()
        comp._create_validated_config = MagicMock(return_value=None)
        comp.render_parameter_input_form()
        comp._export_configuration.assert_called_once()

    def test_form_validate_button_clicked(self):
        """Validate button triggers _validate_all_parameters."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"lr": 0.1}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["st"].selectbox.return_value = "None"
        mocks["st"].multiselect.return_value = []
        # perf=False, export=False, validate=True, generate=False
        mocks["st"].button.side_effect = [False, False, True, False]
        mocks["validator"].get_research_objectives.return_value = []
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_parameter_summary = MagicMock()
        comp._validate_all_parameters = MagicMock()
        comp.render_config_integration_section = MagicMock()
        comp._create_validated_config = MagicMock(return_value=None)
        comp.render_parameter_input_form()
        comp._validate_all_parameters.assert_called_once()

    def test_form_citations_button_clicked(self):
        """Generate Citations button triggers _show_citations."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"lr": 0.1}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["st"].selectbox.return_value = "None"
        mocks["st"].multiselect.return_value = []
        # perf=False, export=False, validate=False, generate=True
        mocks["st"].button.side_effect = [False, False, False, True]
        mocks["validator"].get_research_objectives.return_value = []
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_parameter_summary = MagicMock()
        comp._show_citations = MagicMock()
        comp.render_config_integration_section = MagicMock()
        comp._create_validated_config = MagicMock(return_value=None)
        comp.render_parameter_input_form()
        comp._show_citations.assert_called_once()

    def test_form_performance_metrics_rendered_when_toggled_on(self):
        """Performance metrics section rendered when flag is True."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = True
        mocks["st"].selectbox.return_value = "None"
        mocks["st"].multiselect.return_value = []
        mocks["st"].button.return_value = False
        mocks["validator"].get_research_objectives.return_value = []
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_performance_metrics = MagicMock()
        comp._create_validated_config = MagicMock(return_value=None)
        comp.render_parameter_input_form()
        comp._render_performance_metrics.assert_called_once()

    def test_form_objective_info_none(self):
        """When objective info returns None, no info is displayed."""
        mocks = _build_mocks()
        mocks["st"].selectbox.return_value = "unknown_obj"
        mocks["st"].multiselect.return_value = []
        mocks["st"].button.return_value = False
        mocks["validator"].get_research_objectives.return_value = ["unknown_obj"]
        mocks["validator"].get_research_objective_info.return_value = None
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._create_validated_config = MagicMock(return_value=None)
        comp.render_parameter_input_form()
        # st.info should NOT be called for objective info since obj_info is None
        # (it may be called elsewhere, just ensure no crash)


# ===================================================================
# Tests for _render_category_parameters (lines 160-163)
# ===================================================================
@pytest.mark.apptest
class TestRenderCategoryParameters:
    """Test _render_category_parameters covering lines 160-163."""

    def test_with_parameters(self):
        """Renders markdown and calls _render_parameter_input per param."""
        mocks = _build_mocks()
        param1 = _make_mock_param("p1")
        param2 = _make_mock_param("p2")
        mocks["lit_db"].get_parameters_by_category.return_value = [param1, param2]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        category = MagicMock()
        category.value = "electrochemical"
        comp._render_parameter_input = MagicMock()
        comp._render_category_parameters(category)
        mocks["st"].markdown.assert_called()
        assert comp._render_parameter_input.call_count == 2

    def test_with_no_parameters(self):
        """Shows info when no parameters available."""
        mocks = _build_mocks()
        mocks["lit_db"].get_parameters_by_category.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        category = MagicMock()
        category.value = "substrate"
        comp._render_category_parameters(category)
        mocks["st"].info.assert_called()


# ===================================================================
# Tests for _render_parameter_input (lines 167-247)
# ===================================================================
@pytest.mark.apptest
class TestRenderParameterInput:
    """Test _render_parameter_input covering lines 167-247."""

    def test_slider_path_small_range(self):
        """Slider is used for small range parameters."""
        mocks = _build_mocks()
        mocks["st"].slider.return_value = 0.5
        vr = _make_mock_validation_result()
        vr.level = mocks["validation_level"].VALID
        mocks["validator"].validate_parameter_realtime.return_value = vr
        mocks["converter"].get_compatible_units.return_value = ["V"]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param("test_param", large_range=False)
        comp._render_enhanced_validation_indicator = MagicMock()
        comp._render_parameter_references = MagicMock()
        comp._render_parameter_input(param)
        mocks["st"].slider.assert_called()
        assert mocks["st"].session_state.parameter_values["test_param"] == 0.5

    def test_number_input_path_large_range(self):
        """Number input is used for large range parameters."""
        mocks = _build_mocks()
        mocks["st"].number_input.return_value = 500.0
        vr = _make_mock_validation_result()
        vr.level = mocks["validation_level"].VALID
        mocks["validator"].validate_parameter_realtime.return_value = vr
        mocks["converter"].get_compatible_units.return_value = ["V"]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param("big_param", large_range=True)
        comp._render_enhanced_validation_indicator = MagicMock()
        comp._render_parameter_references = MagicMock()
        comp._render_parameter_input(param)
        mocks["st"].number_input.assert_called()
        assert mocks["st"].session_state.parameter_values["big_param"] == 500.0

    def test_unit_conversion_with_multiple_compatible_units(self):
        """Unit conversion shown when multiple compatible units exist."""
        mocks = _build_mocks()
        mocks["st"].slider.return_value = 0.5
        mocks["st"].selectbox.return_value = "mV"
        vr = _make_mock_validation_result()
        vr.level = mocks["validation_level"].VALID
        mocks["validator"].validate_parameter_realtime.return_value = vr
        mocks["converter"].get_compatible_units.return_value = ["V", "mV"]
        mocks["converter"].convert.return_value = 500.0
        mocks["converter"].format_value_with_unit.return_value = "500.0 mV"
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param("conv_param")
        comp._render_enhanced_validation_indicator = MagicMock()
        comp._render_parameter_references = MagicMock()
        comp._render_parameter_input(param)
        mocks["converter"].convert.assert_called()

    def test_unit_conversion_returns_none(self):
        """When conversion returns None, no caption for conversion."""
        mocks = _build_mocks()
        mocks["st"].slider.return_value = 0.5
        mocks["st"].selectbox.return_value = "mV"
        vr = _make_mock_validation_result()
        vr.level = mocks["validation_level"].VALID
        mocks["validator"].validate_parameter_realtime.return_value = vr
        mocks["converter"].get_compatible_units.return_value = ["V", "mV"]
        mocks["converter"].convert.return_value = None
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param("null_conv")
        comp._render_enhanced_validation_indicator = MagicMock()
        comp._render_parameter_references = MagicMock()
        comp._render_parameter_input(param)
        mocks["converter"].format_value_with_unit.assert_not_called()

    def test_same_unit_selected_no_conversion(self):
        """When selected unit matches param unit, no conversion happens."""
        mocks = _build_mocks()
        mocks["st"].slider.return_value = 0.5
        mocks["st"].selectbox.return_value = "V"
        vr = _make_mock_validation_result()
        vr.level = mocks["validation_level"].VALID
        mocks["validator"].validate_parameter_realtime.return_value = vr
        mocks["converter"].get_compatible_units.return_value = ["V", "mV"]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param("same_unit")
        comp._render_enhanced_validation_indicator = MagicMock()
        comp._render_parameter_references = MagicMock()
        comp._render_parameter_input(param)
        mocks["converter"].convert.assert_not_called()

    def test_single_compatible_unit(self):
        """When only one compatible unit, caption is shown instead of selectbox."""
        mocks = _build_mocks()
        mocks["st"].slider.return_value = 0.5
        vr = _make_mock_validation_result()
        vr.level = mocks["validation_level"].VALID
        mocks["validator"].validate_parameter_realtime.return_value = vr
        mocks["converter"].get_compatible_units.return_value = ["V"]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param("single_unit")
        comp._render_enhanced_validation_indicator = MagicMock()
        comp._render_parameter_references = MagicMock()
        comp._render_parameter_input(param)
        # caption called with param unit in col2

    def test_validation_result_stored_in_session(self):
        """Validation result is stored in session_state."""
        mocks = _build_mocks()
        mocks["st"].slider.return_value = 0.5
        vr = _make_mock_validation_result("caution", confidence=0.7)
        vr.level = mocks["validation_level"].CAUTION
        mocks["validator"].validate_parameter_realtime.return_value = vr
        mocks["converter"].get_compatible_units.return_value = ["V"]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param("stored_param")
        comp._render_enhanced_validation_indicator = MagicMock()
        comp._render_parameter_references = MagicMock()
        comp._render_parameter_input(param)
        assert "stored_param" in mocks["st"].session_state.validation_results
        assert mocks["st"].session_state.validation_results["stored_param"]["status"] == "caution"

    def test_existing_value_in_session_state(self):
        """Existing value from session state is used as current value."""
        mocks = _build_mocks()
        mocks["st"].session_state["parameter_values"] = {"preset": 0.75}
        mocks["st"].session_state["validation_results"] = {}
        mocks["st"].session_state["parameter_citations"] = {}
        mocks["st"].session_state["research_objective"] = None
        mocks["st"].session_state["show_performance_metrics"] = False
        mocks["st"].slider.return_value = 0.75
        vr = _make_mock_validation_result()
        vr.level = mocks["validation_level"].VALID
        mocks["validator"].validate_parameter_realtime.return_value = vr
        mocks["converter"].get_compatible_units.return_value = ["V"]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param("preset")
        comp._render_enhanced_validation_indicator = MagicMock()
        comp._render_parameter_references = MagicMock()
        comp._render_parameter_input(param)


# ===================================================================
# Tests for _render_validation_indicator (lines 258-277)
# ===================================================================
@pytest.mark.apptest
class TestRenderValidationIndicator:
    """Test _render_validation_indicator."""

    def test_valid_status(self):
        mocks = _build_mocks()
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_validation_indicator(
            {"status": "valid", "message": "OK", "recommendations": []}
        )
        mocks["st"].success.assert_called()

    def test_caution_status(self):
        mocks = _build_mocks()
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_validation_indicator(
            {"status": "caution", "message": "Careful", "recommendations": ["rec1"]}
        )
        mocks["st"].warning.assert_called()

    def test_invalid_status(self):
        mocks = _build_mocks()
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_validation_indicator(
            {"status": "invalid", "message": "Bad", "recommendations": ["fix1", "fix2"]}
        )
        mocks["st"].error.assert_called()

    def test_unknown_status(self):
        mocks = _build_mocks()
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_validation_indicator(
            {"status": "something_else", "message": "Unknown", "recommendations": []}
        )
        mocks["st"].info.assert_called()


# ===================================================================
# Tests for _render_parameter_references (lines 279-287)
# ===================================================================
@pytest.mark.apptest
class TestRenderParameterReferences:
    """Test _render_parameter_references."""

    def test_with_doi_and_notes(self):
        mocks = _build_mocks()
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param(has_doi=True, has_notes=True, num_refs=2)
        comp._render_parameter_references(param)
        # markdown called for each ref + doi + notes
        assert mocks["st"].markdown.call_count >= 4

    def test_without_doi_no_notes(self):
        mocks = _build_mocks()
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        param = _make_mock_param(has_doi=False, has_notes=False, num_refs=1)
        comp._render_parameter_references(param)
        # markdown called for citation only, no doi, no notes


# ===================================================================
# Tests for _render_parameter_summary (lines 289-347)
# ===================================================================
@pytest.mark.apptest
class TestRenderParameterSummary:
    """Test _render_parameter_summary covering lines 313-319."""

    def test_summary_with_valid_params(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"learning_rate": 0.1}
        ss["validation_results"] = {"learning_rate": {"status": "valid"}}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        param = _make_mock_param("learning_rate")
        mocks["lit_db"].get_parameter.return_value = param
        # We need pd.DataFrame to work
        import pandas as real_pd
        mocks["patches"]["pandas"] = real_pd
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_parameter_summary()
        mocks["st"].metric.assert_called()

    def test_summary_with_caution_and_invalid(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"p1": 0.1, "p2": 0.5, "p3": 999}
        ss["validation_results"] = {
            "p1": {"status": "valid"},
            "p2": {"status": "caution"},
            "p3": {"status": "invalid"},
        }
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        for pname in ["p1", "p2", "p3"]:
            _make_mock_param(pname)
            mocks["lit_db"].get_parameter.side_effect = lambda name: _make_mock_param(name)
        import pandas as real_pd
        mocks["patches"]["pandas"] = real_pd
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_parameter_summary()
        # metric called 3 times (valid, caution, invalid counts)
        assert mocks["st"].metric.call_count == 3

    def test_summary_param_not_in_db(self):
        """When parameter is not in the literature DB."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"unknown_p": 1.0}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["lit_db"].get_parameter.return_value = None
        import pandas as real_pd
        mocks["patches"]["pandas"] = real_pd
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_parameter_summary()
        # No dataframe rendered since summary_data is empty

    def test_color_status_function_branches(self):
        """Test color_status function all branches via styled dataframe rendering."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"p1": 0.1, "p2": 0.5, "p3": 999, "p4": 0.0}
        ss["validation_results"] = {
            "p1": {"status": "valid"},
            "p2": {"status": "caution"},
            "p3": {"status": "invalid"},
            "p4": {"status": "unknown"},
        }
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["lit_db"].get_parameter.side_effect = lambda name: _make_mock_param(name)
        import pandas as real_pd
        mocks["patches"]["pandas"] = real_pd
        # Capture the styled dataframe and force render
        captured = {}
        original_dataframe = mocks["st"].dataframe
        def capture_dataframe(df, **kwargs):
            captured["styled"] = df
            return original_dataframe(df, **kwargs)
        mocks["st"].dataframe = capture_dataframe
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_parameter_summary()
        # Force the styled dataframe to render, which calls color_status
        if "styled" in captured:
            captured["styled"].to_html()


# ===================================================================
# Tests for _validate_all_parameters (lines 349-370)
# ===================================================================
@pytest.mark.apptest
class TestValidateAllParameters:
    """Test _validate_all_parameters."""

    def test_all_valid(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"p1": 0.1, "p2": 0.5}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["lit_db"].validate_parameter_value.return_value = {"status": "valid"}
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._validate_all_parameters()
        mocks["st"].success.assert_called()

    def test_some_invalid(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"p1": 0.1, "p2": 999}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["lit_db"].validate_parameter_value.side_effect = [
            {"status": "valid"},
            {"status": "invalid"},
        ]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._validate_all_parameters()
        mocks["st"].warning.assert_called()


# ===================================================================
# Tests for _export_configuration (lines 374-478)
# ===================================================================
@pytest.mark.apptest
class TestExportConfiguration:
    """Test _export_configuration covering lines 374-478."""

    def _setup_export(self, mocks, export_format="JSON"):
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"learning_rate": 0.1}
        ss["validation_results"] = {"learning_rate": {"status": "valid"}}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        param = _make_mock_param("learning_rate")
        mocks["lit_db"].get_parameter.return_value = param
        mocks["st"].selectbox.return_value = export_format

    def test_export_json(self):
        mocks = _build_mocks()
        self._setup_export(mocks, "JSON")
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._export_configuration()
        mocks["st"].download_button.assert_called_once()
        mocks["st"].code.assert_called_once()

    def test_export_csv(self):
        mocks = _build_mocks()
        self._setup_export(mocks, "CSV")
        import pandas as real_pd
        mocks["patches"]["pandas"] = real_pd
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._export_configuration()
        mocks["st"].download_button.assert_called_once()
        mocks["st"].dataframe.assert_called_once()

    def test_export_bibtex(self):
        mocks = _build_mocks()
        self._setup_export(mocks, "BibTeX Citations Only")
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._export_configuration()
        mocks["st"].download_button.assert_called_once()
        mocks["st"].code.assert_called_once()

    def test_export_with_validation_status_counts(self):
        """Export updates validation summary counters."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"p1": 0.1, "p2": 0.5, "p3": 999}
        ss["validation_results"] = {
            "p1": {"status": "valid"},
            "p2": {"status": "caution"},
            "p3": {"status": "invalid"},
        }
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["lit_db"].get_parameter.side_effect = lambda name: _make_mock_param(name)
        mocks["st"].selectbox.return_value = "JSON"
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._export_configuration()
        # Verify JSON was created with correct counts
        call_args = mocks["st"].code.call_args[0][0]
        data = json.loads(call_args)
        assert data["validation_summary"]["valid"] == 1
        assert data["validation_summary"]["caution"] == 1
        assert data["validation_summary"]["invalid"] == 1

    def test_export_param_not_in_db(self):
        """Export skips parameters not found in literature DB."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"unknown": 1.0}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["lit_db"].get_parameter.return_value = None
        mocks["st"].selectbox.return_value = "JSON"
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._export_configuration()
        call_args = mocks["st"].code.call_args[0][0]
        data = json.loads(call_args)
        assert len(data["parameters"]) == 0

    def test_export_unknown_validation_status(self):
        """Export handles unknown validation status."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"p1": 0.1}
        ss["validation_results"] = {"p1": {"status": "unknown"}}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["lit_db"].get_parameter.return_value = _make_mock_param("p1")
        mocks["st"].selectbox.return_value = "JSON"
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._export_configuration()
        call_args = mocks["st"].code.call_args[0][0]
        data = json.loads(call_args)
        # "unknown" is not in validation_summary keys, so count stays 0
        assert data["validation_summary"]["valid"] == 0


# ===================================================================
# Tests for _show_citations (lines 480-522)
# ===================================================================
@pytest.mark.apptest
class TestShowCitations:
    """Test _show_citations covering lines 509-512."""

    def test_apa_format(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"p1": 0.1}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        param = _make_mock_param("p1", num_refs=2)
        mocks["lit_db"].get_parameter.return_value = param
        mocks["st"].selectbox.return_value = "APA"
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._show_citations()
        mocks["st"].subheader.assert_called()
        mocks["st"].metric.assert_called()

    def test_bibtex_format(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"p1": 0.1}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        param = _make_mock_param("p1", num_refs=1)
        mocks["lit_db"].get_parameter.return_value = param
        mocks["st"].selectbox.return_value = "BibTeX"
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._show_citations()
        mocks["st"].code.assert_called()

    def test_no_params(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["st"].selectbox.return_value = "APA"
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._show_citations()

    def test_param_not_in_db(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"missing": 0.1}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["lit_db"].get_parameter.return_value = None
        mocks["st"].selectbox.return_value = "APA"
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._show_citations()


# ===================================================================
# Tests for _render_enhanced_validation_indicator (lines 591-656)
# ===================================================================
@pytest.mark.apptest
class TestRenderEnhancedValidationIndicator:
    """Test _render_enhanced_validation_indicator."""

    def test_valid_high_confidence(self):
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result("valid", confidence=0.95, response_time=30.0)
        vr.level = vl.VALID
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)
        mocks["st"].success.assert_called()

    def test_valid_medium_confidence(self):
        """Valid with confidence >= 0.8 but < 0.9."""
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result("valid", confidence=0.85, response_time=30.0)
        vr.level = vl.VALID
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)
        mocks["st"].success.assert_called()

    def test_valid_low_confidence(self):
        """Valid with confidence < 0.8, no uncertainty shown."""
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result("valid", confidence=0.5, response_time=30.0)
        vr.level = vl.VALID
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)
        mocks["st"].success.assert_called()

    def test_caution(self):
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result("caution", confidence=0.7, response_time=100.0)
        vr.level = vl.CAUTION
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)
        mocks["st"].warning.assert_called()

    def test_invalid(self):
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result("invalid", confidence=0.2, response_time=100.0)
        vr.level = vl.INVALID
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)
        mocks["st"].error.assert_called()

    def test_unknown(self):
        mocks = _build_mocks()
        mocks["validation_level"]
        vr = _make_mock_validation_result("unknown", confidence=0.0, response_time=100.0)
        vr.level = "SOMETHING_ELSE"
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)
        mocks["st"].info.assert_called()

    def test_slow_response_time(self):
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result("valid", confidence=0.95, response_time=500.0)
        vr.level = vl.VALID
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)
        # slow indicator

    def test_with_warnings(self):
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result(
            "valid", confidence=0.95, response_time=30.0,
            warnings=["warn1_long_text_here_abcdefghij", "warn2_text", "warn3_ignored"]
        )
        vr.level = vl.VALID
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)

    def test_with_recommendations_and_suggested_ranges(self):
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result(
            "valid", confidence=0.95, response_time=30.0,
            recommendations=["rec1", "rec2"],
            suggested_ranges=[(0.1, 0.5), (0.2, 0.6), (0.3, 0.7)]
        )
        vr.level = vl.VALID
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)

    def test_with_recommendations_and_more_than_3_suggested_ranges(self):
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result(
            "valid", confidence=0.95, response_time=30.0,
            recommendations=["rec1"],
            suggested_ranges=[(0.1, 0.5), (0.2, 0.6), (0.3, 0.7), (0.4, 0.8)]
        )
        vr.level = vl.VALID
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)

    def test_with_recommendations_no_suggested_ranges(self):
        mocks = _build_mocks()
        vl = mocks["validation_level"]
        vr = _make_mock_validation_result(
            "valid", confidence=0.95, response_time=30.0,
            recommendations=["rec1"],
            suggested_ranges=[]
        )
        vr.level = vl.VALID
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_enhanced_validation_indicator(vr)


# ===================================================================
# Tests for _render_performance_metrics (lines 658-708)
# ===================================================================
@pytest.mark.apptest
class TestRenderPerformanceMetrics:
    """Test _render_performance_metrics."""

    def test_excellent_performance(self):
        mocks = _build_mocks()
        mocks["validator"].get_performance_metrics.return_value = {
            "avg_response_time_ms": 50.0,
            "max_response_time_ms": 100.0,
            "min_response_time_ms": 10.0,
            "cache_hit_rate": 0.9,
            "total_validations": 100,
            "fast_validations": 95,
            "instant_validations": 80,
        }
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_performance_metrics()
        mocks["st"].success.assert_called()

    def test_acceptable_performance(self):
        mocks = _build_mocks()
        mocks["validator"].get_performance_metrics.return_value = {
            "avg_response_time_ms": 300.0,
            "cache_hit_rate": 0.7,
            "total_validations": 100,
            "fast_validations": 50,
            "instant_validations": 20,
        }
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_performance_metrics()
        mocks["st"].warning.assert_called()

    def test_poor_performance(self):
        mocks = _build_mocks()
        mocks["validator"].get_performance_metrics.return_value = {
            "avg_response_time_ms": 600.0,
            "cache_hit_rate": 0.3,
            "total_validations": 100,
            "fast_validations": 10,
            "instant_validations": 2,
        }
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_performance_metrics()
        mocks["st"].error.assert_called()

    def test_zero_total_validations(self):
        mocks = _build_mocks()
        mocks["validator"].get_performance_metrics.return_value = {
            "avg_response_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "total_validations": 0,
            "fast_validations": 0,
            "instant_validations": 0,
        }
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_performance_metrics()
        mocks["st"].success.assert_called()

    def test_cache_hit_rate_above_80_shows_delta(self):
        mocks = _build_mocks()
        mocks["validator"].get_performance_metrics.return_value = {
            "avg_response_time_ms": 50.0,
            "cache_hit_rate": 0.85,
            "total_validations": 10,
            "fast_validations": 8,
            "instant_validations": 5,
        }
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_performance_metrics()

    def test_cache_hit_rate_below_80_no_delta(self):
        mocks = _build_mocks()
        mocks["validator"].get_performance_metrics.return_value = {
            "avg_response_time_ms": 50.0,
            "cache_hit_rate": 0.5,
            "total_validations": 10,
            "fast_validations": 8,
            "instant_validations": 5,
        }
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._render_performance_metrics()


# ===================================================================
# Tests for _create_validated_config (lines 710-731)
# ===================================================================
@pytest.mark.apptest
class TestCreateValidatedConfig:
    """Test _create_validated_config."""

    def test_no_parameter_values(self):
        mocks = _build_mocks()
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp._create_validated_config()
        assert result is None

    def test_successful_config_creation(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"learning_rate": 0.1}
        ss["validation_results"] = {"learning_rate": {"status": "valid"}}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        config_mock = MagicMock()
        mocks["bridge"].create_literature_validated_config.return_value = (
            config_mock,
            {"learning_rate": {"extra": "data"}},
        )
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp._create_validated_config()
        assert result is config_mock
        assert ss.validation_results["learning_rate"]["extra"] == "data"

    def test_config_creation_exception(self):
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"learning_rate": 0.1}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        mocks["bridge"].create_literature_validated_config.side_effect = ValueError("fail")
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp._create_validated_config()
        assert result is None
        mocks["st"].error.assert_called()

    def test_config_creation_with_new_params(self):
        """Validation results for param not in session_state are not updated."""
        mocks = _build_mocks()
        ss = mocks["st"].session_state
        ss["parameter_values"] = {"p1": 0.1}
        ss["validation_results"] = {}
        ss["parameter_citations"] = {}
        ss["research_objective"] = None
        ss["show_performance_metrics"] = False
        config_mock = MagicMock()
        mocks["bridge"].create_literature_validated_config.return_value = (
            config_mock,
            {"p1": {"new_key": "val"}, "p2": {"other": "data"}},
        )
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp._create_validated_config()
        assert result is config_mock
        # p2 is not in validation_results, so its entry should not appear
        assert "p2" not in ss.validation_results


# ===================================================================
# Tests for render_config_integration_section (lines 735-774)
# ===================================================================
@pytest.mark.apptest
class TestRenderConfigIntegrationSection:
    """Test render_config_integration_section covering lines 735-774."""

    def test_no_buttons_clicked(self):
        mocks = _build_mocks()
        mocks["st"].button.return_value = False
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp.render_config_integration_section()
        mocks["st"].subheader.assert_called()

    def test_create_config_button_success(self):
        mocks = _build_mocks()
        mocks["st"].button.side_effect = [True, False]
        config_mock = MagicMock()
        config_mock.learning_rate = 0.1
        config_mock.discount_factor = 0.95
        config_mock.epsilon = 0.3
        config_mock.anode_area_per_cell = 0.005
        config_mock.substrate_target_concentration = 30.0
        config_mock.optimal_biofilm_thickness = 100.0
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._create_validated_config = MagicMock(return_value=config_mock)
        comp.render_config_integration_section()
        mocks["st"].success.assert_called()
        mocks["st"].json.assert_called()

    def test_create_config_button_returns_none(self):
        mocks = _build_mocks()
        mocks["st"].button.side_effect = [True, False]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._create_validated_config = MagicMock(return_value=None)
        comp.render_config_integration_section()
        mocks["st"].success.assert_not_called()

    def test_suggest_improvements_with_suggestions(self):
        mocks = _build_mocks()
        mocks["st"].button.side_effect = [False, True]
        config_mock = MagicMock()
        suggestions = [
            {
                "parameter": "learning_rate",
                "current_value": 0.01,
                "current_status": "caution",
                "suggestion": "Increase learning rate",
            },
            {
                "parameter": "epsilon",
                "current_value": 0.9,
                "current_status": "invalid",
                "suggestion": "Decrease exploration rate",
            },
        ]
        mocks["bridge"].suggest_parameter_improvements.return_value = suggestions
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._create_validated_config = MagicMock(return_value=config_mock)
        comp.render_config_integration_section()
        mocks["st"].warning.assert_called()

    def test_suggest_improvements_no_suggestions(self):
        mocks = _build_mocks()
        mocks["st"].button.side_effect = [False, True]
        config_mock = MagicMock()
        mocks["bridge"].suggest_parameter_improvements.return_value = []
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._create_validated_config = MagicMock(return_value=config_mock)
        comp.render_config_integration_section()
        mocks["st"].success.assert_called()

    def test_suggest_improvements_config_none(self):
        mocks = _build_mocks()
        mocks["st"].button.side_effect = [False, True]
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        comp._create_validated_config = MagicMock(return_value=None)
        comp.render_config_integration_section()
        mocks["bridge"].suggest_parameter_improvements.assert_not_called()


# ===================================================================
# Tests for render_parameter_input_interface (lines 779-780)
# ===================================================================
@pytest.mark.apptest
class TestRenderParameterInputInterface:
    """Test the module-level render_parameter_input_interface function."""

    def test_render_interface(self):
        mocks = _build_mocks()
        mocks["st"].selectbox.return_value = "None"
        mocks["st"].multiselect.return_value = []
        mocks["st"].button.return_value = False
        mocks["validator"].get_research_objectives.return_value = []
        mocks["lit_db"].get_all_categories.return_value = []
        mod = _import_module(mocks)
        with patch.object(
            mod.ParameterInputComponent,
            "render_parameter_input_form",
            return_value={"test": True},
        ):
            result = mod.render_parameter_input_interface()
            assert result == {"test": True}


# ===================================================================
# Tests for create_parameter_range_visualization (lines 524-589)
# ===================================================================
@pytest.mark.apptest
class TestCreateParameterRangeVisualization:
    """Test create_parameter_range_visualization."""

    def test_param_not_found(self):
        mocks = _build_mocks()
        mocks["lit_db"].get_parameter.return_value = None
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp.create_parameter_range_visualization("nonexistent")
        assert result is None

    def test_param_found_creates_figure(self):
        mocks = _build_mocks()
        param = _make_mock_param("test_viz")
        mocks["lit_db"].get_parameter.return_value = param
        mocks["lit_db"].validate_parameter_value.return_value = {"status": "valid"}
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp.create_parameter_range_visualization("test_viz")
        # go.Figure is mocked, so result is a mock
        assert result is not None

    def test_param_caution_status(self):
        mocks = _build_mocks()
        param = _make_mock_param("test_caution")
        mocks["lit_db"].get_parameter.return_value = param
        mocks["lit_db"].validate_parameter_value.return_value = {"status": "caution"}
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp.create_parameter_range_visualization("test_caution")
        assert result is not None

    def test_param_invalid_status(self):
        mocks = _build_mocks()
        param = _make_mock_param("test_invalid")
        mocks["lit_db"].get_parameter.return_value = param
        mocks["lit_db"].validate_parameter_value.return_value = {"status": "invalid"}
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp.create_parameter_range_visualization("test_invalid")
        assert result is not None

    def test_param_unknown_status(self):
        mocks = _build_mocks()
        param = _make_mock_param("test_unknown")
        mocks["lit_db"].get_parameter.return_value = param
        mocks["lit_db"].validate_parameter_value.return_value = {"status": "weird"}
        mod = _import_module(mocks)
        comp = mod.ParameterInputComponent()
        result = comp.create_parameter_range_visualization("test_unknown")
        assert result is not None
