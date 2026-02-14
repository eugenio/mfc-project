"""Coverage tests for gui/pages/cell_config.py -- target 99%+.

Covers: render_membrane_configuration with MembraneMaterial.CUSTOM,
membrane_configuration ValueError, cell_config in session for
integration check, checkbox for advanced analysis.
"""
import importlib.util
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

_CELL_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "src", "gui", "pages", "cell_config.py",
)


def _make_col():
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    return col


def _make_st():
    st = MagicMock()
    st.columns.side_effect = lambda *a, **kw: [
        _make_col()
        for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))
    ]
    st.tabs.side_effect = lambda labels: [_make_col() for _ in labels]
    st.selectbox.return_value = "Rectangular Chamber"
    st.number_input.return_value = 10.0
    st.checkbox.return_value = False
    st.session_state = MagicMock()
    return st


@contextmanager
def _cell_config_env(st_mock, extra_modules=None):
    """Context manager that loads cell_config.py with mocked modules.

    Keeps all mocked modules in sys.modules for the duration of the
    context, so that runtime imports inside functions also resolve to mocks.

    Yields the loaded module.
    """
    saved = {}
    keys_to_mock = {
        "streamlit": st_mock,
        "gui.pages.cell_config_helpers": MagicMock(),
        "gui.scientific_widgets": MagicMock(),
        "pandas": MagicMock(),
    }
    if extra_modules:
        keys_to_mock.update(extra_modules)
    for k, v in keys_to_mock.items():
        saved[k] = sys.modules.get(k)
        sys.modules[k] = v

    try:
        spec = importlib.util.spec_from_file_location(
            "_cell_config_isolated",
            _CELL_CONFIG_PATH,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _setup_membrane_mocks(material_is_custom=False, create_raises=None):
    """Build the standard membrane UI and config mocks.

    Returns (mock_ui_instance, extra_modules dict).
    """
    mock_ui_class = MagicMock()
    mock_ui_instance = MagicMock()
    mock_ui_class.return_value = mock_ui_instance

    mock_material_enum = MagicMock()
    mock_material = MagicMock()
    mock_material_enum.CUSTOM = mock_material

    if material_is_custom:
        mock_ui_instance.render_material_selector.return_value = mock_material
        mock_ui_instance.render_custom_membrane_properties.return_value = {
            "conductivity": 0.05
        }
    else:
        non_custom = MagicMock()
        mock_ui_instance.render_material_selector.return_value = non_custom

    mock_ui_instance.render_area_input.return_value = 0.0025
    mock_ui_instance.render_operating_conditions.return_value = (
        303.0,
        7.0,
        7.0,
    )

    mock_mod = MagicMock()
    mock_mod.MembraneConfigurationUI = mock_ui_class

    mock_membrane_config_mod = MagicMock()
    mock_membrane_config_mod.MembraneMaterial = mock_material_enum

    if create_raises:
        mock_membrane_config_mod.create_membrane_config.side_effect = (
            create_raises
        )
    else:
        mock_config = MagicMock()
        mock_config.calculate_resistance.return_value = 0.5
        mock_config.calculate_proton_flux.return_value = 1e-5
        mock_props = MagicMock()
        mock_props.proton_conductivity = 0.1
        mock_props.ion_exchange_capacity = 0.9
        mock_props.permselectivity = 0.95
        mock_props.thickness = 183.0
        mock_props.area_resistance = 2.0
        mock_props.expected_lifetime = 50000.0
        mock_props.reference = "Literature"
        mock_config.properties = mock_props
        mock_membrane_config_mod.create_membrane_config.return_value = (
            mock_config
        )

    extra_modules = {
        "gui.membrane_configuration_ui": mock_mod,
        "config.membrane_config": mock_membrane_config_mod,
    }
    return mock_ui_instance, extra_modules


@pytest.mark.coverage_extra
class TestMembraneConfigCustomMaterial:
    """Cover the CUSTOM material branch in render_membrane_configuration."""

    def test_custom_material_renders(self):
        st = _make_st()
        st.checkbox.return_value = False

        mock_ui_instance, extra = _setup_membrane_mocks(
            material_is_custom=True
        )

        with _cell_config_env(st, extra_modules=extra) as mod:
            mod.render_membrane_configuration()

        mock_ui_instance.render_custom_membrane_properties.assert_called_once()


@pytest.mark.coverage_extra
class TestMembraneConfigValueError:
    """Cover ValueError branch in membrane config."""

    def test_value_error_in_create_config(self):
        st = _make_st()
        st.checkbox.return_value = False

        _, extra = _setup_membrane_mocks(
            create_raises=ValueError("Bad config")
        )

        with _cell_config_env(st, extra_modules=extra) as mod:
            mod.render_membrane_configuration()

        st.error.assert_called()


@pytest.mark.coverage_extra
class TestMembraneWithCellConfigIntegration:
    """Cover integration checks when cell_config is in session state."""

    def test_with_cell_config_area_too_small(self):
        st = _make_st()
        st.checkbox.return_value = False
        session_dict = {
            "cell_config": {"volume": 500.0, "electrode_area": 50.0}
        }
        st.session_state.__contains__ = lambda s, k: k in session_dict
        st.session_state.__getitem__ = lambda s, k: session_dict[k]
        st.session_state.cell_config = session_dict["cell_config"]

        mock_ui_instance, extra = _setup_membrane_mocks()
        mock_ui_instance.render_area_input.return_value = 0.0001

        with _cell_config_env(st, extra_modules=extra) as mod:
            mod.render_membrane_configuration()

        st.warning.assert_called()


@pytest.mark.coverage_extra
class TestMembraneAdvancedCheckbox:
    """Cover the advanced analysis checkbox=True branch."""

    def test_checkbox_triggers_visualization(self):
        st = _make_st()
        st.checkbox.return_value = True

        mock_ui_instance, extra = _setup_membrane_mocks()

        with _cell_config_env(st, extra_modules=extra) as mod:
            mod.render_membrane_configuration()

        mock_ui_instance.render_membrane_visualization.assert_called_once()
