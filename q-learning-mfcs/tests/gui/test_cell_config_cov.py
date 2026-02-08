"""Coverage boost tests for gui/pages/cell_config.py."""
import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


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


# Mock streamlit and heavy deps before import
_mock_st = _make_st()
sys.modules.setdefault("streamlit", _mock_st)

# Mock gui.pages.cell_config_helpers and gui.scientific_widgets
_mock_helpers = MagicMock()
sys.modules.setdefault("gui.pages.cell_config_helpers", _mock_helpers)
_mock_widgets = MagicMock()
sys.modules.setdefault("gui.scientific_widgets", _mock_widgets)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gui.pages.cell_config import (
    CELL_PARAMETERS,
    render_cell_configuration_interface,
    render_cell_configuration_page,
    render_cylindrical_cell_parameters,
    render_h_type_cell_parameters,
    render_mec_cell_parameters,
    render_membrane_configuration,
    render_rectangular_cell_parameters,
    render_simple_geometries,
    render_tubular_cell_parameters,
)


class TestCellParameters:
    def test_parameters_defined(self):
        assert "volume" in CELL_PARAMETERS
        assert "electrode_spacing" in CELL_PARAMETERS
        assert "membrane_area" in CELL_PARAMETERS
        assert "flow_rate" in CELL_PARAMETERS


class TestRenderCellConfigPage:
    def test_renders(self):
        st = _make_st()
        with patch("gui.pages.cell_config.st", st):
            render_cell_configuration_page()
            st.title.assert_called()


class TestRenderCellConfigInterface:
    def test_renders_tabs(self):
        st = _make_st()
        with patch("gui.pages.cell_config.st", st):
            render_cell_configuration_interface()


class TestRenderSimpleGeometries:
    def test_rectangular(self):
        st = _make_st()
        st.selectbox.return_value = "Rectangular Chamber"
        with patch("gui.pages.cell_config.st", st):
            render_simple_geometries()

    def test_cylindrical(self):
        st = _make_st()
        st.selectbox.return_value = "Cylindrical Reactor"
        with patch("gui.pages.cell_config.st", st):
            render_simple_geometries()

    def test_h_type(self):
        st = _make_st()
        st.selectbox.return_value = "H-Type Cell"
        with patch("gui.pages.cell_config.st", st):
            render_simple_geometries()

    def test_tubular(self):
        st = _make_st()
        st.selectbox.return_value = "Tubular MFC"
        with patch("gui.pages.cell_config.st", st):
            render_simple_geometries()

    def test_mec(self):
        st = _make_st()
        st.selectbox.return_value = "Microbial Electrolysis Cell"
        with patch("gui.pages.cell_config.st", st):
            render_simple_geometries()

    def test_custom(self):
        st = _make_st()
        st.selectbox.return_value = "Custom Geometry"
        with patch("gui.pages.cell_config.st", st):
            render_simple_geometries()

    def test_with_cell_config_in_session(self):
        st = _make_st()
        st.selectbox.return_value = "Rectangular Chamber"
        st.session_state.__contains__ = MagicMock(return_value=True)
        with patch("gui.pages.cell_config.st", st):
            render_simple_geometries()


class TestRenderRectangularCellParameters:
    def test_renders(self):
        st = _make_st()
        st.number_input.return_value = 10.0
        with patch("gui.pages.cell_config.st", st):
            render_rectangular_cell_parameters()


class TestRenderCylindricalCellParameters:
    def test_renders(self):
        st = _make_st()
        st.number_input.return_value = 8.0
        with patch("gui.pages.cell_config.st", st):
            render_cylindrical_cell_parameters()


class TestRenderHTypeCellParameters:
    def test_renders(self):
        st = _make_st()
        st.number_input.return_value = 250.0
        with patch("gui.pages.cell_config.st", st):
            render_h_type_cell_parameters()


class TestRenderTubularCellParameters:
    def test_renders(self):
        st = _make_st()
        st.number_input.return_value = 5.0
        with patch("gui.pages.cell_config.st", st):
            render_tubular_cell_parameters()


class TestRenderMecCellParameters:
    def test_renders(self):
        st = _make_st()
        st.number_input.return_value = 500.0
        with patch("gui.pages.cell_config.st", st):
            render_mec_cell_parameters()


class TestRenderMembraneConfiguration:
    def test_fallback_on_import_error(self):
        st = _make_st()
        st.selectbox.return_value = "Nafion 117"
        st.number_input.return_value = 25.0
        with patch("gui.pages.cell_config.st", st):
            # Force ImportError on local import of MembraneConfigurationUI
            with patch.dict(sys.modules, {"gui.membrane_configuration_ui": None}):
                render_membrane_configuration()

    def test_with_membrane_ui(self):
        st = _make_st()
        st.checkbox.return_value = False
        mock_ui_class = MagicMock()
        mock_ui_instance = MagicMock()
        mock_ui_class.return_value = mock_ui_instance
        mock_ui_instance.render_material_selector.return_value = MagicMock()
        mock_ui_instance.render_area_input.return_value = 0.0025
        mock_ui_instance.render_operating_conditions.return_value = (303.0, 7.0, 7.0)
        mock_mod = MagicMock()
        mock_mod.MembraneConfigurationUI = mock_ui_class
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
        with patch("gui.pages.cell_config.st", st):
            with patch.dict(sys.modules, {"gui.membrane_configuration_ui": mock_mod}):
                with patch(
                    "config.membrane_config.create_membrane_config",
                    return_value=mock_config,
                ):
                    render_membrane_configuration()
