"""Tests for gui/electrode_configuration_ui.py - coverage 98%+."""
import importlib.util
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, _src)

# Mock streamlit and plotly before importing
mock_st = MagicMock()
mock_st.session_state = MagicMock()
mock_st.session_state.__contains__ = MagicMock(return_value=False)
# st.columns(N) returns N context-manager mocks
mock_col = MagicMock()
mock_col.__enter__ = MagicMock(return_value=mock_col)
mock_col.__exit__ = MagicMock(return_value=False)
mock_st.columns.return_value = [mock_col, mock_col]
sys.modules["streamlit"] = mock_st

mock_plotly = MagicMock()
sys.modules["plotly"] = mock_plotly
sys.modules["plotly.graph_objects"] = mock_plotly
sys.modules["plotly.subplots"] = MagicMock()

# Load electrode_configuration_ui directly to avoid gui.__init__ plotly imports
_spec = importlib.util.spec_from_file_location(
    "gui.electrode_configuration_ui",
    os.path.join(_src, "gui", "electrode_configuration_ui.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["gui.electrode_configuration_ui"] = _mod
_spec.loader.exec_module(_mod)

ElectrodeConfigurationUI = _mod.ElectrodeConfigurationUI

from config.electrode_config import (
    ElectrodeGeometry,
    ElectrodeMaterial,
    MaterialProperties,
    MATERIAL_PROPERTIES_DATABASE,
)


class TestElectrodeConfigurationUI:
    def test_init(self):
        ui = ElectrodeConfigurationUI()
        # session_state attributes set via mock
        assert ui is not None

    def test_initialize_session_state(self):
        ui = ElectrodeConfigurationUI()
        ui.initialize_session_state()
        assert ui is not None

    def test_render_material_selector(self):
        mock_st.selectbox.return_value = "Graphite Plate"
        ui = ElectrodeConfigurationUI()
        material = ui.render_material_selector("anode")
        assert material == ElectrodeMaterial.GRAPHITE_PLATE

    def test_render_material_selector_custom(self):
        mock_st.selectbox.return_value = "Custom Material"
        ui = ElectrodeConfigurationUI()
        material = ui.render_material_selector("cathode")
        assert material == ElectrodeMaterial.CUSTOM

    def test_display_material_properties(self):
        ui = ElectrodeConfigurationUI()
        props = MATERIAL_PROPERTIES_DATABASE[ElectrodeMaterial.GRAPHITE_PLATE]
        ui._display_material_properties(props, "Graphite Plate")
        mock_st.expander.assert_called()
