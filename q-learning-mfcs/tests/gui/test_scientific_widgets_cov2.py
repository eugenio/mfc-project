"""Coverage tests for gui/scientific_widgets.py -- target 99%+.

Covers: ParameterSpec dataclass, ScientificParameterWidget.render,
_show_validation_feedback (in-range, below, above), literature refs,
create_parameter_section, MFC_ELECTRODE_PARAMETERS.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture(autouse=True)
def _mock_st():
    """Provide a streamlit mock."""
    mock = MagicMock()
    mock.number_input = MagicMock(return_value=50.0)
    mock.subheader = MagicMock()
    mock.success = MagicMock()
    mock.warning = MagicMock()
    mock.expander = MagicMock()
    mock.info = MagicMock()
    with patch.dict("sys.modules", {"streamlit": mock}):
        yield mock


@pytest.mark.coverage_extra
class TestParameterSpec:
    def test_fields(self, _mock_st):
        from gui.scientific_widgets import ParameterSpec

        spec = ParameterSpec(
            name="Conductivity",
            unit="S/m",
            min_value=0.1,
            max_value=1e7,
            typical_range=(100.0, 1e5),
            literature_refs="Logan 2008",
            description="Electrode conductivity",
        )
        assert spec.name == "Conductivity"
        assert spec.unit == "S/m"
        assert spec.min_value == 0.1
        assert spec.typical_range == (100.0, 1e5)


@pytest.mark.coverage_extra
class TestScientificParameterWidget:
    def _make_spec(self, _mock_st):
        from gui.scientific_widgets import ParameterSpec

        return ParameterSpec(
            name="Test",
            unit="V",
            min_value=0.0,
            max_value=100.0,
            typical_range=(10.0, 50.0),
            literature_refs="Ref 2020",
            description="A test param",
        )

    def test_render_returns_value(self, _mock_st):
        from gui.scientific_widgets import ScientificParameterWidget

        spec = self._make_spec(_mock_st)
        _mock_st.number_input.return_value = 25.0
        widget = ScientificParameterWidget(spec, "k1")
        val = widget.render("My Label", 25.0)
        assert val == 25.0
        _mock_st.number_input.assert_called_once()

    def test_validation_in_range(self, _mock_st):
        from gui.scientific_widgets import ScientificParameterWidget

        spec = self._make_spec(_mock_st)
        widget = ScientificParameterWidget(spec, "k2")
        widget._show_validation_feedback(30.0)
        _mock_st.success.assert_called()

    def test_validation_below_range(self, _mock_st):
        from gui.scientific_widgets import ScientificParameterWidget

        spec = self._make_spec(_mock_st)
        widget = ScientificParameterWidget(spec, "k3")
        widget._show_validation_feedback(5.0)
        _mock_st.warning.assert_called()

    def test_validation_above_range(self, _mock_st):
        from gui.scientific_widgets import ScientificParameterWidget

        spec = self._make_spec(_mock_st)
        widget = ScientificParameterWidget(spec, "k4")
        widget._show_validation_feedback(60.0)
        _mock_st.warning.assert_called()

    def test_no_literature_refs(self, _mock_st):
        from gui.scientific_widgets import ParameterSpec, ScientificParameterWidget

        spec = ParameterSpec(
            name="NoRef",
            unit="m",
            min_value=0.0,
            max_value=10.0,
            typical_range=(1.0, 5.0),
            literature_refs="",
            description="No ref",
        )
        widget = ScientificParameterWidget(spec, "k5")
        widget._show_validation_feedback(3.0)
        _mock_st.expander.assert_not_called()


@pytest.mark.coverage_extra
class TestCreateParameterSection:
    def test_creates_values(self, _mock_st):
        from gui.scientific_widgets import (
            ParameterSpec,
            create_parameter_section,
        )

        spec = ParameterSpec(
            name="P1",
            unit="A",
            min_value=0.0,
            max_value=100.0,
            typical_range=(10.0, 50.0),
            literature_refs="Ref",
            description="desc",
        )
        _mock_st.number_input.return_value = 42.0
        params = {
            "param_a": {"spec": spec, "label": "Param A", "default": 42.0},
        }
        result = create_parameter_section("Section Title", params)
        assert "param_a" in result
        assert result["param_a"] == 42.0
        _mock_st.subheader.assert_called_with("Section Title")


@pytest.mark.coverage_extra
class TestMFCElectrodeParameters:
    def test_conductivity_spec(self, _mock_st):
        from gui.scientific_widgets import MFC_ELECTRODE_PARAMETERS

        assert "conductivity" in MFC_ELECTRODE_PARAMETERS
        spec = MFC_ELECTRODE_PARAMETERS["conductivity"]
        assert spec.name == "Electrical Conductivity"
        assert spec.unit == "S/m"
