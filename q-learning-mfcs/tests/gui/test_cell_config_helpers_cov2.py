"""Coverage tests for gui/pages/cell_config_helpers.py -- target 99%+.

Covers: render_validation_analysis geometric, performance, optimization
branches that are still uncovered.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


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
    st.number_input.return_value = 10.0
    st.checkbox.return_value = False
    st.button.return_value = False
    st.expander.return_value = _make_col()
    st.session_state = MagicMock()
    st.download_button = MagicMock()
    return st


_mock_st = _make_st()
sys.modules.setdefault("streamlit", _mock_st)

from gui.pages.cell_config_helpers import (
    render_3d_model_upload,
    render_cell_calculations,
    render_validation_analysis,
)


@pytest.mark.coverage_extra
class TestRender3dModelUpload:
    def test_renders(self):
        st = _make_st()
        with patch("gui.pages.cell_config_helpers.st", st):
            render_3d_model_upload()
            st.info.assert_called()


@pytest.mark.coverage_extra
class TestRenderCellCalculations:
    def test_with_electrode_area(self):
        st = _make_st()
        st.session_state.cell_config = {
            "volume": 480.0,
            "electrode_area": 60.0,
            "electrode_spacing": 5.0,
        }
        with patch("gui.pages.cell_config_helpers.st", st):
            render_cell_calculations()
            st.metric.assert_called()

    def test_zero_electrode_area(self):
        st = _make_st()
        st.session_state.cell_config = {
            "volume": 480.0,
            "electrode_area": 0,
            "electrode_spacing": 5.0,
        }
        with patch("gui.pages.cell_config_helpers.st", st):
            render_cell_calculations()


@pytest.mark.coverage_extra
class TestRenderValidationAnalysis:
    def test_no_cell_config(self):
        """Cover early return when no cell_config."""
        st = _make_st()
        st.session_state.__contains__ = MagicMock(return_value=False)
        with patch("gui.pages.cell_config_helpers.st", st):
            render_validation_analysis()
            st.warning.assert_called()

    def test_small_volume(self):
        """Cover volume < 10 error."""
        st = _make_st()
        st.session_state.__contains__ = MagicMock(return_value=True)
        st.session_state.__bool__ = MagicMock(return_value=True)
        st.session_state.cell_config = {
            "type": "rectangular",
            "volume": 5,
            "electrode_area": 5,
            "electrode_spacing": 0.5,
            "length": 2,
            "width": 1,
            "height": 1,
        }
        st.button.return_value = False
        with patch("gui.pages.cell_config_helpers.st", st):
            render_validation_analysis()

    def test_large_volume(self):
        """Cover volume > 10000 warning."""
        st = _make_st()
        st.session_state.__contains__ = MagicMock(return_value=True)
        st.session_state.__bool__ = MagicMock(return_value=True)
        st.session_state.cell_config = {
            "type": "cylindrical",
            "volume": 15000,
            "electrode_area": 1500,
            "electrode_spacing": 12,
        }
        st.button.return_value = False
        with patch("gui.pages.cell_config_helpers.st", st):
            render_validation_analysis()

    def test_optimal_config(self):
        """Cover success paths for volume, area, spacing."""
        st = _make_st()
        st.session_state.__contains__ = MagicMock(return_value=True)
        st.session_state.__bool__ = MagicMock(return_value=True)
        st.session_state.cell_config = {
            "type": "rectangular",
            "volume": 500,
            "electrode_area": 80,
            "electrode_spacing": 4,
            "length": 10,
            "width": 10,
            "height": 5,
        }
        st.button.return_value = False
        with patch("gui.pages.cell_config_helpers.st", st):
            render_validation_analysis()

    def test_high_aspect_ratio(self):
        """Cover aspect_ratio > 5 warning."""
        st = _make_st()
        st.session_state.__contains__ = MagicMock(return_value=True)
        st.session_state.__bool__ = MagicMock(return_value=True)
        st.session_state.cell_config = {
            "type": "rectangular",
            "volume": 500,
            "electrode_area": 80,
            "electrode_spacing": 4,
            "length": 60,
            "width": 10,
            "height": 5,
        }
        st.button.return_value = False
        with patch("gui.pages.cell_config_helpers.st", st):
            render_validation_analysis()

    def test_custom_type_suggestion(self):
        """Cover custom type optimization suggestion."""
        st = _make_st()
        st.session_state.__contains__ = MagicMock(return_value=True)
        st.session_state.__bool__ = MagicMock(return_value=True)
        st.session_state.cell_config = {
            "type": "custom",
            "volume": 1500,
            "electrode_area": 30,
            "electrode_spacing": 8,
        }
        st.button.return_value = False
        with patch("gui.pages.cell_config_helpers.st", st):
            render_validation_analysis()

    def test_no_suggestions(self):
        """Cover the well-optimized config path."""
        st = _make_st()
        st.session_state.__contains__ = MagicMock(return_value=True)
        st.session_state.__bool__ = MagicMock(return_value=True)
        st.session_state.cell_config = {
            "type": "cylindrical",
            "volume": 500,
            "electrode_area": 80,
            "electrode_spacing": 3,
        }
        st.button.return_value = False
        with patch("gui.pages.cell_config_helpers.st", st):
            render_validation_analysis()

    def test_generate_report_button(self):
        """Cover generate report download button."""
        st = _make_st()
        st.session_state.__contains__ = MagicMock(return_value=True)
        st.session_state.__bool__ = MagicMock(return_value=True)
        st.session_state.cell_config = {
            "type": "rectangular",
            "volume": 500,
            "electrode_area": 80,
            "electrode_spacing": 4,
            "length": 10,
            "width": 10,
            "height": 5,
        }
        # Make the Generate Report button return True
        st.button.side_effect = lambda *a, **kw: (
            "Generate" in str(a)
            or "Learn More" in str(a)
        )
        with patch("gui.pages.cell_config_helpers.st", st):
            render_validation_analysis()

    def test_zero_electrode_area_performance(self):
        """Cover performance tab when electrode_area is 0."""
        st = _make_st()
        st.session_state.__contains__ = MagicMock(return_value=True)
        st.session_state.__bool__ = MagicMock(return_value=True)
        st.session_state.cell_config = {
            "type": "h_type",
            "volume": 500,
            "electrode_area": 0,
            "electrode_spacing": 5,
        }
        st.button.return_value = False
        with patch("gui.pages.cell_config_helpers.st", st):
            render_validation_analysis()
