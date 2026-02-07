"""AppTest tests for ML Optimization page."""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_LABELS


def _navigate_to_ml() -> AppTest:
    """Navigate to the ML Optimization page."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(PAGE_LABELS[4]).run()
    return at


@pytest.mark.apptest
class TestMLOptimizationPage:
    """Tests for the ML Optimization page."""

    def test_ml_loads(self) -> None:
        """ML page loads without exceptions."""
        at = _navigate_to_ml()
        assert len(at.exception) == 0

    def test_ml_title(self) -> None:
        """ML page shows correct title."""
        at = _navigate_to_ml()
        titles = [t.value for t in at.title]
        assert any(
            "ML" in t or "Optimization" in t
            for t in titles
        )

    def test_ml_has_selectbox(self) -> None:
        """ML page has method selection widgets."""
        at = _navigate_to_ml()
        assert len(at.selectbox) >= 1

    def test_ml_sidebar_value(self) -> None:
        """Sidebar shows ML Optimization selected."""
        at = _navigate_to_ml()
        assert at.sidebar.radio[0].value == PAGE_LABELS[4]

    def test_ml_has_number_inputs(self) -> None:
        """ML page has parameter number inputs."""
        at = _navigate_to_ml()
        assert len(at.number_input) >= 1

    def test_ml_has_subheaders(self) -> None:
        """ML page has method section headers."""
        at = _navigate_to_ml()
        assert len(at.subheader) >= 1

    def test_ml_has_checkboxes(self) -> None:
        """ML page has configuration checkboxes."""
        at = _navigate_to_ml()
        assert len(at.checkbox) >= 1


@pytest.mark.apptest
class TestMLOptimizationInteractions:
    """Widget interaction tests for ML Optimization."""

    def test_change_acquisition_function(self) -> None:
        """Change acquisition function selectbox."""
        at = _navigate_to_ml()
        acq = at.selectbox[0]
        assert acq.value == "Expected Improvement"
        acq.set_value("Upper Confidence Bound").run()
        assert len(at.exception) == 0

    def test_change_gp_kernel(self) -> None:
        """Change GP kernel selectbox."""
        at = _navigate_to_ml()
        kernel = at.selectbox[1]
        assert kernel.value == "RBF"
        kernel.set_value("Matern").run()
        assert len(at.exception) == 0

    def test_toggle_parallel_eval(self) -> None:
        """Toggle parallel evaluations checkbox."""
        at = _navigate_to_ml()
        cb = at.checkbox[0]
        original = cb.value
        cb.set_value(not original).run()
        assert at.checkbox[0].value != original

    def test_change_max_iterations(self) -> None:
        """Change maximum iterations number input."""
        at = _navigate_to_ml()
        iters = at.number_input[0]
        iters.set_value(100).run()
        assert at.number_input[0].value == 100
        assert len(at.exception) == 0

    def test_selectbox_options(self) -> None:
        """Verify selectbox options are correct."""
        at = _navigate_to_ml()
        acq_opts = at.selectbox[0].options
        assert "Expected Improvement" in acq_opts
        assert "Upper Confidence Bound" in acq_opts
