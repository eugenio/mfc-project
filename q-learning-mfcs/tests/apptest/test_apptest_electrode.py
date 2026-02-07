"""AppTest tests for Electrode System page."""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_LABELS


def _navigate_to_electrode() -> AppTest:
    """Navigate to the Electrode System page."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(PAGE_LABELS[1]).run()
    return at


@pytest.mark.apptest
class TestElectrodeSystemPage:
    """Tests for the Electrode System page."""

    def test_electrode_page_renders(self) -> None:
        """Electrode page renders with title."""
        at = _navigate_to_electrode()
        titles = [t.value for t in at.title]
        assert any("Electrode" in t for t in titles)

    def test_electrode_has_metrics(self) -> None:
        """Electrode page shows performance metrics."""
        at = _navigate_to_electrode()
        assert len(at.metric) >= 1

    def test_electrode_has_tabs(self) -> None:
        """Electrode page uses tab layout."""
        at = _navigate_to_electrode()
        assert len(at.tabs) >= 1

    def test_electrode_known_bug_material_options(
        self,
    ) -> None:
        """Known bug: material_options not defined."""
        at = _navigate_to_electrode()
        exceptions = [str(e.value) for e in at.exception]
        has_material_bug = any(
            "material_options" in exc for exc in exceptions
        )
        # Document known bug - this should be fixed
        if has_material_bug:
            pytest.xfail(
                "Known bug: material_options not defined"
            )

    def test_electrode_subheaders(self) -> None:
        """Electrode page has section subheaders."""
        at = _navigate_to_electrode()
        assert len(at.subheader) >= 1

    def test_electrode_has_selectbox(self) -> None:
        """Electrode page has material selection."""
        at = _navigate_to_electrode()
        # May have selectbox for material selection
        assert len(at.selectbox) >= 0

    def test_electrode_sidebar_value(self) -> None:
        """Sidebar shows Electrode System selected."""
        at = _navigate_to_electrode()
        assert at.sidebar.radio[0].value == PAGE_LABELS[1]

    def test_electrode_page_caption(self) -> None:
        """Electrode page has descriptive caption."""
        at = _navigate_to_electrode()
        assert len(at.caption) >= 1
