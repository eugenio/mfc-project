"""AppTest tests for core layout functions."""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_LABELS


@pytest.mark.apptest
class TestCoreLayout:
    """Tests for the core layout framework."""

    def test_sidebar_title(self, app: AppTest) -> None:
        """Sidebar shows platform title."""
        titles = [t.value for t in app.title]
        assert any("MFC Platform" in t for t in titles)

    def test_sidebar_has_radio(self, app: AppTest) -> None:
        """Sidebar has navigation radio widget."""
        assert len(app.sidebar.radio) == 1

    def test_sidebar_radio_label(
        self, app: AppTest
    ) -> None:
        """Sidebar radio has navigation label."""
        radio = app.sidebar.radio[0]
        assert radio.label == "Navigate to:"

    def test_sidebar_phase_status(
        self, app: AppTest
    ) -> None:
        """Sidebar shows phase status text."""
        # Phase status is displayed as sidebar text
        assert len(app.sidebar.text) >= 1

    def test_theme_applied(self, app: AppTest) -> None:
        """Enhanced theme CSS is applied via markdown."""
        assert len(app.markdown) >= 1
