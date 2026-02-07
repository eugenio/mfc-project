"""AppTest navigation tests for MFC GUI.

Tests cross-page navigation using Streamlit's AppTest framework.
"""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_KEYS, PAGE_LABELS


@pytest.mark.apptest
class TestNavigationBasic:
    """Basic navigation tests for all 9 pages."""

    def test_app_loads_without_exception(self, app: AppTest) -> None:
        """App initial load should have no exceptions."""
        assert len(app.exception) == 0

    def test_default_page_is_dashboard(self, app: AppTest) -> None:
        """Default page should be Dashboard."""
        assert app.sidebar.radio[0].value == PAGE_LABELS[0]

    def test_sidebar_has_all_pages(self, app: AppTest) -> None:
        """Sidebar radio should contain all 9 page options."""
        radio = app.sidebar.radio[0]
        assert radio.options == PAGE_LABELS

    @pytest.mark.parametrize("page_label", PAGE_LABELS)
    def test_navigate_to_each_page(
        self, page_label: str
    ) -> None:
        """Each page should be navigable without crashing."""
        at = AppTest.from_file(APP_FILE, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value(page_label).run()
        # Page should render (at least the MFC Platform title)
        titles = [t.value for t in at.title]
        assert len(titles) >= 1


@pytest.mark.apptest
class TestNavigationSequential:
    """Test sequential navigation between pages."""

    def test_navigate_dashboard_to_electrode(
        self, app: AppTest
    ) -> None:
        """Navigate from Dashboard to Electrode System."""
        app.sidebar.radio[0].set_value(PAGE_LABELS[1]).run()
        titles = [t.value for t in app.title]
        assert any("Electrode" in t for t in titles)

    def test_navigate_dashboard_to_cell_config(
        self, app: AppTest
    ) -> None:
        """Navigate from Dashboard to Cell Configuration."""
        app.sidebar.radio[0].set_value(PAGE_LABELS[2]).run()
        titles = [t.value for t in app.title]
        assert any("Cell Configuration" in t for t in titles)

    def test_navigate_back_to_dashboard(
        self, app: AppTest
    ) -> None:
        """Navigate away and back to Dashboard."""
        app.sidebar.radio[0].set_value(PAGE_LABELS[3]).run()
        app.sidebar.radio[0].set_value(PAGE_LABELS[0]).run()
        titles = [t.value for t in app.title]
        assert any("Dashboard" in t for t in titles)

    def test_sidebar_radio_value_updates(
        self, app: AppTest
    ) -> None:
        """Sidebar radio value should reflect current page."""
        target = PAGE_LABELS[4]
        app.sidebar.radio[0].set_value(target).run()
        assert app.sidebar.radio[0].value == target
