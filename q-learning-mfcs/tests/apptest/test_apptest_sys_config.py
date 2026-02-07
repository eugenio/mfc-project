"""AppTest tests for System Configuration page."""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_LABELS


def _navigate_to_sys_config() -> AppTest:
    """Navigate to the System Configuration page."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(PAGE_LABELS[8]).run()
    return at


@pytest.mark.apptest
class TestSystemConfigPage:
    """Tests for the System Configuration page."""

    def test_sys_config_loads(self) -> None:
        """System Config page loads without exceptions."""
        at = _navigate_to_sys_config()
        assert len(at.exception) == 0

    def test_sys_config_title(self) -> None:
        """System Config page shows correct title."""
        at = _navigate_to_sys_config()
        titles = [t.value for t in at.title]
        assert any("Configuration" in t for t in titles)

    def test_sys_config_has_tabs(self) -> None:
        """System Config page uses tab layout."""
        at = _navigate_to_sys_config()
        assert len(at.tabs) >= 1

    def test_sys_config_sidebar_value(self) -> None:
        """Sidebar shows Configuration selected."""
        at = _navigate_to_sys_config()
        assert at.sidebar.radio[0].value == PAGE_LABELS[8]

    def test_sys_config_has_subheaders(self) -> None:
        """System Config has section subheaders."""
        at = _navigate_to_sys_config()
        assert len(at.subheader) >= 1
