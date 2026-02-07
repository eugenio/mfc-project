"""AppTest tests for GSM Integration page."""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_LABELS


def _navigate_to_gsm() -> AppTest:
    """Navigate to the GSM Integration page."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(PAGE_LABELS[5]).run()
    return at


@pytest.mark.apptest
class TestGSMIntegrationPage:
    """Tests for the GSM Integration page."""

    def test_gsm_loads(self) -> None:
        """GSM page loads without exceptions."""
        at = _navigate_to_gsm()
        assert len(at.exception) == 0

    def test_gsm_title(self) -> None:
        """GSM page shows correct title."""
        at = _navigate_to_gsm()
        titles = [t.value for t in at.title]
        assert any("GSM" in t for t in titles)

    def test_gsm_has_tabs(self) -> None:
        """GSM page uses tab layout."""
        at = _navigate_to_gsm()
        assert len(at.tabs) >= 1

    def test_gsm_sidebar_value(self) -> None:
        """Sidebar shows GSM Integration selected."""
        at = _navigate_to_gsm()
        assert at.sidebar.radio[0].value == PAGE_LABELS[5]

    def test_gsm_has_subheaders(self) -> None:
        """GSM page has section subheaders."""
        at = _navigate_to_gsm()
        assert len(at.subheader) >= 1
