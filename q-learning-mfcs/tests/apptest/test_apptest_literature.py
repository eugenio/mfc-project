"""AppTest tests for Literature Validation page."""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_LABELS


def _navigate_to_literature() -> AppTest:
    """Navigate to the Literature Validation page."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(PAGE_LABELS[6]).run()
    return at


@pytest.mark.apptest
class TestLiteratureValidationPage:
    """Tests for the Literature Validation page."""

    def test_literature_loads(self) -> None:
        """Literature page loads without exceptions."""
        at = _navigate_to_literature()
        assert len(at.exception) == 0

    def test_literature_title(self) -> None:
        """Literature page shows correct title."""
        at = _navigate_to_literature()
        titles = [t.value for t in at.title]
        assert any("Literature" in t for t in titles)

    def test_literature_has_tabs(self) -> None:
        """Literature page uses tab layout."""
        at = _navigate_to_literature()
        assert len(at.tabs) >= 1

    def test_literature_sidebar_value(self) -> None:
        """Sidebar shows Literature Validation selected."""
        at = _navigate_to_literature()
        assert at.sidebar.radio[0].value == PAGE_LABELS[6]

    def test_literature_has_metrics(self) -> None:
        """Literature page shows validation metrics."""
        at = _navigate_to_literature()
        assert len(at.metric) >= 1
