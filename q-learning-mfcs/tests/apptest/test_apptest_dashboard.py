"""AppTest tests for Dashboard page."""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_LABELS


@pytest.mark.apptest
class TestDashboardPage:
    """Tests for the Dashboard page."""

    def test_dashboard_loads(self, app: AppTest) -> None:
        """Dashboard loads without exceptions."""
        assert len(app.exception) == 0

    def test_dashboard_title(self, app: AppTest) -> None:
        """Dashboard shows correct title."""
        titles = [t.value for t in app.title]
        assert any("Dashboard" in t for t in titles)

    def test_dashboard_has_metrics(self, app: AppTest) -> None:
        """Dashboard displays system metrics."""
        assert len(app.metric) >= 4

    def test_dashboard_phase_status(self, app: AppTest) -> None:
        """Dashboard shows phase status section."""
        subheaders = [s.value for s in app.subheader]
        assert any("Phase" in s for s in subheaders)

    def test_dashboard_has_success_indicators(
        self, app: AppTest
    ) -> None:
        """Dashboard shows status indicators."""
        assert len(app.success) >= 1 or len(app.info) >= 1
