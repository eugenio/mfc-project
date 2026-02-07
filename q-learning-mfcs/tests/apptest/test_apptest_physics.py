"""AppTest tests for Advanced Physics page."""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_LABELS


def _navigate_to_physics() -> AppTest:
    """Navigate to the Advanced Physics page."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(PAGE_LABELS[3]).run()
    return at


@pytest.mark.apptest
class TestAdvancedPhysicsPage:
    """Tests for the Advanced Physics page."""

    def test_physics_loads(self) -> None:
        """Physics page loads without exceptions."""
        at = _navigate_to_physics()
        assert len(at.exception) == 0

    def test_physics_title(self) -> None:
        """Physics page shows correct title."""
        at = _navigate_to_physics()
        titles = [t.value for t in at.title]
        assert any(
            "Physics" in t or "Simulation" in t
            for t in titles
        )

    def test_physics_has_number_inputs(self) -> None:
        """Physics page has simulation parameter inputs."""
        at = _navigate_to_physics()
        assert len(at.number_input) >= 1

    def test_physics_sidebar_value(self) -> None:
        """Sidebar shows Physics Simulation selected."""
        at = _navigate_to_physics()
        assert at.sidebar.radio[0].value == PAGE_LABELS[3]

    def test_physics_has_subheaders(self) -> None:
        """Physics page has section subheaders."""
        at = _navigate_to_physics()
        assert len(at.subheader) >= 1

    def test_physics_has_metrics(self) -> None:
        """Physics page shows simulation metrics."""
        at = _navigate_to_physics()
        assert len(at.metric) >= 1
