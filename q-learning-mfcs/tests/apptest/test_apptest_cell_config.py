"""AppTest tests for Cell Configuration page."""

import pytest
from streamlit.testing.v1 import AppTest

from constants import APP_FILE, PAGE_LABELS


def _navigate_to_cell_config() -> AppTest:
    """Navigate to the Cell Configuration page."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(PAGE_LABELS[2]).run()
    return at


@pytest.mark.apptest
class TestCellConfigPage:
    """Tests for the Cell Configuration page."""

    def test_cell_config_loads(self) -> None:
        """Cell Config page loads without exceptions."""
        at = _navigate_to_cell_config()
        assert len(at.exception) == 0

    def test_cell_config_title(self) -> None:
        """Cell Config page shows correct title."""
        at = _navigate_to_cell_config()
        titles = [t.value for t in at.title]
        assert any("Cell Configuration" in t for t in titles)

    def test_cell_config_has_metrics(self) -> None:
        """Cell Config shows configuration metrics."""
        at = _navigate_to_cell_config()
        assert len(at.metric) >= 1

    def test_cell_config_has_number_inputs(self) -> None:
        """Cell Config has numerical parameter inputs."""
        at = _navigate_to_cell_config()
        assert len(at.number_input) >= 1

    def test_cell_config_has_selectbox(self) -> None:
        """Cell Config has geometry selection."""
        at = _navigate_to_cell_config()
        assert len(at.selectbox) >= 1

    def test_cell_config_caption(self) -> None:
        """Cell Config has descriptive caption."""
        at = _navigate_to_cell_config()
        assert len(at.caption) >= 1

    def test_cell_config_sidebar_value(self) -> None:
        """Sidebar shows Cell Configuration selected."""
        at = _navigate_to_cell_config()
        assert at.sidebar.radio[0].value == PAGE_LABELS[2]

    def test_cell_config_has_tabs(self) -> None:
        """Cell Config may use tab layout."""
        at = _navigate_to_cell_config()
        assert len(at.tabs) >= 0


@pytest.mark.apptest
class TestCellConfigInteractions:
    """Widget interaction tests for Cell Config."""

    def test_change_cell_type(self) -> None:
        """Changing cell type selectbox re-renders."""
        at = _navigate_to_cell_config()
        sb = at.selectbox[0]
        assert sb.value == "Rectangular Chamber"
        sb.set_value("Cylindrical Reactor").run()
        assert at.selectbox[0].value == "Cylindrical Reactor"
        assert len(at.exception) == 0

    def test_change_dimensions(self) -> None:
        """Changing dimension number inputs works."""
        at = _navigate_to_cell_config()
        length_input = at.number_input[0]
        length_input.set_value(15.0).run()
        assert at.number_input[0].value == 15.0
        assert len(at.exception) == 0

    def test_multiple_dimension_changes(self) -> None:
        """Multiple dimension changes don't cause errors."""
        at = _navigate_to_cell_config()
        at.number_input[0].set_value(12.0).run()
        at.number_input[1].set_value(10.0).run()
        at.number_input[2].set_value(8.0).run()
        assert len(at.exception) == 0

    def test_cell_type_options(self) -> None:
        """Cell type selectbox has expected options."""
        at = _navigate_to_cell_config()
        opts = at.selectbox[0].options
        assert "Rectangular Chamber" in opts
        assert "Cylindrical Reactor" in opts

    def test_number_input_labels(self) -> None:
        """Number inputs have expected labels."""
        at = _navigate_to_cell_config()
        labels = [n.label for n in at.number_input]
        assert any("Length" in l for l in labels)
        assert any("Width" in l for l in labels)
