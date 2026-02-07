"""E2E tests: Form interactions."""

import pytest
from playwright.sync_api import Page

from conftest import click_radio_page


@pytest.mark.playwright
class TestFormInteractions:
    """Test form input and submission interactions."""

    def test_cell_config_selectbox(
        self, streamlit_page: Page
    ) -> None:
        """Cell Config page selectbox is interactive."""
        click_radio_page(
            streamlit_page, "Cell Configuration"
        )
        selectboxes = streamlit_page.locator(
            '[data-testid="stSelectbox"]'
        )
        if selectboxes.count() > 0:
            selectboxes.first.click()
            streamlit_page.wait_for_timeout(500)

    def test_cell_config_number_inputs(
        self, streamlit_page: Page
    ) -> None:
        """Cell Config page number inputs are present."""
        click_radio_page(
            streamlit_page, "Cell Configuration"
        )
        inputs = streamlit_page.locator(
            '[data-testid="stNumberInput"]'
        )
        assert inputs.count() >= 1

    def test_cell_config_tabs(
        self, streamlit_page: Page
    ) -> None:
        """Cell Config page has tab navigation."""
        click_radio_page(
            streamlit_page, "Cell Configuration"
        )
        tabs = streamlit_page.locator('[data-testid="stTabs"]')
        assert tabs.count() >= 1

    def test_ml_optimization_inputs(
        self, streamlit_page: Page
    ) -> None:
        """ML Optimization page inputs are visible."""
        click_radio_page(
            streamlit_page, "ML Optimization"
        )
        inputs = streamlit_page.locator(
            '[data-testid="stNumberInput"]'
        )
        assert inputs.count() >= 1

    def test_ml_optimization_selectbox(
        self, streamlit_page: Page
    ) -> None:
        """ML Optimization page has selectbox."""
        click_radio_page(
            streamlit_page, "ML Optimization"
        )
        selectboxes = streamlit_page.locator(
            '[data-testid="stSelectbox"]'
        )
        assert selectboxes.count() >= 1

    def test_config_page_checkboxes(
        self, streamlit_page: Page
    ) -> None:
        """Config page has interactive checkboxes."""
        click_radio_page(
            streamlit_page, "Configuration"
        )
        checkboxes = streamlit_page.locator(
            '[data-testid="stCheckbox"]'
        )
        assert checkboxes.count() >= 1

    def test_config_page_tabs(
        self, streamlit_page: Page
    ) -> None:
        """Config page has tab navigation."""
        click_radio_page(
            streamlit_page, "Configuration"
        )
        tabs = streamlit_page.locator(
            '[data-testid="stTabs"]'
        )
        assert tabs.count() >= 1

    def test_physics_number_inputs(
        self, streamlit_page: Page
    ) -> None:
        """Physics page has parameter inputs."""
        click_radio_page(
            streamlit_page, "Physics Simulation"
        )
        inputs = streamlit_page.locator(
            '[data-testid="stNumberInput"]'
        )
        assert inputs.count() >= 1

    def test_dashboard_metrics_display(
        self, streamlit_page: Page
    ) -> None:
        """Dashboard displays metric widgets."""
        click_radio_page(streamlit_page, "Dashboard")
        metrics = streamlit_page.locator(
            '[data-testid="stMetric"]'
        )
        assert metrics.count() >= 1

    def test_gsm_tabs_navigation(
        self, streamlit_page: Page
    ) -> None:
        """GSM page tabs are clickable."""
        click_radio_page(
            streamlit_page, "GSM Integration"
        )
        tabs = streamlit_page.locator(
            '[data-testid="stTabs"]'
        )
        if tabs.count() > 0:
            tab_buttons = tabs.locator(
                '[role="tab"]'
            )
            if tab_buttons.count() > 1:
                tab_buttons.nth(1).click()
                streamlit_page.wait_for_timeout(500)
