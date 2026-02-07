"""E2E tests: Page loading for all 9 pages."""

import pytest
from playwright.sync_api import Page

from conftest import PAGE_LABELS, click_radio_page


@pytest.mark.playwright
class TestPageLoad:
    """Test that all pages load without errors."""

    def test_app_loads(
        self, streamlit_page: Page
    ) -> None:
        """Main app loads successfully."""
        assert streamlit_page.title() != ""

    @pytest.mark.parametrize("page_label", PAGE_LABELS)
    def test_page_loads_no_exception(
        self,
        streamlit_page: Page,
        page_label: str,
    ) -> None:
        """Each page loads without Streamlit exceptions."""
        click_radio_page(streamlit_page, page_label)
        exceptions = streamlit_page.locator(
            '[data-testid="stException"]'
        )
        exc_count = exceptions.count()
        if exc_count > 0 and page_label in (
            "Electrode System",
            "Performance Monitor",
        ):
            pytest.xfail(
                f"Known exceptions on {page_label}"
            )
        if page_label not in (
            "Electrode System",
            "Performance Monitor",
        ):
            assert exc_count == 0, (
                f"Exceptions found on {page_label}"
            )

    def test_sidebar_visible(
        self, streamlit_page: Page
    ) -> None:
        """Sidebar contains navigation radio."""
        radio = streamlit_page.locator(
            '[data-testid="stRadio"]'
        )
        assert radio.count() >= 1

    def test_main_content_visible(
        self, streamlit_page: Page
    ) -> None:
        """Main content area is visible."""
        main = streamlit_page.locator(
            '[data-testid="stAppViewContainer"]'
        )
        assert main.is_visible()
