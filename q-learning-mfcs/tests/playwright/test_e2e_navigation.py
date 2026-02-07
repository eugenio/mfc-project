"""E2E tests: Sidebar navigation."""

import pytest
from playwright.sync_api import Page

from conftest import PAGE_LABELS, _LABEL_TO_RADIO, click_radio_page


@pytest.mark.playwright
class TestNavigation:
    """Test sidebar navigation between pages."""

    def test_sidebar_has_all_pages(
        self, streamlit_page: Page
    ) -> None:
        """Sidebar radio contains all 9 navigation options."""
        radio = streamlit_page.locator(
            '[data-testid="stRadio"]'
        )
        for label in PAGE_LABELS:
            radio_label = _LABEL_TO_RADIO[label]
            option = radio.get_by_text(
                radio_label, exact=True
            )
            assert option.count() >= 1, (
                f"Missing radio option: {label}"
            )

    @pytest.mark.parametrize(
        "from_page,to_page",
        [
            ("Dashboard", "Electrode System"),
            ("Electrode System", "Cell Configuration"),
            ("Cell Configuration", "ML Optimization"),
            ("ML Optimization", "Dashboard"),
        ],
    )
    def test_navigate_between_pages(
        self,
        streamlit_page: Page,
        from_page: str,
        to_page: str,
    ) -> None:
        """Navigate between specific page pairs."""
        click_radio_page(streamlit_page, from_page)
        click_radio_page(streamlit_page, to_page)
        h1 = streamlit_page.locator("h1").first
        assert h1.is_visible()

    def test_navigate_all_pages_sequentially(
        self, streamlit_page: Page
    ) -> None:
        """Navigate through all pages in sequence."""
        for label in PAGE_LABELS:
            click_radio_page(streamlit_page, label)
            main = streamlit_page.locator(
                '[data-testid="stAppViewContainer"]'
            )
            assert main.is_visible()

    def test_sidebar_radio_label(
        self, streamlit_page: Page
    ) -> None:
        """Sidebar radio has 'Navigate to:' label."""
        radio = streamlit_page.locator(
            '[data-testid="stRadio"]'
        )
        label = radio.locator("label").first
        assert "Navigate" in (label.text_content() or "")
