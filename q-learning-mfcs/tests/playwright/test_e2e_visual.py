"""E2E tests: Visual screenshot capture."""

import os

import pytest
from playwright.sync_api import Page

from conftest import PAGE_LABELS, SCREENSHOT_DIR, click_radio_page


@pytest.mark.playwright
class TestVisualScreenshots:
    """Capture screenshots for visual inspection."""

    @pytest.mark.parametrize("page_label", PAGE_LABELS)
    def test_capture_page_screenshot(
        self,
        streamlit_page: Page,
        page_label: str,
    ) -> None:
        """Capture screenshot of each page."""
        click_radio_page(streamlit_page, page_label)

        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        safe_name = (
            page_label.lower().replace(" ", "_")
        )
        path = os.path.join(
            SCREENSHOT_DIR, f"{safe_name}.png"
        )
        streamlit_page.screenshot(path=path)
        assert os.path.exists(path)
