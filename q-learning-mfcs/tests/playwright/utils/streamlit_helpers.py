"""Streamlit-specific Playwright helper utilities."""

from playwright.sync_api import Page


def wait_for_render(page: Page, timeout: int = 10000) -> None:
    """Wait for Streamlit app to finish rendering."""
    page.wait_for_selector(
        '[data-testid="stAppViewContainer"]',
        timeout=timeout,
    )
    page.wait_for_load_state("networkidle")


def navigate_sidebar(page: Page, label: str) -> None:
    """Navigate to a page via sidebar radio button."""
    radio = page.locator('[data-testid="stRadio"]')
    radio_option = radio.get_by_text(label, exact=False)
    radio_option.click()
    wait_for_render(page)


def check_no_exceptions(page: Page) -> bool:
    """Check that no Streamlit exceptions are displayed."""
    exceptions = page.locator('[data-testid="stException"]')
    return exceptions.count() == 0


def get_page_title(page: Page) -> str:
    """Get the main page title text."""
    title = page.locator("h1").first
    return title.text_content() or ""
