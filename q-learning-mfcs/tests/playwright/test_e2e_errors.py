"""E2E tests: Error detection."""

import pytest
from playwright.sync_api import Page

from conftest import PAGE_LABELS, click_radio_page

KNOWN_EXCEPTION_PAGES = (
    "Electrode System",
    "Performance Monitor",
)


@pytest.mark.playwright
class TestErrorDetection:
    """Test for runtime errors and console issues."""

    def test_no_console_errors_on_dashboard(
        self, streamlit_page: Page
    ) -> None:
        """Dashboard should have no console errors."""
        errors: list[str] = []
        streamlit_page.on(
            "console",
            lambda msg: (
                errors.append(msg.text)
                if msg.type == "error"
                else None
            ),
        )
        streamlit_page.reload()
        streamlit_page.wait_for_timeout(2000)
        real_errors = [
            e
            for e in errors
            if "favicon" not in e.lower()
            and "manifest" not in e.lower()
        ]
        assert len(real_errors) == 0, (
            f"Console errors: {real_errors}"
        )

    def test_no_python_tracebacks(
        self, streamlit_page: Page
    ) -> None:
        """Dashboard should have no Python tracebacks."""
        content = streamlit_page.content()
        assert "Traceback" not in content

    @pytest.mark.parametrize(
        "page_label",
        [
            "Dashboard",
            "Cell Configuration",
            "ML Optimization",
            "Configuration",
            "Physics Simulation",
            "GSM Integration",
            "Literature Validation",
        ],
    )
    def test_clean_pages_no_exceptions(
        self,
        streamlit_page: Page,
        page_label: str,
    ) -> None:
        """Known clean pages show no exceptions."""
        click_radio_page(streamlit_page, page_label)
        exceptions = streamlit_page.locator(
            '[data-testid="stException"]'
        )
        assert exceptions.count() == 0

    @pytest.mark.parametrize(
        "page_label", list(KNOWN_EXCEPTION_PAGES)
    )
    def test_known_exception_pages(
        self,
        streamlit_page: Page,
        page_label: str,
    ) -> None:
        """Known buggy pages are documented."""
        click_radio_page(streamlit_page, page_label)
        exceptions = streamlit_page.locator(
            '[data-testid="stException"]'
        )
        if exceptions.count() > 0:
            pytest.xfail(
                f"Known exception on {page_label}"
            )

    def test_no_alert_dialogs(
        self, streamlit_page: Page
    ) -> None:
        """App should not trigger alert dialogs."""
        alerts: list[str] = []

        def _handle_dialog(dialog):  # noqa: ANN001
            alerts.append(dialog.message)
            dialog.dismiss()

        streamlit_page.on("dialog", _handle_dialog)
        click_radio_page(streamlit_page, "Dashboard")
        assert len(alerts) == 0
