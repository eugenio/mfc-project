"""Pytest configuration for AppTest integration tests.

Provides fixtures for Streamlit AppTest-based headless integration testing.
Tests run without a browser, using Streamlit's native testing framework.
"""

import sys
from pathlib import Path

# Add this directory and src to sys.path for imports
_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str((Path(__file__).resolve().parent / ".." / ".." / "src").resolve())
for _d in (_THIS_DIR, _SRC_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import pytest  # noqa: E402
from constants import APP_FILE  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def app() -> AppTest:
    """Create a fresh AppTest instance from the main app file.

    Returns an AppTest that has been run once (initial render).
    """
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    return at


@pytest.fixture
def navigate_to(app: AppTest):  # noqa: ANN201
    """Navigate to a specific page by label.

    Usage:
        def test_electrode(navigate_to):
            at = navigate_to("Electrode System")
            # at is now on the electrode page
    """

    def _navigate(page_label: str) -> AppTest:
        at = app
        at.sidebar.radio[0].set_value(page_label).run()
        return at

    return _navigate


def _mock_psutil() -> None:
    """Pre-mock psutil to avoid blocking calls."""
    from unittest.mock import MagicMock  # noqa: PLC0415

    mock_psutil = MagicMock()
    mock_psutil.cpu_percent.return_value = 45.0
    mock_psutil.virtual_memory.return_value = MagicMock(
        percent=60.0,
        total=16_000_000_000,
        available=6_400_000_000,
    )
    mock_psutil.disk_usage.return_value = MagicMock(
        percent=55.0,
        total=500_000_000_000,
        free=225_000_000_000,
    )
    mock_psutil.net_io_counters.return_value = MagicMock(
        bytes_sent=1000, bytes_recv=2000,
    )
    sys.modules.setdefault("psutil", mock_psutil)


@pytest.fixture(autouse=True)
def _setup_environment() -> None:
    """Set up environment for AppTest runs."""
    _mock_psutil()
