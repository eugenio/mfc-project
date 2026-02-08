"""Tests for web_download_server.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture
def mock_chronology():
    """Create a mock chronology manager."""
    entry = MagicMock()
    entry.id = "abc123"
    entry.simulation_name = "Test Sim"
    entry.timestamp = "2025-01-01"
    entry.success = True
    entry.duration_hours = 24.0
    entry.execution_time_seconds = 120.0
    entry.tags = ["test"]
    entry.description = "A test simulation"
    entry.result_files = {"data": "output/data.csv"}
    entry.parameters = {}
    entry.results_summary = {"power": 0.5}

    manager = MagicMock()
    manager.chronology.get_recent_entries.return_value = [entry]
    manager.chronology.get_entry_by_id.return_value = entry
    manager.chronology.entries = [entry]

    return manager, entry


class TestDownloadServer:
    def test_init_no_fastapi(self):
        with patch("web_download_server.FASTAPI_AVAILABLE", False):
            with patch(
                "web_download_server.get_chronology_manager"
            ) as mock_cm:
                mock_cm.return_value = MagicMock()
                from web_download_server import DownloadServer
                server = DownloadServer()
                assert server.port == 8080

    def test_init_with_fastapi(self):
        with patch("web_download_server.FASTAPI_AVAILABLE", True):
            with patch(
                "web_download_server.get_chronology_manager"
            ) as mock_cm:
                mock_cm.return_value = MagicMock()
                from web_download_server import DownloadServer
                server = DownloadServer(port=9090)
                assert server.port == 9090

    def test_create_fastapi_app_not_available(self):
        with patch("web_download_server.FASTAPI_AVAILABLE", False):
            with patch(
                "web_download_server.get_chronology_manager"
            ) as mock_cm:
                mock_cm.return_value = MagicMock()
                from web_download_server import DownloadServer
                server = DownloadServer()
                app = server.create_fastapi_app()
                assert app is None

    def test_generate_download_html_no_entries(self, mock_chronology):
        manager, entry = mock_chronology
        manager.chronology.get_recent_entries.return_value = []
        with patch(
            "web_download_server.get_chronology_manager",
            return_value=manager,
        ):
            from web_download_server import DownloadServer
            server = DownloadServer()
            html = server._generate_download_html()
            assert "No simulations found" in html

    def test_generate_download_html_with_entries(self, mock_chronology):
        manager, entry = mock_chronology
        with patch(
            "web_download_server.get_chronology_manager",
            return_value=manager,
        ):
            from web_download_server import DownloadServer
            server = DownloadServer()
            html = server._generate_download_html()
            assert "Test Sim" in html
            assert "abc123" in html

    def test_generate_download_html_failed_entry(self, mock_chronology):
        manager, entry = mock_chronology
        entry.success = False
        entry.result_files = {}
        with patch(
            "web_download_server.get_chronology_manager",
            return_value=manager,
        ):
            from web_download_server import DownloadServer
            server = DownloadServer()
            html = server._generate_download_html()
            assert "No download files available" in html

    def test_start_server_no_fastapi(self, mock_chronology):
        manager, _ = mock_chronology
        with patch("web_download_server.FASTAPI_AVAILABLE", False):
            with patch(
                "web_download_server.get_chronology_manager",
                return_value=manager,
            ):
                from web_download_server import DownloadServer
                server = DownloadServer()
                server.start_server()  # should return early


class TestStartDownloadInterface:
    def test_fastapi_interface(self, mock_chronology):
        manager, _ = mock_chronology
        with patch(
            "web_download_server.get_chronology_manager",
            return_value=manager,
        ):
            with patch("web_download_server.FASTAPI_AVAILABLE", True):
                from web_download_server import start_download_interface
                with patch(
                    "web_download_server.DownloadServer.start_server"
                ):
                    start_download_interface("fastapi", 8080)

    def test_streamlit_interface(self, mock_chronology):
        manager, _ = mock_chronology
        with patch(
            "web_download_server.get_chronology_manager",
            return_value=manager,
        ):
            with patch("web_download_server.STREAMLIT_AVAILABLE", True):
                with patch(
                    "web_download_server.create_streamlit_download_interface"
                ) as mock_st:
                    from web_download_server import start_download_interface
                    start_download_interface("streamlit", 8080)
                    mock_st.assert_called_once()

    def test_unknown_interface(self, mock_chronology):
        manager, _ = mock_chronology
        with patch(
            "web_download_server.get_chronology_manager",
            return_value=manager,
        ):
            from web_download_server import start_download_interface
            with patch("web_download_server.FASTAPI_AVAILABLE", True):
                with patch("web_download_server.STREAMLIT_AVAILABLE", True):
                    start_download_interface("unknown", 8080)
