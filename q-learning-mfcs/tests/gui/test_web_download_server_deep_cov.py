"""Tests for web_download_server.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock streamlit before importing
mock_st = MagicMock()
sys.modules["streamlit"] = mock_st

# Mock fastapi/uvicorn
mock_fastapi = MagicMock()
mock_uvicorn = MagicMock()
sys.modules["uvicorn"] = mock_uvicorn
sys.modules["fastapi"] = mock_fastapi
sys.modules["fastapi.responses"] = MagicMock()

from web_download_server import (
    DownloadServer,
    create_streamlit_download_interface,
    start_download_interface,
    STREAMLIT_AVAILABLE,
    FASTAPI_AVAILABLE,
)


class TestDownloadServer:
    def test_init(self):
        with patch("web_download_server.get_chronology_manager") as mock_cm:
            mock_cm.return_value = MagicMock()
            server = DownloadServer(port=9090)
            assert server.port == 9090

    def test_generate_download_html_no_entries(self):
        with patch("web_download_server.get_chronology_manager") as mock_cm:
            mock_mgr = MagicMock()
            mock_mgr.chronology.get_recent_entries.return_value = []
            mock_cm.return_value = mock_mgr
            server = DownloadServer()
            html = server._generate_download_html()
            assert "No simulations found" in html

    def test_generate_download_html_with_entries(self):
        with patch("web_download_server.get_chronology_manager") as mock_cm:
            mock_entry = MagicMock()
            mock_entry.id = "test-123"
            mock_entry.simulation_name = "Test Sim"
            mock_entry.timestamp = "2025-01-01"
            mock_entry.success = True
            mock_entry.duration_hours = 1.5
            mock_entry.execution_time_seconds = 30.0
            mock_entry.description = "A test"
            mock_entry.tags = ["tag1", "tag2"]
            mock_entry.result_files = {"csv": "data.csv"}
            mock_mgr = MagicMock()
            mock_mgr.chronology.get_recent_entries.return_value = [mock_entry]
            mock_cm.return_value = mock_mgr
            server = DownloadServer()
            html = server._generate_download_html()
            assert "Test Sim" in html
            assert "tag1" in html

    def test_generate_download_html_failed_entry(self):
        with patch("web_download_server.get_chronology_manager") as mock_cm:
            mock_entry = MagicMock()
            mock_entry.id = "fail-1"
            mock_entry.simulation_name = "Failed Sim"
            mock_entry.timestamp = "2025-01-01"
            mock_entry.success = False
            mock_entry.duration_hours = 0.5
            mock_entry.execution_time_seconds = 10.0
            mock_entry.description = ""
            mock_entry.tags = []
            mock_entry.result_files = {}
            mock_mgr = MagicMock()
            mock_mgr.chronology.get_recent_entries.return_value = [mock_entry]
            mock_cm.return_value = mock_mgr
            server = DownloadServer()
            html = server._generate_download_html()
            assert "Failed Sim" in html
            assert "No download files" in html

    def test_start_server_no_fastapi(self):
        with patch("web_download_server.get_chronology_manager") as mock_cm:
            mock_cm.return_value = MagicMock()
            with patch("web_download_server.FASTAPI_AVAILABLE", False):
                server = DownloadServer()
                server.start_server()

    def test_start_server_no_app(self):
        with patch("web_download_server.get_chronology_manager") as mock_cm:
            mock_cm.return_value = MagicMock()
            server = DownloadServer()
            server.create_fastapi_app = MagicMock(return_value=None)
            server.start_server()


class TestStartDownloadInterface:
    def test_fastapi_interface(self):
        with patch("web_download_server.get_chronology_manager") as mock_cm:
            mock_cm.return_value = MagicMock()
            with patch("web_download_server.DownloadServer") as mock_ds:
                mock_server = MagicMock()
                mock_ds.return_value = mock_server
                start_download_interface("fastapi", 8080)
                mock_server.start_server.assert_called_once()

    def test_unknown_interface(self):
        start_download_interface("unknown", 8080)


class TestCreateStreamlitDownloadInterface:
    def test_not_available(self):
        with patch("web_download_server.STREAMLIT_AVAILABLE", False):
            create_streamlit_download_interface()

    def test_no_entries(self):
        with patch("web_download_server.STREAMLIT_AVAILABLE", True):
            with patch("web_download_server.get_chronology_manager") as mock_cm:
                mock_mgr = MagicMock()
                mock_mgr.chronology.get_recent_entries.return_value = []
                mock_cm.return_value = mock_mgr
                create_streamlit_download_interface()
