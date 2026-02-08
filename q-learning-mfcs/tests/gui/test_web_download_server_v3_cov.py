"""Tests for web_download_server.py - comprehensive coverage.

Missing: 74-155, 254-271, 290-371.
Covers: create_fastapi_app routes, start_server, streamlit interface,
start_download_interface.
"""
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class FakeChronologyEntry:
    def __init__(self, eid="e1", success=True):
        self.id = eid
        self.simulation_name = "test_sim"
        self.timestamp = "2025-01-01T00:00:00"
        self.success = success
        self.duration_hours = 1.0
        self.execution_time_seconds = 30.0
        self.tags = ["test", "mfc"]
        self.result_files = {"csv": "test.csv"}
        self.description = "Test simulation"
        self.parameters = {"param1": 1.0}
        self.results_summary = {"power": 0.5}


class FakeChronology:
    def __init__(self):
        self.entries = [FakeChronologyEntry(), FakeChronologyEntry("e2", False)]

    def get_recent_entries(self, n):
        return self.entries[:n]

    def get_entry_by_id(self, eid):
        for e in self.entries:
            if e.id == eid:
                return e
        return None


class FakeChronologyManager:
    def __init__(self):
        self.chronology = FakeChronology()


mock_chronology = MagicMock()
mock_chronology.get_chronology_manager = MagicMock(
    return_value=FakeChronologyManager()
)

with patch.dict(sys.modules, {
    "config.simulation_chronology": mock_chronology,
}):
    from web_download_server import (
        DownloadServer,
        FASTAPI_AVAILABLE,
        STREAMLIT_AVAILABLE,
    )
    import web_download_server as _wds


class TestDownloadServerInit:
    def test_init(self):
        server = DownloadServer()
        assert server.port == 8080

    def test_init_custom_port(self):
        server = DownloadServer(port=9090)
        assert server.port == 9090


class TestGenerateDownloadHtml:
    def test_html_contains_entries(self):
        server = DownloadServer()
        html = server._generate_download_html()
        assert "test_sim" in html
        assert "MFC Simulation Downloads" in html

    def test_html_empty_entries(self):
        server = DownloadServer()
        server.chronology_manager.chronology.entries = []
        html = server._generate_download_html()
        assert "No simulations found" in html

    def test_html_with_no_result_files(self):
        server = DownloadServer()
        entry = FakeChronologyEntry()
        entry.result_files = {}
        entry.description = ""
        server.chronology_manager.chronology.entries = [entry]
        html = server._generate_download_html()
        assert "No download files available" in html


class TestCreateFastapiApp:
    def test_returns_none_when_not_available(self, monkeypatch):
        monkeypatch.setattr(_wds, "FASTAPI_AVAILABLE", False)
        server = DownloadServer()
        result = server.create_fastapi_app()
        assert result is None

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
    def test_returns_app(self):
        server = DownloadServer()
        app = server.create_fastapi_app()
        assert app is not None

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
    def test_app_routes_exist(self):
        server = DownloadServer()
        app = server.create_fastapi_app()
        routes = [r.path for r in app.routes]
        assert "/" in routes
        assert "/api/simulations" in routes


class TestStartServer:
    def test_start_server_no_fastapi(self, monkeypatch):
        monkeypatch.setattr(_wds, "FASTAPI_AVAILABLE", False)
        server = DownloadServer()
        server.start_server()  # should return early

    def test_start_server_no_app(self):
        server = DownloadServer()
        with patch.object(server, "create_fastapi_app", return_value=None):
            server.start_server()

    def test_start_server_with_browser(self, monkeypatch):
        mock_uv = MagicMock()
        monkeypatch.setattr(_wds, "uvicorn", mock_uv)
        server = DownloadServer()
        mock_app = MagicMock()
        with patch.object(server, "create_fastapi_app", return_value=mock_app):
            server.start_server(open_browser=True)
            mock_uv.run.assert_called_once()

    def test_start_server_keyboard_interrupt(self, monkeypatch):
        mock_uv = MagicMock()
        mock_uv.run.side_effect = KeyboardInterrupt
        monkeypatch.setattr(_wds, "uvicorn", mock_uv)
        server = DownloadServer()
        mock_app = MagicMock()
        with patch.object(server, "create_fastapi_app", return_value=mock_app):
            server.start_server(open_browser=False)

    def test_start_server_exception(self, monkeypatch):
        mock_uv = MagicMock()
        mock_uv.run.side_effect = RuntimeError("err")
        monkeypatch.setattr(_wds, "uvicorn", mock_uv)
        server = DownloadServer()
        mock_app = MagicMock()
        with patch.object(server, "create_fastapi_app", return_value=mock_app):
            server.start_server(open_browser=False)


class TestStartDownloadInterface:
    def test_fastapi_interface(self):
        with patch("web_download_server.FASTAPI_AVAILABLE", True):
            with patch("web_download_server.DownloadServer") as mock_cls:
                mock_inst = MagicMock()
                mock_cls.return_value = mock_inst
                from web_download_server import start_download_interface
                start_download_interface("fastapi", 9090)
                mock_inst.start_server.assert_called_once()

    def test_unknown_interface(self):
        with patch("web_download_server.FASTAPI_AVAILABLE", True):
            with patch("web_download_server.STREAMLIT_AVAILABLE", True):
                from web_download_server import start_download_interface
                start_download_interface("unknown", 8080)


class TestStreamlitInterface:
    def test_streamlit_not_available(self):
        with patch("web_download_server.STREAMLIT_AVAILABLE", False):
            from web_download_server import create_streamlit_download_interface
            result = create_streamlit_download_interface()
            assert result is None

    def test_streamlit_available(self):
        mock_st = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.sidebar = MagicMock()
        mock_st.sidebar.multiselect.return_value = []
        mock_st.sidebar.checkbox.return_value = True

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            with patch("web_download_server.STREAMLIT_AVAILABLE", True):
                with patch("web_download_server.st", mock_st, create=True):
                    from web_download_server import (
                        create_streamlit_download_interface,
                    )
                    create_streamlit_download_interface()
                    mock_st.title.assert_called_once()

    def test_streamlit_no_entries(self):
        mock_st = MagicMock()
        mock_chronology_mgr = FakeChronologyManager()
        mock_chronology_mgr.chronology.entries = []

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            with patch("web_download_server.STREAMLIT_AVAILABLE", True):
                with patch("web_download_server.st", mock_st, create=True):
                    with patch(
                        "web_download_server.get_chronology_manager",
                        return_value=mock_chronology_mgr,
                    ):
                        from web_download_server import (
                            create_streamlit_download_interface,
                        )
                        create_streamlit_download_interface()
                        mock_st.warning.assert_called()
