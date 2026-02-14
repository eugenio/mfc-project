"""Coverage tests for web_download_server.py -- target 99%+.

Covers remaining uncovered paths: create_fastapi_app route handlers,
streamlit download interface with entries and file downloads,
start_download_interface streamlit branch, entry with results_summary
containing int/float and other types.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _make_col():
    """Create a MagicMock that works as a context manager."""
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    return col


def _make_st():
    """Build a streamlit mock with proper columns/expander support."""
    mock_st = MagicMock()

    def columns_side_effect(n_or_list):
        if isinstance(n_or_list, int):
            return [_make_col() for _ in range(n_or_list)]
        return [_make_col() for _ in range(len(n_or_list))]

    mock_st.columns = MagicMock(side_effect=columns_side_effect)
    mock_st.expander.return_value = _make_col()
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.multiselect.return_value = []
    mock_st.sidebar.checkbox.return_value = True
    return mock_st


class FakeEntry:
    """Fake chronology entry for testing."""

    def __init__(self, eid="e1", success=True, has_files=True):
        self.id = eid
        self.simulation_name = "test_sim"
        self.timestamp = "2025-01-01T00:00:00"
        self.success = success
        self.duration_hours = 1.0
        self.execution_time_seconds = 30.0
        self.tags = ["test", "mfc"]
        self.description = "Test simulation"
        self.parameters = {"param1": 1.0}
        self.results_summary = {
            "power": 0.5,
            "efficiency": 42,
            "label": "good",
        }
        if has_files:
            self.result_files = {"csv": "test.csv", "json": "test.json"}
        else:
            self.result_files = {}


class FakeChronology:
    def __init__(self, entries=None):
        self.entries = entries or [
            FakeEntry("e1", True),
            FakeEntry("e2", False, has_files=False),
        ]

    def get_recent_entries(self, n):
        return self.entries[:n]

    def get_entry_by_id(self, eid):
        for e in self.entries:
            if e.id == eid:
                return e
        return None


class FakeChronologyManager:
    def __init__(self, entries=None):
        self.chronology = FakeChronology(entries)


@pytest.fixture
def mock_chronology_module():
    """Provide fake chronology module in sys.modules."""
    mock_mod = MagicMock()
    mock_mod.get_chronology_manager = MagicMock(
        return_value=FakeChronologyManager()
    )
    return mock_mod


@pytest.mark.coverage_extra
class TestStreamlitDownloadInterfaceWithEntries:
    """Cover create_streamlit_download_interface with entries."""

    def test_with_entries_and_files(self, mock_chronology_module, tmp_path):
        """Cover the full entry rendering including file download buttons."""
        mock_st = _make_st()

        # Create a temp file for download
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2\n")

        entry_with_file = FakeEntry("e1", True)
        entry_with_file.result_files = {"csv": str(test_file)}
        mgr = FakeChronologyManager(
            entries=[entry_with_file, FakeEntry("e2", False, has_files=False)]
        )
        mock_chronology_module.get_chronology_manager.return_value = mgr

        with patch.dict(
            sys.modules,
            {
                "streamlit": mock_st,
                "config.simulation_chronology": mock_chronology_module,
            },
        ):
            for k in list(sys.modules.keys()):
                if "web_download_server" in k:
                    del sys.modules[k]

            from web_download_server import create_streamlit_download_interface

            with patch("web_download_server.STREAMLIT_AVAILABLE", True), patch(
                "web_download_server.st", mock_st
            ):
                create_streamlit_download_interface()
                mock_st.title.assert_called()

    def test_with_tag_filter(self, mock_chronology_module):
        """Cover tag filtering branch."""
        mock_st = _make_st()
        mock_st.sidebar.multiselect.return_value = ["test"]

        with patch.dict(
            sys.modules,
            {
                "streamlit": mock_st,
                "config.simulation_chronology": mock_chronology_module,
            },
        ):
            for k in list(sys.modules.keys()):
                if "web_download_server" in k:
                    del sys.modules[k]

            from web_download_server import create_streamlit_download_interface

            with patch("web_download_server.STREAMLIT_AVAILABLE", True), patch(
                "web_download_server.st", mock_st
            ):
                create_streamlit_download_interface()

    def test_hide_failed(self, mock_chronology_module):
        """Cover show_failed=False filtering branch."""
        mock_st = _make_st()
        mock_st.sidebar.checkbox.return_value = False

        with patch.dict(
            sys.modules,
            {
                "streamlit": mock_st,
                "config.simulation_chronology": mock_chronology_module,
            },
        ):
            for k in list(sys.modules.keys()):
                if "web_download_server" in k:
                    del sys.modules[k]

            from web_download_server import create_streamlit_download_interface

            with patch("web_download_server.STREAMLIT_AVAILABLE", True), patch(
                "web_download_server.st", mock_st
            ):
                create_streamlit_download_interface()

    def test_file_not_found(self, mock_chronology_module):
        """Cover the file-not-found error branch."""
        mock_st = _make_st()

        entry = FakeEntry("e1", True)
        entry.result_files = {"csv": "/nonexistent/path.csv"}
        mgr = FakeChronologyManager(entries=[entry])
        mock_chronology_module.get_chronology_manager.return_value = mgr

        with patch.dict(
            sys.modules,
            {
                "streamlit": mock_st,
                "config.simulation_chronology": mock_chronology_module,
            },
        ):
            for k in list(sys.modules.keys()):
                if "web_download_server" in k:
                    del sys.modules[k]

            from web_download_server import create_streamlit_download_interface

            with patch("web_download_server.STREAMLIT_AVAILABLE", True), patch(
                "web_download_server.st", mock_st
            ):
                create_streamlit_download_interface()


@pytest.mark.coverage_extra
class TestStartDownloadInterfaceStreamlit:
    """Cover start_download_interface streamlit branch."""

    def test_streamlit_interface(self, mock_chronology_module):
        mock_st = _make_st()

        with patch.dict(
            sys.modules,
            {
                "streamlit": mock_st,
                "config.simulation_chronology": mock_chronology_module,
            },
        ):
            for k in list(sys.modules.keys()):
                if "web_download_server" in k:
                    del sys.modules[k]

            from web_download_server import start_download_interface

            with patch("web_download_server.STREAMLIT_AVAILABLE", True), patch(
                "web_download_server.create_streamlit_download_interface"
            ) as mock_create:
                start_download_interface("streamlit", 8080)
                mock_create.assert_called_once()


@pytest.mark.coverage_extra
class TestDownloadServerFastapiRoutes:
    """Cover create_fastapi_app route handler functions."""

    def test_fastapi_app_created(self, mock_chronology_module):
        """Cover create_fastapi_app when FASTAPI_AVAILABLE is True."""
        with patch.dict(
            sys.modules,
            {
                "config.simulation_chronology": mock_chronology_module,
            },
        ):
            for k in list(sys.modules.keys()):
                if "web_download_server" in k:
                    del sys.modules[k]

            from web_download_server import FASTAPI_AVAILABLE

            if not FASTAPI_AVAILABLE:
                pytest.skip("FastAPI not installed")

            from web_download_server import DownloadServer

            server = DownloadServer()
            app = server.create_fastapi_app()
            assert app is not None
            routes = [r.path for r in app.routes]
            assert "/" in routes
            assert "/api/simulations" in routes

    def test_html_with_no_description_no_tags(
        self, mock_chronology_module
    ):
        """Cover HTML gen with entry having no description and no tags."""
        entry = FakeEntry("e3", True)
        entry.description = ""
        entry.tags = []
        mgr = FakeChronologyManager(entries=[entry])
        mock_chronology_module.get_chronology_manager.return_value = mgr

        with patch.dict(
            sys.modules,
            {
                "config.simulation_chronology": mock_chronology_module,
            },
        ):
            for k in list(sys.modules.keys()):
                if "web_download_server" in k:
                    del sys.modules[k]

            from web_download_server import DownloadServer

            server = DownloadServer()
            html = server._generate_download_html()
            assert "test_sim" in html
