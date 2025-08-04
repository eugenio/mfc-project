"""Test suite for monitoring and API integration modules."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from web_download_server import DownloadServer


# Mock ChronologyEntry since it doesn't exist in the actual implementation
class ChronologyEntry:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestWebDownloadServerAPI:
    """Comprehensive test suite for web download server API integration."""

    def setup_method(self):
        """Setup test environment before each test."""
        self.test_output_dir = Path(tempfile.mkdtemp())
        self.server = DownloadServer(output_dir=self.test_output_dir, port=8081)

        # Mock chronology manager
        self.mock_chronology = Mock()
        self.mock_chronology_manager = Mock()
        self.mock_chronology_manager.chronology = self.mock_chronology

        with patch('web_download_server.get_chronology_manager', return_value=self.mock_chronology_manager):
            self.server = DownloadServer(output_dir=self.test_output_dir, port=8081)

        # Create mock entries
        self.mock_entry = ChronologyEntry(
            id="test-sim-001",
            simulation_name="Test Simulation",
            description="Test simulation description",
            timestamp="2024-01-01T10:00:00",
            success=True,
            duration_hours=12.5,
            execution_time_seconds=45000.0,
            tags=["test", "api"],
            parameters={"test_param": "value"},
            results_summary={"final_power": 1.5},
            result_files={
                "config": "test_config.json",
                "results": "test_results.csv"
            }
        )

    def test_fastapi_app_creation(self):
        """Test FastAPI application creation and configuration."""
        app = self.server.create_fastapi_app()

        assert app is not None
        assert app.title == "MFC Simulation Download Server"
        assert app.version == "1.0.0"

    def test_download_index_endpoint(self):
        """Test the main download page endpoint."""
        app = self.server.create_fastapi_app()
        client = TestClient(app)

        # Mock chronology entries
        self.mock_chronology.get_recent_entries.return_value = [self.mock_entry]

        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "MFC Simulation Downloads" in response.text
        assert "Test Simulation" in response.text

    def test_list_simulations_api_endpoint(self):
        """Test the simulations listing API endpoint."""
        app = self.server.create_fastapi_app()
        client = TestClient(app)

        # Mock chronology entries
        self.mock_chronology.get_recent_entries.return_value = [self.mock_entry]

        response = client.get("/api/simulations")

        assert response.status_code == 200
        data = response.json()

        assert "simulations" in data
        assert len(data["simulations"]) == 1

        sim = data["simulations"][0]
        assert sim["id"] == "test-sim-001"
        assert sim["name"] == "Test Simulation"
        assert sim["success"]
        assert sim["duration_hours"] == 12.5
        assert sim["tags"] == ["test", "api"]

    def test_get_simulation_details_endpoint(self):
        """Test getting detailed simulation information."""
        app = self.server.create_fastapi_app()
        client = TestClient(app)

        # Mock chronology manager
        self.mock_chronology.get_entry_by_id.return_value = self.mock_entry

        response = client.get("/api/simulation/test-sim-001")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == "test-sim-001"
        assert data["name"] == "Test Simulation"
        assert data["description"] == "Test simulation description"
        assert data["success"]
        assert data["parameters"] == {"test_param": "value"}
        assert data["results_summary"] == {"final_power": 1.5}

    def test_get_simulation_not_found(self):
        """Test handling of non-existent simulation requests."""
        app = self.server.create_fastapi_app()
        client = TestClient(app)

        # Mock not found
        self.mock_chronology.get_entry_by_id.return_value = None

        response = client.get("/api/simulation/nonexistent")

        assert response.status_code == 404
        assert response.json()["detail"] == "Simulation not found"

    def test_download_security_path_traversal(self):
        """Test security against path traversal attacks."""
        app = self.server.create_fastapi_app()
        client = TestClient(app)

        # Mock entry with malicious path
        malicious_entry = ChronologyEntry(
            id="malicious-001",
            simulation_name="Malicious Simulation",
            timestamp="2024-01-01T10:00:00",
            success=True,
            duration_hours=1.0,
            execution_time_seconds=3600.0,
            result_files={
                "config": "../../../etc/passwd"  # Path traversal attempt
            }
        )

        self.mock_chronology.get_entry_by_id.return_value = malicious_entry

        response = client.get("/download/malicious-001/config")

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

