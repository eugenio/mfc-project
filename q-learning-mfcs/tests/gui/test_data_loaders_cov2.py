"""Coverage tests for gui/data_loaders.py — target 99%+.

Covers: load_simulation_data (success, no files, exception),
load_recent_simulations (no dir, valid dirs, exception in json,
non-matching dirs).
"""
import gzip
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
class TestLoadSimulationData:
    def test_loads_csv_gz(self, tmp_path, _mock_streamlit):
        """Happy path: compressed CSV is loaded correctly."""
        from gui.data_loaders import load_simulation_data

        df_orig = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        gz_path = tmp_path / "sim_data_001.csv.gz"
        with gzip.open(gz_path, "wt") as f:
            df_orig.to_csv(f, index=False)

        result = load_simulation_data(str(tmp_path))
        assert result is not None
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_no_csv_files(self, tmp_path, _mock_streamlit):
        """Returns None when no matching .csv.gz files exist."""
        from gui.data_loaders import load_simulation_data

        result = load_simulation_data(str(tmp_path))
        assert result is None

    def test_corrupt_gz(self, tmp_path, _mock_streamlit):
        """Returns None and calls st.error on corrupt file."""
        from gui.data_loaders import load_simulation_data

        bad_file = tmp_path / "corrupt_data_001.csv.gz"
        bad_file.write_bytes(b"not-gzip-content")

        result = load_simulation_data(str(tmp_path))
        assert result is None
        _mock_streamlit.error.assert_called()


# ---- load_recent_simulations ----

class TestLoadRecentSimulations:
    def test_no_data_dir(self, _mock_streamlit, monkeypatch):
        """Returns [] when data directory does not exist."""
        from gui.data_loaders import load_recent_simulations

        # Patch the Path constructor used inside the function
        monkeypatch.chdir("/tmp")
        with patch(
            "gui.data_loaders.Path",
            wraps=Path,
        ) as mock_path_cls:
            fake_data_dir = MagicMock(spec=Path)
            fake_data_dir.exists.return_value = False
            mock_path_cls.return_value = fake_data_dir

            result = load_recent_simulations()
            assert result == []

    def test_with_valid_simulation_dirs(self, tmp_path, _mock_streamlit):
        """Returns sorted list of valid simulation metadata."""
        from gui.data_loaders import load_recent_simulations

        # Create a simulation directory
        sim_dir = tmp_path / "gpu_test_sim"
        sim_dir.mkdir()

        # Create a results JSON
        results = {
            "simulation_info": {
                "timestamp": "2025-01-15T10:00:00",
                "duration_hours": 24,
            },
            "performance_metrics": {
                "final_reservoir_concentration": 18.5,
                "control_effectiveness_2mM": 92.0,
            },
        }
        results_file = sim_dir / "sim_results.json"
        results_file.write_text(json.dumps(results))

        # Create a compressed CSV
        gz_file = sim_dir / "sim_data.csv.gz"
        with gzip.open(gz_file, "wt") as f:
            f.write("a,b\n1,2\n")

        # Patch Path to use our tmp directory
        with patch("gui.data_loaders.Path") as mock_path_cls:
            mock_path_cls.return_value = tmp_path
            result = load_recent_simulations()

        assert len(result) == 1
        assert result[0]["name"] == "gpu_test_sim"
        assert result[0]["duration"] == 24
        assert result[0]["final_conc"] == 18.5
        assert result[0]["control_effectiveness"] == 92.0

    def test_with_multiple_dirs_sorted(self, tmp_path, _mock_streamlit):
        """Multiple dirs are sorted by timestamp descending."""
        from gui.data_loaders import load_recent_simulations

        for i, prefix in enumerate(["gpu_a", "lactate_b"]):
            sim_dir = tmp_path / prefix
            sim_dir.mkdir()
            results = {
                "simulation_info": {
                    "timestamp": f"2025-01-{15 + i}T10:00:00",
                    "duration_hours": i + 1,
                },
                "performance_metrics": {
                    "final_reservoir_concentration": 20.0 + i,
                    "control_effectiveness_2mM": 80.0 + i,
                },
            }
            (sim_dir / "results.json").write_text(json.dumps(results))
            with gzip.open(sim_dir / "data.csv.gz", "wt") as f:
                f.write("col\n1\n")

        with patch("gui.data_loaders.Path") as mock_path_cls:
            mock_path_cls.return_value = tmp_path
            result = load_recent_simulations()

        assert len(result) == 2
        # Most recent first
        assert result[0]["timestamp"] > result[1]["timestamp"]

    def test_skips_non_matching_dirs(self, tmp_path, _mock_streamlit):
        """Directories not starting with gpu_, lactate_, gui_ are ignored."""
        from gui.data_loaders import load_recent_simulations

        (tmp_path / "random_dir").mkdir()

        with patch("gui.data_loaders.Path") as mock_path_cls:
            mock_path_cls.return_value = tmp_path
            result = load_recent_simulations()

        assert result == []

    def test_skips_dirs_without_results(self, tmp_path, _mock_streamlit):
        """Dirs with matching prefix but no results files are skipped."""
        from gui.data_loaders import load_recent_simulations

        sim_dir = tmp_path / "gpu_no_results"
        sim_dir.mkdir()

        with patch("gui.data_loaders.Path") as mock_path_cls:
            mock_path_cls.return_value = tmp_path
            result = load_recent_simulations()

        assert result == []

    def test_exception_in_json_parsing(self, tmp_path, _mock_streamlit):
        """Corrupt JSON is caught and dir is skipped (continue)."""
        from gui.data_loaders import load_recent_simulations

        sim_dir = tmp_path / "gui_bad_json"
        sim_dir.mkdir()
        (sim_dir / "results.json").write_text("not valid json{{{")
        with gzip.open(sim_dir / "data.csv.gz", "wt") as f:
            f.write("a\n1\n")

        with patch("gui.data_loaders.Path") as mock_path_cls:
            mock_path_cls.return_value = tmp_path
            result = load_recent_simulations()

        assert result == []
@pytest.fixture(autouse=True)
def _mock_streamlit():
    """Mock streamlit before importing data_loaders."""
    mock_st = MagicMock()
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        yield mock_st


# ---- load_simulation_data ----

