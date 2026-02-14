"""Extra coverage tests for experiment_tracker.py.

Targets remaining uncovered lines:
  263-264 - log_artifact exception path (save failure)
  296-297 - load_artifact exception path (load failure)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mlops.experiment_tracker import (
    ExperimentTracker,
    ExperimentTrackerError,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tracker(tmp_dir):
    return ExperimentTracker(base_path=tmp_dir)


@pytest.mark.coverage_extra
class TestLogArtifactSaveFailure:
    """Cover lines 263-264."""

    def test_pickle_save_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        run_dir = tracker.artifacts_path / tracker.current_run.run_id
        run_dir.mkdir(exist_ok=True)
        with patch("builtins.open", side_effect=OSError("disk full")):
            with pytest.raises(ExperimentTrackerError, match="Failed to save"):
                tracker.log_artifact("x", {"k": "v"}, "pickle")

    def test_json_save_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        with patch("json.dump", side_effect=TypeError("bad")):
            with pytest.raises(ExperimentTrackerError, match="Failed to save"):
                tracker.log_artifact("x", object(), "json")

    def test_numpy_save_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        with patch("numpy.save", side_effect=OSError("no space")):
            with pytest.raises(ExperimentTrackerError, match="Failed to save"):
                tracker.log_artifact("x", np.array([1]), "numpy")


@pytest.mark.coverage_extra
class TestLoadArtifactLoadFailure:
    """Cover lines 296-297."""

    def test_pickle_load_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        p = tracker.log_artifact("d", {"k": "v"}, "pickle")
        p.write_bytes(b"corrupt")
        with pytest.raises(ExperimentTrackerError, match="Failed to load"):
            tracker.load_artifact("d")

    def test_json_load_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        p = tracker.log_artifact("j", {"a": 1}, "json")
        p.write_text("bad json{{{")
        with pytest.raises(ExperimentTrackerError, match="Failed to load"):
            tracker.load_artifact("j")

    def test_numpy_load_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        p = tracker.log_artifact("n", np.array([1, 2]), "numpy")
        p.write_bytes(b"not numpy")
        with pytest.raises(ExperimentTrackerError, match="Failed to load"):
            tracker.load_artifact("n")

    def test_model_load_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        p = tracker.log_artifact("m", {"w": 1}, "model")
        p.write_bytes(b"corrupt model")
        with pytest.raises(ExperimentTrackerError, match="Failed to load"):
            tracker.load_artifact("m")
