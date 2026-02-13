"""Extra coverage tests for experiment_tracker.py.

Targets remaining uncovered lines:
  263-264 - log_artifact exception path (save failure)
  296-297 - load_artifact exception path (load failure)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlops.experiment_tracker import (
class TestLogArtifactException:
    """Cover lines 263-264: exception during artifact save."""

    def test_pickle_save_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        # Make the artifact directory read-only so open() fails
        run_artifacts_path = tracker.artifacts_path / tracker.current_run.run_id
        run_artifacts_path.mkdir(exist_ok=True)

        # Patch open to raise an IOError during pickle.dump
        with patch("builtins.open", side_effect=OSError("disk full")):
            with pytest.raises(ExperimentTrackerError, match="Failed to save artifact"):
                tracker.log_artifact("broken", {"data": "value"}, "pickle")

    def test_json_save_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")

        # Patch json.dump to raise
        with patch("json.dump", side_effect=TypeError("not serializable")):
            with pytest.raises(ExperimentTrackerError, match="Failed to save artifact"):
                tracker.log_artifact("broken", object(), "json")

    def test_numpy_save_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")

        with patch("numpy.save", side_effect=OSError("no space")):
            with pytest.raises(ExperimentTrackerError, match="Failed to save artifact"):
                tracker.log_artifact("broken", np.array([1]), "numpy")

class TestLoadArtifactException:
    """Cover lines 296-297: exception during artifact load."""

    def test_pickle_load_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        # Save a valid artifact first
        path = tracker.log_artifact("d", {"k": "v"}, "pickle")
        # Corrupt the file
        path.write_bytes(b"corrupted data that is not valid pickle")
        with pytest.raises(ExperimentTrackerError, match="Failed to load artifact"):
            tracker.load_artifact("d")

    def test_json_load_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        path = tracker.log_artifact("j", {"k": "v"}, "json")
        # Corrupt the json file
        path.write_text("not valid json {{{")
        with pytest.raises(ExperimentTrackerError, match="Failed to load artifact"):
            tracker.load_artifact("j")

    def test_numpy_load_fails(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        path = tracker.log_artifact("n", np.array([1, 2, 3]), "numpy")
        # Corrupt the numpy file
        path.write_bytes(b"not a numpy file")
        with pytest.raises(ExperimentTrackerError, match="Failed to load artifact"):
            tracker.load_artifact("n")

    def test_other_type_load_fails(self, tracker):
        """Test loading an artifact with 'other' type that has corrupted data."""
        tracker.create_experiment("e")
        tracker.start_run("e")
        path = tracker.log_artifact("o", {"data": 1}, "model")
        # Corrupt the model file
        path.write_bytes(b"corrupted model data")
        with pytest.raises(ExperimentTrackerError, match="Failed to load artifact"):
            tracker.load_artifact("o")
@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tracker(tmp_dir):
    return ExperimentTracker(base_path=tmp_dir)

