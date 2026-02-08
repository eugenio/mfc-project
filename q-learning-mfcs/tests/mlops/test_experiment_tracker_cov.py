"""Tests for experiment_tracker module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mlops.experiment_tracker import (
    ArtifactInfo, ExperimentRun, ExperimentTracker,
    ExperimentTrackerError, MetricValue,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tracker(tmp_dir):
    return ExperimentTracker(base_path=tmp_dir)


class TestMetricValue:
    def test_create_int(self):
        assert MetricValue(42).value == 42.0

    def test_create_float_with_step(self):
        assert MetricValue(3.14, step=5).step == 5

    def test_create_with_timestamp(self):
        ts = datetime(2025, 1, 1)
        assert MetricValue(1.0, timestamp=ts).timestamp == ts

    def test_invalid_nan(self):
        with pytest.raises(ExperimentTrackerError):
            MetricValue(float("nan"))

    def test_invalid_string(self):
        with pytest.raises(ExperimentTrackerError):
            MetricValue("bad")

    def test_to_dict(self):
        d = MetricValue(1.5, step=3).to_dict()
        assert d["value"] == 1.5 and d["step"] == 3

    def test_from_dict(self):
        d = {"value": 2.5, "timestamp": "2025-01-01T00:00:00", "step": 7}
        assert MetricValue.from_dict(d).value == 2.5

    def test_from_dict_no_step(self):
        d = {"value": 1.0, "timestamp": "2025-01-01T00:00:00"}
        assert MetricValue.from_dict(d).step is None


class TestArtifactInfo:
    def test_create_existing_file(self, tmp_dir):
        p = Path(tmp_dir) / "test.pkl"
        p.write_bytes(b"data")
        assert ArtifactInfo("test", p, "pickle").size_bytes == 4

    def test_create_nonexistent(self, tmp_dir):
        assert ArtifactInfo("t", Path(tmp_dir) / "x.pkl", "p").size_bytes is None

    def test_to_dict(self, tmp_dir):
        ai = ArtifactInfo("t", Path(tmp_dir) / "t.pkl", "p", size_bytes=10)
        assert ai.to_dict()["name"] == "t"

    def test_from_dict(self):
        d = {"name": "t", "file_path": "/tmp/t.pkl", "artifact_type": "pickle",
             "size_bytes": 10, "created_at": "2025-01-01T00:00:00"}
        assert ArtifactInfo.from_dict(d).name == "t"


class TestExperimentRun:
    def test_create_minimal(self):
        assert ExperimentRun("r1", "exp", datetime.now()).status == "running"

    def test_to_dict_with_end(self):
        run = ExperimentRun("r1", "exp", datetime(2025, 1, 1),
                            end_time=datetime(2025, 1, 2), status="done",
                            metrics={"l": [MetricValue(0.5)]}, tags=["v1"])
        assert run.to_dict()["end_time"] is not None

    def test_to_dict_no_end(self):
        assert ExperimentRun("r1", "e", datetime.now()).to_dict()["end_time"] is None

    def test_from_dict(self):
        d = {"run_id": "r1", "experiment_name": "exp",
             "start_time": "2025-01-01T00:00:00",
             "end_time": "2025-01-02T00:00:00", "status": "done",
             "hyperparameters": {}, "metrics": {
                 "l": [{"value": 0.5, "timestamp": "2025-01-01T00:00:00", "step": 1}]
             }, "artifacts": {}, "tags": [], "notes": None}
        assert ExperimentRun.from_dict(d).run_id == "r1"

    def test_from_dict_no_end(self):
        d = {"run_id": "r1", "experiment_name": "e",
             "start_time": "2025-01-01T00:00:00", "end_time": None, "status": "r"}
        assert ExperimentRun.from_dict(d).end_time is None


class TestExperimentTracker:
    def test_init(self, tmp_dir):
        assert ExperimentTracker(base_path=tmp_dir).experiments_path.exists()

    def test_create_experiment(self, tracker):
        tracker.create_experiment("exp1", "desc1")
        assert (tracker.experiments_path / "exp1.json").exists()

    def test_create_experiment_duplicate(self, tracker):
        tracker.create_experiment("exp1")
        tracker.create_experiment("exp1")

    def test_list_experiments(self, tracker):
        tracker.create_experiment("exp1", "d")
        assert "exp1" in tracker.list_experiments()

    def test_list_experiments_bad_json(self, tracker):
        (tracker.experiments_path / "bad.json").write_text("{bad")
        assert "bad" not in tracker.list_experiments()

    def test_start_run(self, tracker):
        tracker.create_experiment("exp1")
        assert tracker.start_run("exp1", {"lr": 0.01}, ["t"], "n") is not None

    def test_start_run_already_active(self, tracker):
        tracker.create_experiment("exp1")
        tracker.start_run("exp1")
        with pytest.raises(ExperimentTrackerError):
            tracker.start_run("exp1")

    def test_start_run_no_experiment(self, tracker):
        with pytest.raises(ExperimentTrackerError):
            tracker.start_run("no")

    def test_log_metric(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.log_metric("l", 0.5, step=1)
        assert "l" in tracker.current_run.metrics

    def test_log_metric_no_run(self, tracker):
        with pytest.raises(ExperimentTrackerError):
            tracker.log_metric("l", 0.5)

    def test_log_metric_nan(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        with pytest.raises(ExperimentTrackerError):
            tracker.log_metric("l", float("nan"))

    def test_log_hyperparameters(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.log_hyperparameters({"lr": 0.01})
        assert tracker.current_run.hyperparameters["lr"] == 0.01

    def test_log_hyperparameters_no_run(self, tracker):
        with pytest.raises(ExperimentTrackerError):
            tracker.log_hyperparameters({})

    def test_log_artifact_pickle(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        assert tracker.log_artifact("d", {"k": "v"}, "pickle").exists()

    def test_log_artifact_numpy(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        assert tracker.log_artifact("a", np.array([1]), "numpy").exists()

    def test_log_artifact_json(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        assert tracker.log_artifact("c", {"a": 1}, "json").exists()

    def test_log_artifact_other(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        assert tracker.log_artifact("d", b"x", "other").exists()

    def test_log_artifact_no_run(self, tracker):
        with pytest.raises(ExperimentTrackerError):
            tracker.log_artifact("d", {})

    def test_log_model(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        assert tracker.log_model("m", {"w": [1]}).exists()

    def test_load_artifact_pickle(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.log_artifact("d", {"k": "v"}, "pickle")
        assert tracker.load_artifact("d")["k"] == "v"

    def test_load_artifact_numpy(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        a = np.array([1, 2])
        tracker.log_artifact("a", a, "numpy")
        np.testing.assert_array_equal(tracker.load_artifact("a"), a)

    def test_load_artifact_json(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.log_artifact("c", {"a": 1}, "json")
        assert tracker.load_artifact("c")["a"] == 1

    def test_load_artifact_no_run(self, tracker):
        with pytest.raises(ExperimentTrackerError):
            tracker.load_artifact("d")

    def test_load_artifact_not_found(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        with pytest.raises(ExperimentTrackerError):
            tracker.load_artifact("missing")

    def test_load_artifact_file_deleted(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.log_artifact("d", {"k": "v"}, "pickle")
        tracker.current_run.artifacts["d"].file_path.unlink()
        with pytest.raises(ExperimentTrackerError):
            tracker.load_artifact("d")

    def test_end_run(self, tracker):
        tracker.create_experiment("e")
        rid = tracker.start_run("e")
        tracker.end_run("completed")
        assert tracker.current_run is None
        assert (tracker.runs_path / f"{rid}.json").exists()

    def test_end_run_no_active(self, tracker):
        tracker.end_run()

    def test_get_run(self, tracker):
        tracker.create_experiment("e")
        rid = tracker.start_run("e")
        tracker.end_run()
        assert tracker.get_run(rid) is not None

    def test_get_run_not_found(self, tracker):
        assert tracker.get_run("x") is None

    def test_get_run_bad_json(self, tracker):
        (tracker.runs_path / "bad.json").write_text("{bad")
        assert tracker.get_run("bad") is None

    def test_compare_runs(self, tracker):
        tracker.create_experiment("e")
        i1 = tracker.start_run("e")
        tracker.log_metric("l", 0.5)
        tracker.end_run()
        i2 = tracker.start_run("e")
        tracker.log_metric("l", 0.3)
        tracker.end_run()
        assert len(tracker.compare_runs([i1, i2])) == 2

    def test_compare_runs_missing(self, tracker):
        assert len(tracker.compare_runs(["x"])) == 0

    def test_get_best_run_max(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.log_metric("a", 0.8)
        tracker.end_run()
        tracker.start_run("e")
        tracker.log_metric("a", 0.9)
        tracker.end_run()
        assert tracker.get_best_run("e", "a", True) is not None

    def test_get_best_run_min(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.log_metric("l", 0.8)
        tracker.end_run()
        tracker.start_run("e")
        tracker.log_metric("l", 0.3)
        tracker.end_run()
        assert tracker.get_best_run("e", "l", False) is not None

    def test_get_best_run_no_runs(self, tracker):
        assert tracker.get_best_run("x", "a") is None

    def test_get_best_run_no_metric(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.end_run()
        assert tracker.get_best_run("e", "missing") is None

    def test_search_runs_hp(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e", {"lr": 0.01})
        tracker.end_run()
        tracker.start_run("e", {"lr": 0.1})
        tracker.end_run()
        assert len(tracker.search_runs("e", hyperparameters={"lr": 0.01})) == 1

    def test_search_runs_tags(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e", tags=["gpu"])
        tracker.end_run()
        tracker.start_run("e", tags=["cpu"])
        tracker.end_run()
        assert len(tracker.search_runs("e", tags=["gpu"])) == 1

    def test_search_runs_status(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.end_run("completed")
        tracker.start_run("e")
        tracker.end_run("failed")
        assert len(tracker.search_runs("e", status="completed")) == 1

    def test_search_runs_no_match(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e", {"lr": 0.01})
        tracker.end_run()
        assert len(tracker.search_runs("e", hyperparameters={"lr": 999})) == 0

    def test_export_json(self, tracker, tmp_dir):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.log_metric("l", 0.5)
        tracker.end_run()
        out = Path(tmp_dir) / "exp.json"
        tracker.export_experiment("e", out, format="json")
        assert out.exists()

    def test_export_csv(self, tracker, tmp_dir):
        tracker.create_experiment("e")
        tracker.start_run("e", {"lr": 0.01}, ["t1"])
        tracker.log_metric("l", 0.5)
        tracker.end_run()
        out = Path(tmp_dir) / "exp.csv"
        tracker.export_experiment("e", out, format="csv")
        assert out.exists()

    def test_export_unsupported(self, tracker, tmp_dir):
        tracker.create_experiment("e")
        with pytest.raises(ExperimentTrackerError):
            tracker.export_experiment("e", Path(tmp_dir) / "x", format="xyz")

    def test_get_experiment_summary(self, tracker):
        tracker.create_experiment("e")
        tracker.start_run("e")
        tracker.log_metric("a", 0.9)
        tracker.end_run("completed")
        tracker.start_run("e")
        tracker.log_metric("a", 0.8)
        tracker.end_run("failed")
        s = tracker.get_experiment_summary("e")
        assert s["total_runs"] == 2

    def test_get_experiment_summary_no_runs(self, tracker):
        assert tracker.get_experiment_summary("x")["total_runs"] == 0

    def test_get_experiment_runs_bad(self, tracker):
        (tracker.experiments_path / "bad.json").write_text("{bad")
        assert tracker._get_experiment_runs("bad") == []
