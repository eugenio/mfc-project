"""Tests for config/experimental_data_integration.py - coverage target 98%+."""
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.experimental_data_integration import (
    DataFormat,
    DataQuality,
    CalibrationMethod,
    ExperimentalDataset,
    CalibrationResult,
    DataLoader,
    DataPreprocessor,
    ModelCalibrator,
    ExperimentalDataManager,
    calculate_model_validation_metrics,
    detect_change_points,
    align_time_series,
)


class TestEnums:
    def test_data_format(self):
        assert DataFormat.CSV.value == "csv"
        assert DataFormat.HDF5.value == "hdf5"

    def test_data_quality(self):
        assert DataQuality.EXCELLENT.value == "excellent"

    def test_calibration_method(self):
        assert CalibrationMethod.LEAST_SQUARES.value == "least_squares"


class TestExperimentalDataset:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds = ExperimentalDataset(name="t", data=df)
        assert ds.name == "t"

    def test_with_timestamp(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "v": [1.0, 2.0, 3.0],
        })
        ds = ExperimentalDataset(name="ts", data=df)
        assert ds.start_time is not None

    def test_with_duration(self):
        from datetime import datetime
        df = pd.DataFrame({"a": [1]})
        ds = ExperimentalDataset(
            name="d", data=df,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
        )
        assert ds.duration is not None


class TestDataLoader:
    def test_load_csv(self):
        loader = DataLoader()
        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", delete=False
        ) as f:
            f.write("a,b\n1,2\n3,4\n")
            path = f.name
        try:
            ds = loader.load_data(path)
            assert "a" in ds.data.columns
        finally:
            os.unlink(path)

    def test_load_json_list(self):
        loader = DataLoader()
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump([{"a": 1}], f)
            path = f.name
        try:
            ds = loader.load_data(path)
            assert len(ds.data) == 1
        finally:
            os.unlink(path)

    def test_load_json_dict(self):
        loader = DataLoader()
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"a": 1}, f)
            path = f.name
        try:
            ds = loader.load_data(path)
            assert len(ds.data) == 1
        finally:
            os.unlink(path)

    def test_load_json_invalid(self):
        loader = DataLoader()
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump("string", f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="list or dictionary"):
                loader.load_data(path)
        finally:
            os.unlink(path)

    def test_load_parquet(self):
        loader = DataLoader()
        df = pd.DataFrame({"x": [1.0, 2.0]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        df.to_parquet(path)
        try:
            ds = loader.load_data(path)
            assert "x" in ds.data.columns
        finally:
            os.unlink(path)

    def test_load_not_found(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_data("/tmp/no_such_file_xyz.csv")

    def test_load_unsupported(self):
        loader = DataLoader()
        with tempfile.NamedTemporaryFile(
            suffix=".xyz", mode="w", delete=False
        ) as f:
            f.write("x")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                loader.load_data(path)
        finally:
            os.unlink(path)

    def test_detect_formats(self):
        loader = DataLoader()
        assert loader._detect_format(Path("t.csv")) == DataFormat.CSV
        assert loader._detect_format(Path("t.json")) == DataFormat.JSON
        assert loader._detect_format(Path("t.h5")) == DataFormat.HDF5
        assert loader._detect_format(Path("t.xlsx")) == DataFormat.EXCEL
        assert loader._detect_format(Path("t.db")) == DataFormat.SQLITE
        assert loader._detect_format(Path("t.parquet")) == DataFormat.PARQUET
        assert loader._detect_format(Path("t.xyz")) == DataFormat.CSV

    def test_timestamp_auto_detect(self):
        loader = DataLoader()
        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", delete=False
        ) as f:
            f.write("time_col,v\n2024-01-01,1\n2024-01-02,2\n")
            path = f.name
        try:
            ds = loader.load_data(path)
            assert "timestamp" in ds.data.columns
        finally:
            os.unlink(path)


class TestDataPreprocessor:
    def test_default_ops(self):
        pp = DataPreprocessor()
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        })
        ds = ExperimentalDataset(name="t", data=df)
        r = pp.preprocess_dataset(ds)
        assert len(r.processing_history) > 0

    def test_unknown_op(self):
        pp = DataPreprocessor()
        df = pd.DataFrame({"a": [1.0]})
        ds = ExperimentalDataset(name="t", data=df)
        r = pp.preprocess_dataset(ds, operations=["nonexistent_op"])
        assert len(r.processing_history) == 0

    def test_clean_data(self):
        pp = DataPreprocessor()
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
        c = pp._clean_data(df)
        assert "b" not in c.columns

    def test_outliers_iqr(self):
        pp = DataPreprocessor()
        df = pd.DataFrame({"a": list(range(20)) + [1000]})
        r = pp._detect_outliers(df, method="iqr")
        assert "a_outlier" in r.columns

    def test_interpolate(self):
        pp = DataPreprocessor()
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        r = pp._interpolate_missing(df)
        assert not r["a"].isna().any()

    def test_smooth(self):
        pp = DataPreprocessor()
        df = pd.DataFrame({"a": np.random.normal(0, 1, 20)})
        r = pp._smooth_data(df, window_size=3)
        assert len(r) == 20

    def test_resample_no_ts(self):
        pp = DataPreprocessor()
        df = pd.DataFrame({"a": [1.0, 2.0]})
        r = pp._resample_data(df)
        assert len(r) == 2

    def test_quality_score(self):
        pp = DataPreprocessor()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        s = pp._calculate_quality_score(df)
        assert 0.0 <= s <= 1.0

    def test_quality_issues(self):
        pp = DataPreprocessor()
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan, np.nan]})
        issues = pp._identify_quality_issues(df)
        assert any("missing" in i.lower() for i in issues)


class TestModelCalibrator:
    def test_r_squared(self):
        mc = ModelCalibrator()
        o = np.array([1.0, 2.0, 3.0])
        p = np.array([1.0, 2.0, 3.0])
        assert mc._calculate_r_squared(o, p) == pytest.approx(1.0)

    def test_r_squared_const(self):
        mc = ModelCalibrator()
        o = np.array([5.0, 5.0, 5.0])
        p = np.array([5.0, 5.0, 5.0])
        assert mc._calculate_r_squared(o, p) == 1.0

    def test_r_squared_const_bad(self):
        mc = ModelCalibrator()
        o = np.array([5.0, 5.0, 5.0])
        p = np.array([6.0, 6.0, 6.0])
        assert mc._calculate_r_squared(o, p) == 0.0

    def test_bayesian(self):
        mc = ModelCalibrator()
        df = pd.DataFrame({"v": [1.0, 2.0]})
        ds = ExperimentalDataset(name="t", data=df)
        r = mc.calibrate_model(
            lambda x: {"v": 0.0}, ds, ["p1"], {"p1": (0, 10)},
            method=CalibrationMethod.BAYESIAN,
        )
        assert r.method == CalibrationMethod.BAYESIAN

    def test_max_likelihood(self):
        mc = ModelCalibrator()
        df = pd.DataFrame({"v": [1.0]})
        ds = ExperimentalDataset(name="t", data=df)
        r = mc.calibrate_model(
            lambda x: {"v": 0.0}, ds, ["p1"], {"p1": (0, 10)},
            method=CalibrationMethod.MAXIMUM_LIKELIHOOD,
        )
        assert r.method == CalibrationMethod.MAXIMUM_LIKELIHOOD

    def test_unsupported(self):
        mc = ModelCalibrator()
        df = pd.DataFrame({"v": [1.0]})
        ds = ExperimentalDataset(name="t", data=df)
        with pytest.raises(NotImplementedError):
            mc.calibrate_model(
                lambda x: {}, ds, ["p"], {"p": (0, 1)},
                method=CalibrationMethod.ROBUST,
            )


class TestManager:
    def test_init(self):
        with tempfile.TemporaryDirectory() as d:
            m = ExperimentalDataManager(d)
            assert m.data_directory.exists()

    def test_load_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "d.csv")
            pd.DataFrame({"a": [1, 2, 3]}).to_csv(p, index=False)
            m = ExperimentalDataManager(d)
            n = m.load_experimental_data(p)
            s = m.get_dataset_summary(n)
            assert "shape" in s

    def test_summary_not_found(self):
        m = ExperimentalDataManager()
        with pytest.raises(ValueError):
            m.get_dataset_summary("x")

    def test_calibrate_not_found(self):
        m = ExperimentalDataManager()
        with pytest.raises(ValueError):
            m.calibrate_model_against_data("x", lambda: {}, [], {})

    def test_lists(self):
        m = ExperimentalDataManager()
        assert isinstance(m.list_datasets(), list)
        assert isinstance(m.list_calibration_results(), list)

    def test_export_not_found(self):
        m = ExperimentalDataManager()
        with pytest.raises(ValueError):
            m.export_calibration_results("x", "/tmp/x.json")

    def test_export(self):
        m = ExperimentalDataManager()
        r = CalibrationResult(
            method=CalibrationMethod.LEAST_SQUARES,
            dataset_name="t",
            calibrated_parameters={"p": 1.0},
            parameter_uncertainties={"p": 0.1},
            r_squared=0.9, rmse=0.1, mae=0.08, aic=10, bic=12,
            residuals=np.array([0.01]),
            standardized_residuals=np.array([0.5]),
        )
        m.calibration_results["r1"] = r
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            m.export_calibration_results("r1", path)
            with open(path) as f:
                d = json.load(f)
            assert d["result_id"] == "r1"
        finally:
            os.unlink(path)

    def test_compare(self):
        m = ExperimentalDataManager()
        r = CalibrationResult(
            method=CalibrationMethod.LEAST_SQUARES,
            dataset_name="d",
            calibrated_parameters={"p": 1.5},
            parameter_uncertainties={"p": 0.1},
            r_squared=0.9, rmse=0.1, mae=0.08, aic=10, bic=12,
            residuals=np.array([]),
            standardized_residuals=np.array([]),
        )
        m.calibration_results["r1"] = r
        df = m.compare_calibration_results(["r1"])
        assert len(df) == 1


class TestValidationMetrics:
    def test_basic(self):
        o = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = np.array([1.1, 2.1, 2.9, 4.0, 5.1])
        m = calculate_model_validation_metrics(o, p)
        assert "r_squared" in m
        assert "spearman_r" in m


class TestChangePoints:
    def test_stationary(self):
        np.random.seed(42)
        r = detect_change_points(np.random.normal(0, 1, 100))
        assert isinstance(r, list)

    def test_with_change(self):
        d = np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(0, 5, 50),
        ])
        r = detect_change_points(d)
        assert isinstance(r, list)


class TestAlignTimeSeries:
    def test_nearest(self):
        t1 = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="s"),
            "a": [1, 2, 3, 4, 5],
        })
        t2 = pd.DataFrame({
            "timestamp": pd.date_range(
                "2024-01-01 00:00:00.5", periods=5, freq="s"
            ),
            "b": [10, 20, 30, 40, 50],
        })
        a1, a2 = align_time_series(t1, t2, method="nearest")
        assert len(a2) == len(t1)

    def test_interpolate(self):
        t1 = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="s"),
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        t2 = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="s"),
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        a1, a2 = align_time_series(t1, t2, method="interpolate")
        assert len(a1) == len(a2)

    def test_unknown(self):
        t1 = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
            "a": [1, 2, 3],
        })
        t2 = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
            "b": [4, 5, 6],
        })
        a1, a2 = align_time_series(t1, t2, method="other")
        assert len(a1) == 3

    def test_missing_col(self):
        t1 = pd.DataFrame({"a": [1]})
        t2 = pd.DataFrame({"b": [2]})
        with pytest.raises(ValueError, match="not found"):
            align_time_series(t1, t2)
