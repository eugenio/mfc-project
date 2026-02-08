"""Tests for experimental_data_integration.py - coverage part 3.

Missing lines: DataPreprocessor._detect_outliers (zscore),
_interpolate_missing (spline, forward_fill, backward_fill),
_smooth_data, _resample_data, _calculate_quality_score,
_identify_quality_issues, ModelCalibrator.calibrate_model
(bayesian/ml methods), _create_parameter_array, _calculate_r_squared,
ExperimentalDataManager init.
"""
import sys
import os
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.experimental_data_integration import (
    CalibrationMethod,
    CalibrationResult,
    DataLoader,
    DataPreprocessor,
    DataQuality,
    ExperimentalDataManager,
    ExperimentalDataset,
    ModelCalibrator,
)


def _make_dataset(n=20, with_outliers=False, with_missing=False):
    """Create a test dataset."""
    data = pd.DataFrame({
        "voltage": np.random.uniform(0.3, 0.6, n),
        "current": np.random.uniform(0.001, 0.01, n),
        "power": np.random.uniform(0.001, 0.006, n),
    })
    if with_outliers:
        data.loc[0, "voltage"] = 100.0
        data.loc[1, "current"] = -50.0
    if with_missing:
        data.loc[5, "voltage"] = np.nan
        data.loc[10, "current"] = np.nan
        data.loc[15, "power"] = np.nan
    return ExperimentalDataset(
        name="test_dataset",
        data=data,
        metadata={"source": "test"},
    )


class TestDataPreprocessorClean:
    def test_clean_data(self):
        pp = DataPreprocessor()
        ds = _make_dataset(with_missing=True)
        result = pp.preprocess_dataset(ds, ["clean_data"])
        assert "processed" in result.name


class TestDataPreprocessorOutliers:
    def test_iqr_outlier_detection(self):
        pp = DataPreprocessor()
        ds = _make_dataset(with_outliers=True)
        result = pp.preprocess_dataset(ds, ["detect_outliers"])
        outlier_cols = [c for c in result.data.columns if c.endswith("_outlier")]
        assert len(outlier_cols) > 0

    def test_zscore_outlier_detection(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": np.random.uniform(0.3, 0.6, 50),
        })
        data.loc[0, "voltage"] = 100.0
        result = pp._detect_outliers(data, method="zscore")
        assert "voltage_outlier" in result.columns


class TestDataPreprocessorInterpolation:
    def test_linear_interpolation(self):
        pp = DataPreprocessor()
        ds = _make_dataset(with_missing=True)
        result = pp.preprocess_dataset(ds, ["interpolate_missing"])
        assert result.data["voltage"].isna().sum() < 3

    def test_spline_interpolation(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": [1.0, np.nan, 3.0, np.nan, 5.0] + [6.0] * 10,
        })
        result = pp._interpolate_missing(data, method="spline")
        assert result["voltage"].isna().sum() == 0

    def test_forward_fill(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": [1.0, np.nan, 3.0, np.nan, 5.0],
        })
        result = pp._interpolate_missing(data, method="forward_fill")
        assert result["voltage"].isna().sum() < 2

    def test_backward_fill(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": [np.nan, 2.0, np.nan, 4.0, 5.0],
        })
        result = pp._interpolate_missing(data, method="backward_fill")
        assert result["voltage"].isna().sum() < 2


class TestDataPreprocessorSmooth:
    def test_smooth_data(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": np.random.uniform(0.3, 0.6, 20),
            "current": np.random.uniform(0.001, 0.01, 20),
        })
        result = pp._smooth_data(data, window_size=3)
        assert isinstance(result, pd.DataFrame)

    def test_smooth_ignores_outlier_cols(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": np.random.uniform(0.3, 0.6, 20),
            "voltage_outlier": [True] * 20,
        })
        result = pp._smooth_data(data, window_size=3)
        assert (result["voltage_outlier"] == True).all()  # noqa: E712


class TestDataPreprocessorResample:
    def test_resample_with_timestamp(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="1s"),
            "voltage": np.random.uniform(0.3, 0.6, 100),
        })
        result = pp._resample_data(data, frequency="5s")
        assert len(result) < 100

    def test_resample_no_timestamp(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": np.random.uniform(0.3, 0.6, 20),
        })
        result = pp._resample_data(data, frequency="1s")
        assert len(result) == 20


class TestDataPreprocessorQuality:
    def test_quality_score_clean_data(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": np.random.uniform(0.3, 0.6, 20),
        })
        score = pp._calculate_quality_score(data)
        assert 0.0 <= score <= 1.0

    def test_quality_score_with_outliers(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": np.random.uniform(0.3, 0.6, 20),
            "voltage_outlier": [True] * 20,
        })
        score = pp._calculate_quality_score(data)
        assert 0.0 <= score <= 1.0

    def test_quality_score_with_timestamp(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=20, freq="1s"),
            "voltage": np.random.uniform(0.3, 0.6, 20),
        })
        score = pp._calculate_quality_score(data)
        assert 0.0 <= score <= 1.0

    def test_identify_issues_clean(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": np.random.uniform(0.3, 0.6, 20),
        })
        issues = pp._identify_quality_issues(data)
        assert isinstance(issues, list)

    def test_identify_issues_missing(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": [np.nan] * 15 + [0.5] * 5,
        })
        issues = pp._identify_quality_issues(data)
        assert any("missing" in i.lower() for i in issues)

    def test_identify_issues_outliers(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": [0.5] * 20,
            "voltage_outlier": [True] * 20,
        })
        issues = pp._identify_quality_issues(data)
        assert any("outlier" in i.lower() for i in issues)

    def test_identify_issues_duplicates(self):
        pp = DataPreprocessor()
        data = pd.DataFrame({
            "voltage": [0.5] * 20,
        })
        issues = pp._identify_quality_issues(data)
        assert any("duplicate" in i.lower() for i in issues)


class TestModelCalibrator:
    def test_bayesian_calibration(self):
        mc = ModelCalibrator()
        ds = _make_dataset(n=10)
        result = mc.calibrate_model(
            model_function=lambda x: {"voltage": 0.5},
            dataset=ds,
            parameters_to_calibrate=["param1"],
            parameter_bounds={"param1": (0.0, 1.0)},
            method=CalibrationMethod.BAYESIAN,
        )
        assert result.method == CalibrationMethod.BAYESIAN

    def test_maximum_likelihood_calibration(self):
        mc = ModelCalibrator()
        ds = _make_dataset(n=10)
        result = mc.calibrate_model(
            model_function=lambda x: {"voltage": 0.5},
            dataset=ds,
            parameters_to_calibrate=["param1"],
            parameter_bounds={"param1": (0.0, 1.0)},
            method=CalibrationMethod.MAXIMUM_LIKELIHOOD,
        )
        assert result.method == CalibrationMethod.MAXIMUM_LIKELIHOOD

    def test_unsupported_method(self):
        mc = ModelCalibrator()
        ds = _make_dataset(n=10)
        with pytest.raises(NotImplementedError):
            mc.calibrate_model(
                model_function=lambda x: {"voltage": 0.5},
                dataset=ds,
                parameters_to_calibrate=["param1"],
                parameter_bounds={"param1": (0.0, 1.0)},
                method=CalibrationMethod.ROBUST,
            )

    def test_create_parameter_array(self):
        mc = ModelCalibrator()
        result = mc._create_parameter_array(
            {"param1": 1.0, "param2": 2.0},
            pd.Series({"voltage": 0.5}),
        )
        assert len(result) == 2
        assert result[0] == 1.0

    def test_calculate_r_squared(self):
        mc = ModelCalibrator()
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        r2 = mc._calculate_r_squared(obs, pred)
        assert r2 > 0.9

    def test_r_squared_perfect(self):
        mc = ModelCalibrator()
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        r2 = mc._calculate_r_squared(obs, pred)
        assert r2 == 1.0

    def test_r_squared_constant(self):
        mc = ModelCalibrator()
        obs = np.array([1.0, 1.0, 1.0])
        pred = np.array([1.0, 1.0, 1.0])
        r2 = mc._calculate_r_squared(obs, pred)
        assert r2 == 1.0


class TestExperimentalDataManager:
    def test_init_default(self, tmp_path):
        mgr = ExperimentalDataManager(str(tmp_path))
        assert mgr.data_directory == Path(str(tmp_path))

    def test_init_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mgr = ExperimentalDataManager()
        assert "experimental_data" in str(mgr.data_directory)


class TestDataLoader:
    def test_load_json_list(self, tmp_path):
        loader = DataLoader()
        data = [{"voltage": 0.5, "current": 0.01}]
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(data))
        ds = loader.load_data(fp)
        assert "voltage" in ds.data.columns

    def test_load_json_dict(self, tmp_path):
        loader = DataLoader()
        data = {"voltage": 0.5, "current": 0.01}
        fp = tmp_path / "test.json"
        fp.write_text(json.dumps(data))
        ds = loader.load_data(fp)
        assert "voltage" in ds.data.columns

    def test_detect_format(self):
        loader = DataLoader()
        from config.experimental_data_integration import DataFormat
        assert loader._detect_format(Path("test.csv")) == DataFormat.CSV
        assert loader._detect_format(Path("test.json")) == DataFormat.JSON
        assert loader._detect_format(Path("test.xlsx")) == DataFormat.EXCEL
        assert loader._detect_format(Path("test.db")) == DataFormat.SQLITE
        assert loader._detect_format(Path("test.parquet")) == DataFormat.PARQUET
        assert loader._detect_format(Path("test.unknown")) == DataFormat.CSV

    def test_load_csv(self, tmp_path):
        loader = DataLoader()
        fp = tmp_path / "test.csv"
        df = pd.DataFrame({"voltage": [0.5], "current": [0.01]})
        df.to_csv(fp, index=False)
        ds = loader.load_data(fp)
        assert len(ds.data) == 1

    def test_load_file_not_found(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_data("/nonexistent/path.csv")

    def test_load_unsupported_format(self, tmp_path):
        loader = DataLoader()
        fp = tmp_path / "test.xyz"
        fp.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_data(fp)


class TestPreprocessFullPipeline:
    def test_unknown_operation(self):
        pp = DataPreprocessor()
        ds = _make_dataset()
        result = pp.preprocess_dataset(ds, ["nonexistent_op"])
        assert len(result.processing_history) == 0

    def test_default_operations(self):
        pp = DataPreprocessor()
        ds = _make_dataset(with_missing=True, with_outliers=True)
        result = pp.preprocess_dataset(ds)
        assert len(result.processing_history) >= 3
