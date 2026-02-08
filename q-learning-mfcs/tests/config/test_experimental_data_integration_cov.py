import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path

from config.experimental_data_integration import (
    DataFormat, DataQuality, CalibrationMethod,
    ExperimentalDataset, CalibrationResult, DataLoader,
    DataPreprocessor, ModelCalibrator, ExperimentalDataManager,
    calculate_model_validation_metrics, detect_change_points,
    align_time_series,
)


class TestEnums:
    def test_data_format(self):
        assert DataFormat.CSV.value == "csv"
        assert DataFormat.JSON.value == "json"
        assert DataFormat.HDF5.value == "hdf5"
        assert DataFormat.EXCEL.value == "excel"
        assert DataFormat.SQLITE.value == "sqlite"
        assert DataFormat.PARQUET.value == "parquet"
        assert DataFormat.MATLAB.value == "matlab"

    def test_data_quality(self):
        assert DataQuality.EXCELLENT.value == "excellent"
        assert DataQuality.GOOD.value == "good"
        assert DataQuality.ACCEPTABLE.value == "acceptable"
        assert DataQuality.POOR.value == "poor"
        assert DataQuality.INVALID.value == "invalid"

    def test_calibration_method(self):
        assert CalibrationMethod.LEAST_SQUARES.value == "least_squares"
        assert CalibrationMethod.MAXIMUM_LIKELIHOOD.value == "maximum_likelihood"
        assert CalibrationMethod.BAYESIAN.value == "bayesian"
        assert CalibrationMethod.ROBUST.value == "robust"
        assert CalibrationMethod.WEIGHTED.value == "weighted"


class TestExperimentalDataset:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2]})
        ds = ExperimentalDataset(name="t", data=df)
        assert ds.duration is None
        assert ds.quality_score == 0.0
        assert ds.quality_issues == []
        assert ds.measurement_units == {}
        assert ds.measurement_uncertainty == {}
        assert ds.sampling_frequency is None
        assert ds.start_time is None
        assert ds.end_time is None
        assert ds.experimental_conditions == {}
        assert ds.processing_history == []

    def test_with_timestamps(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "value": [1.0, 2.0, 3.0],
        })
        ds = ExperimentalDataset(name="ts", data=df)
        assert ds.start_time is not None
        assert ds.end_time is not None

    def test_with_start_end(self):
        from datetime import datetime, timedelta
        df = pd.DataFrame({"a": [1]})
        ds = ExperimentalDataset(
            name="t", data=df,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
        )
        assert ds.duration == timedelta(days=1)


class TestCalibrationResult:
    def test_creation(self):
        r = CalibrationResult(
            method=CalibrationMethod.LEAST_SQUARES,
            dataset_name="ds1",
            calibrated_parameters={"a": 1.0},
            parameter_uncertainties={"a": 0.1},
            r_squared=0.95,
            rmse=0.1,
            mae=0.08,
            aic=-10.0,
            bic=-8.0,
            residuals=np.array([0.1, -0.1]),
            standardized_residuals=np.array([1.0, -1.0]),
        )
        assert r.method == CalibrationMethod.LEAST_SQUARES
        assert r.parameter_covariance is None
        assert r.prediction_bands is None
        assert r.cross_validation_score is None
        assert r.validation_residuals is None
        assert r.calibration_time == 0.0
        assert r.convergence_info == {}
        assert r.created_at is not None


class TestDataLoader:
    @pytest.fixture
    def loader(self):
        return DataLoader()

    def test_load_csv(self, loader, tmp_path):
        f = tmp_path / "d.csv"
        f.write_text("a,b\n1,2\n3,4\n")
        ds = loader.load_data(f)
        assert "a" in ds.data.columns

    def test_load_json_list(self, loader, tmp_path):
        f = tmp_path / "d.json"
        f.write_text(json.dumps([{"a": 1}]))
        ds = loader.load_data(f)
        assert len(ds.data) == 1

    def test_load_json_dict(self, loader, tmp_path):
        f = tmp_path / "d.json"
        f.write_text(json.dumps({"a": 1}))
        ds = loader.load_data(f)
        assert len(ds.data) == 1

    def test_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_data("/nonexistent.csv")

    def test_unsupported(self, loader, tmp_path):
        f = tmp_path / "d.xyz"
        f.write_text("x")
        with pytest.raises(ValueError):
            loader.load_data(f)

    def test_detect_format(self, loader):
        assert loader._detect_format(Path("f.csv")) == DataFormat.CSV
        assert loader._detect_format(Path("f.json")) == DataFormat.JSON
        assert loader._detect_format(Path("f.xlsx")) == DataFormat.EXCEL
        assert loader._detect_format(Path("f.h5")) == DataFormat.HDF5
        assert loader._detect_format(Path("f.db")) == DataFormat.SQLITE
        assert loader._detect_format(Path("f.parquet")) == DataFormat.PARQUET
        assert loader._detect_format(Path("f.unknown")) == DataFormat.CSV

    def test_auto_timestamp(self, loader, tmp_path):
        f = tmp_path / "d.csv"
        f.write_text("time,v\n2024-01-01,1\n2024-01-02,2\n")
        ds = loader.load_data(f)
        assert "timestamp" in ds.data.columns

    def test_json_invalid_type(self, loader, tmp_path):
        f = tmp_path / "d.json"
        f.write_text('"just a string"')
        with pytest.raises(ValueError):
            loader.load_data(f)


class TestDataPreprocessor:
    @pytest.fixture
    def pp(self):
        return DataPreprocessor()

    def test_preprocess_default(self, pp):
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [1, 2, 3, 4]})
        ds = ExperimentalDataset(name="t", data=df)
        result = pp.preprocess_dataset(ds)
        assert "processed" in result.name

    def test_unknown_op(self, pp):
        df = pd.DataFrame({"a": [1]})
        ds = ExperimentalDataset(name="t", data=df)
        result = pp.preprocess_dataset(ds, operations=["bad_op"])
        assert result is not None

    def test_clean_data(self, pp):
        df = pd.DataFrame({"a": ["1", "2"], "b": [None, None]})
        cleaned = pp._clean_data(df)
        assert cleaned.shape[1] <= df.shape[1]

    def test_outliers_iqr(self, pp):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
        r = pp._detect_outliers(df, method="iqr")
        assert "a_outlier" in r.columns

    def test_interpolate(self, pp):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        r = pp._interpolate_missing(df)
        assert not r["a"].isna().any()

    def test_smooth(self, pp):
        df = pd.DataFrame({"a": list(range(10))})
        r = pp._smooth_data(df, window_size=3)
        assert len(r) == 10

    def test_resample_no_ts(self, pp):
        df = pd.DataFrame({"a": [1, 2]})
        r = pp._resample_data(df)
        assert len(r) == 2

    def test_quality_score(self, pp):
        df = pd.DataFrame({"a": [1, 2, 3]})
        s = pp._calculate_quality_score(df)
        assert 0 <= s <= 1

    def test_quality_issues(self, pp):
        df = pd.DataFrame({"a": [1] + [None] * 9})
        issues = pp._identify_quality_issues(df)
        assert any("missing" in i.lower() for i in issues)

    def test_quality_issues_no_issues(self, pp):
        df = pd.DataFrame({"a": list(range(100))})
        issues = pp._identify_quality_issues(df)
        assert isinstance(issues, list)

    def test_quality_score_with_outlier_cols(self, pp):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "a_outlier": [True, False, False, False, False],
        })
        s = pp._calculate_quality_score(df)
        assert 0 <= s <= 1


class TestModelCalibrator:
    @pytest.fixture
    def cal(self):
        return ModelCalibrator()

    def test_r2_perfect(self, cal):
        o = np.array([1, 2, 3])
        assert cal._calculate_r_squared(o, o) == pytest.approx(1.0)

    def test_r2_zero_sstot(self, cal):
        o = np.array([5, 5, 5])
        assert cal._calculate_r_squared(o, o) == 1.0

    def test_r2_zero_sstot_nonzero(self, cal):
        assert cal._calculate_r_squared(
            np.array([5, 5, 5]), np.array([4, 5, 6])
        ) == 0.0

    def test_create_parameter_array(self, cal):
        row = pd.Series({"x": 10})
        result = cal._create_parameter_array({"a": 1.0, "b": 2.0}, row)
        assert len(result) == 2

    def test_bayesian(self, cal):
        df = pd.DataFrame({"o": [1.0]})
        r = cal._bayesian_calibration(
            lambda x: {}, df, ["a"], {"a": (0, 1)}, ["o"]
        )
        assert r.method == CalibrationMethod.BAYESIAN

    def test_ml(self, cal):
        df = pd.DataFrame({"o": [1.0]})
        r = cal._maximum_likelihood_calibration(
            lambda x: {}, df, ["a"], {"a": (0, 1)}, ["o"]
        )
        assert r.method == CalibrationMethod.MAXIMUM_LIKELIHOOD

    def test_unsupported(self, cal):
        ds = ExperimentalDataset(
            name="t", data=pd.DataFrame({"a": [1]})
        )
        with pytest.raises(NotImplementedError):
            cal.calibrate_model(
                lambda x: {}, ds, ["p"], {"p": (0, 1)},
                method=CalibrationMethod.ROBUST,
            )


class TestUtilityFunctions:
    def test_metrics(self):
        o = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        m = calculate_model_validation_metrics(o, p)
        assert "rmse" in m
        assert "r_squared" in m
        assert "mae" in m
        assert "bias" in m
        assert "nash_sutcliffe" in m

    def test_change_points(self):
        ts = np.concatenate([np.ones(50), np.ones(50) * 10])
        assert isinstance(detect_change_points(ts), list)

    def test_change_points_uniform(self):
        ts = np.ones(100)
        result = detect_change_points(ts)
        assert isinstance(result, list)

    def test_align_nearest(self):
        ts = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "v": [1, 2, 3],
        })
        r1, r2 = align_time_series(ts.copy(), ts.copy(), method="nearest")
        assert len(r1) == 3

    def test_align_missing_col(self):
        with pytest.raises(ValueError):
            align_time_series(
                pd.DataFrame({"a": [1]}),
                pd.DataFrame({"b": [2]}),
            )

    def test_align_other(self):
        ts = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
            "v": [1, 2],
        })
        r1, r2 = align_time_series(
            ts.copy(), ts.copy(), method="other"
        )
        assert len(r1) == 2


class TestDataManager:
    def test_init(self, tmp_path):
        m = ExperimentalDataManager(data_directory=tmp_path)
        assert m.data_directory == tmp_path

    def test_init_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        m = ExperimentalDataManager()
        assert m.data_directory.name == "experimental_data"

    def test_list_empty(self, tmp_path):
        m = ExperimentalDataManager(data_directory=tmp_path)
        assert m.list_datasets() == []
        assert m.list_calibration_results() == []

    def test_load_and_summary(self, tmp_path):
        csv_file = tmp_path / "d.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        m = ExperimentalDataManager(data_directory=tmp_path)
        n = m.load_experimental_data(csv_file, dataset_name="my")
        assert "my" in n
        s = m.get_dataset_summary(n)
        assert "my" in s["name"]

    def test_load_no_preprocess(self, tmp_path):
        csv_file = tmp_path / "d.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        m = ExperimentalDataManager(data_directory=tmp_path)
        n = m.load_experimental_data(csv_file, preprocess=False)
        assert n is not None

    def test_summary_not_found(self, tmp_path):
        m = ExperimentalDataManager(data_directory=tmp_path)
        with pytest.raises(ValueError):
            m.get_dataset_summary("nope")

    def test_calibrate_not_found(self, tmp_path):
        m = ExperimentalDataManager(data_directory=tmp_path)
        with pytest.raises(ValueError):
            m.calibrate_model_against_data(
                "x", lambda: {}, ["p"], {"p": (0, 1)}
            )

    def test_export_not_found(self, tmp_path):
        m = ExperimentalDataManager(data_directory=tmp_path)
        with pytest.raises(ValueError):
            m.export_calibration_results("x", tmp_path / "o.json")

    def test_compare_empty(self, tmp_path):
        m = ExperimentalDataManager(data_directory=tmp_path)
        r = m.compare_calibration_results(["x"])
        assert len(r) == 0
