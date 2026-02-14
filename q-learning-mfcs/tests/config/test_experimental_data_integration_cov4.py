"""Tests for experimental_data_integration.py - coverage part 4.

Targets missing lines: HDF5 import (49), _load_excel (296),
_load_hdf5 (300-304), _load_sqlite (308-312),
_least_squares_calibration (597-666),
calibrate_model_against_data (860-881),
detect_change_points no scipy (1024-1025),
calibrate_model least_squares method dispatch (551).
"""
import sys
import os
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.experimental_data_integration import (
    CalibrationMethod,
    CalibrationResult,
    DataFormat,
    DataLoader,
    DataPreprocessor,
    ExperimentalDataManager,
    ExperimentalDataset,
    ModelCalibrator,
    calculate_model_validation_metrics,
    detect_change_points,
    align_time_series,
)


@pytest.mark.coverage_extra
class TestDataLoaderExcel:
    """Cover _load_excel (line 296)."""

    def test_load_excel(self, tmp_path):
        """Cover _load_excel by mocking pd.read_excel (openpyxl may not be installed)."""
        loader = DataLoader()
        fp = tmp_path / "test.xlsx"
        fp.write_bytes(b"dummy")
        expected_df = pd.DataFrame({"voltage": [0.5, 0.6], "current": [0.01, 0.02]})
        with patch("pandas.read_excel", return_value=expected_df):
            result = loader._load_excel(fp)
        assert "voltage" in result.columns
        assert len(result) == 2

    def test_load_xls_format_detected(self):
        loader = DataLoader()
        fmt = loader._detect_format(Path("test.xls"))
        assert fmt == DataFormat.EXCEL


@pytest.mark.coverage_extra
class TestDataLoaderHDF5:
    """Cover _load_hdf5 (lines 300-304)."""

    def test_load_hdf5_no_h5py(self, tmp_path):
        """Cover ImportError when h5py not available."""
        loader = DataLoader()
        fp = tmp_path / "test.h5"
        fp.write_bytes(b"dummy")
        with patch("config.experimental_data_integration.HAS_HDF5", False):
            with pytest.raises(ImportError, match="h5py"):
                loader._load_hdf5(fp)

    def test_load_hdf5_success(self, tmp_path):
        """Cover successful HDF5 loading by mocking pd.read_hdf (pytables may not be installed)."""
        loader = DataLoader()
        fp = tmp_path / "test.h5"
        fp.write_bytes(b"dummy")
        expected_df = pd.DataFrame({"voltage": [0.5, 0.6]})
        with patch("config.experimental_data_integration.HAS_HDF5", True), \
             patch("pandas.read_hdf", return_value=expected_df):
            result = loader._load_hdf5(fp, key="data")
        assert "voltage" in result.columns


@pytest.mark.coverage_extra
class TestDataLoaderSQLite:
    """Cover _load_sqlite (lines 308-312)."""

    def test_load_sqlite(self, tmp_path):
        loader = DataLoader()
        fp = tmp_path / "test.db"
        conn = sqlite3.connect(fp)
        df = pd.DataFrame({"voltage": [0.5, 0.6], "current": [0.01, 0.02]})
        df.to_sql("measurements", conn, index=False)
        conn.close()
        result = loader._load_sqlite(fp, table="measurements")
        assert "voltage" in result.columns
        assert len(result) == 2

    def test_load_sqlite_via_load_data(self, tmp_path):
        """Load via the main load_data method with .db extension."""
        loader = DataLoader()
        fp = tmp_path / "test.db"
        conn = sqlite3.connect(fp)
        df = pd.DataFrame({"voltage": [0.5, 0.6]})
        df.to_sql("data_table", conn, index=False)
        conn.close()
        ds = loader.load_data(fp, table="data_table")
        assert "voltage" in ds.data.columns

    def test_load_sqlite_extension(self, tmp_path):
        """Load via .sqlite extension."""
        loader = DataLoader()
        fp = tmp_path / "test.sqlite"
        conn = sqlite3.connect(fp)
        df = pd.DataFrame({"val": [1, 2, 3]})
        df.to_sql("tbl", conn, index=False)
        conn.close()
        ds = loader.load_data(fp, table="tbl")
        assert "val" in ds.data.columns


@pytest.mark.coverage_extra
class TestModelCalibratorLeastSquares:
    """Cover _least_squares_calibration (lines 597-666)."""

    def test_least_squares_calibration(self):
        mc = ModelCalibrator()
        # Simple linear model: y = 2*x
        df = pd.DataFrame({
            "x_input": [1.0, 2.0, 3.0, 4.0, 5.0],
            "output": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        ds = ExperimentalDataset(name="test_ls", data=df)

        def model_fn(params):
            return {"output": params[0] * 2}

        result = mc.calibrate_model(
            model_function=model_fn,
            dataset=ds,
            parameters_to_calibrate=["slope"],
            parameter_bounds={"slope": (0.5, 5.0)},
            method=CalibrationMethod.LEAST_SQUARES,
        )
        assert result.method == CalibrationMethod.LEAST_SQUARES
        assert "slope" in result.calibrated_parameters
        assert result.r_squared is not None
        assert result.rmse >= 0
        assert result.mae >= 0
        assert result.residuals is not None
        assert result.standardized_residuals is not None

    def test_least_squares_with_failed_evaluations(self):
        """Cover the exception handling in objective (line 617)."""
        mc = ModelCalibrator()
        call_count = [0]

        def flaky_model(params):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise RuntimeError("flaky failure")
            return {"output": params[0]}

        df = pd.DataFrame({"output": [1.0, 2.0, 3.0]})
        ds = ExperimentalDataset(name="test_flaky", data=df)

        result = mc.calibrate_model(
            model_function=flaky_model,
            dataset=ds,
            parameters_to_calibrate=["p1"],
            parameter_bounds={"p1": (0.0, 10.0)},
            method=CalibrationMethod.LEAST_SQUARES,
        )
        assert result is not None

    def test_least_squares_zero_std_residuals(self):
        """Cover zero-std residuals branch (line 681)."""
        mc = ModelCalibrator()
        # Model that produces exact predictions
        df = pd.DataFrame({"output": [1.0, 1.0, 1.0]})
        ds = ExperimentalDataset(name="test_exact", data=df)

        def perfect_model(params):
            return {"output": 1.0}

        result = mc.calibrate_model(
            model_function=perfect_model,
            dataset=ds,
            parameters_to_calibrate=["p1"],
            parameter_bounds={"p1": (0.0, 10.0)},
            method=CalibrationMethod.LEAST_SQUARES,
        )
        assert result is not None


@pytest.mark.coverage_extra
class TestExperimentalDataManagerCalibrate:
    """Cover calibrate_model_against_data (lines 860-881)."""

    def test_calibrate_model_against_data(self, tmp_path):
        mgr = ExperimentalDataManager(str(tmp_path))

        # Create and load dataset
        fp = tmp_path / "data.csv"
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "output": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        df.to_csv(fp, index=False)
        ds_name = mgr.load_experimental_data(fp, dataset_name="cal_data")

        def model_fn(params):
            return {"output": params[0]}

        result_id = mgr.calibrate_model_against_data(
            dataset_name=ds_name,
            model_function=model_fn,
            parameters_to_calibrate=["p1"],
            parameter_bounds={"p1": (0.0, 20.0)},
            method=CalibrationMethod.BAYESIAN,
        )
        assert result_id is not None
        assert result_id in mgr.calibration_results


@pytest.mark.coverage_extra
class TestDetectChangePointsNoScipy:
    """Cover detect_change_points no scipy branch (lines 1024-1025)."""

    def test_no_scipy(self):
        with patch("config.experimental_data_integration.HAS_SCIPY", False):
            result = detect_change_points(np.ones(100))
            assert result == []


@pytest.mark.coverage_extra
class TestExperimentalDataManagerLoadWithName:
    """Cover load_experimental_data with custom name and preprocess=False."""

    def test_load_no_preprocess_with_name(self, tmp_path):
        mgr = ExperimentalDataManager(str(tmp_path))
        fp = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(fp, index=False)
        name = mgr.load_experimental_data(fp, dataset_name="custom_name", preprocess=False)
        assert name == "custom_name"
        assert "custom_name" in mgr.datasets


@pytest.mark.coverage_extra
class TestExperimentalDataManagerExportAndCompare:
    """Cover export and compare methods."""

    def test_export_and_compare(self, tmp_path):
        mgr = ExperimentalDataManager(str(tmp_path))

        r1 = CalibrationResult(
            method=CalibrationMethod.LEAST_SQUARES,
            dataset_name="d1",
            calibrated_parameters={"p1": 2.5, "p2": 3.0},
            parameter_uncertainties={"p1": 0.1, "p2": 0.2},
            r_squared=0.95,
            rmse=0.05,
            mae=0.03,
            aic=-10.0,
            bic=-8.0,
            residuals=np.array([0.01, -0.01, 0.02]),
            standardized_residuals=np.array([0.5, -0.5, 1.0]),
            calibration_time=1.5,
        )
        r2 = CalibrationResult(
            method=CalibrationMethod.BAYESIAN,
            dataset_name="d1",
            calibrated_parameters={"p1": 2.6},
            parameter_uncertainties={"p1": 0.15},
            r_squared=0.90,
            rmse=0.08,
            mae=0.05,
            aic=-8.0,
            bic=-6.0,
            residuals=np.array([0.02, -0.02]),
            standardized_residuals=np.array([1.0, -1.0]),
            calibration_time=3.0,
        )
        mgr.calibration_results["r1"] = r1
        mgr.calibration_results["r2"] = r2

        # Compare
        comparison = mgr.compare_calibration_results(["r1", "r2"])
        assert len(comparison) == 2
        assert "param_p1" in comparison.columns

        # Export
        out = tmp_path / "export.json"
        mgr.export_calibration_results("r1", out)
        with open(out) as f:
            data = json.load(f)
        assert data["result_id"] == "r1"
        assert data["goodness_of_fit"]["r_squared"] == 0.95


@pytest.mark.coverage_extra
class TestCalculateModelValidationMetricsComprehensive:
    """Cover comprehensive metrics including spearman_r."""

    def test_all_metrics_present(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.1, 2.2, 2.8, 4.1, 5.0])
        metrics = calculate_model_validation_metrics(obs, pred)
        assert "r_squared" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "pearson_r" in metrics
        assert "spearman_r" in metrics
        assert "bias" in metrics
        assert "relative_bias" in metrics
        assert "nash_sutcliffe" in metrics
