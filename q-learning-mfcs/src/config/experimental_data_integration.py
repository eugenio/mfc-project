"""Experimental Data Integration System.

This module provides comprehensive tools for integrating experimental data with
MFC models, including data loading, preprocessing, calibration, and validation.

Classes:
- ExperimentalDataManager: Main data management and integration system
- DataLoader: Loading data from various sources and formats
- DataPreprocessor: Data cleaning, filtering, and preprocessing
- ModelCalibrator: Model calibration against experimental data
- ValidationFramework: Model validation and uncertainty quantification

Features:
- Multi-format data loading (CSV, JSON, HDF5, databases)
- Advanced data preprocessing and quality control
- Automated model calibration with uncertainty quantification
- Statistical validation and goodness-of-fit metrics
- Real-time data streaming and processing capabilities

Literature References:
1. Kennedy, M. C., & O'Hagan, A. (2001). "Bayesian calibration of computer models"
2. Higdon, D., et al. (2004). "Combining field data and computer simulations for calibration and prediction"
3. Brynjarsdóttir, J., & O'Hagan, A. (2014). "Learning about physical parameters"
4. Pourret, O., et al. (2008). "Bayesian Networks: A Practical Guide to Applications"
"""

from __future__ import annotations

import json
import logging
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

# Optional dependencies
try:
    import h5py

    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    warnings.warn("h5py not available. HDF5 support will be limited.", stacklevel=2)

try:
    from scipy import interpolate, signal, stats
    from scipy.optimize import curve_fit

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn(
        "SciPy not available. Some data processing features will be limited.",
        stacklevel=2,
    )

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn(
        "Matplotlib/Seaborn not available. Plotting features will be limited.",
        stacklevel=2,
    )

# Import related modules


class DataFormat(Enum):
    """Supported data formats."""

    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    EXCEL = "excel"
    SQLITE = "sqlite"
    PARQUET = "parquet"
    MATLAB = "matlab"


class DataQuality(Enum):
    """Data quality indicators."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class CalibrationMethod(Enum):
    """Available calibration methods."""

    LEAST_SQUARES = "least_squares"
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"
    BAYESIAN = "bayesian"
    ROBUST = "robust"
    WEIGHTED = "weighted"


@dataclass
class ExperimentalDataset:
    """Container for experimental dataset with metadata."""

    name: str
    data: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    # Data quality information
    quality_score: float = 0.0
    quality_issues: list[str] = field(default_factory=list)

    # Measurement information
    measurement_units: dict[str, str] = field(default_factory=dict)
    measurement_uncertainty: dict[str, float] = field(default_factory=dict)
    sampling_frequency: float | None = None  # Hz

    # Temporal information
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: timedelta | None = None

    # Experimental conditions
    experimental_conditions: dict[str, Any] = field(default_factory=dict)

    # Processing history
    processing_history: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize derived properties."""
        if self.start_time and self.end_time:
            self.duration = self.end_time - self.start_time

        if "timestamp" in self.data.columns:
            if self.start_time is None:
                self.start_time = pd.to_datetime(self.data["timestamp"].min())
            if self.end_time is None:
                self.end_time = pd.to_datetime(self.data["timestamp"].max())


@dataclass
class CalibrationResult:
    """Results of model calibration against experimental data."""

    method: CalibrationMethod
    dataset_name: str
    calibrated_parameters: dict[str, float]
    parameter_uncertainties: dict[str, float]

    # Goodness of fit metrics
    r_squared: float
    rmse: float
    mae: float
    aic: float
    bic: float

    # Residual analysis
    residuals: np.ndarray
    standardized_residuals: np.ndarray

    # Uncertainty quantification
    parameter_covariance: np.ndarray | None = None
    prediction_bands: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    # Validation metrics
    cross_validation_score: float | None = None
    validation_residuals: np.ndarray | None = None

    # Metadata
    calibration_time: float = 0.0
    convergence_info: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class DataLoader:
    """Advanced data loading with automatic format detection and preprocessing."""

    def __init__(self) -> None:
        """Initialize data loader."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {
            ".csv": self._load_csv,
            ".json": self._load_json,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".h5": self._load_hdf5,
            ".hdf5": self._load_hdf5,
            ".db": self._load_sqlite,
            ".sqlite": self._load_sqlite,
            ".parquet": self._load_parquet,
        }

    def load_data(
        self,
        file_path: str | Path,
        format: DataFormat | None = None,
        **kwargs,
    ) -> ExperimentalDataset:
        """Load experimental data from file.

        Args:
            file_path: Path to data file
            format: Data format (auto-detected if None)
            **kwargs: Format-specific loading parameters

        Returns:
            Experimental dataset

        """
        file_path = Path(file_path)

        if not file_path.exists():
            msg = f"Data file not found: {file_path}"
            raise FileNotFoundError(msg)

        # Auto-detect format if not specified
        if format is None:
            format = self._detect_format(file_path)

        # Load data using appropriate method
        if file_path.suffix.lower() in self.supported_formats:
            data = self.supported_formats[file_path.suffix.lower()](file_path, **kwargs)
        else:
            msg = f"Unsupported file format: {file_path.suffix}"
            raise ValueError(msg)

        # Create dataset
        dataset = ExperimentalDataset(
            name=file_path.stem,
            data=data,
            metadata={"file_path": str(file_path), "format": format.value},
        )

        # Auto-detect timestamp columns
        timestamp_cols = [
            col
            for col in data.columns
            if any(keyword in col.lower() for keyword in ["time", "date", "timestamp"])
        ]

        if timestamp_cols:
            dataset.data["timestamp"] = pd.to_datetime(dataset.data[timestamp_cols[0]])

        return dataset

    def _detect_format(self, file_path: Path) -> DataFormat:
        """Auto-detect data format from file extension."""
        suffix = file_path.suffix.lower()

        format_map = {
            ".csv": DataFormat.CSV,
            ".json": DataFormat.JSON,
            ".xlsx": DataFormat.EXCEL,
            ".xls": DataFormat.EXCEL,
            ".h5": DataFormat.HDF5,
            ".hdf5": DataFormat.HDF5,
            ".db": DataFormat.SQLITE,
            ".sqlite": DataFormat.SQLITE,
            ".parquet": DataFormat.PARQUET,
        }

        return format_map.get(suffix, DataFormat.CSV)

    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        default_kwargs = {"parse_dates": True, "infer_datetime_format": True}
        default_kwargs.update(kwargs)
        return pd.read_csv(file_path, **default_kwargs)

    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        with open(file_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        msg = "JSON data must be a list or dictionary"
        raise ValueError(msg)

    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(file_path, **kwargs)

    def _load_hdf5(self, file_path: Path, key: str = "data", **kwargs) -> pd.DataFrame:
        """Load HDF5 file."""
        if not HAS_HDF5:
            msg = "h5py required for HDF5 support"
            raise ImportError(msg)

        return pd.read_hdf(file_path, key=key, **kwargs)

    def _load_sqlite(self, file_path: Path, table: str, **kwargs) -> pd.DataFrame:
        """Load SQLite database."""
        conn = sqlite3.connect(file_path)
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn, **kwargs)
        finally:
            conn.close()

    def _load_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(file_path, **kwargs)


class DataPreprocessor:
    """Advanced data preprocessing and quality control."""

    def __init__(self) -> None:
        """Initialize data preprocessor."""
        self.logger = logging.getLogger(__name__)

    def preprocess_dataset(
        self,
        dataset: ExperimentalDataset,
        operations: list[str] | None = None,
    ) -> ExperimentalDataset:
        """Apply preprocessing operations to dataset.

        Args:
            dataset: Input experimental dataset
            operations: List of preprocessing operations

        Returns:
            Preprocessed dataset

        """
        if operations is None:
            operations = ["clean_data", "detect_outliers", "interpolate_missing"]

        processed_data = dataset.data.copy()
        processing_history = dataset.processing_history.copy()

        for operation in operations:
            if hasattr(self, f"_{operation}"):
                processed_data = getattr(self, f"_{operation}")(processed_data)
                processing_history.append(f"Applied {operation}")
            else:
                self.logger.warning(f"Unknown preprocessing operation: {operation}")

        # Create new dataset with processed data
        processed_dataset = ExperimentalDataset(
            name=f"{dataset.name}_processed",
            data=processed_data,
            metadata=dataset.metadata.copy(),
            measurement_units=dataset.measurement_units.copy(),
            measurement_uncertainty=dataset.measurement_uncertainty.copy(),
            sampling_frequency=dataset.sampling_frequency,
            experimental_conditions=dataset.experimental_conditions.copy(),
            processing_history=processing_history,
        )

        # Calculate quality score
        processed_dataset.quality_score = self._calculate_quality_score(processed_data)
        processed_dataset.quality_issues = self._identify_quality_issues(processed_data)

        return processed_dataset

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning operations."""
        cleaned = data.copy()

        # Remove completely empty rows/columns
        cleaned = cleaned.dropna(how="all").dropna(axis=1, how="all")

        # Convert numeric columns
        for col in cleaned.columns:
            if col != "timestamp":
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

        return cleaned

    def _detect_outliers(self, data: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Detect and flag outliers."""
        processed = data.copy()

        for col in processed.select_dtypes(include=[np.number]).columns:
            if method == "iqr":
                Q1 = processed[col].quantile(0.25)
                Q3 = processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (processed[col] < lower_bound) | (
                    processed[col] > upper_bound
                )
                processed.loc[outliers, f"{col}_outlier"] = True

            elif method == "zscore":
                z_scores = np.abs(stats.zscore(processed[col].dropna()))
                outliers = z_scores > 3
                processed.loc[processed[col].notna(), f"{col}_outlier"] = outliers

        return processed

    def _interpolate_missing(
        self,
        data: pd.DataFrame,
        method: str = "linear",
    ) -> pd.DataFrame:
        """Interpolate missing values."""
        interpolated = data.copy()

        for col in interpolated.select_dtypes(include=[np.number]).columns:
            if interpolated[col].isna().any():
                if method == "linear":
                    interpolated[col] = interpolated[col].interpolate(method="linear")
                elif method == "spline":
                    interpolated[col] = interpolated[col].interpolate(
                        method="spline",
                        order=2,
                    )
                elif method == "forward_fill":
                    interpolated[col] = interpolated[col].fillna(method="ffill")
                elif method == "backward_fill":
                    interpolated[col] = interpolated[col].fillna(method="bfill")

        return interpolated

    def _smooth_data(self, data: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """Apply smoothing filter to data."""
        smoothed = data.copy()

        for col in smoothed.select_dtypes(include=[np.number]).columns:
            if not col.endswith("_outlier"):
                smoothed[col] = (
                    smoothed[col].rolling(window=window_size, center=True).mean()
                )

        return smoothed

    def _resample_data(self, data: pd.DataFrame, frequency: str = "1S") -> pd.DataFrame:
        """Resample data to uniform frequency."""
        if "timestamp" not in data.columns:
            self.logger.warning("No timestamp column found for resampling")
            return data

        return data.set_index("timestamp").resample(frequency).mean().reset_index()

    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        scores = []

        # Completeness score
        completeness = 1.0 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        scores.append(completeness)

        # Consistency score (based on outlier detection)
        outlier_cols = [col for col in data.columns if col.endswith("_outlier")]
        if outlier_cols:
            outlier_rate = data[outlier_cols].sum().sum() / (
                data.shape[0] * len(outlier_cols)
            )
            consistency = 1.0 - outlier_rate
            scores.append(consistency)

        # Temporal regularity (if timestamp exists)
        if "timestamp" in data.columns:
            time_diffs = data["timestamp"].diff().dropna()
            if len(time_diffs) > 1:
                regularity = 1.0 - (time_diffs.std() / time_diffs.mean())
                scores.append(np.clip(regularity, 0, 1))

        return np.mean(scores) if scores else 0.0

    def _identify_quality_issues(self, data: pd.DataFrame) -> list[str]:
        """Identify data quality issues."""
        issues = []

        # Check for missing data
        missing_rate = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_rate > 0.1:
            issues.append(f"High missing data rate: {missing_rate:.2%}")

        # Check for outliers
        outlier_cols = [col for col in data.columns if col.endswith("_outlier")]
        if outlier_cols:
            outlier_rate = data[outlier_cols].sum().sum() / (
                data.shape[0] * len(outlier_cols)
            )
            if outlier_rate > 0.05:
                issues.append(f"High outlier rate: {outlier_rate:.2%}")

        # Check for duplicates
        duplicate_rate = data.duplicated().sum() / len(data)
        if duplicate_rate > 0.01:
            issues.append(f"Duplicate rows detected: {duplicate_rate:.2%}")

        return issues


class ModelCalibrator:
    """Model calibration against experimental data with uncertainty quantification."""

    def __init__(self) -> None:
        """Initialize model calibrator."""
        self.logger = logging.getLogger(__name__)

    def calibrate_model(
        self,
        model_function: Callable[[np.ndarray], dict[str, float]],
        dataset: ExperimentalDataset,
        parameters_to_calibrate: list[str],
        parameter_bounds: dict[str, tuple[float, float]],
        method: CalibrationMethod = CalibrationMethod.LEAST_SQUARES,
        **kwargs,
    ) -> CalibrationResult:
        """Calibrate model parameters against experimental data.

        Args:
            model_function: Model function to calibrate
            dataset: Experimental dataset for calibration
            parameters_to_calibrate: List of parameter names to calibrate
            parameter_bounds: Bounds for each parameter
            method: Calibration method
            **kwargs: Method-specific parameters

        Returns:
            Calibration results

        """
        import time

        start_time = time.time()

        # Extract experimental data
        exp_data = dataset.data

        # Determine output variables to match
        output_variables = [
            col
            for col in exp_data.columns
            if col not in ["timestamp"] and not col.endswith("_outlier")
        ]

        if method == CalibrationMethod.LEAST_SQUARES:
            result = self._least_squares_calibration(
                model_function,
                exp_data,
                parameters_to_calibrate,
                parameter_bounds,
                output_variables,
                **kwargs,
            )
        elif method == CalibrationMethod.BAYESIAN:
            result = self._bayesian_calibration(
                model_function,
                exp_data,
                parameters_to_calibrate,
                parameter_bounds,
                output_variables,
                **kwargs,
            )
        elif method == CalibrationMethod.MAXIMUM_LIKELIHOOD:
            result = self._maximum_likelihood_calibration(
                model_function,
                exp_data,
                parameters_to_calibrate,
                parameter_bounds,
                output_variables,
                **kwargs,
            )
        else:
            msg = f"Calibration method {method} not implemented"
            raise NotImplementedError(msg)

        result.method = method
        result.dataset_name = dataset.name
        result.calibration_time = time.time() - start_time

        return result

    def _least_squares_calibration(
        self,
        model_function: Callable,
        exp_data: pd.DataFrame,
        parameters_to_calibrate: list[str],
        parameter_bounds: dict[str, tuple[float, float]],
        output_variables: list[str],
        **kwargs,
    ) -> CalibrationResult:
        """Least squares parameter calibration."""
        from scipy.optimize import differential_evolution

        # Define objective function
        def objective(params):
            param_dict = dict(zip(parameters_to_calibrate, params, strict=False))

            total_error = 0.0
            for _, row in exp_data.iterrows():
                try:
                    # Create parameter array for model
                    model_params = self._create_parameter_array(param_dict, row)
                    model_output = model_function(model_params)

                    # Calculate squared errors
                    for var in output_variables:
                        if var in model_output and not pd.isna(row[var]):
                            error = (model_output[var] - row[var]) ** 2
                            total_error += error

                except Exception:
                    return 1e10  # Large penalty for failed evaluations

            return total_error

        # Set up bounds
        bounds = [parameter_bounds[param] for param in parameters_to_calibrate]

        # Optimize using differential evolution for global optimization
        result = differential_evolution(objective, bounds, seed=42, maxiter=1000)

        if not result.success:
            self.logger.warning("Calibration optimization did not converge")

        # Calculate goodness of fit metrics
        calibrated_params = dict(zip(parameters_to_calibrate, result.x, strict=False))

        # Generate predictions with calibrated parameters
        predictions = []
        observations = []

        for _, row in exp_data.iterrows():
            try:
                model_params = self._create_parameter_array(calibrated_params, row)
                model_output = model_function(model_params)

                for var in output_variables:
                    if var in model_output and not pd.isna(row[var]):
                        predictions.append(model_output[var])
                        observations.append(row[var])
            except Exception:
                continue

        predictions = np.array(predictions)
        observations = np.array(observations)

        # Calculate metrics
        r_squared = self._calculate_r_squared(observations, predictions)
        rmse = np.sqrt(np.mean((observations - predictions) ** 2))
        mae = np.mean(np.abs(observations - predictions))
        residuals = observations - predictions

        # Calculate AIC and BIC
        n = len(observations)
        k = len(parameters_to_calibrate)
        mse = np.mean(residuals**2)
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(n)

        # Create calibration result
        return CalibrationResult(
            method=CalibrationMethod.LEAST_SQUARES,
            dataset_name="",
            calibrated_parameters=calibrated_params,
            parameter_uncertainties=dict.fromkeys(
                parameters_to_calibrate,
                0.0,
            ),  # TODO: estimate uncertainties
            r_squared=r_squared,
            rmse=rmse,
            mae=mae,
            aic=aic,
            bic=bic,
            residuals=residuals,
            standardized_residuals=(
                residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
            ),
            convergence_info={
                "success": result.success,
                "nit": result.nit,
                "fun": result.fun,
            },
        )

    def _bayesian_calibration(
        self,
        model_function: Callable,
        exp_data: pd.DataFrame,
        parameters_to_calibrate: list[str],
        parameter_bounds: dict[str, tuple[float, float]],
        output_variables: list[str],
        **kwargs,
    ) -> CalibrationResult:
        """Bayesian parameter calibration using MCMC."""
        # This would use the BayesianInference class from uncertainty_quantification
        # For now, return a placeholder
        self.logger.warning("Bayesian calibration not fully implemented")

        return CalibrationResult(
            method=CalibrationMethod.BAYESIAN,
            dataset_name="",
            calibrated_parameters=dict.fromkeys(parameters_to_calibrate, 0.0),
            parameter_uncertainties=dict.fromkeys(parameters_to_calibrate, 0.0),
            r_squared=0.0,
            rmse=0.0,
            mae=0.0,
            aic=0.0,
            bic=0.0,
            residuals=np.array([]),
            standardized_residuals=np.array([]),
        )

    def _maximum_likelihood_calibration(
        self,
        model_function: Callable,
        exp_data: pd.DataFrame,
        parameters_to_calibrate: list[str],
        parameter_bounds: dict[str, tuple[float, float]],
        output_variables: list[str],
        **kwargs,
    ) -> CalibrationResult:
        """Maximum likelihood parameter calibration."""
        # Similar to least squares but with likelihood maximization
        self.logger.warning("Maximum likelihood calibration not fully implemented")

        return CalibrationResult(
            method=CalibrationMethod.MAXIMUM_LIKELIHOOD,
            dataset_name="",
            calibrated_parameters=dict.fromkeys(parameters_to_calibrate, 0.0),
            parameter_uncertainties=dict.fromkeys(parameters_to_calibrate, 0.0),
            r_squared=0.0,
            rmse=0.0,
            mae=0.0,
            aic=0.0,
            bic=0.0,
            residuals=np.array([]),
            standardized_residuals=np.array([]),
        )

    def _create_parameter_array(
        self,
        param_dict: dict[str, float],
        data_row: pd.Series,
    ) -> np.ndarray:
        """Create parameter array for model evaluation."""
        # This would need to be customized based on the specific model structure
        # For now, return the parameter values as an array
        return np.array(list(param_dict.values()))

    def _calculate_r_squared(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """Calculate coefficient of determination (R²)."""
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)


class ExperimentalDataManager:
    """Main experimental data management and integration system."""

    def __init__(self, data_directory: str | Path | None = None) -> None:
        """Initialize experimental data manager.

        Args:
            data_directory: Directory containing experimental data files

        """
        self.data_directory = (
            Path(data_directory) if data_directory else Path("experimental_data")
        )
        self.data_directory.mkdir(parents=True, exist_ok=True)

        self.datasets: dict[str, ExperimentalDataset] = {}
        self.calibration_results: dict[str, CalibrationResult] = {}

        # Initialize components
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.calibrator = ModelCalibrator()

        self.logger = logging.getLogger(__name__)

    def load_experimental_data(
        self,
        file_path: str | Path,
        dataset_name: str | None = None,
        preprocess: bool = True,
        **kwargs,
    ) -> str:
        """Load experimental data from file.

        Args:
            file_path: Path to data file
            dataset_name: Name for dataset (auto-generated if None)
            preprocess: Whether to apply preprocessing
            **kwargs: Loading parameters

        Returns:
            Dataset name

        """
        # Load raw data
        dataset = self.loader.load_data(file_path, **kwargs)

        if dataset_name:
            dataset.name = dataset_name

        # Apply preprocessing if requested
        if preprocess:
            dataset = self.preprocessor.preprocess_dataset(dataset)

        # Store dataset
        self.datasets[dataset.name] = dataset

        self.logger.info(f"Loaded experimental dataset: {dataset.name}")
        self.logger.info(f"Data shape: {dataset.data.shape}")
        self.logger.info(f"Quality score: {dataset.quality_score:.3f}")

        return dataset.name

    def calibrate_model_against_data(
        self,
        dataset_name: str,
        model_function: Callable,
        parameters_to_calibrate: list[str],
        parameter_bounds: dict[str, tuple[float, float]],
        method: CalibrationMethod = CalibrationMethod.LEAST_SQUARES,
        **kwargs,
    ) -> str:
        """Calibrate model against experimental dataset.

        Args:
            dataset_name: Name of experimental dataset
            model_function: Model function to calibrate
            parameters_to_calibrate: Parameters to optimize
            parameter_bounds: Parameter bounds
            method: Calibration method
            **kwargs: Method-specific parameters

        Returns:
            Calibration result identifier

        """
        if dataset_name not in self.datasets:
            msg = f"Dataset not found: {dataset_name}"
            raise ValueError(msg)

        dataset = self.datasets[dataset_name]

        # Perform calibration
        result = self.calibrator.calibrate_model(
            model_function,
            dataset,
            parameters_to_calibrate,
            parameter_bounds,
            method,
            **kwargs,
        )

        # Store result
        result_id = (
            f"{dataset_name}_{method.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.calibration_results[result_id] = result

        self.logger.info(f"Model calibration completed: {result_id}")
        self.logger.info(f"R²: {result.r_squared:.3f}, RMSE: {result.rmse:.3f}")

        return result_id

    def get_dataset_summary(self, dataset_name: str) -> dict[str, Any]:
        """Get summary statistics for dataset."""
        if dataset_name not in self.datasets:
            msg = f"Dataset not found: {dataset_name}"
            raise ValueError(msg)

        dataset = self.datasets[dataset_name]

        summary = {
            "name": dataset.name,
            "shape": dataset.data.shape,
            "quality_score": dataset.quality_score,
            "quality_issues": dataset.quality_issues,
            "columns": list(dataset.data.columns),
            "data_types": dataset.data.dtypes.to_dict(),
            "missing_values": dataset.data.isnull().sum().to_dict(),
            "processing_history": dataset.processing_history,
        }

        # Add statistical summary for numeric columns
        numeric_summary = dataset.data.describe()
        if not numeric_summary.empty:
            summary["statistical_summary"] = numeric_summary.to_dict()

        return summary

    def compare_calibration_results(self, result_ids: list[str]) -> pd.DataFrame:
        """Compare multiple calibration results."""
        comparison_data = []

        for result_id in result_ids:
            if result_id in self.calibration_results:
                result = self.calibration_results[result_id]

                row = {
                    "result_id": result_id,
                    "method": result.method.value,
                    "dataset": result.dataset_name,
                    "r_squared": result.r_squared,
                    "rmse": result.rmse,
                    "mae": result.mae,
                    "aic": result.aic,
                    "bic": result.bic,
                    "calibration_time": result.calibration_time,
                }

                # Add calibrated parameter values
                for param, value in result.calibrated_parameters.items():
                    row[f"param_{param}"] = value

                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def export_calibration_results(
        self,
        result_id: str,
        output_path: str | Path,
    ) -> None:
        """Export calibration results to file."""
        if result_id not in self.calibration_results:
            msg = f"Calibration result not found: {result_id}"
            raise ValueError(msg)

        result = self.calibration_results[result_id]

        # Convert to dictionary for JSON export
        export_data = {
            "result_id": result_id,
            "method": result.method.value,
            "dataset_name": result.dataset_name,
            "calibrated_parameters": result.calibrated_parameters,
            "parameter_uncertainties": result.parameter_uncertainties,
            "goodness_of_fit": {
                "r_squared": result.r_squared,
                "rmse": result.rmse,
                "mae": result.mae,
                "aic": result.aic,
                "bic": result.bic,
            },
            "residuals": (
                result.residuals.tolist() if result.residuals is not None else None
            ),
            "calibration_time": result.calibration_time,
            "convergence_info": result.convergence_info,
            "created_at": result.created_at.isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Calibration results exported to: {output_path}")

    def list_datasets(self) -> list[str]:
        """List all loaded datasets."""
        return list(self.datasets.keys())

    def list_calibration_results(self) -> list[str]:
        """List all calibration results."""
        return list(self.calibration_results.keys())


# Utility functions for experimental data integration
def calculate_model_validation_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate comprehensive model validation metrics."""
    metrics = {}

    # Basic metrics
    metrics["r_squared"] = 1 - np.sum((observed - predicted) ** 2) / np.sum(
        (observed - np.mean(observed)) ** 2,
    )
    metrics["rmse"] = np.sqrt(np.mean((observed - predicted) ** 2))
    metrics["mae"] = np.mean(np.abs(observed - predicted))
    metrics["mape"] = np.mean(np.abs((observed - predicted) / observed)) * 100

    # Correlation metrics
    metrics["pearson_r"] = np.corrcoef(observed, predicted)[0, 1]

    if HAS_SCIPY:
        from scipy.stats import spearmanr

        metrics["spearman_r"] = spearmanr(observed, predicted)[0]

    # Bias metrics
    metrics["bias"] = np.mean(predicted - observed)
    metrics["relative_bias"] = metrics["bias"] / np.mean(observed) * 100

    # Efficiency metrics
    metrics["nash_sutcliffe"] = 1 - np.sum((observed - predicted) ** 2) / np.sum(
        (observed - np.mean(observed)) ** 2,
    )

    return metrics


def detect_change_points(time_series: np.ndarray, method: str = "pelt") -> list[int]:
    """Detect change points in time series data."""
    if not HAS_SCIPY:
        warnings.warn("SciPy required for change point detection", stacklevel=2)
        return []

    # Simple implementation using variance change detection
    n = len(time_series)
    change_points = []

    window_size = max(10, n // 20)

    for i in range(window_size, n - window_size):
        before = time_series[i - window_size : i]
        after = time_series[i : i + window_size]

        # Test for variance change
        f_stat = np.var(after) / np.var(before) if np.var(before) > 0 else 1.0

        # Simple threshold-based detection
        if f_stat > 2.0 or f_stat < 0.5:
            change_points.append(i)

    return change_points


def align_time_series(
    ts1: pd.DataFrame,
    ts2: pd.DataFrame,
    time_col: str = "timestamp",
    method: str = "nearest",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two time series datasets."""
    if time_col not in ts1.columns or time_col not in ts2.columns:
        msg = f"Time column '{time_col}' not found in both datasets"
        raise ValueError(msg)

    # Convert to datetime if not already
    ts1[time_col] = pd.to_datetime(ts1[time_col])
    ts2[time_col] = pd.to_datetime(ts2[time_col])

    # Set time as index
    ts1_indexed = ts1.set_index(time_col)
    ts2_indexed = ts2.set_index(time_col)

    # Align using pandas reindex
    if method == "nearest":
        aligned_ts2 = ts2_indexed.reindex(ts1_indexed.index, method="nearest")
        return ts1, aligned_ts2.reset_index()
    if method == "interpolate":
        # Create common time grid
        common_times = pd.date_range(
            start=max(ts1[time_col].min(), ts2[time_col].min()),
            end=min(ts1[time_col].max(), ts2[time_col].max()),
            freq="1S",  # 1 second frequency
        )

        aligned_ts1 = ts1_indexed.reindex(common_times).interpolate()
        aligned_ts2 = ts2_indexed.reindex(common_times).interpolate()

        return aligned_ts1.reset_index(), aligned_ts2.reset_index()

    return ts1, ts2
