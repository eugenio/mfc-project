"""Model Validation and Comparison Framework.

This module provides comprehensive **ML/statistical model validation** tools for MFC systems,
including cross-validation, performance metrics, statistical tests, and benchmarking.

.. note::
    This module is for validating **predictive models** (e.g., ML models, regression models).
    For validating **configuration parameters**, use:

    - ``parameter_validation`` - Q-learning and sensor config validation
    - ``biological_validation`` - Biological/electrochemical config validation

    All validation modules are accessible via the config package:
    ``from config import validate_qlearning_config, validate_sensor_config``

Classes:
- ModelValidator: Main validation framework with cross-validation methods
- PerformanceMetrics: Comprehensive performance evaluation metrics
- StatisticalTester: Statistical significance testing for model comparisons
- BenchmarkSuite: Standardized benchmarking tools
- ValidationResult: Results container for validation analyses

Features:
- K-fold and time series cross-validation
- Comprehensive performance metrics (RMSE, MAE, R², etc.)
- Statistical significance testing (t-tests, Wilcoxon, etc.)
- Model selection and comparison tools
- Residual analysis and diagnostic plots
- Bootstrap confidence intervals
- Information criteria (AIC, BIC) calculations
- Model ensemble validation

Literature References:
1. Hastie, T., et al. (2009). "The Elements of Statistical Learning"
2. Bergmeir, C., & Benítez, J. M. (2012). "On the use of cross-validation for time series predictor evaluation"
3. Diebold, F. X. (2015). "Comparing predictive accuracy, twenty years later"
4. Hyndman, R. J., & Athanasopoulos, G. (2018). "Forecasting: principles and practice"
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Statistical dependencies
try:
    from scipy import stats
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn(
        "SciPy not available. Some statistical tests will be limited.",
        stacklevel=2,
    )

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn(
        "Scikit-learn not available. Some validation features will be limited.",
        stacklevel=2,
    )

# Plotting dependencies
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

# Import configuration classes


class ValidationMethod(Enum):
    """Available validation methods."""

    K_FOLD = "k_fold"
    TIME_SERIES_SPLIT = "time_series_split"
    LEAVE_ONE_OUT = "leave_one_out"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"
    HOLDOUT = "holdout"


class MetricType(Enum):
    """Types of performance metrics."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    CUSTOM = "custom"


class StatisticalTest(Enum):
    """Statistical tests for model comparison."""

    T_TEST_PAIRED = "t_test_paired"
    T_TEST_INDEPENDENT = "t_test_independent"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    MANN_WHITNEY_U = "mann_whitney_u"
    DIEBOLD_MARIANO = "diebold_mariano"
    FRIEDMAN = "friedman"
    NEMENYI = "nemenyi"


@dataclass
class PerformanceMetric:
    """Definition of a performance metric."""

    name: str
    metric_type: MetricType
    higher_is_better: bool
    metric_function: Callable[[np.ndarray, np.ndarray], float]
    description: str = ""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluate the metric."""
        try:
            return self.metric_function(y_true, y_pred)
        except Exception as e:
            logging.warning(f"Error evaluating metric {self.name}: {e}")
            return np.nan


@dataclass
class ValidationResult:
    """Results container for model validation."""

    # Validation metadata
    validation_method: ValidationMethod
    model_name: str
    dataset_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # Cross-validation results
    cv_scores: dict[str, np.ndarray] = field(default_factory=dict)
    cv_mean_scores: dict[str, float] = field(default_factory=dict)
    cv_std_scores: dict[str, float] = field(default_factory=dict)

    # Overall performance
    test_scores: dict[str, float] = field(default_factory=dict)
    train_scores: dict[str, float] = field(default_factory=dict)

    # Predictions and residuals
    predictions: np.ndarray | None = None
    residuals: np.ndarray | None = None
    prediction_intervals: tuple[np.ndarray, np.ndarray] | None = None

    # Model diagnostics
    model_complexity: dict[str, Any] | None = None
    information_criteria: dict[str, float] = field(default_factory=dict)

    # Statistical tests
    normality_tests: dict[str, dict[str, float]] = field(default_factory=dict)
    heteroscedasticity_tests: dict[str, dict[str, float]] = field(default_factory=dict)
    autocorrelation_tests: dict[str, dict[str, float]] = field(default_factory=dict)

    # Bootstrap results
    bootstrap_scores: dict[str, np.ndarray] = field(default_factory=dict)
    bootstrap_confidence_intervals: dict[str, tuple[float, float]] = field(
        default_factory=dict,
    )

    # Computation metadata
    computation_time: float = 0.0
    n_folds: int = 0
    n_samples: int = 0
    n_features: int = 0

    def set_end_time(self) -> None:
        """Set the end time of validation."""
        self.end_time = datetime.now()

    def get_validation_time(self) -> float:
        """Get total validation time in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class ComparisonResult:
    """Results container for model comparison."""

    # Models being compared
    model_names: list[str]
    comparison_method: str

    # Statistical test results
    test_statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None

    # Pairwise comparisons
    pairwise_results: dict[tuple[str, str], dict[str, float]] = field(
        default_factory=dict,
    )

    # Rankings
    model_rankings: dict[str, float] = field(default_factory=dict)

    # Critical difference (for Nemenyi test)
    critical_difference: float | None = None

    # Confidence level
    alpha: float = 0.05

    # Summary
    best_model: str | None = None
    significant_differences: list[tuple[str, str]] = field(default_factory=list)


class ModelValidator:
    """Main framework for model validation and cross-validation."""

    def __init__(
        self,
        metrics: list[PerformanceMetric] | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Initialize model validator.

        Args:
            metrics: List of performance metrics to evaluate
            random_seed: Random seed for reproducibility

        """
        self.metrics = metrics or self._get_default_metrics()
        self.random_seed = random_seed
        self.logger = logging.getLogger(__name__)

        if random_seed is not None:
            np.random.seed(random_seed)

    def validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        validation_method: ValidationMethod = ValidationMethod.K_FOLD,
        n_folds: int = 5,
        test_size: float = 0.2,
        model_name: str = "model",
        dataset_name: str = "dataset",
        **kwargs,
    ) -> ValidationResult:
        """Perform comprehensive model validation.

        Args:
            model: Model to validate (must have fit/predict methods)
            X: Feature matrix
            y: Target values
            validation_method: Validation method to use
            n_folds: Number of folds for cross-validation
            test_size: Proportion of data for test set
            model_name: Name of the model
            dataset_name: Name of the dataset
            **kwargs: Additional validation parameters

        Returns:
            Validation results

        """
        import time

        start_time = time.time()

        result = ValidationResult(
            validation_method=validation_method,
            model_name=model_name,
            dataset_name=dataset_name,
            n_folds=n_folds,
            n_samples=len(X),
            n_features=X.shape[1] if len(X.shape) > 1 else 1,
        )

        # Perform cross-validation
        if validation_method == ValidationMethod.K_FOLD:
            cv_results = self._k_fold_validation(model, X, y, n_folds, **kwargs)
        elif validation_method == ValidationMethod.TIME_SERIES_SPLIT:
            cv_results = self._time_series_validation(model, X, y, n_folds, **kwargs)
        elif validation_method == ValidationMethod.BOOTSTRAP:
            cv_results = self._bootstrap_validation(model, X, y, n_folds, **kwargs)
        elif validation_method == ValidationMethod.HOLDOUT:
            cv_results = self._holdout_validation(model, X, y, test_size, **kwargs)
        else:
            msg = f"Unsupported validation method: {validation_method}"
            raise ValueError(msg)

        result.cv_scores = cv_results["cv_scores"]
        result.cv_mean_scores = cv_results["cv_mean_scores"]
        result.cv_std_scores = cv_results["cv_std_scores"]
        result.predictions = cv_results.get("predictions")
        result.residuals = cv_results.get("residuals")

        # Calculate information criteria if possible
        if hasattr(model, "score") and result.predictions is not None:
            result.information_criteria = self._calculate_information_criteria(
                y,
                result.predictions,
                result.n_features,
            )

        # Perform diagnostic tests
        if result.residuals is not None:
            result.normality_tests = self._test_normality(result.residuals)
            result.heteroscedasticity_tests = self._test_heteroscedasticity(
                result.predictions,
                result.residuals,
            )
            result.autocorrelation_tests = self._test_autocorrelation(result.residuals)

        # Bootstrap confidence intervals
        if validation_method != ValidationMethod.BOOTSTRAP:
            result.bootstrap_scores, result.bootstrap_confidence_intervals = (
                self._calculate_bootstrap_intervals(model, X, y, n_bootstrap=1000)
            )

        result.computation_time = time.time() - start_time
        result.set_end_time()

        return result

    def _k_fold_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform k-fold cross-validation."""
        if not HAS_SKLEARN:
            msg = "Scikit-learn required for k-fold validation"
            raise ImportError(msg)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)

        cv_scores = {metric.name: [] for metric in self.metrics}
        all_predictions = []
        all_true = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Store predictions
            all_predictions.extend(y_pred)
            all_true.extend(y_val)

            # Calculate metrics
            for metric in self.metrics:
                score = metric.evaluate(y_val, y_pred)
                cv_scores[metric.name].append(score)

        # Calculate statistics
        cv_scores = {name: np.array(scores) for name, scores in cv_scores.items()}
        cv_mean_scores = {
            name: np.nanmean(scores) for name, scores in cv_scores.items()
        }
        cv_std_scores = {name: np.nanstd(scores) for name, scores in cv_scores.items()}

        predictions = np.array(all_predictions)
        residuals = np.array(all_true) - predictions

        return {
            "cv_scores": cv_scores,
            "cv_mean_scores": cv_mean_scores,
            "cv_std_scores": cv_std_scores,
            "predictions": predictions,
            "residuals": residuals,
        }

    def _time_series_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform time series cross-validation."""
        if not HAS_SKLEARN:
            msg = "Scikit-learn required for time series validation"
            raise ImportError(msg)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_scores = {metric.name: [] for metric in self.metrics}
        all_predictions = []
        all_true = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Store predictions
            all_predictions.extend(y_pred)
            all_true.extend(y_val)

            # Calculate metrics
            for metric in self.metrics:
                score = metric.evaluate(y_val, y_pred)
                cv_scores[metric.name].append(score)

        # Calculate statistics
        cv_scores = {name: np.array(scores) for name, scores in cv_scores.items()}
        cv_mean_scores = {
            name: np.nanmean(scores) for name, scores in cv_scores.items()
        }
        cv_std_scores = {name: np.nanstd(scores) for name, scores in cv_scores.items()}

        predictions = np.array(all_predictions)
        residuals = np.array(all_true) - predictions

        return {
            "cv_scores": cv_scores,
            "cv_mean_scores": cv_mean_scores,
            "cv_std_scores": cv_std_scores,
            "predictions": predictions,
            "residuals": residuals,
        }

    def _bootstrap_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform bootstrap validation."""
        cv_scores = {metric.name: [] for metric in self.metrics}

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Out-of-bag samples
            oob_indices = np.setdiff1d(np.arange(len(X)), indices)
            if len(oob_indices) == 0:
                continue

            X_oob = X[oob_indices]
            y_oob = y[oob_indices]

            # Fit and predict
            model.fit(X_boot, y_boot)
            y_pred = model.predict(X_oob)

            # Calculate metrics
            for metric in self.metrics:
                score = metric.evaluate(y_oob, y_pred)
                cv_scores[metric.name].append(score)

        # Calculate statistics
        cv_scores = {name: np.array(scores) for name, scores in cv_scores.items()}
        cv_mean_scores = {
            name: np.nanmean(scores) for name, scores in cv_scores.items()
        }
        cv_std_scores = {name: np.nanstd(scores) for name, scores in cv_scores.items()}

        return {
            "cv_scores": cv_scores,
            "cv_mean_scores": cv_mean_scores,
            "cv_std_scores": cv_std_scores,
        }

    def _holdout_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform holdout validation."""
        # Split data
        n_test = int(len(X) * test_size)
        indices = np.random.permutation(len(X))

        train_indices = indices[n_test:]
        test_indices = indices[:n_test]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        cv_scores = {}
        cv_mean_scores = {}
        cv_std_scores = {}

        for metric in self.metrics:
            score = metric.evaluate(y_test, y_pred)
            cv_scores[metric.name] = np.array([score])
            cv_mean_scores[metric.name] = score
            cv_std_scores[metric.name] = 0.0

        residuals = y_test - y_pred

        return {
            "cv_scores": cv_scores,
            "cv_mean_scores": cv_mean_scores,
            "cv_std_scores": cv_std_scores,
            "predictions": y_pred,
            "residuals": residuals,
        }

    def _calculate_information_criteria(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: int,
    ) -> dict[str, float]:
        """Calculate information criteria (AIC, BIC)."""
        n = len(y_true)
        mse = np.mean((y_true - y_pred) ** 2)
        log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - n / 2

        aic = 2 * n_features - 2 * log_likelihood
        bic = np.log(n) * n_features - 2 * log_likelihood

        return {"AIC": aic, "BIC": bic, "log_likelihood": log_likelihood}

    def _test_normality(self, residuals: np.ndarray) -> dict[str, dict[str, float]]:
        """Test normality of residuals."""
        tests = {}

        if HAS_SCIPY:
            # Shapiro-Wilk test
            if len(residuals) <= 5000:  # Shapiro-Wilk has sample size limit
                stat, p_value = stats.shapiro(residuals)
                tests["shapiro_wilk"] = {"statistic": stat, "p_value": p_value}

            # Jarque-Bera test
            stat, p_value = stats.jarque_bera(residuals)
            tests["jarque_bera"] = {"statistic": stat, "p_value": p_value}

            # Anderson-Darling test
            result = stats.anderson(residuals, dist="norm")
            tests["anderson_darling"] = {
                "statistic": result.statistic,
                "critical_values": result.critical_values.tolist(),
                "significance_levels": result.significance_levels.tolist(),
            }

        return tests

    def _test_heteroscedasticity(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Test for heteroscedasticity in residuals."""
        tests = {}

        if HAS_SCIPY:
            # Breusch-Pagan test (simplified)
            try:
                # Regress squared residuals on predictions
                X = np.column_stack([np.ones(len(predictions)), predictions])
                y = residuals**2

                # Simple linear regression
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ beta

                # Calculate test statistic
                tss = np.sum((y - np.mean(y)) ** 2)
                rss = np.sum((y - y_pred) ** 2)

                r_squared = 1 - rss / tss if tss > 0 else 0
                lm_statistic = len(y) * r_squared
                p_value = 1 - stats.chi2.cdf(lm_statistic, df=1)

                tests["breusch_pagan"] = {"statistic": lm_statistic, "p_value": p_value}
            except Exception as e:
                self.logger.warning(f"Breusch-Pagan test failed: {e}")

        return tests

    def _test_autocorrelation(
        self,
        residuals: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Test for autocorrelation in residuals."""
        tests = {}

        if HAS_SCIPY and len(residuals) > 1:
            # Durbin-Watson test (simplified)
            diff_residuals = np.diff(residuals)
            dw_statistic = np.sum(diff_residuals**2) / np.sum(residuals**2)
            tests["durbin_watson"] = {"statistic": dw_statistic}

            # Ljung-Box test (simplified)
            try:
                n = len(residuals)
                if n > 10:
                    lags = min(10, n // 4)
                    autocorrs = []

                    for lag in range(1, lags + 1):
                        if n - lag > 0:
                            autocorr = np.corrcoef(residuals[:-lag], residuals[lag:])[
                                0,
                                1,
                            ]
                            autocorrs.append(autocorr if not np.isnan(autocorr) else 0)

                    if autocorrs:
                        lb_statistic = (
                            n
                            * (n + 2)
                            * np.sum(
                                [
                                    (autocorr**2) / (n - lag - 1)
                                    for lag, autocorr in enumerate(autocorrs)
                                ],
                            )
                        )
                        p_value = 1 - stats.chi2.cdf(lb_statistic, df=len(autocorrs))

                        tests["ljung_box"] = {
                            "statistic": lb_statistic,
                            "p_value": p_value,
                            "lags": len(autocorrs),
                        }
            except Exception as e:
                self.logger.warning(f"Ljung-Box test failed: {e}")

        return tests

    def _calculate_bootstrap_intervals(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
        """Calculate bootstrap confidence intervals."""
        bootstrap_scores = {metric.name: [] for metric in self.metrics}

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Fit and predict on original data
            model.fit(X_boot, y_boot)
            y_pred = model.predict(X)

            # Calculate metrics
            for metric in self.metrics:
                score = metric.evaluate(y, y_pred)
                bootstrap_scores[metric.name].append(score)

        # Convert to arrays
        bootstrap_scores = {
            name: np.array(scores) for name, scores in bootstrap_scores.items()
        }

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        confidence_intervals = {}

        for name, scores in bootstrap_scores.items():
            valid_scores = scores[~np.isnan(scores)]
            if len(valid_scores) > 0:
                lower = np.percentile(valid_scores, 100 * alpha / 2)
                upper = np.percentile(valid_scores, 100 * (1 - alpha / 2))
                confidence_intervals[name] = (lower, upper)
            else:
                confidence_intervals[name] = (np.nan, np.nan)

        return bootstrap_scores, confidence_intervals

    def _get_default_metrics(self) -> list[PerformanceMetric]:
        """Get default performance metrics."""
        metrics = []

        # Root Mean Square Error
        metrics.append(
            PerformanceMetric(
                name="RMSE",
                metric_type=MetricType.REGRESSION,
                higher_is_better=False,
                metric_function=lambda y_true, y_pred: np.sqrt(
                    np.mean((y_true - y_pred) ** 2),
                ),
                description="Root Mean Square Error",
            ),
        )

        # Mean Absolute Error
        metrics.append(
            PerformanceMetric(
                name="MAE",
                metric_type=MetricType.REGRESSION,
                higher_is_better=False,
                metric_function=lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
                description="Mean Absolute Error",
            ),
        )

        # R-squared
        metrics.append(
            PerformanceMetric(
                name="R2",
                metric_type=MetricType.REGRESSION,
                higher_is_better=True,
                metric_function=lambda y_true, y_pred: 1
                - np.sum((y_true - y_pred) ** 2)
                / np.sum((y_true - np.mean(y_true)) ** 2),
                description="Coefficient of Determination",
            ),
        )

        # Mean Absolute Percentage Error
        metrics.append(
            PerformanceMetric(
                name="MAPE",
                metric_type=MetricType.REGRESSION,
                higher_is_better=False,
                metric_function=lambda y_true, y_pred: np.mean(
                    np.abs((y_true - y_pred) / (y_true + 1e-8)),
                )
                * 100,
                description="Mean Absolute Percentage Error",
            ),
        )

        return metrics


class StatisticalTester:
    """Statistical significance testing for model comparisons."""

    def __init__(self, alpha: float = 0.05) -> None:
        """Initialize statistical tester.

        Args:
            alpha: Significance level

        """
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

    def compare_models(
        self,
        validation_results: list[ValidationResult],
        metric_name: str = "RMSE",
        test_type: StatisticalTest = StatisticalTest.T_TEST_PAIRED,
    ) -> ComparisonResult:
        """Compare multiple models using statistical tests.

        Args:
            validation_results: List of validation results to compare
            metric_name: Name of metric to compare
            test_type: Statistical test to perform

        Returns:
            Comparison results

        """
        if len(validation_results) < 2:
            msg = "At least two models required for comparison"
            raise ValueError(msg)

        model_names = [result.model_name for result in validation_results]

        # Extract scores for the specified metric
        scores = []
        for result in validation_results:
            if metric_name in result.cv_scores:
                scores.append(result.cv_scores[metric_name])
            else:
                msg = f"Metric {metric_name} not found in results"
                raise ValueError(msg)

        result = ComparisonResult(
            model_names=model_names,
            comparison_method=test_type.value,
            alpha=self.alpha,
        )

        if test_type == StatisticalTest.T_TEST_PAIRED:
            result = self._paired_t_test(scores, model_names, result)
        elif test_type == StatisticalTest.WILCOXON_SIGNED_RANK:
            result = self._wilcoxon_test(scores, model_names, result)
        elif test_type == StatisticalTest.FRIEDMAN:
            result = self._friedman_test(scores, model_names, result)
        else:
            msg = f"Unsupported test type: {test_type}"
            raise ValueError(msg)

        return result

    def _paired_t_test(
        self,
        scores: list[np.ndarray],
        model_names: list[str],
        result: ComparisonResult,
    ) -> ComparisonResult:
        """Perform paired t-test between models."""
        if not HAS_SCIPY:
            msg = "SciPy required for t-tests"
            raise ImportError(msg)

        # Pairwise comparisons
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                score1, score2 = scores[i], scores[j]
                name1, name2 = model_names[i], model_names[j]

                # Ensure same length
                min_len = min(len(score1), len(score2))
                score1 = score1[:min_len]
                score2 = score2[:min_len]

                # Perform t-test
                statistic, p_value = stats.ttest_rel(score1, score2)

                # Calculate effect size (Cohen's d)
                diff = score1 - score2
                pooled_std = np.sqrt((np.var(score1) + np.var(score2)) / 2)
                effect_size = np.mean(diff) / (pooled_std + 1e-8)

                result.pairwise_results[(name1, name2)] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "significant": p_value < self.alpha,
                }

                if p_value < self.alpha:
                    result.significant_differences.append((name1, name2))

        # Overall ranking (based on mean scores)
        mean_scores = [np.mean(score) for score in scores]
        sorted_indices = np.argsort(mean_scores)

        for rank, idx in enumerate(sorted_indices):
            result.model_rankings[model_names[idx]] = rank + 1

        result.best_model = model_names[sorted_indices[0]]

        return result

    def _wilcoxon_test(
        self,
        scores: list[np.ndarray],
        model_names: list[str],
        result: ComparisonResult,
    ) -> ComparisonResult:
        """Perform Wilcoxon signed-rank test."""
        if not HAS_SCIPY:
            msg = "SciPy required for Wilcoxon test"
            raise ImportError(msg)

        # Pairwise comparisons
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                score1, score2 = scores[i], scores[j]
                name1, name2 = model_names[i], model_names[j]

                # Ensure same length
                min_len = min(len(score1), len(score2))
                score1 = score1[:min_len]
                score2 = score2[:min_len]

                # Perform Wilcoxon test
                try:
                    statistic, p_value = stats.wilcoxon(score1, score2)

                    result.pairwise_results[(name1, name2)] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < self.alpha,
                    }

                    if p_value < self.alpha:
                        result.significant_differences.append((name1, name2))

                except Exception as e:
                    self.logger.warning(
                        f"Wilcoxon test failed for {name1} vs {name2}: {e}",
                    )

        return result

    def _friedman_test(
        self,
        scores: list[np.ndarray],
        model_names: list[str],
        result: ComparisonResult,
    ) -> ComparisonResult:
        """Perform Friedman test for multiple model comparison."""
        if not HAS_SCIPY:
            msg = "SciPy required for Friedman test"
            raise ImportError(msg)

        # Ensure all score arrays have the same length
        min_len = min(len(score) for score in scores)
        aligned_scores = [score[:min_len] for score in scores]

        # Perform Friedman test
        statistic, p_value = stats.friedmanchisquare(*aligned_scores)

        result.test_statistic = statistic
        result.p_value = p_value

        # Calculate rankings
        score_matrix = np.array(aligned_scores).T  # n_samples x n_models
        rankings = np.argsort(np.argsort(score_matrix, axis=1), axis=1) + 1
        mean_rankings = np.mean(rankings, axis=0)

        for i, name in enumerate(model_names):
            result.model_rankings[name] = mean_rankings[i]

        # Best model (lowest mean rank)
        best_idx = np.argmin(mean_rankings)
        result.best_model = model_names[best_idx]

        return result


# Utility functions for model validation
def calculate_residual_diagnostics(residuals: np.ndarray) -> dict[str, float]:
    """Calculate comprehensive residual diagnostics.

    Args:
        residuals: Model residuals

    Returns:
        Dictionary of diagnostic statistics

    """
    diagnostics = {}

    # Basic statistics
    diagnostics["mean"] = np.mean(residuals)
    diagnostics["std"] = np.std(residuals)
    diagnostics["skewness"] = stats.skew(residuals) if HAS_SCIPY else np.nan
    diagnostics["kurtosis"] = stats.kurtosis(residuals) if HAS_SCIPY else np.nan

    # Quantiles
    diagnostics["q25"] = np.percentile(residuals, 25)
    diagnostics["median"] = np.median(residuals)
    diagnostics["q75"] = np.percentile(residuals, 75)

    # Outlier detection
    iqr = diagnostics["q75"] - diagnostics["q25"]
    lower_bound = diagnostics["q25"] - 1.5 * iqr
    upper_bound = diagnostics["q75"] + 1.5 * iqr
    diagnostics["n_outliers"] = np.sum(
        (residuals < lower_bound) | (residuals > upper_bound),
    )
    diagnostics["outlier_fraction"] = diagnostics["n_outliers"] / len(residuals)

    return diagnostics


def create_validation_report(
    validation_results: list[ValidationResult],
    comparison_result: ComparisonResult | None = None,
) -> str:
    """Create comprehensive validation report.

    Args:
        validation_results: List of validation results
        comparison_result: Optional comparison results

    Returns:
        Formatted validation report

    """
    report = []
    report.append("=" * 80)
    report.append("MODEL VALIDATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Individual model results
    for i, result in enumerate(validation_results):
        report.append(f"Model {i + 1}: {result.model_name}")
        report.append("-" * 40)
        report.append(f"Dataset: {result.dataset_name}")
        report.append(f"Validation Method: {result.validation_method.value}")
        report.append(f"Samples: {result.n_samples}, Features: {result.n_features}")
        report.append(f"Validation Time: {result.get_validation_time():.2f}s")
        report.append("")

        # Cross-validation scores
        report.append("Cross-Validation Results:")
        for metric_name, mean_score in result.cv_mean_scores.items():
            std_score = result.cv_std_scores.get(metric_name, 0)
            report.append(f"  {metric_name}: {mean_score:.4f} ± {std_score:.4f}")
        report.append("")

        # Information criteria
        if result.information_criteria:
            report.append("Information Criteria:")
            for criterion, value in result.information_criteria.items():
                report.append(f"  {criterion}: {value:.2f}")
            report.append("")

        # Bootstrap confidence intervals
        if result.bootstrap_confidence_intervals:
            report.append("Bootstrap Confidence Intervals (95%):")
            for metric_name, (
                lower,
                upper,
            ) in result.bootstrap_confidence_intervals.items():
                report.append(f"  {metric_name}: [{lower:.4f}, {upper:.4f}]")
            report.append("")

        report.append("")

    # Model comparison
    if comparison_result:
        report.append("MODEL COMPARISON")
        report.append("-" * 40)
        report.append(f"Comparison Method: {comparison_result.comparison_method}")
        report.append(f"Significance Level: {comparison_result.alpha}")
        report.append("")

        # Rankings
        report.append("Model Rankings:")
        sorted_rankings = sorted(
            comparison_result.model_rankings.items(),
            key=lambda x: x[1],
        )
        for name, rank in sorted_rankings:
            report.append(f"  {rank}. {name}")
        report.append("")

        # Best model
        if comparison_result.best_model:
            report.append(f"Best Model: {comparison_result.best_model}")
            report.append("")

        # Significant differences
        if comparison_result.significant_differences:
            report.append("Significant Differences:")
            for model1, model2 in comparison_result.significant_differences:
                report.append(f"  {model1} vs {model2}")
            report.append("")

    return "\n".join(report)
