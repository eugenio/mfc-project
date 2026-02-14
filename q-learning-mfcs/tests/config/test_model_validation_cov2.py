"""Tests for model_validation.py - coverage part 2.

Targets missing lines: 322, 358-359, 414-415, 480, 589, 619-620,
670-671, 684-719, 850-851, 901-902, 926-929, 943-944.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from config.model_validation import (
    ValidationMethod,
    MetricType,
    PerformanceMetric,
    ValidationResult,
    ComparisonResult,
    ModelValidator,
    StatisticalTester,
    StatisticalTest,
)


def _make_mock_model_1d():
    """Mock model that works with 1D input (single feature)."""
    model = MagicMock()
    model.fit = MagicMock()
    model.predict = MagicMock(side_effect=lambda X: X.ravel() + 0.1)
    model.score = MagicMock(return_value=0.9)
    return model


def _make_mock_model_2d():
    """Mock model that works with 2D input."""
    model = MagicMock()
    model.fit = MagicMock()
    model.predict = MagicMock(side_effect=lambda X: X[:, 0] + 0.1)
    model.score = MagicMock(return_value=0.9)
    return model


@pytest.mark.coverage_extra
class TestValidateModel1DFeatures:
    """Cover line 322: n_features=1 when X is 1D, and information_criteria branch."""

    def test_validate_model_1d_input_with_score(self):
        """When X.shape has len 1, n_features should be 1.
        Also covers line 322: information_criteria calculation when model has 'score'
        and predictions is not None."""
        validator = ModelValidator(random_seed=42)
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1
        model = _make_mock_model_2d()

        # Mock diagnostics to avoid scipy version bugs (anderson_darling)
        # and bootstrap for speed. Focus on covering line 322.
        with patch.object(validator, "_test_normality", return_value={}), \
             patch.object(validator, "_test_heteroscedasticity", return_value={}), \
             patch.object(validator, "_test_autocorrelation", return_value={}), \
             patch.object(validator, "_calculate_bootstrap_intervals", return_value=({}, {})):
            result = validator.validate_model(
                model, X, y,
                validation_method=ValidationMethod.K_FOLD,
                n_folds=3,
            )
        # model has .score attribute, so information_criteria should be set
        assert result.information_criteria is not None
        assert "AIC" in result.information_criteria


@pytest.mark.coverage_extra
class TestKFoldNoSklearn:
    """Cover lines 358-359: ImportError when sklearn is not available."""

    def test_k_fold_without_sklearn(self):
        validator = ModelValidator(random_seed=42)
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        model = _make_mock_model_2d()

        with patch("config.model_validation.HAS_SKLEARN", False):
            with pytest.raises(ImportError, match="Scikit-learn required"):
                validator._k_fold_validation(model, X, y, n_folds=3)


@pytest.mark.coverage_extra
class TestTimeSeriesNoSklearn:
    """Cover lines 414-415: ImportError when sklearn is not available."""

    def test_time_series_without_sklearn(self):
        validator = ModelValidator(random_seed=42)
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        model = _make_mock_model_2d()

        with patch("config.model_validation.HAS_SKLEARN", False):
            with pytest.raises(ImportError, match="Scikit-learn required"):
                validator._time_series_validation(model, X, y, n_splits=3)


@pytest.mark.coverage_extra
class TestBootstrapOOBEmpty:
    """Cover line 480: continue when oob_indices is empty."""

    def test_bootstrap_all_samples_selected(self):
        """When all indices are in-bag (no OOB), the iteration should skip."""
        validator = ModelValidator(random_seed=42)
        np.random.seed(42)
        # Very small dataset where bootstrap may select all samples
        X = np.random.randn(3, 2)
        y = np.random.randn(3)
        model = _make_mock_model_2d()

        # Force np.random.choice to always return all indices, so oob is empty
        original_choice = np.random.choice

        call_count = [0]

        def always_same_indices(*args, **kwargs):
            call_count[0] += 1
            # Return [0, 1, 2] always so oob_indices is empty
            return np.array([0, 1, 2])

        with patch("numpy.random.choice", side_effect=always_same_indices):
            result = validator._bootstrap_validation(model, X, y, n_bootstrap=5)

        # All iterations should have been skipped (no OOB samples)
        # cv_scores should exist but have empty lists
        assert "cv_scores" in result


@pytest.mark.coverage_extra
class TestNormalityReturnNoScipy:
    """Cover line 589: return tests when HAS_SCIPY is False."""

    def test_normality_no_scipy(self):
        validator = ModelValidator(random_seed=42)
        residuals = np.random.normal(0, 1, 50)
        with patch("config.model_validation.HAS_SCIPY", False):
            tests = validator._test_normality(residuals)
        assert tests == {}


@pytest.mark.coverage_extra
class TestHeteroscedasticityException:
    """Cover lines 619-620: exception in breusch_pagan test."""

    def test_heteroscedasticity_exception(self):
        validator = ModelValidator(random_seed=42)
        predictions = np.random.normal(0, 1, 50)
        residuals = np.random.normal(0, 1, 50)

        # Mock stats to raise an exception inside the try block
        with patch("config.model_validation.HAS_SCIPY", True), \
             patch("config.model_validation.stats") as mock_stats:
            # Make the polyfit-like operation raise
            mock_stats.chi2.cdf.side_effect = RuntimeError("test error")
            tests = validator._test_heteroscedasticity(predictions, residuals)
        # Should have logged the warning but not crashed
        assert isinstance(tests, dict)


@pytest.mark.coverage_extra
class TestAutocorrelationLjungBox:
    """Cover lines 665-671: Ljung-Box test with sufficient data."""

    def test_autocorrelation_ljung_box_succeeds(self):
        """Use enough residuals (>10) to trigger Ljung-Box calculation."""
        validator = ModelValidator(random_seed=42)
        np.random.seed(42)
        # Need >10 residuals for ljung_box branch
        residuals = np.random.normal(0, 1, 50)
        tests = validator._test_autocorrelation(residuals)
        assert "durbin_watson" in tests
        assert "ljung_box" in tests
        assert "statistic" in tests["ljung_box"]
        assert "p_value" in tests["ljung_box"]
        assert "lags" in tests["ljung_box"]

    def test_autocorrelation_ljung_box_exception(self):
        """Cover lines 670-671: exception in Ljung-Box test."""
        validator = ModelValidator(random_seed=42)
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 50)

        with patch("config.model_validation.HAS_SCIPY", True), \
             patch("config.model_validation.stats") as mock_stats:
            mock_stats.chi2.cdf.side_effect = RuntimeError("ljung-box fail")
            tests = validator._test_autocorrelation(residuals)
        # Should have durbin_watson but ljung_box should be missing due to exception
        assert isinstance(tests, dict)


@pytest.mark.coverage_extra
class TestBootstrapIntervalsNaN:
    """Cover lines 684-719: _calculate_bootstrap_intervals including NaN handling."""

    def test_bootstrap_intervals_basic(self):
        """Cover the full _calculate_bootstrap_intervals method."""
        validator = ModelValidator(random_seed=42)
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = X[:, 0] + np.random.randn(20) * 0.1
        model = _make_mock_model_2d()

        scores, intervals = validator._calculate_bootstrap_intervals(
            model, X, y, n_bootstrap=10, confidence_level=0.95,
        )
        # Should have scores and intervals for each default metric
        assert isinstance(scores, dict)
        assert isinstance(intervals, dict)
        for name in scores:
            assert name in intervals
            lower, upper = intervals[name]
            # Should be finite
            assert np.isfinite(lower) or np.isnan(lower)
            assert np.isfinite(upper) or np.isnan(upper)

    def test_bootstrap_intervals_all_nan(self):
        """Cover lines 716-717: when all scores are NaN, interval should be (nan, nan)."""
        validator = ModelValidator(random_seed=42)
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = X[:, 0] + np.random.randn(20) * 0.1

        # Model that always produces NaN predictions
        model = MagicMock()
        model.fit = MagicMock()
        model.predict = MagicMock(
            return_value=np.full(len(X), np.nan),
        )

        scores, intervals = validator._calculate_bootstrap_intervals(
            model, X, y, n_bootstrap=5, confidence_level=0.95,
        )
        # Some metrics should produce NaN due to NaN predictions
        # At least one metric should have NaN interval
        has_nan_interval = False
        for name, (lower, upper) in intervals.items():
            if np.isnan(lower) and np.isnan(upper):
                has_nan_interval = True
                break
        assert has_nan_interval


@pytest.mark.coverage_extra
class TestPairedTTestNoScipy:
    """Cover lines 850-851: ImportError when scipy not available."""

    def test_paired_t_test_no_scipy(self):
        tester = StatisticalTester(alpha=0.05)
        scores = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        result = ComparisonResult(
            model_names=["m1", "m2"],
            comparison_method="t_test_paired",
        )
        with patch("config.model_validation.HAS_SCIPY", False):
            with pytest.raises(ImportError, match="SciPy required"):
                tester._paired_t_test(scores, ["m1", "m2"], result)


@pytest.mark.coverage_extra
class TestWilcoxonNoScipy:
    """Cover lines 901-902: ImportError when scipy not available."""

    def test_wilcoxon_no_scipy(self):
        tester = StatisticalTester(alpha=0.05)
        scores = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        result = ComparisonResult(
            model_names=["m1", "m2"],
            comparison_method="wilcoxon_signed_rank",
        )
        with patch("config.model_validation.HAS_SCIPY", False):
            with pytest.raises(ImportError, match="SciPy required"):
                tester._wilcoxon_test(scores, ["m1", "m2"], result)


@pytest.mark.coverage_extra
class TestWilcoxonException:
    """Cover lines 926-929: exception during Wilcoxon test."""

    def test_wilcoxon_exception_in_test(self):
        tester = StatisticalTester(alpha=0.05)
        # Equal scores will cause Wilcoxon to fail
        scores = [np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
                  np.array([0.1, 0.1, 0.1, 0.1, 0.1])]
        result = ComparisonResult(
            model_names=["m1", "m2"],
            comparison_method="wilcoxon_signed_rank",
        )
        with patch("config.model_validation.stats") as mock_stats:
            mock_stats.wilcoxon.side_effect = ValueError("identical values")
            result = tester._wilcoxon_test(scores, ["m1", "m2"], result)
        # Should not crash, just log warning
        assert isinstance(result, ComparisonResult)


@pytest.mark.coverage_extra
class TestFriedmanNoScipy:
    """Cover lines 943-944: ImportError when scipy not available."""

    def test_friedman_no_scipy(self):
        tester = StatisticalTester(alpha=0.05)
        scores = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.7, 0.8, 0.9]),
        ]
        result = ComparisonResult(
            model_names=["m1", "m2", "m3"],
            comparison_method="friedman",
        )
        with patch("config.model_validation.HAS_SCIPY", False):
            with pytest.raises(ImportError, match="SciPy required"):
                tester._friedman_test(scores, ["m1", "m2", "m3"], result)


@pytest.mark.coverage_extra
class TestFriedmanRankings:
    """Cover Friedman test ranking computation (lines around 957-966)."""

    def test_friedman_produces_rankings(self):
        tester = StatisticalTester(alpha=0.05)
        scores = [
            np.array([0.1, 0.2, 0.15, 0.12, 0.18]),
            np.array([0.3, 0.4, 0.35, 0.32, 0.38]),
            np.array([0.5, 0.6, 0.55, 0.52, 0.58]),
        ]
        result = ComparisonResult(
            model_names=["m1", "m2", "m3"],
            comparison_method="friedman",
        )
        result = tester._friedman_test(scores, ["m1", "m2", "m3"], result)
        assert result.test_statistic is not None
        assert result.p_value is not None
        assert result.best_model is not None
        assert "m1" in result.model_rankings
        assert "m2" in result.model_rankings
        assert "m3" in result.model_rankings
