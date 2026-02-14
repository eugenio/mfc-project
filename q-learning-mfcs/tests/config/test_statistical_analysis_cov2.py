import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np

from config.statistical_analysis import (
    StatisticalAnalyzer,
    DistributionAnalyzer,
    TimeSeriesAnalyzer,
)


@pytest.mark.coverage_extra
class TestDistributionAnalyzer:
    @pytest.fixture
    def da(self):
        from unittest.mock import patch
        from scipy import stats as sp_stats
        fixed_distributions = [
            sp_stats.norm, sp_stats.lognorm, sp_stats.gamma,
            sp_stats.beta, sp_stats.expon, sp_stats.uniform,
            sp_stats.t, sp_stats.chi2,
        ]
        da = DistributionAnalyzer.__new__(DistributionAnalyzer)
        da.logger = __import__("logging").getLogger(__name__)
        da.distributions = fixed_distributions
        return da

    def test_init(self, da):
        assert len(da.distributions) > 0

    def test_fit_distributions(self, da):
        np.random.seed(42)
        data = np.random.normal(5, 1, 200)
        result = da.fit_distributions(data)
        assert "best_distribution" in result
        assert "results" in result
        assert "ranking_by_aic" in result

    def test_fit_too_few_points(self, da):
        with pytest.raises(ValueError, match="at least 10"):
            da.fit_distributions(np.array([1, 2, 3]))

    def test_fit_custom_distributions(self, da):
        from scipy import stats as sp_stats
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        result = da.fit_distributions(data, distributions=[sp_stats.norm])
        assert "norm" in result["results"]

    def test_fit_with_nan(self, da):
        np.random.seed(42)
        data = np.concatenate([np.random.normal(0, 1, 100), [np.nan, np.nan]])
        result = da.fit_distributions(data)
        assert "best_distribution" in result


@pytest.mark.coverage_extra
class TestTimeSeriesAnalyzer:
    @pytest.fixture
    def tsa(self):
        return TimeSeriesAnalyzer()

    def test_stationarity_tests(self, tsa):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        result = tsa.stationarity_tests(data)
        assert "adf" in result or "mean_stability" in result

    def test_stationarity_nonstationary(self, tsa):
        data = np.cumsum(np.random.normal(0, 1, 200))
        result = tsa.stationarity_tests(data)
        assert isinstance(result, dict)

    def test_basic_stationarity_tests(self, tsa):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        result = tsa._basic_stationarity_tests(data)
        assert "mean_stability" in result
        assert "variance_stability" in result

    def test_trend_analysis(self, tsa):
        np.random.seed(42)
        data = np.arange(50) + np.random.normal(0, 0.5, 50)
        result = tsa.trend_analysis(data)
        assert "linear_trend" in result
        assert result["linear_trend"]["slope"] > 0

    def test_trend_analysis_custom_time(self, tsa):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        time_idx = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = tsa.trend_analysis(data, time_index=time_idx)
        assert "linear_trend" in result

    def test_mann_kendall(self, tsa):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = tsa._mann_kendall_test(data)
        assert result["trend"] == "Increasing"
        assert result["p_value"] is not None
        assert result["significant"] is not None

    def test_mann_kendall_no_trend(self, tsa):
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = tsa._mann_kendall_test(data)
        assert result["z_score"] == 0
        assert result["trend"] == "No trend"

    def test_mann_kendall_decreasing(self, tsa):
        data = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = tsa._mann_kendall_test(data)
        assert result["trend"] == "Decreasing"


@pytest.mark.coverage_extra
class TestMultipleComparisons:
    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer(alpha=0.05, random_seed=42)

    def test_tukey(self, analyzer):
        np.random.seed(42)
        data = [
            np.random.normal(5, 1, 20),
            np.random.normal(5.5, 1, 20),
            np.random.normal(6, 1, 20),
        ]
        result = analyzer.multiple_comparisons(data, method="tukey")
        assert result["n_groups"] == 3
        assert "group_names" in result

    def test_bonferroni(self, analyzer):
        np.random.seed(42)
        data = [
            np.random.normal(5, 1, 20),
            np.random.normal(7, 1, 20),
        ]
        result = analyzer.multiple_comparisons(
            data, group_names=["A", "B"], method="bonferroni"
        )
        assert result["method"] == "bonferroni"
        assert "pairwise_comparisons" in result

    def test_holm(self, analyzer):
        np.random.seed(42)
        data = [
            np.random.normal(5, 1, 20),
            np.random.normal(6, 1, 20),
            np.random.normal(7, 1, 20),
        ]
        result = analyzer.multiple_comparisons(data, method="holm")
        assert result["method"] == "holm"

    def test_fdr(self, analyzer):
        np.random.seed(42)
        data = [
            np.random.normal(5, 1, 20),
            np.random.normal(6, 1, 20),
        ]
        result = analyzer.multiple_comparisons(data, method="fdr_bh")
        assert result["method"] == "fdr_bh"

    def test_auto_group_names(self, analyzer):
        data = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        result = analyzer.multiple_comparisons(data, method="bonferroni")
        assert result["group_names"][0] == "Group_1"
