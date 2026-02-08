import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np

from config.statistical_analysis import (
    TestType, HypothesisType, EffectSizeType,
    StatisticalTest, TestResult, DescriptiveStatistics,
    StatisticalAnalyzer,
    calculate_effect_size, interpret_effect_size, power_analysis,
)


class TestEnums:
    def test_test_type(self):
        assert TestType.PARAMETRIC.value == "parametric"
        assert TestType.NON_PARAMETRIC.value == "non_parametric"
        assert TestType.BAYESIAN.value == "bayesian"
        assert TestType.BOOTSTRAP.value == "bootstrap"
        assert TestType.PERMUTATION.value == "permutation"

    def test_hypothesis_type(self):
        assert HypothesisType.ONE_SAMPLE_T.value == "one_sample_t"
        assert HypothesisType.TWO_SAMPLE_T.value == "two_sample_t"
        assert HypothesisType.PAIRED_T.value == "paired_t"
        assert HypothesisType.MANN_WHITNEY_U.value == "mann_whitney_u"
        assert HypothesisType.SHAPIRO_WILK.value == "shapiro_wilk"
        assert HypothesisType.KOLMOGOROV_SMIRNOV.value == "kolmogorov_smirnov"
        assert HypothesisType.ONE_WAY_ANOVA.value == "one_way_anova"

    def test_effect_size_type(self):
        assert EffectSizeType.COHENS_D.value == "cohens_d"
        assert EffectSizeType.HEDGES_G.value == "hedges_g"
        assert EffectSizeType.GLASS_DELTA.value == "glass_delta"
        assert EffectSizeType.ETA_SQUARED.value == "eta_squared"
        assert EffectSizeType.CRAMERS_V.value == "cramers_v"
        assert EffectSizeType.CLIFF_DELTA.value == "cliff_delta"


class TestStatisticalTest:
    def test_valid(self):
        st = StatisticalTest(test_type=HypothesisType.ONE_SAMPLE_T)
        assert st.alpha == 0.05
        assert st.alternative == "two-sided"

    def test_invalid_alpha_low(self):
        with pytest.raises(ValueError, match="Alpha must be between"):
            StatisticalTest(test_type=HypothesisType.ONE_SAMPLE_T, alpha=0.0)

    def test_invalid_alpha_high(self):
        with pytest.raises(ValueError, match="Alpha must be between"):
            StatisticalTest(test_type=HypothesisType.ONE_SAMPLE_T, alpha=1.0)

    def test_invalid_alternative(self):
        with pytest.raises(ValueError, match="Alternative must be"):
            StatisticalTest(
                test_type=HypothesisType.ONE_SAMPLE_T, alternative="bad"
            )

    def test_valid_alternatives(self):
        for alt in ["two-sided", "less", "greater"]:
            st = StatisticalTest(
                test_type=HypothesisType.ONE_SAMPLE_T, alternative=alt
            )
            assert st.alternative == alt


class TestTestResult:
    def test_significant(self):
        r = TestResult(test_name="t", statistic=2.0, p_value=0.01)
        assert r.is_significant is True

    def test_not_significant(self):
        r = TestResult(test_name="t", statistic=0.5, p_value=0.5)
        assert r.is_significant is False

    def test_interpret_reject(self):
        r = TestResult(test_name="t", statistic=2.0, p_value=0.01)
        interpretation = r.interpret_result(alpha=0.05)
        assert "Reject" in interpretation

    def test_interpret_fail(self):
        r = TestResult(test_name="t", statistic=0.5, p_value=0.5)
        interpretation = r.interpret_result(alpha=0.05)
        assert "Fail to reject" in interpretation


class TestDescriptiveStatistics:
    def test_creation(self):
        ds = DescriptiveStatistics(
            n=10, mean=5.0, median=5.0, mode=5.0,
            std=1.0, variance=1.0,
            min_value=3.0, max_value=7.0, range_value=4.0,
            q1=4.0, q3=6.0, iqr=2.0,
            skewness=0.0, kurtosis=0.0,
            coefficient_of_variation=0.2, mad=0.5,
            trimmed_mean=5.0, geometric_mean=4.9, harmonic_mean=4.8,
            mean_ci_95=(4.3, 5.7),
        )
        assert ds.n == 10
        assert ds.normality_p_value is None
        assert ds.outliers_count == 0
        assert ds.outliers_indices == []


class TestStatisticalAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer(alpha=0.05, random_seed=42)

    def test_descriptive_stats(self, analyzer):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = analyzer.descriptive_statistics(data)
        assert result.n == 10
        assert result.mean == pytest.approx(5.5)
        assert result.median == pytest.approx(5.5)
        assert result.std > 0
        assert result.variance > 0
        assert result.min_value == 1.0
        assert result.max_value == 10.0

    def test_descriptive_stats_empty(self, analyzer):
        with pytest.raises(ValueError, match="No valid data"):
            analyzer.descriptive_statistics(np.array([np.nan, np.nan]))

    def test_descriptive_stats_positive_only(self, analyzer):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyzer.descriptive_statistics(data)
        assert result.geometric_mean is not None
        assert result.harmonic_mean is not None

    def test_descriptive_stats_with_negatives(self, analyzer):
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = analyzer.descriptive_statistics(data)
        assert result.geometric_mean is None
        assert result.harmonic_mean is None

    def test_descriptive_stats_single_value(self, analyzer):
        data = np.array([5.0])
        result = analyzer.descriptive_statistics(data)
        assert result.n == 1
        assert result.std == 0.0

    def test_one_sample_t_test(self, analyzer):
        data = np.random.normal(5.0, 1.0, 30)
        config = StatisticalTest(
            test_type=HypothesisType.ONE_SAMPLE_T,
            effect_size_type=EffectSizeType.COHENS_D,
        )
        result = analyzer.hypothesis_test(config, data, mu=5.0)
        assert result.test_name == "One-sample t-test"
        assert result.effect_size is not None

    def test_two_sample_t_test(self, analyzer):
        d1 = np.random.normal(5, 1, 30)
        d2 = np.random.normal(5, 1, 30)
        config = StatisticalTest(
            test_type=HypothesisType.TWO_SAMPLE_T,
            effect_size_type=EffectSizeType.COHENS_D,
        )
        result = analyzer.hypothesis_test(config, d1, d2)
        assert result.test_name == "Two-sample t-test"

    def test_paired_t_test(self, analyzer):
        d1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        config = StatisticalTest(
            test_type=HypothesisType.PAIRED_T,
            effect_size_type=EffectSizeType.COHENS_D,
        )
        result = analyzer.hypothesis_test(config, d1, d2)
        assert result.test_name == "Paired t-test"

    def test_mann_whitney(self, analyzer):
        d1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d2 = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        config = StatisticalTest(
            test_type=HypothesisType.MANN_WHITNEY_U,
            effect_size_type=EffectSizeType.CLIFF_DELTA,
        )
        result = analyzer.hypothesis_test(config, d1, d2)
        assert result.test_name == "Mann-Whitney U test"

    def test_wilcoxon(self, analyzer):
        d1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        config = StatisticalTest(test_type=HypothesisType.WILCOXON_SIGNED_RANK)
        result = analyzer.hypothesis_test(config, d1, d2)
        assert result.test_name == "Wilcoxon signed-rank test"

    def test_wilcoxon_one_sample(self, analyzer):
        d1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        config = StatisticalTest(test_type=HypothesisType.WILCOXON_SIGNED_RANK)
        result = analyzer.hypothesis_test(config, d1)
        assert "Wilcoxon" in result.test_name

    def test_shapiro_wilk(self, analyzer):
        data = np.random.normal(0, 1, 50)
        config = StatisticalTest(test_type=HypothesisType.SHAPIRO_WILK)
        result = analyzer.hypothesis_test(config, data)
        assert result.test_name == "Shapiro-Wilk normality test"

    def test_ks_test(self, analyzer):
        data = np.random.normal(0, 1, 50)
        config = StatisticalTest(test_type=HypothesisType.KOLMOGOROV_SMIRNOV)
        result = analyzer.hypothesis_test(config, data)
        assert result.test_name == "Kolmogorov-Smirnov test"

    def test_unsupported_test(self, analyzer):
        config = StatisticalTest(test_type=HypothesisType.ONE_WAY_ANOVA)
        with pytest.raises(ValueError, match="Unsupported"):
            analyzer.hypothesis_test(config, np.array([1, 2, 3]))

    def test_bootstrap_one_sample(self, analyzer):
        data = np.random.normal(5, 1, 30)
        result = analyzer.bootstrap_test(data, n_bootstrap=100)
        assert "observed_statistic" in result
        assert "p_value" in result
        assert "confidence_interval" in result

    def test_bootstrap_two_sample(self, analyzer):
        d1 = np.random.normal(5, 1, 30)
        d2 = np.random.normal(6, 1, 30)
        result = analyzer.bootstrap_test(d1, d2, n_bootstrap=100)
        assert "observed_statistic" in result


class TestUtilityFunctions:
    def test_cohens_d(self):
        d1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        es = calculate_effect_size(d1, d2, EffectSizeType.COHENS_D)
        assert isinstance(es, float)

    def test_hedges_g(self):
        d1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        es = calculate_effect_size(d1, d2, EffectSizeType.HEDGES_G)
        assert isinstance(es, float)

    def test_glass_delta(self):
        d1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        es = calculate_effect_size(d1, d2, EffectSizeType.GLASS_DELTA)
        assert isinstance(es, float)

    def test_unsupported_effect(self):
        d1 = np.array([1.0, 2.0])
        d2 = np.array([3.0, 4.0])
        with pytest.raises(ValueError, match="Unsupported"):
            calculate_effect_size(d1, d2, EffectSizeType.ETA_SQUARED)

    def test_interpret_negligible(self):
        assert interpret_effect_size(0.1) == "Negligible"

    def test_interpret_small(self):
        assert interpret_effect_size(0.3) == "Small"

    def test_interpret_medium(self):
        assert interpret_effect_size(0.6) == "Medium"

    def test_interpret_large(self):
        assert interpret_effect_size(1.0) == "Large"

    def test_interpret_unknown(self):
        result = interpret_effect_size(0.5, EffectSizeType.CRAMERS_V)
        assert result == "Unknown interpretation"

    def test_power_analysis(self):
        power = power_analysis(0.5, 100)
        assert 0 <= power <= 1

    def test_power_analysis_generic(self):
        power = power_analysis(0.5, 100, test_type="other")
        assert 0 <= power <= 1
