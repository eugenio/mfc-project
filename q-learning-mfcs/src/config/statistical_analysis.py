"""
Statistical Analysis and Hypothesis Testing Tools

This module provides comprehensive statistical analysis and hypothesis testing tools
for MFC systems, including descriptive statistics, inferential statistics, and
advanced statistical methods for biological system analysis.

Classes:
- StatisticalAnalyzer: Main statistical analysis framework
- HypothesisTest: Hypothesis testing framework with multiple test types
- DescriptiveStats: Comprehensive descriptive statistics calculator
- DistributionAnalyzer: Distribution fitting and analysis tools
- TimeSeriesStats: Time series statistical analysis
- ExperimentalDesign: Statistical experimental design tools

Features:
- Comprehensive descriptive statistics
- Parametric and non-parametric hypothesis tests
- Distribution fitting and goodness-of-fit tests
- Time series analysis (trend, seasonality, stationarity)
- Experimental design (ANOVA, factorial designs)
- Multiple comparison corrections
- Effect size calculations
- Bootstrap and permutation tests
- Bayesian statistical inference

Literature References:
1. Montgomery, D. C. (2017). "Design and Analysis of Experiments"
2. Wilcox, R. R. (2017). "Introduction to Robust Estimation and Hypothesis Testing"
3. Efron, B., & Hastie, T. (2016). "Computer Age Statistical Inference"
4. Gelman, A., et al. (2013). "Bayesian Data Analysis"
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Statistical dependencies
try:
    from scipy import stats
    from scipy.stats import chi2_contingency, fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Statistical analysis will be limited.")

# Advanced statistical analysis
try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("Statsmodels not available. Some advanced statistical tests will be limited.")

# Machine learning for statistical analysis
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available. Some multivariate analysis will be limited.")


class TestType(Enum):
    """Types of statistical tests."""
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non_parametric"
    BAYESIAN = "bayesian"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


class HypothesisType(Enum):
    """Types of hypothesis tests."""
    ONE_SAMPLE_T = "one_sample_t"
    TWO_SAMPLE_T = "two_sample_t"
    PAIRED_T = "paired_t"
    WELCH_T = "welch_t"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    CHI_SQUARE_GOODNESS = "chi_square_goodness"
    CHI_SQUARE_INDEPENDENCE = "chi_square_independence"
    FISHER_EXACT = "fisher_exact"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    SHAPIRO_WILK = "shapiro_wilk"
    ANDERSON_DARLING = "anderson_darling"
    JARQUE_BERA = "jarque_bera"
    ONE_WAY_ANOVA = "one_way_anova"
    TWO_WAY_ANOVA = "two_way_anova"
    REPEATED_MEASURES_ANOVA = "repeated_measures_anova"


class EffectSizeType(Enum):
    """Types of effect size measures."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    CRAMERS_V = "cramers_v"
    CLIFF_DELTA = "cliff_delta"


@dataclass
class StatisticalTest:
    """Configuration for statistical test."""
    test_type: HypothesisType
    alpha: float = 0.05
    alternative: str = "two-sided"  # "two-sided", "less", "greater"
    correction_method: Optional[str] = None  # "bonferroni", "holm", "fdr_bh"
    bootstrap_samples: int = 1000
    effect_size_type: Optional[EffectSizeType] = None

    def __post_init__(self):
        """Validate test configuration."""
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("Alpha must be between 0 and 1")

        valid_alternatives = ["two-sided", "less", "greater"]
        if self.alternative not in valid_alternatives:
            raise ValueError(f"Alternative must be one of {valid_alternatives}")


@dataclass
class TestResult:
    """Results of statistical test."""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[float] = None
    critical_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    power: Optional[float] = None
    sample_size: Optional[int] = None
    assumptions_met: Optional[Dict[str, bool]] = None
    test_description: str = ""

    @property
    def is_significant(self) -> bool:
        """Check if test result is statistically significant."""
        return self.p_value < 0.05  # Default alpha

    def interpret_result(self, alpha: float = 0.05) -> str:
        """Interpret the test result."""
        if self.p_value < alpha:
            return f"Reject null hypothesis (p = {self.p_value:.4f} < α = {alpha})"
        else:
            return f"Fail to reject null hypothesis (p = {self.p_value:.4f} ≥ α = {alpha})"


@dataclass
class DescriptiveStatistics:
    """Comprehensive descriptive statistics."""
    n: int
    mean: float
    median: float
    mode: Optional[float]
    std: float
    variance: float
    min_value: float
    max_value: float
    range_value: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float

    # Additional statistics
    coefficient_of_variation: float
    mad: float  # Median Absolute Deviation
    trimmed_mean: float
    geometric_mean: Optional[float]
    harmonic_mean: Optional[float]

    # Confidence intervals
    mean_ci_95: Tuple[float, float]
    median_ci_95: Optional[Tuple[float, float]] = None

    # Distribution characteristics
    normality_p_value: Optional[float] = None
    outliers_count: int = 0
    outliers_indices: List[int] = field(default_factory=list)


class StatisticalAnalyzer:
    """Main framework for statistical analysis."""

    def __init__(self, alpha: float = 0.05, random_seed: Optional[int] = None):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for tests
            random_seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.random_seed = random_seed
        self.logger = logging.getLogger(__name__)

        if random_seed is not None:
            np.random.seed(random_seed)

    def descriptive_statistics(self, data: np.ndarray,
                             confidence_level: float = 0.95) -> DescriptiveStatistics:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            data: Input data array
            confidence_level: Confidence level for intervals
            
        Returns:
            Descriptive statistics
        """
        data = np.asarray(data)
        data_clean = data[~np.isnan(data)]

        if len(data_clean) == 0:
            raise ValueError("No valid data points")

        n = len(data_clean)

        # Basic statistics
        mean_val = np.mean(data_clean)
        median_val = np.median(data_clean)
        std_val = np.std(data_clean, ddof=1) if n > 1 else 0.0
        var_val = np.var(data_clean, ddof=1) if n > 1 else 0.0

        # Mode (most frequent value)
        if HAS_SCIPY:
            mode_result = stats.mode(data_clean, keepdims=True)
            mode_val = mode_result.mode[0] if mode_result.count[0] > 1 else None
        else:
            mode_val = None

        # Percentiles
        q1 = np.percentile(data_clean, 25)
        q3 = np.percentile(data_clean, 75)
        iqr = q3 - q1

        # Shape statistics
        if HAS_SCIPY and n > 2:
            skewness = stats.skew(data_clean)
            kurtosis_val = stats.kurtosis(data_clean)
        else:
            skewness = 0.0
            kurtosis_val = 0.0

        # Additional statistics
        cv = std_val / mean_val if mean_val != 0 else np.inf
        mad = np.median(np.abs(data_clean - median_val))
        trimmed_mean = stats.trim_mean(data_clean, 0.1) if HAS_SCIPY else mean_val

        # Geometric and harmonic means (for positive data)
        if np.all(data_clean > 0):
            geometric_mean = stats.gmean(data_clean) if HAS_SCIPY else np.exp(np.mean(np.log(data_clean)))
            harmonic_mean = stats.hmean(data_clean) if HAS_SCIPY else len(data_clean) / np.sum(1.0/data_clean)
        else:
            geometric_mean = None
            harmonic_mean = None

        # Confidence interval for mean
        if HAS_SCIPY and n > 1:
            sem = stats.sem(data_clean)
            t_crit = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin_error = t_crit * sem
            mean_ci = (mean_val - margin_error, mean_val + margin_error)
        else:
            mean_ci = (mean_val, mean_val)

        # Outliers detection (IQR method)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_mask = (data_clean < lower_bound) | (data_clean > upper_bound)
        outliers_indices = np.where(outliers_mask)[0].tolist()

        # Normality test
        normality_p = None
        if HAS_SCIPY and n >= 3:
            try:
                if n <= 5000:  # Shapiro-Wilk has sample size limitation
                    _, normality_p = stats.shapiro(data_clean)
                else:
                    _, normality_p = stats.jarque_bera(data_clean)
            except Exception:
                pass

        return DescriptiveStatistics(
            n=n,
            mean=mean_val,
            median=median_val,
            mode=mode_val,
            std=std_val,
            variance=var_val,
            min_value=np.min(data_clean),
            max_value=np.max(data_clean),
            range_value=np.max(data_clean) - np.min(data_clean),
            q1=q1,
            q3=q3,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurtosis_val,
            coefficient_of_variation=cv,
            mad=mad,
            trimmed_mean=trimmed_mean,
            geometric_mean=geometric_mean,
            harmonic_mean=harmonic_mean,
            mean_ci_95=mean_ci,
            normality_p_value=normality_p,
            outliers_count=len(outliers_indices),
            outliers_indices=outliers_indices
        )

    def hypothesis_test(self, test_config: StatisticalTest,
                       data1: np.ndarray,
                       data2: Optional[np.ndarray] = None,
                       **kwargs) -> TestResult:
        """
        Perform hypothesis test.
        
        Args:
            test_config: Test configuration
            data1: First data sample
            data2: Second data sample (if applicable)
            **kwargs: Additional test parameters
            
        Returns:
            Test results
        """
        if not HAS_SCIPY:
            raise ImportError("SciPy required for hypothesis testing")

        # Clean data
        data1_clean = np.asarray(data1)[~np.isnan(data1)]
        data2_clean = np.asarray(data2)[~np.isnan(data2)] if data2 is not None else None

        # Dispatch to appropriate test
        if test_config.test_type == HypothesisType.ONE_SAMPLE_T:
            return self._one_sample_t_test(data1_clean, test_config, **kwargs)
        elif test_config.test_type == HypothesisType.TWO_SAMPLE_T:
            return self._two_sample_t_test(data1_clean, data2_clean, test_config, **kwargs)
        elif test_config.test_type == HypothesisType.PAIRED_T:
            return self._paired_t_test(data1_clean, data2_clean, test_config, **kwargs)
        elif test_config.test_type == HypothesisType.MANN_WHITNEY_U:
            return self._mann_whitney_test(data1_clean, data2_clean, test_config, **kwargs)
        elif test_config.test_type == HypothesisType.WILCOXON_SIGNED_RANK:
            return self._wilcoxon_test(data1_clean, data2_clean, test_config, **kwargs)
        elif test_config.test_type == HypothesisType.SHAPIRO_WILK:
            return self._shapiro_wilk_test(data1_clean, test_config, **kwargs)
        elif test_config.test_type == HypothesisType.KOLMOGOROV_SMIRNOV:
            return self._ks_test(data1_clean, test_config, **kwargs)
        else:
            raise ValueError(f"Unsupported test type: {test_config.test_type}")

    def _one_sample_t_test(self, data: np.ndarray, config: StatisticalTest,
                          mu: float = 0.0) -> TestResult:
        """Perform one-sample t-test."""
        n = len(data)
        if n < 2:
            raise ValueError("Need at least 2 observations for t-test")

        statistic, p_value = stats.ttest_1samp(data, mu, alternative=config.alternative)

        # Degrees of freedom
        df = n - 1

        # Confidence interval
        mean_val = np.mean(data)
        sem = stats.sem(data)
        t_crit = stats.t.ppf((1 + 0.95) / 2, df)
        ci = (mean_val - t_crit * sem, mean_val + t_crit * sem)

        # Effect size (Cohen's d)
        effect_size = None
        if config.effect_size_type == EffectSizeType.COHENS_D:
            effect_size = (mean_val - mu) / np.std(data, ddof=1)

        return TestResult(
            test_name="One-sample t-test",
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            confidence_interval=ci,
            effect_size=effect_size,
            sample_size=n,
            test_description=f"Testing if mean equals {mu}"
        )

    def _two_sample_t_test(self, data1: np.ndarray, data2: np.ndarray,
                          config: StatisticalTest,
                          equal_var: bool = True) -> TestResult:
        """Perform two-sample t-test."""
        if data2 is None:
            raise ValueError("Two samples required for two-sample t-test")

        n1, n2 = len(data1), len(data2)
        if n1 < 2 or n2 < 2:
            raise ValueError("Need at least 2 observations in each group")

        statistic, p_value = stats.ttest_ind(data1, data2,
                                           equal_var=equal_var,
                                           alternative=config.alternative)

        # Degrees of freedom
        if equal_var:
            df = n1 + n2 - 2
        else:
            # Welch's t-test degrees of freedom
            s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))

        # Effect size (Cohen's d)
        effect_size = None
        if config.effect_size_type == EffectSizeType.COHENS_D:
            mean1, mean2 = np.mean(data1), np.mean(data2)
            if equal_var:
                pooled_std = np.sqrt(((n1-1)*np.var(data1, ddof=1) + (n2-1)*np.var(data2, ddof=1)) / (n1+n2-2))
            else:
                pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
            effect_size = (mean1 - mean2) / pooled_std

        test_name = "Welch's t-test" if not equal_var else "Two-sample t-test"

        return TestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=effect_size,
            sample_size=n1 + n2,
            test_description="Testing if two population means are equal"
        )

    def _paired_t_test(self, data1: np.ndarray, data2: np.ndarray,
                      config: StatisticalTest) -> TestResult:
        """Perform paired t-test."""
        if data2 is None:
            raise ValueError("Two samples required for paired t-test")

        if len(data1) != len(data2):
            raise ValueError("Paired samples must have equal length")

        statistic, p_value = stats.ttest_rel(data1, data2, alternative=config.alternative)

        n = len(data1)
        df = n - 1

        # Effect size (Cohen's d for paired samples)
        effect_size = None
        if config.effect_size_type == EffectSizeType.COHENS_D:
            diff = data1 - data2
            effect_size = np.mean(diff) / np.std(diff, ddof=1)

        return TestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=effect_size,
            sample_size=n,
            test_description="Testing if paired differences have zero mean"
        )

    def _mann_whitney_test(self, data1: np.ndarray, data2: np.ndarray,
                          config: StatisticalTest) -> TestResult:
        """Perform Mann-Whitney U test."""
        if data2 is None:
            raise ValueError("Two samples required for Mann-Whitney test")

        statistic, p_value = stats.mannwhitneyu(data1, data2,
                                              alternative=config.alternative)

        # Effect size (rank biserial correlation)
        effect_size = None
        if config.effect_size_type == EffectSizeType.CLIFF_DELTA:
            n1, n2 = len(data1), len(data2)
            # Cliff's delta approximation
            effect_size = 2 * statistic / (n1 * n2) - 1

        return TestResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=len(data1) + len(data2),
            test_description="Non-parametric test for comparing two independent samples"
        )

    def _wilcoxon_test(self, data1: np.ndarray, data2: np.ndarray,
                      config: StatisticalTest) -> TestResult:
        """Perform Wilcoxon signed-rank test."""
        if data2 is None:
            # One-sample Wilcoxon
            statistic, p_value = stats.wilcoxon(data1, alternative=config.alternative)
            description = "One-sample Wilcoxon signed-rank test"
        else:
            # Paired Wilcoxon
            statistic, p_value = stats.wilcoxon(data1, data2, alternative=config.alternative)
            description = "Wilcoxon signed-rank test for paired samples"

        return TestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            sample_size=len(data1),
            test_description=description
        )

    def _shapiro_wilk_test(self, data: np.ndarray, config: StatisticalTest) -> TestResult:
        """Perform Shapiro-Wilk normality test."""
        if len(data) > 5000:
            raise ValueError("Shapiro-Wilk test limited to 5000 observations")

        statistic, p_value = stats.shapiro(data)

        return TestResult(
            test_name="Shapiro-Wilk normality test",
            statistic=statistic,
            p_value=p_value,
            sample_size=len(data),
            test_description="Testing if data comes from normal distribution"
        )

    def _ks_test(self, data: np.ndarray, config: StatisticalTest,
                distribution: str = "norm") -> TestResult:
        """Perform Kolmogorov-Smirnov goodness-of-fit test."""
        if distribution == "norm":
            # Fit normal distribution
            mean_est, std_est = stats.norm.fit(data)
            statistic, p_value = stats.kstest(data, lambda x: stats.norm.cdf(x, mean_est, std_est))
            desc = "K-S test for normality"
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return TestResult(
            test_name="Kolmogorov-Smirnov test",
            statistic=statistic,
            p_value=p_value,
            sample_size=len(data),
            test_description=desc
        )

    def multiple_comparisons(self, data: List[np.ndarray],
                           group_names: Optional[List[str]] = None,
                           method: str = "tukey") -> Dict[str, Any]:
        """
        Perform multiple comparisons between groups.
        
        Args:
            data: List of data arrays for each group
            group_names: Names of groups
            method: Multiple comparison method ("tukey", "bonferroni", "holm")
            
        Returns:
            Multiple comparison results
        """
        if not HAS_STATSMODELS and method == "tukey":
            method = "bonferroni"  # Fallback

        n_groups = len(data)
        if group_names is None:
            group_names = [f"Group_{i+1}" for i in range(n_groups)]

        # Prepare data for analysis
        all_data = []
        all_groups = []

        for i, group_data in enumerate(data):
            clean_data = np.asarray(group_data)[~np.isnan(group_data)]
            all_data.extend(clean_data)
            all_groups.extend([group_names[i]] * len(clean_data))

        # Convert to pandas DataFrame for easier handling
        df = pd.DataFrame({
            'value': all_data,
            'group': all_groups
        })

        results = {
            'method': method,
            'n_groups': n_groups,
            'group_names': group_names,
            'pairwise_comparisons': {}
        }

        if method == "tukey" and HAS_STATSMODELS:
            # Tukey HSD
            mc = MultiComparison(df['value'], df['group'])
            tukey_result = mc.tukeyhsd()

            results['tukey_summary'] = str(tukey_result)
            results['tukey_table'] = tukey_result.summary().as_text()

        else:
            # Pairwise tests with correction
            all_p_values = []
            comparisons = []

            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    data1 = np.asarray(data[i])[~np.isnan(data[i])]
                    data2 = np.asarray(data[j])[~np.isnan(data[j])]

                    # Perform t-test
                    if HAS_SCIPY:
                        statistic, p_val = stats.ttest_ind(data1, data2)
                        all_p_values.append(p_val)
                        comparisons.append((group_names[i], group_names[j]))

                        results['pairwise_comparisons'][f"{group_names[i]}_vs_{group_names[j]}"] = {
                            'statistic': statistic,
                            'p_value': p_val
                        }

            # Apply multiple comparison correction
            if HAS_STATSMODELS:
                from statsmodels.stats.multitest import multipletests

                if method == "bonferroni":
                    rejected, p_corrected, _, _ = multipletests(all_p_values, method='bonferroni')
                elif method == "holm":
                    rejected, p_corrected, _, _ = multipletests(all_p_values, method='holm')
                else:
                    rejected, p_corrected, _, _ = multipletests(all_p_values, method='fdr_bh')

                for i, (comp, p_corr, is_rejected) in enumerate(zip(comparisons, p_corrected, rejected)):
                    comp_key = f"{comp[0]}_vs_{comp[1]}"
                    results['pairwise_comparisons'][comp_key]['p_value_corrected'] = p_corr
                    results['pairwise_comparisons'][comp_key]['significant'] = is_rejected

        return results

    def bootstrap_test(self, data1: np.ndarray, data2: Optional[np.ndarray] = None,
                      statistic_func: Callable = np.mean,
                      n_bootstrap: int = 1000,
                      confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Perform bootstrap hypothesis test.
        
        Args:
            data1: First sample
            data2: Second sample (optional)
            statistic_func: Function to calculate test statistic
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Bootstrap test results
        """
        data1_clean = np.asarray(data1)[~np.isnan(data1)]

        if data2 is None:
            # One-sample bootstrap
            observed_stat = statistic_func(data1_clean)

            bootstrap_stats = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data1_clean, size=len(data1_clean), replace=True)
                bootstrap_stats.append(statistic_func(bootstrap_sample))

            bootstrap_stats = np.array(bootstrap_stats)

        else:
            # Two-sample bootstrap
            data2_clean = np.asarray(data2)[~np.isnan(data2)]
            observed_stat = statistic_func(data1_clean) - statistic_func(data2_clean)

            # Pool data for null distribution
            pooled_data = np.concatenate([data1_clean, data2_clean])
            n1, n2 = len(data1_clean), len(data2_clean)

            bootstrap_stats = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(pooled_data, size=n1+n2, replace=True)
                sample1 = bootstrap_sample[:n1]
                sample2 = bootstrap_sample[n1:]
                bootstrap_stats.append(statistic_func(sample1) - statistic_func(sample2))

            bootstrap_stats = np.array(bootstrap_stats)

        # Calculate p-value (two-tailed)
        p_value = 2 * min(np.mean(bootstrap_stats >= observed_stat),
                         np.mean(bootstrap_stats <= observed_stat))

        # Confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return {
            'observed_statistic': observed_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'bootstrap_distribution': bootstrap_stats,
            'n_bootstrap': n_bootstrap
        }


class DistributionAnalyzer:
    """Distribution fitting and analysis tools."""

    def __init__(self):
        """Initialize distribution analyzer."""
        self.logger = logging.getLogger(__name__)

        # Common distributions to test
        if HAS_SCIPY:
            self.distributions = [
                stats.norm, stats.lognorm, stats.gamma, stats.beta,
                stats.exponential, stats.uniform, stats.t, stats.chi2
            ]
        else:
            self.distributions = []

    def fit_distributions(self, data: np.ndarray,
                         distributions: Optional[List] = None) -> Dict[str, Any]:
        """
        Fit multiple distributions to data and compare goodness of fit.
        
        Args:
            data: Input data
            distributions: List of distributions to test
            
        Returns:
            Distribution fitting results
        """
        if not HAS_SCIPY:
            raise ImportError("SciPy required for distribution fitting")

        data_clean = np.asarray(data)[~np.isnan(data)]
        if len(data_clean) < 10:
            raise ValueError("Need at least 10 observations for distribution fitting")

        test_distributions = distributions or self.distributions
        results = {}

        for dist in test_distributions:
            try:
                # Fit distribution
                params = dist.fit(data_clean)

                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = stats.kstest(data_clean, dist.cdf, args=params)

                # Anderson-Darling test (if available)
                ad_statistic, ad_p_value = None, None
                try:
                    if hasattr(stats, 'anderson') and dist.name in ['norm', 'expon', 'logistic']:
                        ad_result = stats.anderson(data_clean, dist=dist.name[:4])
                        ad_statistic = ad_result.statistic
                        # Approximate p-value (not exact)
                        ad_p_value = 0.05 if ad_statistic > ad_result.critical_values[2] else 0.1
                except Exception:
                    pass

                # Log-likelihood and information criteria
                log_likelihood = np.sum(dist.logpdf(data_clean, *params))
                n_params = len(params)
                aic = 2 * n_params - 2 * log_likelihood
                bic = np.log(len(data_clean)) * n_params - 2 * log_likelihood

                results[dist.name] = {
                    'parameters': params,
                    'ks_statistic': ks_statistic,
                    'ks_p_value': ks_p_value,
                    'ad_statistic': ad_statistic,
                    'ad_p_value': ad_p_value,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'distribution_object': dist
                }

            except Exception as e:
                self.logger.warning(f"Failed to fit {dist.name}: {e}")
                continue

        # Rank distributions by AIC
        if results:
            sorted_results = sorted(results.items(), key=lambda x: x[1]['aic'])
            best_distribution = sorted_results[0]

            return {
                'results': results,
                'best_distribution': best_distribution[0],
                'best_fit_summary': best_distribution[1],
                'ranking_by_aic': [name for name, _ in sorted_results]
            }
        else:
            return {'results': {}, 'error': 'No distributions could be fitted'}


class TimeSeriesAnalyzer:
    """Time series statistical analysis."""

    def __init__(self):
        """Initialize time series analyzer."""
        self.logger = logging.getLogger(__name__)

    def stationarity_tests(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Test stationarity of time series.
        
        Args:
            data: Time series data
            
        Returns:
            Stationarity test results
        """
        if not HAS_STATSMODELS:
            self.logger.warning("Statsmodels not available. Using basic stationarity tests.")
            return self._basic_stationarity_tests(data)

        data_clean = np.asarray(data)[~np.isnan(data)]
        results = {}

        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(data_clean)
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
            }
        except Exception as e:
            self.logger.warning(f"ADF test failed: {e}")

        # KPSS test
        try:
            kpss_result = kpss(data_clean)
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'interpretation': 'Non-stationary' if kpss_result[1] < 0.05 else 'Stationary'
            }
        except Exception as e:
            self.logger.warning(f"KPSS test failed: {e}")

        return results

    def _basic_stationarity_tests(self, data: np.ndarray) -> Dict[str, Any]:
        """Basic stationarity tests without statsmodels."""
        data_clean = np.asarray(data)[~np.isnan(data)]

        # Split data into halves and compare means/variances
        n = len(data_clean)
        first_half = data_clean[:n//2]
        second_half = data_clean[n//2:]

        results = {}

        if HAS_SCIPY:
            # Test for equal means
            t_stat, p_val_mean = stats.ttest_ind(first_half, second_half)
            results['mean_stability'] = {
                'statistic': t_stat,
                'p_value': p_val_mean,
                'interpretation': 'Stable mean' if p_val_mean > 0.05 else 'Unstable mean'
            }

            # Test for equal variances
            f_stat, p_val_var = stats.levene(first_half, second_half)
            results['variance_stability'] = {
                'statistic': f_stat,
                'p_value': p_val_var,
                'interpretation': 'Stable variance' if p_val_var > 0.05 else 'Unstable variance'
            }

        return results

    def trend_analysis(self, data: np.ndarray, time_index: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze trend in time series.
        
        Args:
            data: Time series data
            time_index: Time index (optional)
            
        Returns:
            Trend analysis results
        """
        data_clean = np.asarray(data)[~np.isnan(data)]

        if time_index is None:
            time_index = np.arange(len(data_clean))

        results = {}

        if HAS_SCIPY:
            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, data_clean)

            results['linear_trend'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err,
                'trend_direction': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'No trend',
                'significant': p_value < 0.05
            }

            # Mann-Kendall trend test (simplified)
            results['mann_kendall'] = self._mann_kendall_test(data_clean)

        return results

    def _mann_kendall_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Simplified Mann-Kendall trend test."""
        n = len(data)
        s = 0

        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1

        # Variance calculation (simplified)
        var_s = n * (n - 1) * (2 * n + 5) / 18

        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # Two-tailed p-value
        if HAS_SCIPY:
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            p_value = None

        return {
            'statistic': s,
            'z_score': z,
            'p_value': p_value,
            'trend': 'Increasing' if z > 0 else 'Decreasing' if z < 0 else 'No trend',
            'significant': p_value < 0.05 if p_value is not None else None
        }


# Utility functions for statistical analysis
def calculate_effect_size(data1: np.ndarray, data2: np.ndarray,
                         effect_type: EffectSizeType = EffectSizeType.COHENS_D) -> float:
    """
    Calculate effect size between two groups.
    
    Args:
        data1: First group data
        data2: Second group data
        effect_type: Type of effect size to calculate
        
    Returns:
        Effect size value
    """
    data1_clean = np.asarray(data1)[~np.isnan(data1)]
    data2_clean = np.asarray(data2)[~np.isnan(data2)]

    mean1, mean2 = np.mean(data1_clean), np.mean(data2_clean)

    if effect_type == EffectSizeType.COHENS_D:
        # Cohen's d
        n1, n2 = len(data1_clean), len(data2_clean)
        pooled_std = np.sqrt(((n1-1)*np.var(data1_clean, ddof=1) + (n2-1)*np.var(data2_clean, ddof=1)) / (n1+n2-2))
        return (mean1 - mean2) / pooled_std

    elif effect_type == EffectSizeType.HEDGES_G:
        # Hedges' g (bias-corrected Cohen's d)
        n1, n2 = len(data1_clean), len(data2_clean)
        pooled_std = np.sqrt(((n1-1)*np.var(data1_clean, ddof=1) + (n2-1)*np.var(data2_clean, ddof=1)) / (n1+n2-2))
        d = (mean1 - mean2) / pooled_std
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        return d * correction_factor

    elif effect_type == EffectSizeType.GLASS_DELTA:
        # Glass's delta
        return (mean1 - mean2) / np.std(data2_clean, ddof=1)

    else:
        raise ValueError(f"Unsupported effect size type: {effect_type}")


def interpret_effect_size(effect_size: float, effect_type: EffectSizeType = EffectSizeType.COHENS_D) -> str:
    """
    Interpret effect size magnitude.
    
    Args:
        effect_size: Effect size value
        effect_type: Type of effect size
        
    Returns:
        Interpretation string
    """
    abs_effect = abs(effect_size)

    if effect_type in [EffectSizeType.COHENS_D, EffectSizeType.HEDGES_G]:
        if abs_effect < 0.2:
            return "Negligible"
        elif abs_effect < 0.5:
            return "Small"
        elif abs_effect < 0.8:
            return "Medium"
        else:
            return "Large"

    elif effect_type == EffectSizeType.GLASS_DELTA:
        if abs_effect < 0.2:
            return "Negligible"
        elif abs_effect < 0.5:
            return "Small"
        elif abs_effect < 0.8:
            return "Medium"
        else:
            return "Large"

    else:
        return "Unknown interpretation"


def power_analysis(effect_size: float, sample_size: int, alpha: float = 0.05,
                  test_type: str = "two_sample_t") -> float:
    """
    Calculate statistical power for a given effect size and sample size.
    
    Args:
        effect_size: Expected effect size
        sample_size: Total sample size
        alpha: Significance level
        test_type: Type of statistical test
        
    Returns:
        Statistical power (1 - β)
    """
    if not HAS_SCIPY:
        return np.nan

    try:
        if test_type == "two_sample_t":
            # Two-sample t-test power calculation (approximation)
            n_per_group = sample_size // 2
            ncp = effect_size * np.sqrt(n_per_group / 2)  # Non-centrality parameter
            df = sample_size - 2
            t_crit = stats.t.ppf(1 - alpha/2, df)

            # Power = P(|T| > t_crit | H1 is true)
            power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
            return power

        else:
            # Generic approximation
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = effect_size * np.sqrt(sample_size/4) - z_alpha
            power = stats.norm.cdf(z_beta)
            return power

    except Exception:
        return np.nan
