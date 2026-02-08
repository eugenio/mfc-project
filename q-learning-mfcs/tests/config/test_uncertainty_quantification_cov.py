import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from config.uncertainty_quantification import (
    UncertaintyMethod,
    DistributionType,
    UncertainParameter,
    UncertaintyResult,
    MonteCarloAnalyzer,
    BayesianInference,
    PolynomialChaosAnalyzer,
    calculate_reliability,
    calculate_sensitivity_indices_from_pce,
    bootstrap_uncertainty,
)


class TestEnums:
    def test_uncertainty_method(self):
        assert UncertaintyMethod.MONTE_CARLO.value == "monte_carlo"
        assert UncertaintyMethod.LATIN_HYPERCUBE.value == "latin_hypercube"
        assert UncertaintyMethod.POLYNOMIAL_CHAOS.value == "polynomial_chaos"
        assert UncertaintyMethod.BAYESIAN_INFERENCE.value == "bayesian_inference"
        assert UncertaintyMethod.BOOTSTRAP.value == "bootstrap"
        assert UncertaintyMethod.QUASI_MONTE_CARLO.value == "quasi_monte_carlo"

    def test_distribution_type(self):
        assert DistributionType.NORMAL.value == "normal"
        assert DistributionType.UNIFORM.value == "uniform"
        assert DistributionType.LOGNORMAL.value == "lognormal"
        assert DistributionType.BETA.value == "beta"
        assert DistributionType.GAMMA.value == "gamma"
        assert DistributionType.EXPONENTIAL.value == "exponential"
        assert DistributionType.TRIANGULAR.value == "triangular"


class TestUncertainParameter:
    def test_normal_sample(self):
        p = UncertainParameter("x", DistributionType.NORMAL, {"mean": 0.0, "std": 1.0})
        s = p.sample(100, seed=42)
        assert len(s) == 100

    def test_uniform_sample(self):
        p = UncertainParameter("x", DistributionType.UNIFORM, {"low": 0.0, "high": 1.0})
        s = p.sample(100, seed=42)
        assert np.all(s >= 0.0) and np.all(s <= 1.0)

    def test_lognormal_sample(self):
        p = UncertainParameter("x", DistributionType.LOGNORMAL, {"mean": 0.0, "sigma": 0.5})
        s = p.sample(50, seed=42)
        assert len(s) == 50

    def test_beta_sample(self):
        p = UncertainParameter("x", DistributionType.BETA, {"alpha": 2.0, "beta": 5.0})
        s = p.sample(50, seed=42)
        assert len(s) == 50

    def test_gamma_sample(self):
        p = UncertainParameter("x", DistributionType.GAMMA, {"shape": 2.0, "scale": 1.0})
        s = p.sample(50, seed=42)
        assert len(s) == 50

    def test_exponential_sample(self):
        p = UncertainParameter("x", DistributionType.EXPONENTIAL, {"scale": 1.0})
        s = p.sample(50, seed=42)
        assert len(s) == 50

    def test_triangular_sample(self):
        p = UncertainParameter(
            "x", DistributionType.TRIANGULAR, {"left": 0.0, "mode": 0.5, "right": 1.0}
        )
        s = p.sample(50, seed=42)
        assert len(s) == 50

    def test_bounds_clipping(self):
        p = UncertainParameter(
            "x", DistributionType.NORMAL, {"mean": 0.0, "std": 10.0}, bounds=(-1.0, 1.0)
        )
        s = p.sample(1000, seed=42)
        assert np.all(s >= -1.0) and np.all(s <= 1.0)

    def test_no_seed(self):
        p = UncertainParameter("x", DistributionType.NORMAL, {"mean": 0.0, "std": 1.0})
        s = p.sample(10)
        assert len(s) == 10

    def test_pdf_normal(self):
        p = UncertainParameter("x", DistributionType.NORMAL, {"mean": 0.0, "std": 1.0})
        x = np.array([0.0, 1.0])
        pdf_vals = p.pdf(x)
        assert len(pdf_vals) == 2
        assert pdf_vals[0] > pdf_vals[1]

    def test_pdf_uniform(self):
        p = UncertainParameter("x", DistributionType.UNIFORM, {"low": 0.0, "high": 1.0})
        x = np.array([0.5])
        pdf_vals = p.pdf(x)
        assert pdf_vals[0] == pytest.approx(1.0)

    def test_pdf_lognormal(self):
        p = UncertainParameter("x", DistributionType.LOGNORMAL, {"mean": 0.0, "sigma": 0.5})
        x = np.array([1.0])
        pdf_vals = p.pdf(x)
        assert pdf_vals[0] > 0

    def test_pdf_unsupported(self):
        p = UncertainParameter("x", DistributionType.BETA, {"alpha": 2.0, "beta": 5.0})
        with pytest.raises(NotImplementedError):
            p.pdf(np.array([0.5]))

    def test_unsupported_distribution_sample(self):
        p = UncertainParameter.__new__(UncertainParameter)
        p.name = "x"
        p.distribution = "invalid_dist"
        p.parameters = {}
        p.bounds = None
        p.description = ""
        with pytest.raises(ValueError, match="Unsupported distribution"):
            p.sample(10)


class TestMonteCarloAnalyzer:
    def _make_params(self):
        return [
            UncertainParameter("a", DistributionType.NORMAL, {"mean": 1.0, "std": 0.1}),
            UncertainParameter("b", DistributionType.UNIFORM, {"low": 0.0, "high": 2.0}),
        ]

    def _model_fn(self, params):
        return {"output": params[0] + params[1]}

    def test_random_sampling(self):
        mc = MonteCarloAnalyzer(self._make_params(), sampling_method="random", random_seed=42)
        result = mc.propagate_uncertainty(self._model_fn, n_samples=20, parallel=False)
        assert result.method == UncertaintyMethod.MONTE_CARLO
        assert result.n_samples == 20
        assert result.output_mean is not None
        assert "output" in result.output_mean

    def test_lhs_sampling(self):
        mc = MonteCarloAnalyzer(self._make_params(), sampling_method="latin_hypercube", random_seed=42)
        result = mc.propagate_uncertainty(self._model_fn, n_samples=16, parallel=False)
        assert result.n_samples == 16

    def test_sobol_sampling(self):
        mc = MonteCarloAnalyzer(self._make_params(), sampling_method="sobol", random_seed=42)
        result = mc.propagate_uncertainty(self._model_fn, n_samples=16, parallel=False)
        assert result.n_samples == 16

    def test_statistics_calculated(self):
        mc = MonteCarloAnalyzer(self._make_params(), random_seed=42)
        result = mc.propagate_uncertainty(self._model_fn, n_samples=50, parallel=False)
        assert result.output_std is not None
        assert result.output_var is not None
        assert result.output_skewness is not None
        assert result.output_kurtosis is not None

    def test_confidence_intervals(self):
        mc = MonteCarloAnalyzer(self._make_params(), random_seed=42)
        result = mc.propagate_uncertainty(self._model_fn, n_samples=50, parallel=False)
        assert result.confidence_intervals is not None
        assert "output" in result.confidence_intervals

    def test_percentiles(self):
        mc = MonteCarloAnalyzer(self._make_params(), random_seed=42)
        result = mc.propagate_uncertainty(self._model_fn, n_samples=50, parallel=False)
        assert result.percentiles is not None

    def test_correlations(self):
        mc = MonteCarloAnalyzer(self._make_params(), random_seed=42)
        result = mc.propagate_uncertainty(self._model_fn, n_samples=50, parallel=False)
        assert result.parameter_correlations is not None
        assert result.output_correlations is not None
        assert result.parameter_output_correlations is not None

    def test_model_failure_handling(self):
        def failing_model(params):
            raise RuntimeError("fail")

        mc = MonteCarloAnalyzer(self._make_params(), random_seed=42)
        with pytest.raises((ValueError, RuntimeError)):
            mc.propagate_uncertainty(failing_model, n_samples=5, parallel=False)


class TestUtilityFunctions:
    def test_calculate_reliability_greater(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = calculate_reliability(samples, 3.0, "greater")
        assert r == pytest.approx(0.6)

    def test_calculate_reliability_less(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = calculate_reliability(samples, 3.0, "less")
        assert r == pytest.approx(0.6)

    def test_bootstrap_uncertainty(self):
        np.random.seed(42)
        samples = np.random.normal(0, 1, 100)
        stat, ci = bootstrap_uncertainty(samples, np.mean, n_bootstrap=100)
        assert ci[0] < ci[1]

    def test_sensitivity_indices_from_pce(self):
        coeffs = np.array([1.0, 0.5, 0.3, 0.1])
        multi_indices = [[0, 0], [1, 0], [0, 1], [1, 1]]
        indices = calculate_sensitivity_indices_from_pce(coeffs, multi_indices)
        assert "S1_0" in indices
        assert "S1_1" in indices

    def test_sensitivity_indices_zero_variance(self):
        coeffs = np.array([1.0])
        multi_indices = [[0, 0]]
        indices = calculate_sensitivity_indices_from_pce(coeffs, multi_indices)
        assert len(indices) == 2

    def test_sensitivity_indices_empty(self):
        coeffs = np.array([1.0])
        multi_indices = []
        indices = calculate_sensitivity_indices_from_pce(coeffs, multi_indices)
        assert len(indices) == 0
