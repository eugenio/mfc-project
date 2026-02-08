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
    UncertaintyQuantifier,
    MonteCarloAnalyzer,
    BayesianInference,
    PolynomialChaosAnalyzer,
)


class TestUncertaintyResult:
    def test_default_values(self):
        r = UncertaintyResult(
            method=UncertaintyMethod.MONTE_CARLO,
            parameter_names=["a"],
            output_names=["out"],
            n_samples=10,
        )
        assert r.parameter_samples is None
        assert r.output_samples is None
        assert r.output_mean is None
        assert r.posterior_samples is None
        assert r.evidence is None
        assert r.pce_coefficients is None
        assert r.computation_time == 0.0


def _make_params():
    return [
        UncertainParameter("a", DistributionType.NORMAL, {"mean": 1.0, "std": 0.1}),
        UncertainParameter("b", DistributionType.UNIFORM, {"low": 0.0, "high": 2.0}),
    ]


def _model_fn(params):
    return {"output": params[0] + params[1]}


class TestUncertaintyQuantifierBase:
    def test_sample_parameters(self):
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        samples = mc._sample_parameters(20)
        assert samples.shape == (20, 2)

    def test_evaluate_model_batch_sequential(self):
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        samples = mc._sample_parameters(5)
        outputs = mc._evaluate_model_batch(_model_fn, samples, parallel=False)
        assert "output" in outputs
        assert len(outputs["output"]) == 5

    def test_calculate_statistics(self):
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        output_samples = {"out": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        stats = mc._calculate_statistics(output_samples)
        assert stats["out"]["mean"] == pytest.approx(3.0)
        assert stats["out"]["std"] > 0

    def test_calculate_statistics_all_nan(self):
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        output_samples = {"out": np.array([np.nan, np.nan])}
        stats = mc._calculate_statistics(output_samples)
        assert np.isnan(stats["out"]["mean"])

    def test_calculate_confidence_intervals(self):
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        output_samples = {"out": np.random.normal(0, 1, 100)}
        ci = mc._calculate_confidence_intervals(output_samples)
        assert "out" in ci
        assert "0.95" in ci["out"]

    def test_calculate_confidence_intervals_empty(self):
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        output_samples = {"out": np.array([np.nan, np.nan])}
        ci = mc._calculate_confidence_intervals(output_samples)
        assert np.isnan(ci["out"]["0.95"][0])

    def test_calculate_percentiles(self):
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        output_samples = {"out": np.random.normal(0, 1, 100)}
        pctiles = mc._calculate_percentiles(output_samples)
        assert "out" in pctiles
        assert "p50" in pctiles["out"]

    def test_calculate_percentiles_empty(self):
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        output_samples = {"out": np.array([np.nan])}
        pctiles = mc._calculate_percentiles(output_samples)
        assert np.isnan(pctiles["out"]["p50"])

    def test_evaluate_model_batch_with_nan_on_failure(self):
        call_count = [0]

        def sometimes_failing(params):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("fail on second")
            return {"output": params[0]}

        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        samples = mc._sample_parameters(3)
        outputs = mc._evaluate_model_batch(sometimes_failing, samples, parallel=False)
        assert np.isnan(outputs["output"][1])


class ConcreteBayesianInference(BayesianInference):
    def propagate_uncertainty(self, model_function, n_samples, **kwargs):
        return UncertaintyResult(
            method=UncertaintyMethod.BAYESIAN_INFERENCE,
            parameter_names=self.parameter_names,
            output_names=[],
            n_samples=n_samples,
        )


class TestBayesianInference:
    def test_init_default_priors(self):
        params = _make_params()
        bi = ConcreteBayesianInference(params, random_seed=42)
        assert bi.prior_distributions is params

    def test_init_custom_priors(self):
        params = _make_params()
        priors = _make_params()
        bi = ConcreteBayesianInference(params, prior_distributions=priors, random_seed=42)
        assert bi.prior_distributions is priors

    def test_calibrate_parameters(self):
        params = _make_params()
        bi = ConcreteBayesianInference(params, random_seed=42)
        exp_data = {"output": np.array([2.0, 2.1, 1.9])}
        noise = {"output": 0.1}
        result = bi.calibrate_parameters(
            _model_fn, exp_data, noise, n_samples=10, n_chains=1
        )
        assert result.method == UncertaintyMethod.BAYESIAN_INFERENCE
        assert result.posterior_samples is not None
        assert result.evidence is not None

    def test_unsupported_algorithm(self):
        params = _make_params()
        bi = ConcreteBayesianInference(params, random_seed=42)
        exp_data = {"output": np.array([2.0])}
        noise = {"output": 0.1}
        with pytest.raises(NotImplementedError):
            bi.calibrate_parameters(
                _model_fn, exp_data, noise, n_samples=5, n_chains=1,
                algorithm="gibbs"
            )

    def test_metropolis_hastings_acceptance(self):
        params = [
            UncertainParameter("a", DistributionType.NORMAL, {"mean": 1.0, "std": 0.1}),
        ]
        bi = ConcreteBayesianInference(params, random_seed=42)
        exp_data = {"output": np.array([1.0])}

        def simple_model(p):
            return {"output": p[0]}

        result = bi.calibrate_parameters(
            simple_model, exp_data, {"output": 0.5}, n_samples=20, n_chains=1
        )
        assert result.posterior_samples.shape == (20, 1)


class TestPolynomialChaosAnalyzer:
    def test_propagate_uncertainty(self):
        params = _make_params()
        pca = PolynomialChaosAnalyzer(params, polynomial_order=2, random_seed=42)
        result = pca.propagate_uncertainty(_model_fn)
        assert result.method == UncertaintyMethod.POLYNOMIAL_CHAOS
        assert result.pce_coefficients is not None
        assert result.output_mean is not None
        assert result.output_std is not None

    def test_custom_n_samples(self):
        params = _make_params()
        pca = PolynomialChaosAnalyzer(params, polynomial_order=2, random_seed=42)
        result = pca.propagate_uncertainty(_model_fn, n_samples=30)
        assert result.n_samples == 30

    def test_calculate_n_terms(self):
        params = _make_params()
        pca = PolynomialChaosAnalyzer(params, polynomial_order=2, random_seed=42)
        n_terms = pca._calculate_n_terms()
        assert n_terms == 6  # C(2+2, 2) = 6

    def test_generate_multi_indices(self):
        params = _make_params()
        pca = PolynomialChaosAnalyzer(params, polynomial_order=2, random_seed=42)
        indices = pca._generate_multi_indices()
        assert len(indices) == 6

    def test_construct_polynomial_basis(self):
        params = _make_params()
        pca = PolynomialChaosAnalyzer(params, polynomial_order=2, random_seed=42)
        samples = pca._sample_parameters(10)
        basis = pca._construct_polynomial_basis(samples)
        assert basis.shape[0] == 10
        assert basis.shape[1] == pca._calculate_n_terms()

    def test_variance_contributions(self):
        params = _make_params()
        pca = PolynomialChaosAnalyzer(params, polynomial_order=2, random_seed=42)
        coeffs = np.array([1.0, 0.5, 0.3, 0.1, 0.05, 0.02])
        vc = pca._calculate_variance_contributions(coeffs)
        assert len(vc) == 5

    def test_pce_statistics(self):
        params = _make_params()
        pca = PolynomialChaosAnalyzer(params, polynomial_order=2, random_seed=42)
        coeffs = {"out": np.array([2.0, 0.5, 0.3, 0.1, 0.05, 0.02])}
        vc = {"out": np.array([0.25, 0.09, 0.01, 0.0025, 0.0004])}
        stats = pca._calculate_pce_statistics(coeffs, vc)
        assert stats["mean"]["out"] == pytest.approx(2.0)
        assert stats["variance"]["out"] > 0


class TestMonteCarloParallel:
    def test_parallel_evaluation(self):
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        samples = mc._sample_parameters(12)
        outputs = mc._evaluate_model_batch(_model_fn, samples, parallel=True)
        assert "output" in outputs
        assert len(outputs["output"]) == 12
