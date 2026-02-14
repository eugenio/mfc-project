"""Additional tests for config/uncertainty_quantification.py to achieve 99%+ coverage.

Covers missing lines: 313-318 (parallel model evaluation exception handling),
636-637 (log_likelihood exception in Bayesian calibration).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import logging
from unittest.mock import MagicMock, patch, PropertyMock
from concurrent.futures import Future

import numpy as np
import pytest

import config.uncertainty_quantification as uq_mod

from config.uncertainty_quantification import (
    UncertaintyMethod,
    DistributionType,
    UncertainParameter,
    UncertaintyResult,
    MonteCarloAnalyzer,
    BayesianInference,
)


def _make_params():
    return [
        UncertainParameter("a", DistributionType.NORMAL, {"mean": 1.0, "std": 0.1}),
        UncertainParameter("b", DistributionType.UNIFORM, {"low": 0.0, "high": 2.0}),
    ]


def _model_fn(params):
    return {"output": params[0] + params[1]}


# ---------------------------------------------------------------------------
# Test parallel model evaluation exception handling (lines 313-318)
# ---------------------------------------------------------------------------

@pytest.mark.coverage_extra
class TestParallelModelEvalFailure:
    """Cover the except branch in _evaluate_model_batch parallel path.

    The parallel path uses ProcessPoolExecutor and as_completed.
    Local functions can't be pickled, so we mock the executor and futures
    to simulate the parallel execution with controlled exceptions.
    """

    def test_parallel_exception_with_output_names_set(self):
        """Cover lines 313-318: exception when output_names is already set.

        First future succeeds (sets output_names), subsequent futures raise.
        """
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        samples = mc._sample_parameters(15)

        # Create futures: first succeeds, rest raise
        future_ok = MagicMock(spec=Future)
        future_ok.result.return_value = {"output": 1.5}

        future_fail = MagicMock(spec=Future)
        future_fail.result.side_effect = RuntimeError("Intentional failure")

        # Build list: first succeeds, rest fail
        futures_list = [future_ok] + [MagicMock(spec=Future) for _ in range(14)]
        for f in futures_list[1:]:
            f.result.side_effect = RuntimeError("Intentional failure")

        # as_completed yields futures in completion order; simulate first ok, then fails
        def fake_as_completed(fs):
            yield future_ok
            for f in futures_list[1:]:
                yield f

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.side_effect = lambda fn, params: futures_list.pop(0) if futures_list else MagicMock()

        # Rebuild futures_list for submit calls
        all_futures = [future_ok] + [MagicMock(spec=Future) for _ in range(14)]
        for f in all_futures[1:]:
            f.result.side_effect = RuntimeError("Intentional failure")

        submit_idx = [0]
        def fake_submit(fn, params):
            idx = submit_idx[0]
            submit_idx[0] += 1
            return all_futures[idx]

        mock_executor.submit.side_effect = fake_submit

        def fake_as_completed_v2(fs):
            # Yield the first (success) future, then the rest (failures)
            yield all_futures[0]
            for f in all_futures[1:]:
                yield f

        with patch.object(uq_mod, "ProcessPoolExecutor", return_value=mock_executor), \
             patch.object(uq_mod, "as_completed", side_effect=fake_as_completed_v2):
            outputs = mc._evaluate_model_batch(_model_fn, samples, parallel=True)

        assert "output" in outputs
        assert len(outputs["output"]) == 15
        # First value should be set, rest should be NaN
        nan_count = np.sum(np.isnan(outputs["output"]))
        assert nan_count == 14

    def test_parallel_exception_with_output_names_none(self):
        """Cover lines 313-314: exception when output_names is still None.

        All futures raise, so output_names never gets set. The 'if output_names
        is not None' check on line 316 takes the false branch (no NaN setting).
        """
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        samples = mc._sample_parameters(15)

        # All futures raise
        all_futures = [MagicMock(spec=Future) for _ in range(15)]
        for f in all_futures:
            f.result.side_effect = RuntimeError("All fail")

        submit_idx = [0]
        def fake_submit(fn, params):
            idx = submit_idx[0]
            submit_idx[0] += 1
            return all_futures[idx]

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.side_effect = fake_submit

        def fake_as_completed(fs):
            for f in all_futures:
                yield f

        with patch.object(uq_mod, "ProcessPoolExecutor", return_value=mock_executor), \
             patch.object(uq_mod, "as_completed", side_effect=fake_as_completed):
            outputs = mc._evaluate_model_batch(_model_fn, samples, parallel=True)

        # output_names was never set, so all_outputs stays empty
        assert outputs == {}

    def test_parallel_mixed_success_failure_pattern(self):
        """Cover both branches: some succeed, some fail in interleaved pattern."""
        mc = MonteCarloAnalyzer(_make_params(), random_seed=42)
        samples = mc._sample_parameters(15)

        all_futures = [MagicMock(spec=Future) for _ in range(15)]
        # Pattern: success, fail, success, fail, ...
        for i, f in enumerate(all_futures):
            if i % 2 == 0:
                f.result.return_value = {"output": float(i)}
            else:
                f.result.side_effect = RuntimeError("Fail")

        submit_idx = [0]
        def fake_submit(fn, params):
            idx = submit_idx[0]
            submit_idx[0] += 1
            return all_futures[idx]

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.side_effect = fake_submit

        def fake_as_completed(fs):
            for f in all_futures:
                yield f

        with patch.object(uq_mod, "ProcessPoolExecutor", return_value=mock_executor), \
             patch.object(uq_mod, "as_completed", side_effect=fake_as_completed):
            outputs = mc._evaluate_model_batch(_model_fn, samples, parallel=True)

        assert "output" in outputs
        assert len(outputs["output"]) == 15


# ---------------------------------------------------------------------------
# Test log_likelihood exception in Bayesian calibration (lines 636-637)
# ---------------------------------------------------------------------------

class ConcreteBayesianInference(BayesianInference):
    """Concrete subclass to instantiate abstract BayesianInference."""

    def propagate_uncertainty(self, model_function, n_samples, **kwargs):
        return UncertaintyResult(
            method=UncertaintyMethod.BAYESIAN_INFERENCE,
            parameter_names=self.parameter_names,
            output_names=[],
            n_samples=n_samples,
        )


@pytest.mark.coverage_extra
class TestLogLikelihoodException:
    """Cover the except block in log_likelihood that returns -np.inf."""

    def test_model_exception_in_likelihood(self):
        """When model_function raises, log_likelihood should return -inf."""
        params = [
            UncertainParameter(
                "a", DistributionType.NORMAL, {"mean": 1.0, "std": 0.1},
            ),
        ]
        bi = ConcreteBayesianInference(params, random_seed=42)

        call_count = [0]

        def sometimes_failing_model(p):
            call_count[0] += 1
            if call_count[0] % 5 == 0:
                raise RuntimeError("Model crashed")
            return {"output": p[0]}

        exp_data = {"output": np.array([1.0])}
        noise = {"output": 0.5}

        # Run with enough samples that some will hit the exception
        result = bi.calibrate_parameters(
            sometimes_failing_model,
            exp_data,
            noise,
            n_samples=20,
            n_chains=1,
        )
        # Should still produce posterior samples despite some failures
        assert result.posterior_samples is not None
        assert result.posterior_samples.shape == (20, 1)

    def test_model_always_fails_in_likelihood(self):
        """When model always fails, log_likelihood always returns -inf."""
        params = [
            UncertainParameter(
                "a", DistributionType.NORMAL, {"mean": 1.0, "std": 0.1},
            ),
        ]
        bi = ConcreteBayesianInference(params, random_seed=42)

        def always_failing_model(p):
            raise RuntimeError("Always fails")

        exp_data = {"output": np.array([1.0])}
        noise = {"output": 0.5}

        # Should not crash; all proposals will be rejected
        result = bi.calibrate_parameters(
            always_failing_model,
            exp_data,
            noise,
            n_samples=10,
            n_chains=1,
        )
        assert result.posterior_samples is not None
