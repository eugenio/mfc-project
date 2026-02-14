"""Tests for config/sensitivity_analysis.py - coverage target 98%+."""
import sys
import os

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.sensitivity_analysis import (
    SensitivityAnalyzer, ParameterSpace, ParameterDefinition, ParameterBounds,
    SensitivityResult, SensitivityMethod, SamplingMethod, SensitivityVisualizer,
)


def make_params():
    return [
        ParameterDefinition(name="p1", bounds=ParameterBounds(0.0, 1.0, nominal_value=0.5),
            config_path=["a"], units="m"),
        ParameterDefinition(name="p2", bounds=ParameterBounds(1.0, 10.0, nominal_value=5.0),
            config_path=["b"], units="kg"),
    ]


def simple_model(params):
    if params.ndim == 1:
        params = params.reshape(1, -1)
    return {"y": np.sum(params, axis=1)}


@pytest.fixture
def pspace():
    return ParameterSpace(make_params())


@pytest.fixture
def analyzer(pspace):
    return SensitivityAnalyzer(pspace, simple_model, ["y"])


@pytest.mark.coverage_extra
class TestParameterBounds:
    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            ParameterBounds(10.0, 1.0)

    def test_nominal_out_of_range(self):
        with pytest.raises(ValueError):
            ParameterBounds(0.0, 1.0, nominal_value=5.0)


@pytest.mark.coverage_extra
class TestParameterSpace:
    def test_empty_params_raises(self):
        with pytest.raises(ValueError):
            ParameterSpace([])

    def test_sample_random(self, pspace):
        s = pspace.sample(10, SamplingMethod.RANDOM, seed=42)
        assert s.shape == (10, 2)

    def test_sample_lhs(self, pspace):
        s = pspace.sample(10, SamplingMethod.LATIN_HYPERCUBE, seed=42)
        assert s.shape == (10, 2)

    def test_sample_sobol(self, pspace):
        s = pspace.sample(8, SamplingMethod.SOBOL_SEQUENCE, seed=42)
        assert s.shape == (8, 2)

    def test_sample_halton(self, pspace):
        s = pspace.sample(10, SamplingMethod.HALTON, seed=42)
        assert s.shape == (10, 2)

    def test_sample_grid(self, pspace):
        s = pspace.sample(10, SamplingMethod.GRID)
        assert s.shape[1] == 2

    def test_sample_unknown(self, pspace):
        with pytest.raises(ValueError):
            pspace.sample(10, "unknown_method")

    def test_scale_normal_distribution(self):
        params = [ParameterDefinition(name="p1",
            bounds=ParameterBounds(0.0, 10.0, distribution="normal", nominal_value=5.0),
            config_path=["a"])]
        ps = ParameterSpace(params)
        s = ps.sample(10, SamplingMethod.RANDOM, seed=42)
        assert s.shape == (10, 1)

    def test_scale_lognormal_distribution(self):
        params = [ParameterDefinition(name="p1",
            bounds=ParameterBounds(0.1, 10.0, distribution="lognormal", nominal_value=1.0),
            config_path=["a"])]
        ps = ParameterSpace(params)
        s = ps.sample(10, SamplingMethod.RANDOM, seed=42)
        assert s.shape == (10, 1)

    def test_get_parameter_by_name(self, pspace):
        p = pspace.get_parameter_by_name("p1")
        assert p.name == "p1"

    def test_get_parameter_not_found(self, pspace):
        with pytest.raises(ValueError):
            pspace.get_parameter_by_name("nonexistent")


@pytest.mark.coverage_extra
class TestSensitivityAnalyzer:
    def test_oat_analysis(self, analyzer):
        result = analyzer.analyze_sensitivity(SensitivityMethod.ONE_AT_A_TIME, n_samples=10)
        assert result.local_sensitivities is not None

    def test_gradient_based(self, analyzer):
        result = analyzer.analyze_sensitivity(SensitivityMethod.GRADIENT_BASED)
        assert result.local_sensitivities is not None

    def test_morris(self, analyzer):
        result = analyzer.analyze_sensitivity(SensitivityMethod.MORRIS, n_samples=5)
        assert result.morris_means is not None

    def test_sobol(self, analyzer):
        result = analyzer.analyze_sensitivity(SensitivityMethod.SOBOL, n_samples=16)
        assert result.first_order_indices is not None

    def test_unsupported_method(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.analyze_sensitivity(SensitivityMethod.FAST)

    def test_cache_hit(self, analyzer):
        params = analyzer.parameter_space.nominal_values.reshape(1, -1)
        r1 = analyzer._evaluate_model(params)
        r2 = analyzer._evaluate_model(params)
        assert np.array_equal(r1["y"], r2["y"])

    def test_rank_parameters_total_order(self, analyzer):
        result = analyzer.analyze_sensitivity(SensitivityMethod.SOBOL, n_samples=16)
        ranking = analyzer.rank_parameters(result, "y", "total_order")
        assert len(ranking) == 2

    def test_rank_parameters_first_order(self, analyzer):
        result = analyzer.analyze_sensitivity(SensitivityMethod.SOBOL, n_samples=16)
        ranking = analyzer.rank_parameters(result, "y", "first_order")
        assert len(ranking) == 2

    def test_rank_parameters_morris(self, analyzer):
        result = analyzer.analyze_sensitivity(SensitivityMethod.MORRIS, n_samples=5)
        ranking = analyzer.rank_parameters(result, "y", "morris_mean_star")
        assert len(ranking) == 2

    def test_rank_parameters_local(self, analyzer):
        result = analyzer.analyze_sensitivity(SensitivityMethod.ONE_AT_A_TIME, n_samples=10)
        ranking = analyzer.rank_parameters(result, "y", "local")
        assert len(ranking) == 2

    def test_rank_parameters_unavailable_metric(self, analyzer):
        result = SensitivityResult(method=SensitivityMethod.SOBOL, parameter_names=["p1"], output_names=["y"])
        with pytest.raises(ValueError):
            analyzer.rank_parameters(result, "y", "nonexistent")


@pytest.mark.coverage_extra
class TestSensitivityVisualizer:
    def test_plot_sensitivity_indices(self, analyzer, tmp_path):
        result = analyzer.analyze_sensitivity(SensitivityMethod.SOBOL, n_samples=16)
        viz = SensitivityVisualizer()
        fig = viz.plot_sensitivity_indices(result, "y", str(tmp_path / "si.png"))
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_morris_results(self, analyzer, tmp_path):
        result = analyzer.analyze_sensitivity(SensitivityMethod.MORRIS, n_samples=5)
        viz = SensitivityVisualizer()
        fig = viz.plot_morris_results(result, "y", str(tmp_path / "m.png"))
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_morris_no_results(self):
        result = SensitivityResult(method=SensitivityMethod.MORRIS, parameter_names=[], output_names=[])
        viz = SensitivityVisualizer()
        with pytest.raises(ValueError):
            viz.plot_morris_results(result, "y")

    def test_plot_parameter_ranking(self, analyzer, tmp_path):
        result = analyzer.analyze_sensitivity(SensitivityMethod.SOBOL, n_samples=16)
        viz = SensitivityVisualizer()
        fig = viz.plot_parameter_ranking(result, "y", "total_order", top_n=2, save_path=str(tmp_path / "r.png"))
        import matplotlib.pyplot as plt
        plt.close(fig)
