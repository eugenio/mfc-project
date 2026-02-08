import sys
import os
import importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Force-clean stale mocks from other test files and ensure fresh imports
for _stale in ["path_config", "flow_rate_optimization"]:
    if _stale in sys.modules:
        del sys.modules[_stale]

mock_pc = MagicMock()
mock_pc.get_figure_path = MagicMock(return_value="/tmp/fig.png")
sys.modules["path_config"] = mock_pc

from flow_rate_optimization import (
    MFCParameters, MFCFlowOptimizer,
    RealisticMFCParameters, RealisticFlowOptimizer,
    plot_optimization_results, run_optimization,
)


class TestMFCParameters:
    def test_defaults(self):
        p = MFCParameters()
        assert p.V_a == 5.5e-5
        assert p.A_m == 5.0e-4
        assert p.n_cells == 5
        assert p.C_AC_in == 1.56


class TestMFCFlowOptimizer:
    def setup_method(self):
        self.opt = MFCFlowOptimizer()

    def test_init_default(self):
        assert self.opt.params is not None

    def test_init_custom(self):
        p = MFCParameters(n_cells=3)
        o = MFCFlowOptimizer(p)
        assert o.params.n_cells == 3

    def test_residence_time(self):
        tau = self.opt.residence_time(1e-5)
        assert tau > 0

    def test_steady_state_acetate(self):
        c = self.opt.steady_state_acetate(1e-5)
        assert c >= 0

    def test_steady_state_acetate_custom(self):
        c = self.opt.steady_state_acetate(1e-5, X=2.0, biofilm=1.5)
        assert c >= 0

    def test_substrate_consumption_rate(self):
        r = self.opt.substrate_consumption_rate(1e-5, 0.5)
        assert isinstance(r, float)

    def test_power_output(self):
        p = self.opt.power_output(1e-5)
        assert p >= 0

    def test_power_output_custom(self):
        p = self.opt.power_output(1e-5, X=2.0, biofilm=1.5)
        assert p >= 0

    def test_biofilm_growth_factor(self):
        f = self.opt.biofilm_growth_factor(3600)
        assert f >= 1.0
        assert f <= 2.0

    def test_biofilm_growth_factor_long(self):
        f = self.opt.biofilm_growth_factor(360000)
        assert f <= 2.0

    def test_objective_function(self):
        obj = self.opt.objective_function(1e-5)
        assert isinstance(obj, float)

    def test_objective_function_extreme_low(self):
        obj = self.opt.objective_function(1e-7)
        assert obj == 1e6

    def test_objective_function_extreme_high(self):
        obj = self.opt.objective_function(1e-2)
        assert obj == 1e6

    def test_optimize_flow_rate(self):
        result = self.opt.optimize_flow_rate()
        assert "Q_optimal" in result
        assert "Q_optimal_mL_h" in result
        assert "residence_time_min" in result
        assert "power_W" in result
        assert "substrate_efficiency" in result
        assert result["Q_optimal"] > 0

    def test_analyze_flow_range(self):
        results = self.opt.analyze_flow_range(n_points=10)
        assert len(results["Q"]) == 10
        assert len(results["power"]) == 10
        assert len(results["efficiency"]) == 10


class TestRealisticMFCParameters:
    def test_defaults(self):
        p = RealisticMFCParameters()
        assert p.V_cell_max == 0.8
        assert p.i_max == 10.0


class TestRealisticFlowOptimizer:
    def setup_method(self):
        self.opt = RealisticFlowOptimizer()

    def test_init_default(self):
        assert self.opt.params is not None

    def test_init_custom(self):
        p = RealisticMFCParameters(n_cells=3)
        o = RealisticFlowOptimizer(p)
        assert o.params.n_cells == 3

    def test_residence_time(self):
        tau = self.opt.residence_time(1e-5)
        assert tau > 0

    def test_biofilm_factor(self):
        bf = self.opt.biofilm_factor(3600)
        assert bf >= 1.0
        assert bf <= 2.0

    def test_steady_state_concentrations(self):
        c, x = self.opt.steady_state_concentrations(1e-5)
        assert c >= 0
        assert x >= 0

    def test_steady_state_concentrations_biofilm(self):
        c, x = self.opt.steady_state_concentrations(1e-5, biofilm=1.5)
        assert c >= 0

    def test_steady_state_concentrations_negative_disc(self):
        # Use extreme params to potentially get negative discriminant
        p = RealisticMFCParameters(V_a=1e-10, C_AC_in=0.001)
        o = RealisticFlowOptimizer(p)
        c, x = o.steady_state_concentrations(1e-5)
        assert c >= 0

    def test_calculate_power(self):
        p = self.opt.calculate_power(1e-5, 0.5, 1.0, 1.0)
        assert p >= 0

    def test_calculate_power_max_current(self):
        p = self.opt.calculate_power(1e-5, 1.5, 5.0, 1.0)
        assert p >= 0

    def test_substrate_efficiency(self):
        eff = self.opt.substrate_efficiency(0.5)
        assert 0 <= eff <= 1.0

    def test_objective_function(self):
        obj = self.opt.objective_function(1e-5)
        assert isinstance(obj, float)

    def test_objective_function_extreme_low(self):
        obj = self.opt.objective_function(1e-7)
        assert obj == 1e10

    def test_objective_function_extreme_high(self):
        obj = self.opt.objective_function(1e-3)
        assert obj == 1e10

    def test_objective_function_short_tau(self):
        obj = self.opt.objective_function(4.9e-5)  # Very fast flow
        assert isinstance(obj, float)

    def test_optimize(self):
        result = self.opt.optimize()
        assert "Q_optimal" in result
        assert "X" in result
        assert "biofilm_factor" in result
        assert result["Q_optimal"] > 0

    def test_analyze_flow_range(self):
        results = self.opt.analyze_flow_range(n_points=10)
        assert len(results["Q"]) == 10
        assert len(results["power"]) == 10


def _mock_axes_2x3():
    """Create a mock axes array that supports [i, j] indexing."""
    axes = MagicMock()
    mock_ax = MagicMock()
    axes.__getitem__ = MagicMock(return_value=mock_ax)
    return axes


class TestPlotOptimizationResults:
    @patch("flow_rate_optimization.plt")
    def test_basic_mode(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), _mock_axes_2x3())
        opt = MFCFlowOptimizer()
        results = opt.analyze_flow_range(n_points=10)
        optimal = opt.optimize_flow_rate()
        plot_optimization_results(results, optimal, realistic=False)

    @patch("flow_rate_optimization.plt")
    def test_realistic_mode(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), _mock_axes_2x3())
        opt = RealisticFlowOptimizer()
        results = opt.analyze_flow_range(n_points=10)
        optimal = opt.optimize()
        plot_optimization_results(results, optimal, realistic=True)


class TestRunOptimization:
    @patch("flow_rate_optimization.plt")
    def test_basic(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), _mock_axes_2x3())
        optimal, results = run_optimization(realistic=False)
        assert "Q_optimal" in optimal
        assert "Q" in results

    @patch("flow_rate_optimization.plt")
    def test_realistic(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), _mock_axes_2x3())
        optimal, results = run_optimization(realistic=True)
        assert "Q_optimal" in optimal
        assert "Q" in results

class TestMain:
    @patch("flow_rate_optimization.plt")
    def test_basic(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), _mock_axes_2x3())
        with patch("sys.argv", ["prog"]):
            from flow_rate_optimization import main
            main()

    @patch("flow_rate_optimization.plt")
    def test_realistic(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), _mock_axes_2x3())
        with patch("sys.argv", ["prog", "--realistic"]):
            from flow_rate_optimization import main
            main()


class TestNegativeDiscriminant:
    def test_realistic_negative_disc(self):
        p = RealisticMFCParameters(V_a=1e-12, C_AC_in=0.0001, K_AC=1e-8)
        o = RealisticFlowOptimizer(p)
        c, x = o.steady_state_concentrations(1e-3)
        assert c >= 0
