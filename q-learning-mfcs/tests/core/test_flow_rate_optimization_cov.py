"""Tests for flow_rate_optimization.py."""
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

with patch('matplotlib.pyplot.savefig'), \
     patch('matplotlib.pyplot.close'), \
     patch('path_config.get_figure_path', return_value='/tmp/fig.png'):
    from flow_rate_optimization import (
        MFCFlowOptimizer,
        MFCParameters,
        RealisticFlowOptimizer,
        RealisticMFCParameters,
        plot_optimization_results,
        run_optimization,
    )


class TestMFCParameters:
    def test_defaults(self):
        p = MFCParameters()
        assert p.V_a == 5.5e-5


class TestMFCFlowOptimizer:
    @pytest.fixture
    def opt(self):
        return MFCFlowOptimizer()

    def test_residence_time(self, opt):
        assert opt.residence_time(1e-5) > 0

    def test_steady_state_acetate(self, opt):
        assert opt.steady_state_acetate(1e-5) >= 0

    def test_steady_state_with_biofilm(self, opt):
        assert opt.steady_state_acetate(1e-5, X=0.5, biofilm=1.5) >= 0

    def test_substrate_consumption_rate(self, opt):
        assert isinstance(opt.substrate_consumption_rate(1e-5, 0.5), float)

    def test_power_output(self, opt):
        assert opt.power_output(1e-5) >= 0

    def test_power_output_biofilm(self, opt):
        assert opt.power_output(1e-5, X=0.5, biofilm=1.5) >= 0

    def test_biofilm_growth_factor(self, opt):
        f = opt.biofilm_growth_factor(3600)
        assert 1.0 <= f <= opt.params.biofilm_max

    def test_objective_valid(self, opt):
        assert isinstance(opt.objective_function(1e-5), float)

    def test_objective_too_low(self, opt):
        assert opt.objective_function(1e-7) == 1e6

    def test_objective_too_high(self, opt):
        assert opt.objective_function(1e-2) == 1e6

    def test_objective_low_eff(self, opt):
        assert isinstance(opt.objective_function(5e-5, time_hours=1), float)

    def test_optimize(self, opt):
        r = opt.optimize_flow_rate()
        assert r["Q_optimal"] > 0

    def test_analyze(self, opt):
        r = opt.analyze_flow_range(n_points=10)
        assert len(r["power"]) == 10


class TestRealisticFlowOptimizer:
    @pytest.fixture
    def ropt(self):
        return RealisticFlowOptimizer()

    def test_residence_time(self, ropt):
        assert ropt.residence_time(1e-5) > 0

    def test_biofilm_factor(self, ropt):
        assert 1.0 <= ropt.biofilm_factor(3600) <= ropt.params.biofilm_max

    def test_steady_state(self, ropt):
        c, x = ropt.steady_state_concentrations(1e-5)
        assert c >= 0 and x >= 0

    def test_neg_discriminant(self, ropt):
        ropt.params.V_a = 1e10
        c, x = ropt.steady_state_concentrations(1e-10, biofilm=100)
        assert c >= 0

    def test_power(self, ropt):
        assert ropt.calculate_power(1e-5, 0.5, 1.0, 1.0) >= 0

    def test_efficiency(self, ropt):
        assert 0 <= ropt.substrate_efficiency(0.5) <= 1

    def test_objective_valid(self, ropt):
        assert isinstance(ropt.objective_function(1e-5), float)

    def test_objective_bounds(self, ropt):
        assert ropt.objective_function(1e-7) == 1e10

    def test_optimize(self, ropt):
        assert ropt.optimize()["Q_optimal"] > 0

    def test_analyze(self, ropt):
        assert len(ropt.analyze_flow_range(10)["power"]) == 10


class TestPlotAndRun:
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('path_config.get_figure_path', return_value='/tmp/f.png')
    def test_plot_basic(self, *_):
        o = MFCFlowOptimizer()
        plot_optimization_results(o.analyze_flow_range(10), o.optimize_flow_rate())

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('path_config.get_figure_path', return_value='/tmp/f.png')
    def test_plot_realistic(self, *_):
        o = RealisticFlowOptimizer()
        plot_optimization_results(o.analyze_flow_range(10), o.optimize(), realistic=True)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('path_config.get_figure_path', return_value='/tmp/f.png')
    def test_run_basic(self, *_):
        assert run_optimization(False)[0]["Q_optimal"] > 0

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('path_config.get_figure_path', return_value='/tmp/f.png')
    def test_run_realistic(self, *_):
        assert run_optimization(True)[0]["Q_optimal"] > 0


class TestMain:
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('path_config.get_figure_path', return_value='/tmp/f.png')
    @patch('sys.argv', ['prog'])
    def test_main_basic(self, *_):
        from flow_rate_optimization import main
        main()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('path_config.get_figure_path', return_value='/tmp/f.png')
    @patch('sys.argv', ['prog', '--realistic'])
    def test_main_realistic(self, *_):
        from flow_rate_optimization import main
        main()
