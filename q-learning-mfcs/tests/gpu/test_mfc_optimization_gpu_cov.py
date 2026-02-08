"""Tests for mfc_optimization_gpu module - targeting 98%+ coverage."""
import sys
import os
import json
import time
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock matplotlib to avoid display issues
import matplotlib
matplotlib.use("Agg")

# Module imports gpu_acceleration at top level
import mfc_optimization_gpu as mog


class TestMFCOptimizationSimulation:
    def _make_sim(self, use_gpu=False):
        """Create a short simulation for testing."""
        # Override the default to use small arrays
        with patch.object(mog.MFCOptimizationSimulation, '__init__', lambda self_, ug=False: None):
            sim = mog.MFCOptimizationSimulation.__new__(mog.MFCOptimizationSimulation)
        sim.use_gpu = use_gpu
        sim.num_cells = 3
        sim.dt = 10.0
        sim.total_time = 100.0  # Very short
        sim.num_steps = int(sim.total_time / sim.dt)  # 10 steps
        sim.V_a = 5.5e-5
        sim.A_m = 5.0e-4
        sim.F = 96485.0
        sim.r_max = 5.787e-5
        sim.K_AC = 0.592
        sim.K_dec = 8.33e-4
        sim.Y_ac = 0.05
        sim.optimal_biofilm_thickness = 1.3
        sim.flow_rate_bounds = (5.0e-6, 50.0e-6)
        sim.w_power = 0.5
        sim.w_biofilm = 0.3
        sim.w_substrate = 0.2
        sim.initialize_arrays()
        return sim

    def test_init(self):
        sim = self._make_sim()
        assert sim.num_cells == 3
        assert sim.num_steps == 10

    def test_initialize_arrays(self):
        sim = self._make_sim()
        assert sim.cell_voltages.shape == (10, 3)
        assert sim.biofilm_thickness.shape == (10, 3)
        assert sim.stack_voltages.shape == (10,)
        assert sim.biofilm_thickness[0, 0] == 1.0
        assert sim.acetate_concentrations[0, 0] == 1.56

    def test_biofilm_factor_numpy(self):
        sim = self._make_sim(use_gpu=False)
        factor = sim.biofilm_factor(1.3)
        assert factor == pytest.approx(1.0, abs=0.01)  # At optimal, delta_opt ~ 0
        factor2 = sim.biofilm_factor(2.3)
        assert factor2 > 1.0  # Away from optimal

    def test_biofilm_factor_gpu(self):
        sim = self._make_sim(use_gpu=True)
        factor = sim.biofilm_factor(np.float64(1.3))
        assert float(factor) == pytest.approx(1.0, abs=0.01)

    def test_reaction_rate_numpy(self):
        sim = self._make_sim(use_gpu=False)
        rate = sim.reaction_rate(1.0, 1.3)
        assert rate > 0

    def test_reaction_rate_gpu(self):
        sim = self._make_sim(use_gpu=True)
        rate = sim.reaction_rate(np.float64(1.0), np.float64(1.3))
        assert float(rate) > 0

    def test_reaction_rate_at_optimal(self):
        sim = self._make_sim(use_gpu=False)
        rate_opt = sim.reaction_rate(1.0, 1.3)
        rate_off = sim.reaction_rate(1.0, 2.0)
        # At optimal thickness, should get enhancement
        assert rate_opt > rate_off

    def test_update_cell_numpy(self):
        sim = self._make_sim(use_gpu=False)
        result = sim.update_cell(0, 1.56, 20.0e-6, 1.3)
        assert "outlet_concentration" in result
        assert "current_density" in result
        assert "voltage" in result
        assert "power" in result
        assert "substrate_consumed" in result
        assert result["outlet_concentration"] <= 1.56

    def test_update_cell_gpu(self):
        sim = self._make_sim(use_gpu=True)
        result = sim.update_cell(0, np.float64(1.56), np.float64(20.0e-6), np.float64(1.3))
        assert float(result["outlet_concentration"]) <= 1.56

    def test_update_biofilm_numpy(self):
        sim = self._make_sim(use_gpu=False)
        sim.update_biofilm(1, 10.0)
        assert all(0.5 <= sim.biofilm_thickness[1, i] <= 3.0 for i in range(3))

    def test_update_biofilm_gpu(self):
        sim = self._make_sim(use_gpu=True)
        sim.update_biofilm(1, 10.0)
        assert all(0.5 <= float(sim.biofilm_thickness[1, i]) <= 3.0 for i in range(3))

    def test_update_biofilm_above_optimal(self):
        sim = self._make_sim(use_gpu=False)
        sim.biofilm_thickness[0, :] = 2.0  # Above optimal
        sim.update_biofilm(1, 10.0)

    def test_update_biofilm_below_optimal(self):
        sim = self._make_sim(use_gpu=False)
        sim.biofilm_thickness[0, :] = 0.5  # Below optimal * 0.8
        sim.update_biofilm(1, 10.0)

    def test_calculate_objective_function_numpy(self):
        sim = self._make_sim(use_gpu=False)
        # Set some values
        sim.stack_powers[1] = 2.0
        sim.biofilm_thickness[1, :] = 1.3
        sim.acetate_concentrations[1, :] = 0.5
        obj = sim.calculate_objective_function(1)
        assert 0 <= obj <= 1.5  # Weighted sum of objectives

    def test_calculate_objective_function_gpu(self):
        sim = self._make_sim(use_gpu=True)
        sim.stack_powers[1] = 2.0
        sim.biofilm_thickness[1, :] = 1.3
        sim.acetate_concentrations[1, :] = 0.5
        obj = sim.calculate_objective_function(1)
        assert float(obj) >= 0

    def test_calculate_flow_rate_phase1(self):
        sim = self._make_sim()
        # Phase 1: 0-200 hours
        fr = sim.calculate_flow_rate(100 * 3600)  # 100 hours
        assert sim.flow_rate_bounds[0] <= fr <= sim.flow_rate_bounds[1]

    def test_calculate_flow_rate_phase2(self):
        sim = self._make_sim()
        fr = sim.calculate_flow_rate(300 * 3600)  # 300 hours
        assert sim.flow_rate_bounds[0] <= fr <= sim.flow_rate_bounds[1]

    def test_calculate_flow_rate_phase3(self):
        sim = self._make_sim()
        fr = sim.calculate_flow_rate(600 * 3600)  # 600 hours
        assert sim.flow_rate_bounds[0] <= fr <= sim.flow_rate_bounds[1]

    def test_calculate_flow_rate_phase3_high_deviation(self):
        sim = self._make_sim()
        # Extend arrays to fit step
        n = int(600 * 3600 / sim.dt)
        sim.num_steps = n + 10
        sim.biofilm_thickness = np.ones((sim.num_steps, sim.num_cells)) * 2.0  # High deviation
        fr = sim.calculate_flow_rate(600 * 3600)
        assert fr >= sim.flow_rate_bounds[0]

    def test_calculate_flow_rate_phase3_low_deviation(self):
        sim = self._make_sim()
        n = int(600 * 3600 / sim.dt)
        sim.num_steps = n + 10
        sim.biofilm_thickness = np.ones((sim.num_steps, sim.num_cells)) * 1.3  # At optimal
        fr = sim.calculate_flow_rate(600 * 3600)
        assert fr >= sim.flow_rate_bounds[0]

    def test_calculate_flow_rate_phase4(self):
        sim = self._make_sim()
        fr = sim.calculate_flow_rate(900 * 3600)  # 900 hours
        assert sim.flow_rate_bounds[0] <= fr <= sim.flow_rate_bounds[1]

    def test_simulate_step_zero(self):
        sim = self._make_sim()
        sim.simulate_step(0)  # Should return early

    def test_simulate_step_nonzero(self):
        sim = self._make_sim()
        sim.simulate_step(1)
        assert sim.stack_powers[1] > 0

    def test_simulate_step_multiple(self):
        sim = self._make_sim()
        for i in range(1, sim.num_steps):
            sim.simulate_step(i)
        assert sim.stack_powers[-1] > 0

    def test_run_simulation(self):
        sim = self._make_sim()
        sim.run_simulation()
        # After simulation, arrays should be populated
        assert sim.stack_powers[-1] != 0 or sim.stack_powers[1] != 0

    def test_save_data(self, tmp_path):
        sim = self._make_sim()
        sim.run_simulation()
        # Patch os.makedirs and output dirs
        with patch("mfc_optimization_gpu.os.makedirs"):
            with patch("mfc_optimization_gpu.pd.DataFrame.to_csv"):
                with patch("builtins.open", MagicMock()):
                    with patch("mfc_optimization_gpu.json.dump"):
                        ts = sim.save_data()
                        assert ts is not None

    def test_generate_plots(self, tmp_path):
        sim = self._make_sim()
        sim.run_simulation()
        import matplotlib.pyplot as plt
        with patch.object(sim, 'generate_detailed_plots'):
            sim.generate_plots("test_ts")
        plt.close("all")

    def test_generate_detailed_plots(self, tmp_path):
        sim = self._make_sim()
        sim.run_simulation()
        import matplotlib.pyplot as plt
        time_hours = np.arange(sim.num_steps) * sim.dt / 3600
        sim.generate_detailed_plots("test_ts", time_hours)
        plt.close("all")


    def test_real_init(self):
        """Test real __init__ with use_gpu=False and small simulation."""
        with patch("mfc_optimization_gpu.os.makedirs"):
            sim = mog.MFCOptimizationSimulation(use_gpu=False)
        assert sim.num_cells == 5
        assert sim.use_gpu is False
        assert sim.num_steps > 0

    def test_calculate_flow_rate_phase3_step_zero(self):
        sim = self._make_sim()
        # Set dt very large so current_step = int(time_seconds / dt) = 0
        sim.dt = 3_000_000.0  # Larger than 600*3600=2160000
        fr = sim.calculate_flow_rate(600 * 3600)
        assert sim.flow_rate_bounds[0] <= fr <= sim.flow_rate_bounds[1]

    def test_calculate_flow_rate_phase3_no_biofilm(self):
        sim = self._make_sim()
        # Remove biofilm_thickness to test else branch
        del sim.biofilm_thickness
        fr = sim.calculate_flow_rate(600 * 3600)
        assert sim.flow_rate_bounds[0] <= fr <= sim.flow_rate_bounds[1]

    def test_run_simulation_gpu_to_cpu(self):
        sim = self._make_sim(use_gpu=True)
        sim.run_simulation()
        # After simulation with use_gpu=True, arrays should be converted


class TestMain:
    def test_main(self):
        with patch("mfc_optimization_gpu.os.makedirs"):
            with patch.object(mog.MFCOptimizationSimulation, 'run_simulation'):
                with patch.object(mog.MFCOptimizationSimulation, 'save_data', return_value="ts"):
                    with patch.object(mog.MFCOptimizationSimulation, 'generate_plots'):
                        mog.main()
