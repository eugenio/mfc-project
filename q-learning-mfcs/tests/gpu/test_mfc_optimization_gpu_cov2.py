"""Extended coverage tests for mfc_optimization_gpu module.

Covers remaining edge cases and branches not hit by existing test suite.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mfc_optimization_gpu as mog


@pytest.mark.coverage_extra
class TestMFCOptimizationGPUExtraCoverage:
    """Additional tests to cover any remaining lines in mfc_optimization_gpu.py."""

    def _make_sim(self, use_gpu=False, num_cells=3, num_steps=10):
        """Create a compact simulation for testing."""
        with patch.object(
            mog.MFCOptimizationSimulation,
            "__init__",
            lambda self_, ug=False: None,
        ):
            sim = mog.MFCOptimizationSimulation.__new__(
                mog.MFCOptimizationSimulation,
            )
        sim.use_gpu = use_gpu
        sim.num_cells = num_cells
        sim.dt = 10.0
        sim.total_time = num_steps * sim.dt
        sim.num_steps = num_steps
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

    # ---- GPU branches for biofilm_factor, reaction_rate, update_cell ----

    def test_biofilm_factor_gpu_branch(self):
        """Cover gpu_accelerator.abs path in biofilm_factor (lines 97-98)."""
        sim = self._make_sim(use_gpu=True)
        # The GPU_AVAILABLE flag is False, so use_gpu is always False
        # Test with explicit use_gpu set to True to cover the branch
        factor = sim.biofilm_factor(np.array([0.5, 1.3, 2.5]))
        assert all(f >= 1.0 for f in np.atleast_1d(factor))

    def test_reaction_rate_gpu_branch_near_optimal(self):
        """Cover gpu_accelerator.where in reaction_rate (lines 112-116)."""
        sim = self._make_sim(use_gpu=True)
        # Biofilm near optimal to trigger enhancement
        rate = sim.reaction_rate(np.float64(1.0), np.float64(1.35))
        assert float(rate) > 0

    def test_reaction_rate_gpu_branch_far_from_optimal(self):
        """Cover gpu_accelerator.where else branch in reaction_rate."""
        sim = self._make_sim(use_gpu=True)
        rate = sim.reaction_rate(np.float64(1.0), np.float64(3.0))
        assert float(rate) > 0

    def test_update_cell_gpu_branch(self):
        """Cover gpu_accelerator paths in update_cell (lines 137-163)."""
        sim = self._make_sim(use_gpu=True)
        result = sim.update_cell(
            0,
            np.float64(1.56),
            np.float64(20.0e-6),
            np.float64(1.3),
        )
        assert float(result["voltage"]) > 0
        assert float(result["power"]) >= 0

    def test_update_biofilm_gpu_all_branches(self):
        """Cover gpu_accelerator.where and clip in update_biofilm (lines 199-227)."""
        sim = self._make_sim(use_gpu=True)

        # Above optimal
        sim.biofilm_thickness[0, :] = 2.0
        sim.acetate_concentrations[0, :] = 1.0
        sim.flow_rates[0] = 20.0e-6
        sim.update_biofilm(1, 10.0)
        assert all(
            0.5 <= float(sim.biofilm_thickness[1, i]) <= 3.0
            for i in range(sim.num_cells)
        )

        # Below optimal * 0.8
        sim.biofilm_thickness[0, :] = 0.5
        sim.update_biofilm(1, 10.0)

        # At optimal
        sim.biofilm_thickness[0, :] = 1.3
        sim.update_biofilm(1, 10.0)

    def test_calculate_objective_function_gpu_branch(self):
        """Cover gpu_accelerator.mean and abs in objective func (lines 236-241)."""
        sim = self._make_sim(use_gpu=True)
        sim.stack_powers[1] = 3.0
        sim.biofilm_thickness[1, :] = 1.3
        sim.acetate_concentrations[1, :] = 0.3
        obj = sim.calculate_objective_function(1)
        assert float(obj) >= 0

    # ---- Flow rate phase edge cases ----

    def test_calculate_flow_rate_phase1_boundary(self):
        """Cover phase 1 at exactly t=0 and t=200h."""
        sim = self._make_sim()
        fr0 = sim.calculate_flow_rate(0)
        assert sim.flow_rate_bounds[0] <= fr0 <= sim.flow_rate_bounds[1]
        fr200 = sim.calculate_flow_rate(200 * 3600)
        assert sim.flow_rate_bounds[0] <= fr200 <= sim.flow_rate_bounds[1]

    def test_calculate_flow_rate_phase2_boundary(self):
        """Cover phase 2 at exactly t=200h and t=500h."""
        sim = self._make_sim()
        fr = sim.calculate_flow_rate(200.001 * 3600)
        assert sim.flow_rate_bounds[0] <= fr <= sim.flow_rate_bounds[1]
        fr500 = sim.calculate_flow_rate(500 * 3600)
        assert sim.flow_rate_bounds[0] <= fr500 <= sim.flow_rate_bounds[1]

    def test_calculate_flow_rate_phase3_step_at_boundary(self):
        """Cover phase 3 when current_step is exactly num_steps - 1."""
        sim = self._make_sim()
        # Create arrays that are large enough for the step
        large_steps = int(700 * 3600 / sim.dt) + 5
        sim.num_steps = large_steps
        sim.biofilm_thickness = np.ones((sim.num_steps, sim.num_cells)) * 1.5
        fr = sim.calculate_flow_rate(700 * 3600)
        assert fr >= sim.flow_rate_bounds[0]

    def test_calculate_flow_rate_phase4_boundary(self):
        """Cover phase 4 at boundary t=800h."""
        sim = self._make_sim()
        fr = sim.calculate_flow_rate(800.001 * 3600)
        assert sim.flow_rate_bounds[0] <= fr <= sim.flow_rate_bounds[1]

    # ---- run_simulation with GPU to_cpu conversion ----

    def test_run_simulation_with_gpu_to_cpu_conversion(self):
        """Cover the gpu_accelerator.to_cpu calls in run_simulation (lines 388-405)."""
        sim = self._make_sim(use_gpu=True)
        sim.run_simulation()
        # After simulation, arrays should be numpy
        assert isinstance(sim.stack_powers, np.ndarray)

    # ---- simulate_step full coverage ----

    def test_simulate_step_all_steps(self):
        """Run all steps to cover all simulate_step branches."""
        sim = self._make_sim()
        for step in range(sim.num_steps):
            sim.simulate_step(step)
        # Ensure substrate utilization is calculated
        assert sim.substrate_utilizations[1] != 0 or sim.substrate_utilizations[-1] != 0

    # ---- save_data thorough coverage ----

    def test_save_data_creates_correct_structure(self, tmp_path):
        """Cover save_data JSON and CSV paths (lines 407-493)."""
        sim = self._make_sim()
        for step in range(sim.num_steps):
            sim.simulate_step(step)

        # Capture the json.dump call to verify structure
        import json

        captured_json = {}

        def capture_json(data, f, **kwargs):
            captured_json.update(data)

        with patch("mfc_optimization_gpu.os.makedirs"):
            with patch("mfc_optimization_gpu.pd.DataFrame.to_csv"):
                with patch("builtins.open", MagicMock()):
                    with patch("mfc_optimization_gpu.json.dump", side_effect=capture_json):
                        ts = sim.save_data()
                        assert ts is not None
                        assert "simulation_info" in captured_json
                        assert "parameters" in captured_json
                        assert "results" in captured_json

    # ---- generate_plots and generate_detailed_plots ----

    def test_generate_plots_full(self):
        """Cover the entire generate_plots method including detailed plots."""
        import matplotlib.pyplot as plt

        sim = self._make_sim()
        for step in range(sim.num_steps):
            sim.simulate_step(step)
        sim.generate_plots("test_ts_full")
        plt.close("all")

    def test_generate_detailed_plots_standalone(self):
        """Cover generate_detailed_plots as standalone call."""
        import matplotlib.pyplot as plt

        sim = self._make_sim()
        for step in range(sim.num_steps):
            sim.simulate_step(step)
        time_hours = np.arange(sim.num_steps) * sim.dt / 3600
        sim.generate_detailed_plots("test_ts_detail", time_hours)
        plt.close("all")

    # ---- main function ----

    def test_main_function(self):
        """Cover main() function (lines 763-778)."""
        with patch("mfc_optimization_gpu.os.makedirs"):
            with patch.object(
                mog.MFCOptimizationSimulation, "run_simulation"
            ):
                with patch.object(
                    mog.MFCOptimizationSimulation,
                    "save_data",
                    return_value="ts123",
                ):
                    with patch.object(
                        mog.MFCOptimizationSimulation, "generate_plots"
                    ):
                        mog.main()

    # ---- Module-level globals ----

    def test_module_level_gpu_accelerator_exists(self):
        """Cover module-level gpu_accelerator and GPU_AVAILABLE (lines 23-24)."""
        assert hasattr(mog, "gpu_accelerator")
        assert hasattr(mog, "GPU_AVAILABLE")
        assert isinstance(mog.GPU_AVAILABLE, bool)

    # ---- Real init ----

    def test_real_init_creates_directories(self):
        """Cover real __init__ os.makedirs calls (lines 67-68)."""
        with patch("mfc_optimization_gpu.os.makedirs") as mock_makedirs:
            sim = mog.MFCOptimizationSimulation(use_gpu=False)
            assert mock_makedirs.call_count >= 2
            assert sim.num_cells == 5

    def test_real_init_with_gpu_flag(self):
        """Cover use_gpu logic in __init__ (line 30)."""
        with patch("mfc_optimization_gpu.os.makedirs"):
            sim = mog.MFCOptimizationSimulation(use_gpu=True)
            # GPU_AVAILABLE is False, so use_gpu should be False
            assert sim.use_gpu is False

    # ---- Biofilm update edge cases ----

    def test_update_biofilm_numpy_at_optimal(self):
        """Cover the else branch (control_factor=1.0) in update_biofilm (line 210)."""
        sim = self._make_sim(use_gpu=False)
        # Set biofilm exactly at optimal * 0.8 < thickness <= optimal
        sim.biofilm_thickness[0, :] = 1.1  # Between 0.8*1.3=1.04 and 1.3
        sim.acetate_concentrations[0, :] = 1.0
        sim.flow_rates[0] = 20.0e-6
        sim.update_biofilm(1, 10.0)
        # control_factor should be 1.0
        for i in range(sim.num_cells):
            assert 0.5 <= sim.biofilm_thickness[1, i] <= 3.0

    # ---- Objective function edge cases ----

    def test_calculate_objective_high_power(self):
        """Cover power_objective min capping (line 233)."""
        sim = self._make_sim()
        sim.stack_powers[1] = 10.0  # > 5.0, so power_objective = 1.0
        sim.biofilm_thickness[1, :] = 1.3
        sim.acetate_concentrations[1, :] = 0.01
        obj = sim.calculate_objective_function(1)
        assert obj >= 0

    def test_calculate_objective_high_biofilm_deviation(self):
        """Cover biofilm_objective max capping (line 248)."""
        sim = self._make_sim()
        sim.stack_powers[1] = 1.0
        sim.biofilm_thickness[1, :] = 3.0  # Far from optimal
        sim.acetate_concentrations[1, :] = 1.0
        obj = sim.calculate_objective_function(1)
        assert obj >= 0

    def test_calculate_objective_substrate_high_util(self):
        """Cover substrate_objective capping (line 256)."""
        sim = self._make_sim()
        sim.stack_powers[1] = 2.0
        sim.biofilm_thickness[1, :] = 1.3
        sim.acetate_concentrations[1, -1] = 0.01  # Very low -> high utilization
        obj = sim.calculate_objective_function(1)
        assert obj >= 0

    # ---- Update cell with minimum clamping ----

    def test_update_cell_low_concentration(self):
        """Cover np.maximum clamping at 0.01 (line 143-146)."""
        sim = self._make_sim(use_gpu=False)
        # Very low inlet concentration to trigger clamping
        result = sim.update_cell(0, 0.02, 5.0e-6, 1.3)
        assert result["outlet_concentration"] >= 0.01

    def test_update_cell_low_voltage_clamping(self):
        """Cover np.maximum clamping at 0.1 for voltage (lines 169-172)."""
        sim = self._make_sim(use_gpu=False)
        # Large biofilm to cause large voltage loss
        result = sim.update_cell(0, 0.02, 5.0e-6, 10.0)
        assert result["voltage"] >= 0.1
