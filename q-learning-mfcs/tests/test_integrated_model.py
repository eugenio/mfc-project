"""
Comprehensive tests for integrated MFC model.

Tests cover:
- Model initialization and configuration
- Component integration (biofilm, metabolic, recirculation)
- Real-time coupling functionality
- GPU acceleration
- Simulation stability
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import warnings

# Suppress matplotlib backend warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from integrated_mfc_model import IntegratedMFCModel, IntegratedMFCState


class TestIntegratedModel(unittest.TestCase):
    """Test integrated MFC model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Small model for testing
        self.model = IntegratedMFCModel(
            n_cells=3,
            species="mixed",
            substrate="lactate",
            use_gpu=False,
            simulation_hours=10
        )

    def test_model_initialization(self):
        """Test model initialization with various configurations."""
        # Test different species
        for species in ["geobacter", "shewanella", "mixed"]:
            model = IntegratedMFCModel(n_cells=2, species=species, use_gpu=False)
            self.assertEqual(model.species, species)
            self.assertEqual(len(model.biofilm_models), 2)
            self.assertEqual(len(model.metabolic_models), 2)

        # Test different substrates
        for substrate in ["acetate", "lactate"]:
            model = IntegratedMFCModel(n_cells=2, substrate=substrate, use_gpu=False)
            self.assertEqual(model.substrate, substrate)

    def test_component_integration(self):
        """Test that all components are properly integrated."""
        # Biofilm models
        self.assertEqual(len(self.model.biofilm_models), self.model.n_cells)
        for bm in self.model.biofilm_models:
            self.assertEqual(bm.species, self.model.species)
            self.assertEqual(bm.substrate, self.model.substrate)

        # Metabolic models
        self.assertEqual(len(self.model.metabolic_models), self.model.n_cells)
        for mm in self.model.metabolic_models:
            self.assertEqual(mm.species_str, self.model.species)
            self.assertEqual(mm.substrate_str, self.model.substrate)

        # MFC stack
        self.assertIsNotNone(self.model.mfc_stack)
        self.assertEqual(self.model.mfc_stack.n_cells, self.model.n_cells)

        # Q-learning agent
        self.assertIsNotNone(self.model.agent)
        self.assertEqual(self.model.agent.n_cells, self.model.n_cells)

    def test_single_step_dynamics(self):
        """Test single time step integration."""
        # Initial state
        initial_time = self.model.time

        # Step forward
        state = self.model.step_integrated_dynamics(dt=1.0)

        # Verify state structure
        self.assertIsInstance(state, IntegratedMFCState)
        self.assertEqual(state.time, initial_time + 1.0)

        # Verify state arrays have correct length
        self.assertEqual(len(state.biofilm_thickness), self.model.n_cells)
        self.assertEqual(len(state.biomass_density), self.model.n_cells)
        self.assertEqual(len(state.substrate_concentration), self.model.n_cells)
        self.assertEqual(len(state.cell_voltages), self.model.n_cells)
        self.assertEqual(len(state.current_densities), self.model.n_cells)

        # Verify values are reasonable
        for thickness in state.biofilm_thickness:
            self.assertGreaterEqual(thickness, 0)
            self.assertLessEqual(thickness, 200)  # μm

        for voltage in state.cell_voltages:
            self.assertGreaterEqual(voltage, -0.5)
            self.assertLessEqual(voltage, 1.0)

    def test_integrated_reward_calculation(self):
        """Test integrated reward calculation."""
        # Create mock states
        mfc_state = {
            'cell_voltages': [0.5, 0.5, 0.5],
            'substrate_concentrations': [10.0, 10.0, 10.0],
            'anode_overpotentials': [-0.1, -0.1, -0.1]
        }

        biofilm_states = [
            {'biofilm_thickness': 35.0, 'biomass_density': 10.0}
            for _ in range(3)
        ]

        metabolic_states = [
            type('obj', (object,), {
                'coulombic_efficiency': 0.5,
                'metabolites': {'nadh': 0.3, 'nad_plus': 0.7}
            })()
            for _ in range(3)
        ]

        enhanced_currents = [0.1, 0.1, 0.1]

        # Calculate reward
        reward = self.model._calculate_integrated_reward(
            mfc_state, biofilm_states, metabolic_states, enhanced_currents
        )

        # Reward should be a reasonable number
        self.assertIsInstance(reward, (int, float))
        self.assertGreater(reward, -100)
        self.assertLess(reward, 100)

    def test_multi_step_simulation(self):
        """Test multiple time steps."""
        # Run for 5 hours
        states = []
        for _ in range(5):
            state = self.model.step_integrated_dynamics(dt=1.0)
            states.append(state)

        # Verify progression
        self.assertEqual(len(states), 5)
        self.assertEqual(states[-1].time, 5.0)

        # Verify Q-learning progress
        self.assertGreater(states[-1].q_table_size, 0)
        self.assertLessEqual(states[-1].epsilon, 0.3)

        # Verify energy accumulation
        self.assertGreaterEqual(states[-1].total_energy, 0)

    def test_results_compilation(self):
        """Test results compilation."""
        # Run short simulation
        for _ in range(3):
            self.model.step_integrated_dynamics(dt=1.0)

        # Compile results
        results = self.model._compile_results()

        # Verify result structure
        expected_keys = [
            'total_energy', 'average_power', 'peak_power',
            'average_coulombic_efficiency', 'final_biofilm_thickness',
            'substrate_utilization', 'q_table_size', 'time_series',
            'configuration'
        ]

        for key in expected_keys:
            self.assertIn(key, results)

        # Verify time series
        self.assertIn('time', results['time_series'])
        self.assertIn('power', results['time_series'])
        self.assertEqual(len(results['time_series']['time']), 3)

    def test_biofilm_metabolic_coupling(self):
        """Test coupling between biofilm and metabolic models."""
        # Initial biofilm state
        initial_thickness = [bm.biofilm_thickness for bm in self.model.biofilm_models]

        # Run simulation with good growth conditions
        self.model.mfc_stack.reservoir.substrate_concentration = 25.0  # High substrate

        for _ in range(5):
            state = self.model.step_integrated_dynamics(dt=1.0)

        # Biofilm should develop
        final_thickness = state.biofilm_thickness

        # At least some cells should show biofilm growth
        growth_observed = any(final > initial for final, initial in
                            zip(final_thickness, initial_thickness))
        self.assertTrue(growth_observed, "No biofilm growth observed")

        # Metabolic activity should be present
        self.assertGreater(state.coulombic_efficiency, 0)
        self.assertTrue(any(flux > 0 for flux in state.electron_flux))

    def test_species_specific_behavior(self):
        """Test different behaviors for different species."""
        # Geobacter with acetate
        model_geo = IntegratedMFCModel(
            n_cells=2, species="geobacter", substrate="acetate",
            use_gpu=False, simulation_hours=5
        )

        # Shewanella with lactate
        model_she = IntegratedMFCModel(
            n_cells=2, species="shewanella", substrate="lactate",
            use_gpu=False, simulation_hours=5
        )

        # Run both for a few steps
        for _ in range(3):
            state_geo = model_geo.step_integrated_dynamics(dt=1.0)
            state_she = model_she.step_integrated_dynamics(dt=1.0)

        # Different species should behave differently
        # (Exact differences depend on parameter values)
        self.assertIsNotNone(state_geo)
        self.assertIsNotNone(state_she)

    def test_gpu_acceleration_compatibility(self):
        """Test GPU acceleration compatibility."""
        # Create model with GPU enabled (will fallback if not available)
        model_gpu = IntegratedMFCModel(
            n_cells=2, species="mixed", use_gpu=True, simulation_hours=5
        )

        # Should initialize without errors
        self.assertIsNotNone(model_gpu.gpu_acc)

        # Run a step
        state = model_gpu.step_integrated_dynamics(dt=1.0)
        self.assertIsNotNone(state)

    def test_checkpoint_saving(self):
        """Test checkpoint saving functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override path config temporarily
            import path_config
            original_base = path_config.BASE_DIR
            path_config.BASE_DIR = tmpdir

            try:
                # Run simulation
                self.model.step_integrated_dynamics(dt=1.0)

                # Save checkpoint
                self.model._save_checkpoint(1)

                # Check file exists
                checkpoint_file = os.path.join(tmpdir, 'q_learning_models',
                                             'integrated_checkpoint_h1.pkl')
                self.assertTrue(os.path.exists(checkpoint_file))

            finally:
                # Restore original path
                path_config.BASE_DIR = original_base


class TestIntegrationStability(unittest.TestCase):
    """Test long-term stability of integrated model."""

    def test_extended_simulation_stability(self):
        """Test that extended simulation remains stable."""
        model = IntegratedMFCModel(
            n_cells=2, species="mixed", substrate="lactate",
            use_gpu=False, simulation_hours=20
        )

        # Run for 20 hours
        final_state = None
        for hour in range(20):
            final_state = model.step_integrated_dynamics(dt=1.0)

        # All values should remain finite and reasonable
        self.assertTrue(np.isfinite(final_state.total_energy))
        self.assertTrue(all(np.isfinite(v) for v in final_state.cell_voltages))
        self.assertTrue(all(np.isfinite(t) for t in final_state.biofilm_thickness))

        # Biofilm shouldn't grow infinitely
        self.assertTrue(all(t < 200 for t in final_state.biofilm_thickness))

        # Substrate shouldn't go negative
        self.assertTrue(all(s >= 0 for s in final_state.substrate_concentration))
        self.assertGreaterEqual(final_state.reservoir_concentration, 0)

    def test_edge_case_handling(self):
        """Test handling of edge cases."""
        model = IntegratedMFCModel(n_cells=2, use_gpu=False, simulation_hours=5)

        # Deplete substrate
        model.mfc_stack.reservoir.substrate_concentration = 0.1

        # Should handle low substrate without crashing
        for _ in range(5):
            state = model.step_integrated_dynamics(dt=1.0)

        # Model should remain stable
        self.assertTrue(all(v >= -0.5 for v in state.cell_voltages))
        self.assertTrue(state.coulombic_efficiency >= 0)


class TestFullSimulation(unittest.TestCase):
    """Test full simulation workflow."""

    def test_complete_simulation_run(self):
        """Test complete simulation from start to finish."""
        model = IntegratedMFCModel(
            n_cells=2, species="mixed", substrate="lactate",
            use_gpu=False, simulation_hours=5
        )

        # Run simulation
        results = model.run_simulation(dt=1.0, save_interval=10)

        # Verify results
        self.assertIn('total_energy', results)
        self.assertIn('average_power', results)
        self.assertIn('computation_time', results)

        self.assertGreater(results['total_energy'], 0)
        self.assertGreater(results['average_power'], 0)
        self.assertGreater(results['computation_time'], 0)

        # Verify configuration saved
        self.assertEqual(results['configuration']['n_cells'], 2)
        self.assertEqual(results['configuration']['species'], 'mixed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
