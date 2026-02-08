"""Gap coverage tests for membrane_transport.py -- covers missing lines."""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from metabolic_model.membrane_transport import MembraneTransport


class TestInvalidMembraneGrade(unittest.TestCase):
    """Cover lines 128-133: ValueError for unknown membrane grade."""

    def test_unknown_grade(self):
        with self.assertRaises(ValueError) as ctx:
            MembraneTransport("Nafion-999", use_gpu=False)
        self.assertIn("Nafion-999", str(ctx.exception))
        self.assertIn("Available", str(ctx.exception))


class TestOxygenCrossoverGPU(unittest.TestCase):
    """Cover lines 172-176: GPU branch of oxygen crossover."""

    def test_crossover_cpu_explicit(self):
        mt = MembraneTransport("Nafion-117", use_gpu=False)
        flux = mt.calculate_oxygen_crossover(0.001, 0.25, 303.0)
        self.assertGreater(flux, 0.0)

    def test_crossover_gpu_mocked(self):
        """Cover lines 172-176 by mocking GPU acceleration."""
        import numpy as np

        mt = MembraneTransport("Nafion-117", use_gpu=False)
        # Mock GPU availability and acceleration
        mock_gpu = MagicMock()
        mock_gpu.array = lambda x: np.array(x)
        mock_gpu.to_cpu = lambda x: np.array(x).flatten()
        mt.gpu_available = True
        mt.gpu_acc = mock_gpu
        flux = mt.calculate_oxygen_crossover(0.001, 0.25, 303.0)
        self.assertGreater(flux, 0.0)


class TestPotentialDrop(unittest.TestCase):
    """Cover lines 300-308: calculate_potential_drop."""

    def test_basic(self):
        mt = MembraneTransport("Nafion-117", use_gpu=False)
        drop = mt.calculate_potential_drop(
            current_density=1000.0, area=0.01, temperature=303.0
        )
        self.assertGreater(drop, 0.0)

    def test_zero_current(self):
        mt = MembraneTransport("Nafion-117", use_gpu=False)
        drop = mt.calculate_potential_drop(current_density=0.0, area=0.01)
        self.assertAlmostEqual(drop, 0.0)


class TestOxygenConsumptionLoss(unittest.TestCase):
    """Cover line 342: zero substrate_flux -> efficiency_loss = 0."""

    def test_zero_substrate_flux(self):
        mt = MembraneTransport("Nafion-117", use_gpu=False)
        loss = mt.calculate_oxygen_consumption_loss(
            oxygen_flux=1e-6, area=0.01, substrate_flux=0.0
        )
        self.assertAlmostEqual(loss, 0.0)

    def test_normal_substrate_flux(self):
        mt = MembraneTransport("Nafion-117", use_gpu=False)
        loss = mt.calculate_oxygen_consumption_loss(
            oxygen_flux=1e-6, area=0.01, substrate_flux=1e-4
        )
        self.assertGreater(loss, 0.0)
        self.assertLessEqual(loss, 1.0)


class TestGetMassTransportEquations(unittest.TestCase):
    """Cover line 348: get_mass_transport_equations."""

    def test_equations(self):
        mt = MembraneTransport("Nafion-117", use_gpu=False)
        eqs = mt.get_mass_transport_equations()
        self.assertIn("oxygen_flux", eqs)
        self.assertIn("proton_conductivity", eqs)
        self.assertIn("membrane_resistance", eqs)
        self.assertIn("water_transport", eqs)
        self.assertIn("potential_drop", eqs)


class TestOptimizeMembraneThickness(unittest.TestCase):
    """Cover lines 389-397."""

    def test_basic(self):
        mt = MembraneTransport("Nafion-117", use_gpu=False)
        thickness = mt.optimize_membrane_thickness(
            target_resistance=0.1, area=0.01
        )
        self.assertGreater(thickness, 0.0)

    def test_with_temp_rh(self):
        mt = MembraneTransport("Nafion-117", use_gpu=False)
        thickness = mt.optimize_membrane_thickness(
            target_resistance=0.05, area=0.005, temperature=323.0, relative_humidity=80.0
        )
        self.assertGreater(thickness, 0.0)


class TestCrossoverCurrentLoss(unittest.TestCase):
    """Cover lines 415-416."""

    def test_basic(self):
        mt = MembraneTransport("Nafion-117", use_gpu=False)
        loss = mt.calculate_crossover_current_loss(
            oxygen_crossover=1e-6, area=0.01
        )
        self.assertGreater(loss, 0.0)
        # 4 * 1e-6 * 0.01 * 96485 = expected
        expected = 4 * 1e-6 * 0.01 * 96485
        self.assertAlmostEqual(loss, expected, places=3)


class TestGetAvailableMembranes(unittest.TestCase):
    """Cover line 421."""

    def test_available(self):
        membranes = MembraneTransport.get_available_membranes()
        self.assertIn("Nafion-117", membranes)
        self.assertIn("Nafion-115", membranes)
        self.assertIn("Nafion-212", membranes)


if __name__ == "__main__":
    unittest.main()
