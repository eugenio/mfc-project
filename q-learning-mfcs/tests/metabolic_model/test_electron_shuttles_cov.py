"""Gap coverage tests for electron_shuttles.py -- covers missing lines."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from metabolic_model.electron_shuttles import ElectronShuttleModel, ShuttleType


class TestShuttleDiffusion(unittest.TestCase):
    """Cover lines 272-275: calculate_shuttle_diffusion."""

    def test_diffusion_flux(self):
        model = ElectronShuttleModel()
        flux = model.calculate_shuttle_diffusion(
            concentration_gradient=1.0,
            shuttle_type=ShuttleType.RIBOFLAVIN,
            distance=0.001,
        )
        self.assertGreater(flux, 0.0)

    def test_diffusion_zero_gradient(self):
        model = ElectronShuttleModel()
        flux = model.calculate_shuttle_diffusion(
            concentration_gradient=0.0,
            shuttle_type=ShuttleType.FLAVIN_MONONUCLEOTIDE,
            distance=0.001,
        )
        self.assertAlmostEqual(flux, 0.0)


class TestEstimateOptimalConcentrationEdgeCases(unittest.TestCase):
    """Cover lines 451-453, 457: negative driving force and zero flux branches."""

    def test_negative_driving_force(self):
        """When electrode_potential < redox_potential, driving_force <= 0 -> line 455."""
        model = ElectronShuttleModel()
        optimal = model.estimate_optimal_shuttle_concentration(
            target_current=0.01,
            electrode_potential=-0.5,  # Below all shuttle potentials
            volume=0.001,
            area=0.01,
        )
        self.assertIsInstance(optimal, dict)
        # With negative driving force, concentrations should be 0
        for val in optimal.values():
            self.assertAlmostEqual(val, 0.0)

    def test_very_high_target(self):
        """Very high target flux -> denominator <= 0 -> 100.0 max."""
        model = ElectronShuttleModel()
        optimal = model.estimate_optimal_shuttle_concentration(
            target_current=1000.0,  # Very high
            electrode_potential=0.5,
            volume=0.001,
            area=0.01,
        )
        self.assertIsInstance(optimal, dict)
        # Some shuttles may hit max of 100.0
        has_max = any(v == 100.0 for v in optimal.values())
        self.assertTrue(has_max)

    def test_zero_target_current(self):
        """target_current=0 -> target_flux=0 -> shuttle_target_flux=0 -> line 457."""
        model = ElectronShuttleModel()
        optimal = model.estimate_optimal_shuttle_concentration(
            target_current=0.0,
            electrode_potential=0.5,
            volume=0.001,
            area=0.01,
        )
        self.assertIsInstance(optimal, dict)
        for val in optimal.values():
            self.assertAlmostEqual(val, 0.0)

    def test_factor_zero_via_underflow(self):
        """Force factor=0 via floating point underflow to cover line 453.

        When driving_force is ~1e-300, exp(-df/0.025) underflows to 1.0
        and factor = 1 - 1.0 = 0.0, triggering the factor <= 0 branch.
        """
        model = ElectronShuttleModel()
        # Set all shuttle redox potentials to 0.0 and use electrode_potential=1e-300
        # This gives driving_force = 1e-300 > 0 but factor = 0.0
        for shuttle in model.shuttles.values():
            shuttle.redox_potential = 0.0
        optimal = model.estimate_optimal_shuttle_concentration(
            target_current=0.01,
            electrode_potential=1e-300,
            volume=0.001,
            area=0.01,
        )
        self.assertIsInstance(optimal, dict)
        # With factor=0, all should get 0.0
        for val in optimal.values():
            self.assertAlmostEqual(val, 0.0)


if __name__ == "__main__":
    unittest.main()
