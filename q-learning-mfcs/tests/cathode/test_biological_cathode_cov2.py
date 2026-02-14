"""Additional coverage tests for biological_cathode.py — targeting line 311.

The else branch (current_density = 0.0) is hit when the biofilm resistance
voltage drop exceeds the applied overpotential, clamping effective_overpotential
to 0.0 via jnp.maximum.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from cathode_models.biological_cathode import (
    BiologicalCathodeModel,
    BiologicalParameters,
)
from cathode_models.base_cathode import CathodeParameters


@pytest.mark.coverage_extra
class TestBiologicalCathodeHighResistance:
    """Tests to cover the else branch at line 311 of biological_cathode.py.

    When biofilm_voltage_drop >= overpotential, effective_overpotential is
    clamped to 0.0 and the else branch returns current_density = 0.0.
    """

    def test_current_density_returns_zero_when_biofilm_resistance_exceeds_overpotential(
        self,
    ):
        """A very thick biofilm with extremely low conductivity produces
        enough ohmic drop to swallow a small overpotential entirely.
        """
        params = CathodeParameters(temperature_K=303.15, ph=7.0)

        # Extremely low conductivity -> huge resistance per unit thickness
        # Very thick biofilm -> large biofilm_resistance = thickness / conductivity
        bio_params = BiologicalParameters(
            biofilm_conductivity=1e-12,  # S/m — near-insulating biofilm
            initial_biofilm_thickness=500e-6,  # 500 um — maximum thickness
            exchange_current_density_base=1e-3,  # Higher base i0 to boost kinetic current
            biomass_activity_factor=1e5,  # Amplify exchange current even more
            initial_biomass_density=100.0,  # High biomass density
        )

        model = BiologicalCathodeModel(params, bio_params)

        # biofilm_resistance = thickness / conductivity = 500e-6 / 1e-12 = 5e8 ohm*m^2
        # Even a tiny kinetic current times 5e8 will exceed a small overpotential.
        # Use a small but positive overpotential.
        cd = model.calculate_current_density(0.001)

        # The biofilm voltage drop should exceed the overpotential,
        # so the else branch fires and returns 0.0.
        assert cd == 0.0

    def test_current_density_returns_zero_with_moderate_overpotential_high_resistance(
        self,
    ):
        """Even with moderate overpotential, extremely high resistance
        still triggers the else branch.
        """
        params = CathodeParameters(temperature_K=303.15, ph=7.0)
        bio_params = BiologicalParameters(
            biofilm_conductivity=1e-14,  # Extremely low conductivity
            initial_biofilm_thickness=500e-6,
            exchange_current_density_base=1e-2,
            biomass_activity_factor=1e6,
            initial_biomass_density=200.0,
        )

        model = BiologicalCathodeModel(params, bio_params)

        # Even 0.3 V overpotential should be overwhelmed by resistance
        cd = model.calculate_current_density(0.3)
        assert cd == 0.0

    def test_current_density_nonzero_when_resistance_is_low(self):
        """Sanity check: with normal conductivity, current density is positive."""
        params = CathodeParameters(temperature_K=303.15, ph=7.0)
        bio_params = BiologicalParameters(
            biofilm_conductivity=5e-5,  # Normal conductivity
            initial_biofilm_thickness=10e-6,  # Thin biofilm
        )

        model = BiologicalCathodeModel(params, bio_params)
        cd = model.calculate_current_density(0.3)
        assert cd > 0.0
