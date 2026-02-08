import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest

from cathode_models.base_cathode import (
    BaseCathodeModel,
    ButlerVolmerKinetics,
    CathodeParameters,
)


class ConcreteCathode(BaseCathodeModel):
    """Concrete implementation for testing abstract base class."""

    def _setup_kinetic_parameters(self):
        self.exchange_current_density = 1e-4
        self.transfer_coefficient = 0.5

    def calculate_current_density(self, overpotential, oxygen_conc=None):
        if overpotential <= 0:
            return 0.0
        return self.exchange_current_density * overpotential * 100


class TestCathodeParameters:
    def test_default_values(self):
        params = CathodeParameters()
        assert params.area_m2 == 1e-4
        assert params.temperature_K == 298.15
        assert params.oxygen_concentration == 8.0e-3
        assert params.faraday_constant == 96485.0
        assert params.gas_constant == 8.314
        assert params.ph == 7.0

    def test_custom_values(self):
        params = CathodeParameters(area_m2=2e-4, temperature_K=310.0, ph=6.5)
        assert params.area_m2 == 2e-4
        assert params.temperature_K == 310.0
        assert params.ph == 6.5


class TestBaseCathodeModel:
    def setup_method(self):
        self.params = CathodeParameters()
        self.model = ConcreteCathode(self.params)

    def test_init(self):
        assert self.model.area_m2 == 1e-4
        assert self.model.temperature_K == 298.15

    def test_calculate_equilibrium_potential_defaults(self):
        E_eq = self.model.calculate_equilibrium_potential()
        assert isinstance(E_eq, float)
        assert 0.0 < E_eq < 1.5

    def test_calculate_equilibrium_potential_custom(self):
        E_eq = self.model.calculate_equilibrium_potential(oxygen_conc=4e-3, ph=5.0)
        assert isinstance(E_eq, float)

    def test_calculate_equilibrium_potential_none_args(self):
        E_eq = self.model.calculate_equilibrium_potential(None, None)
        assert isinstance(E_eq, float)

    def test_calculate_overpotential(self):
        eta = self.model.calculate_overpotential(0.2)
        assert isinstance(eta, float)

    def test_calculate_overpotential_custom(self):
        eta = self.model.calculate_overpotential(0.2, oxygen_conc=4e-3, ph=5.0)
        assert isinstance(eta, float)

    def test_calculate_current(self):
        current = self.model.calculate_current(0.3)
        assert current >= 0.0

    def test_calculate_current_zero_overpotential(self):
        current = self.model.calculate_current(0.0)
        assert current == 0.0

    def test_calculate_power_loss(self):
        loss = self.model.calculate_power_loss(0.3)
        assert loss >= 0.0

    def test_calculate_power_loss_custom(self):
        loss = self.model.calculate_power_loss(0.3, oxygen_conc=4e-3)
        assert loss >= 0.0

    def test_calculate_oxygen_consumption_rate(self):
        rate = self.model.calculate_oxygen_consumption_rate(0.3)
        assert rate >= 0.0

    def test_calculate_oxygen_consumption_rate_custom(self):
        rate = self.model.calculate_oxygen_consumption_rate(0.3, oxygen_conc=4e-3)
        assert rate >= 0.0

    def test_update_temperature(self):
        self.model.update_temperature(310.0)
        assert self.model.temperature_K == 310.0
        assert self.model.params.temperature_K == 310.0

    def test_update_area(self):
        self.model.update_area(2e-4)
        assert self.model.area_m2 == 2e-4
        assert self.model.params.area_m2 == 2e-4

    def test_get_model_info(self):
        info = self.model.get_model_info()
        assert info["model_type"] == "ConcreteCathode"
        assert "area_m2" in info
        assert "area_cm2" in info
        assert "temperature_K" in info
        assert "temperature_C" in info
        assert "oxygen_concentration_mol_L" in info
        assert "oxygen_concentration_mg_L" in info
        assert "ph" in info
        assert "equilibrium_potential_V" in info

    def test_repr(self):
        r = repr(self.model)
        assert "ConcreteCathode" in r
        assert "cm" in r


class TestButlerVolmerKinetics:
    def test_calculate_current_density(self):
        cd = ButlerVolmerKinetics.calculate_current_density(
            exchange_current_density=1e-4,
            transfer_coefficient=0.5,
            overpotential=0.3,
            temperature_K=298.15,
            concentration_ratio=1.0,
        )
        assert isinstance(cd, float)
        assert cd > 0

    def test_calculate_current_density_zero_overpotential(self):
        cd = ButlerVolmerKinetics.calculate_current_density(
            exchange_current_density=1e-4,
            transfer_coefficient=0.5,
            overpotential=0.0,
            temperature_K=298.15,
        )
        assert abs(cd) < 1e-10

    def test_calculate_current_density_negative_overpotential(self):
        cd = ButlerVolmerKinetics.calculate_current_density(
            exchange_current_density=1e-4,
            transfer_coefficient=0.5,
            overpotential=-0.3,
            temperature_K=298.15,
        )
        assert cd < 0

    def test_calculate_tafel_current(self):
        cd = ButlerVolmerKinetics.calculate_tafel_current(
            exchange_current_density=1e-4,
            tafel_slope=0.12,
            overpotential=0.3,
        )
        assert cd > 0

    def test_calculate_tafel_current_zero_overpotential(self):
        cd = ButlerVolmerKinetics.calculate_tafel_current(
            exchange_current_density=1e-4,
            tafel_slope=0.12,
            overpotential=0.0,
        )
        assert cd == 0.0

    def test_calculate_tafel_current_negative_overpotential(self):
        cd = ButlerVolmerKinetics.calculate_tafel_current(
            exchange_current_density=1e-4,
            tafel_slope=0.12,
            overpotential=-0.1,
        )
        assert cd == 0.0

    def test_calculate_tafel_current_with_concentration(self):
        cd = ButlerVolmerKinetics.calculate_tafel_current(
            exchange_current_density=1e-4,
            tafel_slope=0.12,
            overpotential=0.3,
            concentration_ratio=0.5,
        )
        assert cd > 0
